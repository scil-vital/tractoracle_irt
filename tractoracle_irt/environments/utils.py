import numpy as np
from dipy.tracking import metrics as tm
from dipy.tracking.streamline import set_number_of_points
from multiprocessing import Pool
from scipy.ndimage import map_coordinates
from nibabel.streamlines.array_sequence import ArraySequence
from typing import Union

from tractoracle_irt.utils.utils import normalize_vectors
from tractoracle_irt.utils.logging import get_logger

LOGGER = get_logger(__name__)

def get_neighborhood_directions(
    radius: float
) -> np.ndarray:
    """ Returns predefined neighborhood directions at exactly `radius` length
        For now: Use the 6 main axes as neighbors directions, plus (0,0,0)
        to keep current position

    Parameters
    ----------
    radius : float
        Distance to neighbors

    Returns
    -------
    directions : `numpy.ndarray` with shape (n_directions, 3)

    Notes
    -----
    Coordinates are in voxel-space
    """
    axes = np.identity(3)
    directions = np.concatenate(([[0, 0, 0]], axes, -axes)) * radius
    return directions


def has_reached_gm(
    streamlines: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.,
    min_nb_steps: int = 10
):
    """ Checks which streamlines have their last coordinates inside a mask and
    are at least longer than a minimum strealine length.

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    mask : 3D `numpy.ndarray`
        3D image defining a stopping mask. The interior of the mask is defined
        by values higher or equal than `threshold` .
    threshold : float
        Voxels with a value higher or equal than this threshold are considered
        as part of the interior of the mask.
    min_length: float
        Minimum streamline length to end

    Returns
    -------
    inside : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline's can end after reaching GM.
    """
    return np.logical_and(is_inside_mask(
        streamlines, mask, threshold),
        np.full(streamlines.shape[0], streamlines.shape[1] > min_nb_steps))


def is_inside_mask(
    streamlines: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.
):
    """ Checks which streamlines have their last coordinates inside a mask.

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    mask : 3D `numpy.ndarray`
        3D image defining a stopping mask. The interior of the mask is defined
        by values higher or equal than `threshold` .
    threshold : float
        Voxels with a value higher or equal than this threshold are considered
        as part of the interior of the mask.

    Returns
    -------
    inside : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline's last coordinate is inside the mask
        or not.
    """
    # Get last streamlines coordinates
    return map_coordinates(
        mask, streamlines[:, -1, :].T - 0.5,
        mode='constant', order=0) >= threshold


def is_outside_mask(
    streamlines: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.
):
    """ Checks which streamlines have their last coordinates outside a mask.

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    mask : 3D `numpy.ndarray`
        3D image defining a stopping mask. The interior of the mask is defined
        by values higher or equal than `threshold` .
    threshold : float
        Voxels with a value higher or equal than this threshold are considered
        as part of the interior of the mask.

    Returns
    -------
    outside : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline's last coordinate is outside the
        mask or not.
    """

    # Get last streamlines coordinates
    return map_coordinates(
        mask, streamlines[:, -1, :].T - 0.5, mode='constant', order=0
    ) < threshold


def is_too_long(streamlines: np.ndarray, max_nb_steps: int):
    """ Checks whether streamlines have exceeded the maximum number of steps

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    max_nb_steps : int
        Maximum number of steps a streamline can have

    Returns
    -------
    too_long : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline is too long or not
    """
    return np.full(streamlines.shape[0], streamlines.shape[1] >= max_nb_steps)


def is_too_curvy(streamlines: np.ndarray, max_theta: float):
    """ Checks whether streamlines have exceeded the maximum angle between the
    last 2 steps

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    max_theta : float
        Maximum angle in degrees that two consecutive segments can have between
        each other.

    Returns
    -------
    too_curvy : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline is too curvy or not
    """
    max_theta_rad = np.deg2rad(max_theta)  # Internally use radian
    if streamlines.shape[1] < 3:
        # Not enough segments to compute curvature
        return np.zeros(streamlines.shape[0], dtype=bool)

    # Compute vectors for the last and before last streamline segments
    u = normalize_vectors(streamlines[:, -1] - streamlines[:, -2])
    v = normalize_vectors(streamlines[:, -2] - streamlines[:, -3])

    # Compute angles
    dot = np.einsum('ij,ij->i', u, v)
    angles = np.arccos(np.clip(dot, -1.0, 1.0))  # Clip to avoid numerical errors
    return angles > max_theta_rad


def winding(nxyz: np.ndarray) -> np.ndarray:
    """ Project lines to best fitting planes. Calculate
    the cummulative signed angle between each segment for each line
    and their previous one

    Adapted from dipy.tracking.metrics.winding to allow multiple
    lines that have the same length

    Parameters
    ------------
    nxyz : np.ndarray of shape (N, M, 3)
        Array representing x,y,z of M points in N tracts.

    Returns
    ---------
    a : np.ndarray
        Total turning angle in degrees for all N tracts.
    """

    directions = np.diff(nxyz, axis=1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    thetas = np.einsum(
        'ijk,ijk->ij', directions[:, :-1], directions[:, 1:]).clip(-1., 1.)
    shape = thetas.shape
    rads = np.arccos(thetas.flatten())
    turns = np.sum(np.reshape(rads, shape), axis=-1)
    return np.rad2deg(turns)

    # # This is causing a major slowdown :(
    # U, s, V = np.linalg.svd(nxyz-np.mean(nxyz, axis=1, keepdims=True), 0)

    # Up = U[:, :, 0:2]
    # # Has to be a better way than iterare over all tracts
    # diags = np.stack([np.diag(sp[0:2]) for sp in s], axis=0)
    # proj = np.einsum('ijk,ilk->ijk', Up, diags)

    # v0 = proj[:, :-1]
    # v1 = proj[:, 1:]
    # v = np.einsum('ijk,ijk->ij', v0, v1) / (
    #     np.linalg.norm(v0, axis=-1, keepdims=True)[..., 0] *
    #     np.linalg.norm(v1, axis=-1, keepdims=True)[..., 0])
    # np.clip(v, -1, 1, out=v)
    # shape = v.shape
    # rads = np.arccos(v.flatten())
    # turns = np.sum(np.reshape(rads, shape), axis=-1)

    # return np.rad2deg(turns)


def is_looping(streamlines: np.ndarray, loop_threshold: float):
    """ Checks whether streamlines are looping

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    looping_threshold: float
        Maximum angle in degrees for the whole streamline

    Returns
    -------
    too_curvy : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline is too curvy or not
    """

    clean_ids = remove_loops_and_sharp_turns(
        streamlines, loop_threshold, num_processes=8)
    mask = np.full(streamlines.shape[0], True)
    mask[clean_ids] = False
    return mask


def remove_loops_and_sharp_turns(streamlines,
                                 max_angle,
                                 num_processes=1):
    """
    Remove loops and sharp turns from a list of streamlines.
    Parameters
    ----------
    streamlines: list of ndarray
        The list of streamlines from which to remove loops and sharp turns.
    max_angle: float
        Maximal winding angle a streamline can have before
        being classified as a loop.
    use_qb: bool
        Set to True if the additional QuickBundles pass is done.
        This will help remove sharp turns. Should only be used on
        bundled streamlines, not on whole-brain tractograms.
    qb_threshold: float
        Quickbundles distance threshold, only used if use_qb is True.
    qb_seed: int
        Seed to initialize randomness in QuickBundles

    Returns
    -------
    list: the ids of clean streamlines
        Only the ids are returned so proper filtering can be done afterwards
    """

    ids = []
    pool = Pool(num_processes)
    windings = pool.map(tm.winding, streamlines)
    pool.close()
    ids = list(np.where(np.array(windings) < max_angle)[0])

    return ids

def resample_streamlines_if_needed(streamlines: Union[ArraySequence, list, np.ndarray],
                                   nb_points: int) -> Union[ArraySequence, np.ndarray]:
    assert nb_points > 3, "nb_points must be greater than 1"

    #print("Streamlines shape: ", streamlines.shape)

    if isinstance(streamlines, ArraySequence) or isinstance(streamlines, list):
        if not np.all([len(sl) == nb_points for sl in streamlines]):
            # print("resample ArraySequence")
            data = set_number_of_points(streamlines, nb_points)
        else:
            data = streamlines
    elif isinstance(streamlines, np.ndarray):
        if streamlines.shape[1] != nb_points:
            # print("resample np.ndarray from {} to {}".format(streamlines.shape[1], nb_points))
            data = ArraySequence(streamlines)
            data = set_number_of_points(data, nb_points)
        else:
            # print("not resampling np.ndarray")
            data = streamlines
    else:
        raise ValueError("streamlines must be a list, ArraySequence or "
                            "np.ndarray")
    
    return data

def fix_streamlines_length(streamlines: np.ndarray,
                           current_streamlines_length: int,
                           streamline_nb_points: int):
    """
    This function is used to fix the length of the streamlines. This is
    especially useful when providing streamlines to a neural network with
    a fixed input size.

    1. If the provided streamlines are shorter than streamline_nb_points,
    pad them with zeros.
    2. If the provided streamlines are longer than streamline_nb_points,
    downsample them to streamline_nb_points.
    3. If the streamlines are contained within already padded zeros, then
    just return the first streamline_nb_points.

    Parameters
    ----------
    streamlines: ndarray
        The list of streamlines from which to remove loops and sharp turns.
    current_streamlines_length: int
        The length of all streamlines provided in the first parameter.
    streamline_nb_points: int
        Target length of the desired outputted streamlines.

    Returns
    -------
    ndarray of shape (nb_streamlines, streamline_nb_points, 3)
        The padded/downsampled streamlines.
    """

    if isinstance(streamlines, ArraySequence):
        # This is probably very slow
        numpy_streamlines = np.zeros((len(streamlines), current_streamlines_length, 3))
        for i, sl in enumerate(streamlines):
            streamlines[i] = fix_streamlines_length(sl, current_streamlines_length, streamline_nb_points)
        streamlines = np.array(streamlines) 

    # Input is shorter: padding.
    if streamlines.shape[1] < streamline_nb_points:
        resampled_next_streamlines = np.zeros((streamlines.shape[0], streamline_nb_points, 3))
        resampled_next_streamlines[:, :streamlines.shape[1]] = streamlines
    # Downsample
    elif current_streamlines_length > streamline_nb_points:
        resampled_next_streamlines = np.asarray(set_number_of_points(list(streamlines[:, :current_streamlines_length]), streamline_nb_points))
    # Input is longer, but the provided streamlines are already padded with zeros
    # (but the input is possibly still longer than streamline_nb_points).
    else:
        resampled_next_streamlines = streamlines[:, :streamline_nb_points]

    assert resampled_next_streamlines.shape[1] == streamline_nb_points
    return resampled_next_streamlines
