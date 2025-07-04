# -*- coding: utf-8 -*-
import logging

import torch

from tractoracle_irt.utils.dwi_ml import \
    extend_coordinates_with_neighborhood

B1 = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],
               [-1, 0, 0, 0, 1, 0, 0, 0],
               [-1, 0, 1, 0, 0, 0, 0, 0],
               [-1, 1, 0, 0, 0, 0, 0, 0],
               [1, 0, -1, 0, -1, 0, 1, 0],
               [1, -1, -1, 1, 0, 0, 0, 0],
               [1, -1, 0, 0, -1, 1, 0, 0],
               [-1, 1, 1, -1, 1, -1, -1, 1]], dtype=torch.float32)

# We will use the 8 voxels surrounding current position to interpolate a
# value. See ref https://spie.org/samples/PM159.pdf. The point p000 = [0, 0, 0]
# is the bottom corner of the current position (using floor).
idx_box = torch.tensor([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1]], dtype=torch.float32)


def torch_nearest_neighbor_interpolation(volume: torch.Tensor,
                                         coords_vox_corner: torch.Tensor):
    """
    Parameters
    ----------
    volume : torch.Tensor with 3D or 4D shape
        The input volume to interpolate from
    coords_vox_corner : torch.Tensor with shape (N,3)
        The coordinates where to interpolate. (Origin = corner, space = vox).

    Returns
    -------
    output : torch.Tensor with shape (N, #modalities)
        The list of interpolated values
    """
    # Coord corner: First voxel is coordinates from 0 to 0.99.
    # Using floor value = becomes 0 = index.
    coords_vox_corner = torch.floor(coords_vox_corner).to(dtype=torch.long)

    return volume[coords_vox_corner[:, 0],
                  coords_vox_corner[:, 1],
                  coords_vox_corner[:, 2]]


def torch_trilinear_interpolation(volume: torch.Tensor,
                                  coords_vox_corner: torch.Tensor,
                                  clear_cache=True):
    """Evaluates the data volume at given coordinates using trilinear
    interpolation on a torch tensor.

    Interpolation is done using the device on which the volume is stored.

    * Note. There is a function in torch:
    torch.nn.functional.interpolation with mode trilinear
    But it resamples volumes, not coordinates.

    Parameters
    ----------
    volume : torch.Tensor with 3D or 4D shape
        The input volume to interpolate from
    coords_vox_corner : torch.Tensor with shape (N,3)
        The coordinates where to interpolate. (Origin = corner, space = vox).
    clear_cache : bool
        If True, will clear the cache after interpolation. This can be useful
        to save memory, but will slow down the function.

    Returns
    -------
    output : torch.Tensor with shape (N, #modalities)
        The list of interpolated values
    coords_to_idx_clipped: the coords after floor and clipping in box.

    References
    ----------
    [1] https://spie.org/samples/PM159.pdf
    """
    device = volume.device

    # Send data to device
    idx_box_torch = idx_box.to(device)
    B1_torch = B1.to(device)

    if volume.dim() <= 2 or volume.dim() >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    # - indices are the floor of coordinates + idx, boxes with 8 corners around
    #   given coordinates. (Floor means origin = corner)
    # - coords + idx_torch shape -> the box of 8 corners around each coord
    #   reshaped as (-1,3) = [n * 8, 3]
    # - torch needs indices to be cast to long
    # - clip indices to make sure we don't go out-of-bounds
    #   Origin = corner means the minimum is 0.
    #                         the maximum is shape.
    # Ex, for shape 150, last voxel is #149, with possible coords up to 149.99.
    lower = torch.as_tensor([0, 0, 0], device=device)
    upper = torch.as_tensor(volume.shape[:3], device=device) - 1
    idx_box_clipped = torch.min(
        torch.max(
            torch.floor(coords_vox_corner[:, None, :] + idx_box_torch
                        ).reshape((-1, 3)).long(),
            lower),
        upper)

    # Setting Q1 such as in equation 9.9
    d = coords_vox_corner - torch.floor(coords_vox_corner)
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    Q1 = torch.stack([torch.ones_like(dx), dx, dy, dz,
                      dx * dy, dy * dz, dx * dz,
                      dx * dy * dz], dim=0)

    # As of now:
    # B1 = 8x8
    # Q1 = 8 x n (GROS)
    # mult B1 * Q1 = 8 x n
    # overwriting Q1 with mult to try and save space
    if volume.dim() == 3:
        Q1 = torch.mm(B1_torch.t(), Q1)

        # Fetch volume data at indices based on equation 9.11.
        p = volume[idx_box_clipped[:, 0],
                   idx_box_clipped[:, 1],
                   idx_box_clipped[:, 2]]
        # Last dim (-1) = the 8 corners
        p = p.reshape((coords_vox_corner.shape[0], -1)).t()

        # Finding coordinates with equation 9.12a.
        return torch.sum(p * Q1, dim=0)

    elif volume.dim() == 4:
        Q1 = torch.mm(B1_torch.t(), Q1).t()[:, :, None]

        # Fetch volume data at indices
        p = volume[idx_box_clipped[:, 0],
                   idx_box_clipped[:, 1],
                   idx_box_clipped[:, 2], :]
        p = p.reshape((coords_vox_corner.shape[0], 8, volume.shape[-1]))

        # p: of shape n x 8 x features
        # Q1: n x 8 x 1
        # This can save a bit of space.
        if clear_cache:
            torch.cuda.empty_cache()

        # return torch.sum(p * Q1, dim=1)
        # Able to have bigger batches by avoiding 3D matrix.
        # Ex: With neighborhood axis [1 2] (13 neighbors), 47 features per
        # point, we can pass from batches of 1250 streamlines to 2300!
        total = torch.sum(p * Q1, dim=1)
        return total

    else:
        raise ValueError("Interpolation: There was a problem with the "
                         "volume's number of dimensions!")


def interpolate_volume_in_neighborhood(
        volume_as_tensor, coords_vox_corner, neighborhood_vectors_vox=None,
        clear_cache=True):
    """
    Params
    ------
    data_tensor: tensor
        The data: a 4D tensor with last dimension F (nb of features).
    coords_vox_corner: torch.Tensor shape (M, 3)
        A list of points (3d coordinates). Neighborhood will be added to these
        points based. Coords must be in voxel world, origin='corner', to use
        trilinear interpolation.
    neighborhood_vectors_vox: np.ndarray[float] with shape (N, 3)
        The neighboors to add to each coord. Do not include the current point
        ([0,0,0]). Values are considered in the same space as
        coords_vox_corner, and should thus be in voxel space.
    clear_cache: bool
        If True, will clear the cache after interpolation. This can be useful
        to save memory, but will slow down the function.

    Returns
    -------
    subj_x_data: tensor of shape (M, F * (N+1))
        The interpolated data: M points with contatenated neighbors.
    coords_vox_corner: tensor of shape (M x (N+1), 3)
        The final coordinates.
    """
    
    if (neighborhood_vectors_vox is not None and
            len(neighborhood_vectors_vox) > 0):
        m_input_points = coords_vox_corner.shape[0]
        n_neighb = neighborhood_vectors_vox.shape[0]
        f_features = volume_as_tensor.shape[-1]

        # Extend the coords array with the neighborhood coordinates
        # coords: shape (M x (N+1), 3)
        coords_vox_corner, tiled_vectors = \
            extend_coordinates_with_neighborhood(coords_vox_corner,
                                                 neighborhood_vectors_vox)

        # Interpolate signal for each (new) point
        # DWI data features for each neighbor are concatenated.
        # Result is of shape: (M * (N+1), F).
        flat_subj_x_data = torch_trilinear_interpolation(volume_as_tensor,
                                                         coords_vox_corner,
                                                         clear_cache)

        # Neighbors become new features of the current point.
        # Reshape signal into (M, (N+1)*F))
        new_nb_features = f_features * n_neighb
        subj_x_data = flat_subj_x_data.reshape(m_input_points, new_nb_features)

    else:  # No neighborhood:
        subj_x_data = torch_trilinear_interpolation(volume_as_tensor,
                                                    coords_vox_corner,
                                                    clear_cache)

    if volume_as_tensor.is_cuda:
        logging.debug("Emptying cache now. Can be a little slow but saves A "
                      "LOT of memory for our\n current trilinear "
                      "interpolation. (We could improve code \na little but "
                      "would loose a lot of readability).")

        # Ex: with 2000 streamlines (134 000 points), with neighborhood axis
        # [1 2] (13 neighbors), 47 features per point, we remove 6.5 GB of
        # memory!
        if clear_cache:
            torch.cuda.empty_cache()

    return subj_x_data, coords_vox_corner
