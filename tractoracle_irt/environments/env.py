import functools
from typing import Callable, Dict, Tuple

import nibabel as nib
import numpy as np
import torch
import os
from dipy.core.sphere import HemiSphere
from dipy.data import get_sphere
from dipy.direction.peaks import reshape_peaks_for_visualization
from dipy.tracking import utils as track_utils
from scilpy.reconst.utils import (find_order_from_nb_coeff,
                                  get_maximas)
from dipy.reconst.shm import sh_to_sf_matrix
from torch.utils.data import DataLoader

from tractoracle_irt.environments.neighborhood_manager import NeighborhoodManager
from tractoracle_irt.utils.logging import get_logger
from tractoracle_irt.datasets.SubjectDataset import SubjectDataset
from tractoracle_irt.datasets.utils import (MRIDataVolume,
                                         convert_length_mm2vox,
                                         set_sh_order_basis,
                                         get_sh_order_and_fullness)
from tractoracle_irt.environments.local_reward import PeaksAlignmentReward
from tractoracle_irt.environments.oracle_reward import OracleReward
from tractoracle_irt.environments.tractometer_reward import TractometerReward
from tractoracle_irt.environments.reward import RewardFunction
from tractoracle_irt.environments.stopping_criteria import (
    BinaryStoppingCriterion, OracleStoppingCriterion,
    StoppingFlags)
from tractoracle_irt.environments.utils import (  # is_looping,
    is_too_curvy, is_too_long, has_reached_gm)
from tractoracle_irt.utils.utils import normalize_vectors, Timer
from tractoracle_irt.environments.rollout_env import RolloutEnvironment
from tractoracle_irt.environments.state import ConvState, State
from tractoracle_irt.algorithms.shared.fodf_encoder import WorkingFodfEncoder, SmallWorkingFodfEncoder, DummyFodfEncoder
from tractoracle_irt.utils.interpolation import calc_neighborhood_grid, neighborhood_interpolation
from scilpy.tractograms.tractogram_operations import transform_warp_sft

LOGGER = get_logger(__name__)

def collate_fn(data):
    return data


class BaseEnv(object):
    """
    Abstract tracking environment. This class should not be used directly.
    Instead, use `TrackingEnvironment` or `InferenceTrackingEnvironment`.

    TractOracleIRT environments are based on OpenAI Gym environments. They
    are used to train reinforcement learning algorithms. They also emulate
    "Trackers" in dipy by handling streamline propagation, stopping criteria,
    and seeds.

    Since many streamlines are propagated in parallel, the environment is
    similar to VectorizedEnvironments in the Gym definition. However, the
    environment is not vectorized in the sense that it does not reset
    trajectories (streamlines) independently.

    TODO: reset trajectories independently ?

    """

    def __init__(
        self,
        subject_data: str,
        split_id: str,
        env_dto: dict,
    ):
        """
        Initialize the environment. This should not be called directly.
        Instead, use `from_dataset` or `from_files`.

        Parameters
        ----------
        dataset_file: str
            Path to the HDF5 file containing the dataset.
        split_id: str
            Name of the split to load (e.g. 'training',
            'validation', 'testing').
        subjects: list
            List of subjects to load.
        env_dto: dict
            DTO containing env. parameters

        """

        # If the subject data is a string, it is assumed to be a path to
        # an HDF5 file. Otherwise, it is assumed to be a list of volumes
        if type(subject_data) is str:
            self.dataset_file = subject_data
            self.split = split_id

            self.dataset = SubjectDataset(
                self.dataset_file, self.split)
            self.loader = DataLoader(self.dataset, 1, shuffle=True,
                                     collate_fn=collate_fn,
                                     num_workers=2)
            self.loader_iter = iter(self.loader)
        else:
            self.subject_data = subject_data
            self.split = split_id

        # Unused: this is from an attempt to normalize the input data
        # as is done by the original PPO impl
        # Does not seem to be necessary here.
        self.normalize_obs = False  # env_dto['normalize']
        self.obs_rms = None

        self._state_size = None  # to be calculated later

        # Tracking parameters
        self.n_dirs = env_dto['n_dirs']
        self.theta = env_dto['theta']
        # Number of seeds per voxel
        self.npv = env_dto['npv']
        # Whether to use CMC or binary stopping criterion
        self.binary_stopping_threshold = env_dto['binary_stopping_threshold']

        # Step-size and min/max lengths are typically defined in mm
        # by the user, but need to be converted to voxels.
        self.step_size_mm = env_dto['step_size']
        self.min_length_mm = env_dto['min_length']
        self.max_length_mm = env_dto['max_length']

        # Oracle parameters
        self.oracle_crit_checkpoint = env_dto['oracle_crit_checkpoint']
        self.oracle_reward_checkpoint = env_dto['oracle_reward_checkpoint']
        self.oracle_stopping_criterion = env_dto['oracle_stopping_criterion']

        # Tractometer parameters
        self.scoring_data = env_dto['scoring_data']

        # Reward parameters
        self.reward_was_init = False
        self.compute_reward = env_dto['compute_reward']

        # "Local" reward parameters
        self.alignment_weighting = env_dto['alignment_weighting']
        # "Sparse" reward parameters
        self.oracle_bonus = env_dto['oracle_bonus']

        # Other parameters
        self.rng = env_dto['rng']
        self.device = env_dto['device']
        self.target_sh_order = env_dto['target_sh_order']
        self.reward_with_gt = env_dto['reward_with_gt']
        self.rollout_env = None

        # Extractor parameters
        self.extractor_target = None
        if env_dto['extractor_target'] is not None:
            self.extractor_target = nib.load(env_dto['extractor_target'])

        # ==========================================
        # State parameters
        # ==========================================
        self.neighborhood_radius = env_dto['neighborhood_radius']
        self.neighborhood_type = env_dto['neighborhood_type']
        self.flatten_state = env_dto['flatten_state'] or env_dto['fodf_encoder_ckpt'] is not None
        self.conv_state = env_dto['conv_state']
        self.fodf_encoder_ckpt = env_dto['fodf_encoder_ckpt']
        self.interpolation = env_dto['interpolation']

        # ==========================================
        # FODF Encoder
        # ==========================================
        self.fodf_encoder = None
        self.exclude_direct_neigh = env_dto['exclude_direct_neigh']
        if self.fodf_encoder_ckpt is not None:
            self.fodf_encoder = SmallWorkingFodfEncoder()
            self.fodf_encoder.load_state_dict(torch.load(self.fodf_encoder_ckpt,
                                                         map_location=self.device))

            # Make sure that we never calculate gradients for this model
            self.fodf_encoder.eval()

            # Freeze the model
            for param in self.fodf_encoder.parameters():
                param.requires_grad = False

            # print("Compiling the fodf encoder")
            # with SimpleTimer() as t:
            #     self.fodf_encoder.compile()
            # print(f"Compilation done in {t.interval:.2f}s")

            self.fodf_encoder = self.fodf_encoder.to(self.device)
            print("Sending the encoder to the device: ", self.device)

        # Print a summary of the state we are about to use for the model
        print("=========================================")
        print("State parameters")
        print("=========================================")
        print(f"Neighborhood radius: {self.neighborhood_radius}")
        print(f"Neighborhood type: {self.neighborhood_type}")
        print(f"Flatten state: {self.flatten_state}")
        print(f"FODF encoder checkpoint: {self.fodf_encoder_ckpt}")
        print(f"Interpolation method: {self.interpolation}")
        print("=========================================")

        # Load one subject as an example
        self.load_subject()

    def load_subject(
        self,
    ):
        """ Load a random subject from the dataset. This is used to
        initialize the environment. """

        if hasattr(self, 'dataset_file'):

            if hasattr(self, 'subject_id') and len(self.dataset) == 1:
                return

            try:
                (sub_id, input_volume, tracking_mask, seeding_mask,
                 peaks, reference, gm_mask, transformation, deformation, fa) = next(self.loader_iter)[0]
            except StopIteration:
                self.loader_iter = iter(self.loader)
                (sub_id, input_volume, tracking_mask, seeding_mask,
                 peaks, reference, gm_mask, transformation, deformation, fa) = next(self.loader_iter)[0]

            self.subject_id = sub_id
            # Affines
            self.reference = reference
            self.fa = fa
            self.affine_vox2rasmm = input_volume.affine_vox2rasmm
            self.affine_rasmm2vox = np.linalg.inv(self.affine_vox2rasmm)

            # Volumes and masks
            self.data_volume = input_volume.data
        else:
            (input_volume, tracking_mask, seeding_mask, peaks,
             reference, gm_mask, transformation, deformation, fa) = self.subject_data

            self.affine_vox2rasmm = input_volume.affine_vox2rasmm
            self.affine_rasmm2vox = np.linalg.inv(self.affine_vox2rasmm)

            # Volumes and masks
            self.data_volume = input_volume.data
            self.reference = reference
            self.fa = fa

        # The SH target order is taken from the hyperparameters in the case of tracking.
        # Otherwise, the SH target order is taken from the input volume by default.
        if self.target_sh_order is None:
            n_coefs = input_volume.shape[-1]
            sh_order, _ = get_sh_order_and_fullness(n_coefs)
            self.target_sh_order = sh_order

        self.tracking_mask = tracking_mask
        self.peaks = peaks
        mask_data = tracking_mask.data.astype(np.uint8)
        self.seeding_data = seeding_mask.data.astype(np.uint8)

        self.gm_data = gm_mask.data.astype(np.uint8) if gm_mask else None

        # Used for registering the streamlines to MNI space
        # (e.g. for extractor_flow)
        self.transformation = transformation
        self.deformation = deformation

        self.step_size = convert_length_mm2vox(
            self.step_size_mm,
            self.affine_vox2rasmm)
        self.min_length = self.min_length_mm
        self.max_length = self.max_length_mm

        # Compute maximum length
        self.max_nb_steps = int(self.max_length / self.step_size_mm)
        self.min_nb_steps = int(self.min_length / self.step_size_mm)

        # Neighborhood used as part of the state
        self.add_neighborhood_vox = convert_length_mm2vox(
            self.step_size_mm,
            self.affine_vox2rasmm)

        # With this manager, we interpolate a bigger neighborhood grid around the current position
        # to provide to the FODF autoencoder. Since the neighborhood is bigger than the step size
        # of the agent, we use a fixed resolution of 1 in voxel space.
        self.neigh_manager = NeighborhoodManager(self.data_volume,
                                                    self.neighborhood_radius,
                                                    1, # We just want a crop of the neighborhood around him.
                                                    False,
                                                    neighborhood_type=self.neighborhood_type,
                                                    method=self.interpolation)
        # This is just to interpolate the direct neighbors
        self.direct_neigh_manager = NeighborhoodManager(self.data_volume,
                                            1,
                                            self.add_neighborhood_vox,
                                            True,
                                            neighborhood_type='axes',
                                            method='dwi_ml')
                                                        

        # Tracking seeds
        self.seeds = track_utils.random_seeds_from_mask(
            self.seeding_data,
            np.eye(4),
            seeds_count=self.npv)
        # print(
        #     '{} has {} seeds.'.format(self.__class__.__name__,
        #                               len(self.seeds)))

        # ===========================================
        # Stopping criteria
        # ===========================================

        # Stopping criteria is a dictionary that maps `StoppingFlags`
        # to functions that indicate whether streamlines should stop or not

        # TODO: Make all stopping criteria classes.
        # TODO?: Use dipy's stopping criteria instead of custom ones ?
        self.stopping_criteria = {}

        # Length criterion
        self.stopping_criteria[StoppingFlags.STOPPING_LENGTH] = \
            functools.partial(is_too_long,
                              max_nb_steps=self.max_nb_steps)

        # Angle between segment (curvature criterion)
        self.stopping_criteria[
            StoppingFlags.STOPPING_CURVATURE] = \
            functools.partial(is_too_curvy, max_theta=self.theta)
        
        # GM criterion
        if self.gm_data is not None:
            self.stopping_criteria[
                StoppingFlags.STOPPING_TARGET] = \
                    functools.partial(has_reached_gm, mask=self.gm_data, threshold=0.5)

        # Stopping criterion according to an oracle
        if self.oracle_crit_checkpoint and self.oracle_stopping_criterion:
            self.stopping_criteria[
                StoppingFlags.STOPPING_ORACLE] = OracleStoppingCriterion(
                self.oracle_crit_checkpoint,
                self.min_nb_steps * 5,
                self.reference,
                self.affine_vox2rasmm,
                self.device)

        # Mask criterion (either binary or CMC)
        binary_criterion = BinaryStoppingCriterion(
            mask_data,
            self.binary_stopping_threshold)
        self.stopping_criteria[StoppingFlags.STOPPING_MASK] = \
            binary_criterion

        # ==========================================
        # Reward function
        # =========================================

        # Reward function and reward factors
        if self.compute_reward and not self.reward_was_init:
            # Reward streamline according to alignment with local peaks
            peaks_reward = PeaksAlignmentReward(self.peaks)
            factors = [peaks_reward]
            weights = [self.alignment_weighting]

            # Reward streamlines based on approximated anatomical validity.
            if self.reward_with_gt and np.abs(self.oracle_bonus) > 0:
                tractometer_reward = TractometerReward(self.scoring_data,
                                                       self.reference,
                                                       self.affine_vox2rasmm)
                factors.append(tractometer_reward)
                weights.append(self.oracle_bonus)
            elif not self.reward_with_gt \
                    and self.oracle_reward_checkpoint \
                    and np.abs(self.oracle_bonus) > 0:
                # Reward streamlines according to the oracle
                oracle_reward = OracleReward(self.oracle_reward_checkpoint,
                                             self.min_nb_steps,
                                             self.reference,
                                             self.affine_vox2rasmm,
                                             self.device)
                factors.append(oracle_reward)
                weights.append(self.oracle_bonus)

            # Combine all reward factors into the reward function
            self.reward_function = RewardFunction(
                factors,
                weights)
            
            self.reward_was_init = True
        elif self.compute_reward:
            # In this case, the reward functions were already initialized,
            # we just need to update the subject
            kwargs = {
                'peaks': self.peaks
            }
            self.reward_function.change_subject(
                self.subject_id,
                self.min_nb_steps,
                self.reference,
                self.affine_vox2rasmm,
                **kwargs)
            

    @classmethod
    def from_dataset(
        cls,
        env_dto: dict,
        split: str,
    ):
        """ Initialize the environment from an HDF5.

        Parameters
        ----------
        env_dto: dict
            DTO containing env. parameters
        split: str
            Name of the split to load (e.g. 'training', 'validation',
            'testing').

        Returns
        -------
        env: BaseEnv
            Environment initialized from a dataset.
        """

        dataset_file = env_dto['dataset_file']

        env = cls(dataset_file, split, env_dto)
        return env

    @classmethod
    def from_files(
        cls,
        env_dto: dict,
    ):
        """ Initialize the environment from files. This is useful for
        tracking from a trained model.

        Parameters
        ----------
        env_dto: dict
            DTO containing env. parameters

        Returns
        -------
        env: BaseEnv
            Environment initialized from files.
        """

        in_odf = env_dto['in_odf']
        in_seed = env_dto['in_seed']
        in_mask = env_dto['in_mask']
        gm_mask = env_dto['gm_mask']
        sh_basis = env_dto['sh_basis']
        is_sh_basis_legacy = env_dto['is_sh_basis_legacy']
        reference = env_dto['reference']
        target_sh_order = env_dto['target_sh_order']
        in_fa = env_dto['in_fa']
        in_peaks = env_dto['in_peaks']

        (input_volume, peaks_volume, tracking_mask, seeding_mask,
         gm_mask, transformation, deformation, fa) = BaseEnv._load_files(
             in_odf,
             in_seed,
             in_mask,
             sh_basis,
             is_sh_basis_legacy,
             target_sh_order,
             gm_mask=gm_mask,
             fa=in_fa,
             in_peaks=in_peaks)

        subj_files = (input_volume, tracking_mask, seeding_mask,
                      peaks_volume, reference, gm_mask, transformation, deformation, fa)

        return cls(subj_files, 'testing', env_dto)

    @classmethod
    def _load_files(
        cls,
        signal_file,
        in_seed,
        in_mask,
        sh_basis,
        is_sh_basis_legacy,
        target_sh_order=6,
        gm_mask=None,
        fa=None,
        in_peaks=None
    ):
        """ Load data volumes and masks from files. This is useful for
        tracking from a trained model.

        If the signal is not in descoteaux07 basis, it will be converted. The
        WM mask will be loaded and concatenated to the signal. Additionally,
        peaks will be computed from the signal.

        Parameters
        ----------
        signal_file: str
            Path to the signal file (e.g. SH coefficients).
        in_seed: str
            Path to the seeding mask.
        in_mask: str
            Path to the tracking mask.
        sh_basis: str
            Basis of the SH coefficients.
        target_sh_order: int
            Target SH order. Should come from the hyperparameters file.

        Returns
        -------
        signal_volume: MRIDataVolume
            Volumetric data containing the SH coefficients
        peaks_volume: MRIDataVolume
            Volume containing the fODFs peaks
        tracking_volume: MRIDataVolume
            Volumetric mask where tracking is allowed
        seeding_volume: MRIDataVolume
            Mask where seeding should be done
        """

        signal = nib.load(signal_file)

        # Assert that the subject has iso voxels, else stuff will get
        # complicated
        if not np.allclose(np.mean(signal.header.get_zooms()[:3]),
                           signal.header.get_zooms()[0], atol=1e-03):
            print('WARNING: ODF SH file is not isotropic. Tracking cannot be '
                  'ran robustly. You are entering undefined behavior '
                  'territory.')

        data = set_sh_order_basis(signal.get_fdata(dtype=np.float32),
                                  sh_basis,
                                  target_order=target_sh_order,
                                  target_basis='descoteaux07')

        if in_peaks is not None:
            peaks = nib.load(in_peaks)
            peaks_volume = MRIDataVolume(
                peaks.get_fdata(), peaks.affine)

        else:
            # No peaks was provided
            with Timer("Computing peaks from ODF"):
                # Compute peaks from signal
                # Does not work if signal is not fODFs
                npeaks = 5
                odf_shape_3d = data.shape[:-1]
                peak_dirs = np.zeros((odf_shape_3d + (npeaks, 3)))
                peak_values = np.zeros((odf_shape_3d + (npeaks, )))

                sphere = HemiSphere.from_sphere(get_sphere("repulsion724")
                                                ).subdivide(0)

                order = find_order_from_nb_coeff(data)

                LOGGER.debug("is_sh_basis_legacy: ", is_sh_basis_legacy)
                LOGGER.debug("order: {}".format(order))
                LOGGER.debug("sh_basis: {}".format(sh_basis))
                LOGGER.debug("target_sh_order: {}".format(target_sh_order))

                b_matrix, _ = sh_to_sf_matrix(sphere, order, "descoteaux07", legacy=is_sh_basis_legacy)
                for idx in np.argwhere(np.sum(data, axis=-1)):
                    idx = tuple(idx)
                    directions, values, indices = get_maximas(data[idx],
                                                            sphere, b_matrix.T,
                                                            0.1, 0)
                    if values.shape[0] != 0:
                        n = min(npeaks, values.shape[0])
                        peak_dirs[idx][:n] = directions[:n]
                        peak_values[idx][:n] = values[:n]

                X, Y, Z, N, P = peak_dirs.shape
                peak_values = np.divide(peak_values, peak_values[..., 0, None],
                                        out=np.zeros_like(peak_values),
                                        where=peak_values[..., 0, None] != 0)
                peak_dirs[...] *= peak_values[..., :, None]
                peak_dirs = reshape_peaks_for_visualization(peak_dirs)
                
                peaks_volume = MRIDataVolume(
                    peak_dirs, signal.affine)

        # Load rest of volumes
        with Timer("Loading volumes"):
            seeding = nib.load(in_seed)
            tracking = nib.load(in_mask)
            signal_data = data
            signal_volume = MRIDataVolume(
                signal_data, signal.affine)

            seeding_volume = MRIDataVolume(
                seeding.get_fdata(), seeding.affine)
            tracking_volume = MRIDataVolume(
                tracking.get_fdata(), tracking.affine)
        
        gm_volume = None
        if gm_mask:
            with Timer("Loading gm mask"):
                gm = nib.load(gm_mask)
                gm_volume = MRIDataVolume(
                    gm.get_fdata(), gm.affine)
        
        transformation = None
        deformation = None

        if fa:
            with Timer("Loading fa"):
                fa = nib.load(fa)
                in_fa = MRIDataVolume(
                    fa.get_fdata(), fa.affine)
                
        else:
            in_fa = None

        print("Finished loading files")

        return (signal_volume, peaks_volume, tracking_volume,
                seeding_volume, gm_volume, transformation, deformation, in_fa)

    def get_state_size(self):
        """ Returns the size of the state space by computing the size of
        an example state.

        Returns
        -------
        state_size: int
            Size of the state space.
        """

        example_state, _ = self.reset(0, 1)
        self._state_size = example_state.shape
        return self._state_size

    def get_action_size(self):
        """ Returns the size of the action space.
        """

        return 3

    def get_target_sh_order(self):
        """ Returns the target SH order. For tracking, this is based on the hyperparameters.json if it's specified.
        Otherwise, it's extracted from the data directly.
        """

        return self.target_sh_order

    def get_voxel_size(self):
        """ Returns the voxel size by taking the mean value of the diagonal
        of the affine. This implies that the vox size is always isometric.

        Returns
        -------
        voxel_size: float
            Voxel size in mm.

        """
        diag = np.diagonal(self.affine_vox2rasmm)[:3]
        voxel_size = np.mean(np.abs(diag))

        return voxel_size

    def setup_rollout_env(self, rollout_env: RolloutEnvironment):
        LOGGER.info('Setting up rollout environment')
        self.rollout_env = rollout_env

    def _format_actions(
        self,
        actions: np.ndarray,
    ):
        """ Format actions to be used by the environment. Scaling
        actions to the step size.
        """
        actions = normalize_vectors(actions) * self.step_size

        return actions

    def _format_state(
        self,
        streamlines: np.ndarray
    ) -> np.ndarray:
        """
        From the last streamlines coordinates, extract the corresponding
        SH coefficients

        Parameters
        ----------
        streamlines: `numpy.ndarry`
            Streamlines from which to get the coordinates

        Returns
        -------
        inputs: `numpy.ndarray`
            Observations of the state, incl. previous directions.
        """
        with torch.no_grad(), torch.autocast(device_type=str(self.device), dtype=torch.float16, enabled=False):
            N, L, P = streamlines.shape

            if N <= 0:
                return []

            # Get the last point of each streamline
            segments = streamlines[:, -1, :][:, None, :]

            # Reshape to get a list of coordinates
            N, H, P = segments.shape
            flat_coords = np.reshape(segments, (N * H, P))
            coords = torch.as_tensor(flat_coords).to(self.device)

            # Get the SH coefficients at the last point of each streamline
            # The neighborhood is used to get the SH coefficients around
            # the last point
            # if self.use_custom_interpolation:
            #     signal = neighborhood_interpolation(
            #         self.data_volume, coords, self.neighborhood_directions)
            #     # display_image(nib.Nifti1Image(signal[0].cpu().numpy(), self.affine_vox2rasmm), default_slice=self.neighborhood_radius, save_to="test_neighborhood.png")
            #     # raise NotImplementedError("This implementation wasn't tested")
            # else:
            #     signal, _ = interpolate_volume_in_neighborhood(
            #         self.data_volume,
            #         coords,
            #         self.neighborhood_directions, clear_cache=False)
            #     N, S = signal.shape

            #     if self.big_neighborhood or self.fodf_encoder is not None:
            #         # Unflatten the signal to use it for convolutions
            #         # Unflatten the signal into (N, W, H, D, C) shape
            #         signal = unflatten_neighborhood(
            #             signal, self.neighborhood_directions, 'grid',
            #             self.neighborhood_radius, self.add_neighborhood_vox)
            #         # signal = custom_unflatten_neighborhood(
            #         #     signal, self.neighborhood_directions, 'grid',
            #         #     self.neighborhood_radius, self.add_neighborhood_vox)

            #         # Permute axes to fit PyTorch's convention of (N, C, D, H, W)
            #         # https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
            #         signal = signal.permute(0, 4, 1, 2, 3)

            #         # display_image(nib.Nifti1Image(signal[0].cpu().numpy(), self.affine_vox2rasmm), default_slice=self.neighborhood_radius, save_to="test_neighborhood_dwi.png")
            #         # raise NotImplementedError("This implementation wasn't tested")


            if self.fodf_encoder is not None:
                signal = self.direct_neigh_manager.get(coords)
                encoded_neighborhood = self._get_neighborhood_grid_encodings(coords)

                # Concatenate the encoded neighborhood to the direct neighbors
                if self.exclude_direct_neigh:
                    signal = encoded_neighborhood
                else:
                    signal = torch.cat([signal, encoded_neighborhood], dim=1)
            elif self.conv_state:
                signal = self.neigh_manager.get(coords, torch_convention=True)
            else:
                signal = self.direct_neigh_manager.get(coords)

            # Flatten the signal as this will be fed to a MLP
            # signal = signal.reshape(N, -1)

            # Placeholder for the previous directions
            previous_dirs = np.zeros((N, self.n_dirs, P), dtype=np.float32)
            if L > 1:
                # Compute directions from the streamlines
                dirs = np.diff(streamlines, axis=1)
                # Fetch the N last directions
                previous_dirs[:, :min(dirs.shape[1], self.n_dirs), :] = \
                    dirs[:, :-(self.n_dirs+1):-1, :]

            # Flatten the directions to fit in the inputs and send to device
            dir_inputs = torch.reshape(
                torch.from_numpy(previous_dirs).to(self.device),
                (N, self.n_dirs * P))

            # Return them separately so we can run convolutions on unflattened
            # but not dir_inputs.
            if not self.flatten_state:
                state = ConvState(signal, dir_inputs, coords, device=self.device)
            else:
                state = State(signal, dir_inputs, coords, device=self.device)
        return state

    def _get_neighborhood_grid_encodings(self, coords):
        """
        This method interpolates the neighborhood grid around the current
        position and encodes it using the FODF encoder.

        However, since we are interpolating a big neighborhood grid, we need
        to interpolate the grid in chunks and then concatenate the results
        to avoid running out of GPU memory.
        """
        N = coords.shape[0]

        # Ideally, the method aims to reproduce the following code
        # (whilst avoiding running out of memory):
        #
        # interpolated_neighborhood = self.neigh_manager.get(coords, torch_convention=True)
        # interpolated_neighborhood = interpolated_neighborhood[:, :, :-1, :-1, :-1]
        # encoded_neighborhood = self.fodf_encoder(interpolated_neighborhood)
        # encoded_neighborhood = encoded_neighborhood.reshape(N, -1)

        batch_size = 128 # TODO: Parametrize
        placeholder = torch.zeros((N, self.fodf_encoder.output_size), device=self.device)
        # print("placeholder size: ", placeholder.shape)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            chunk_coords = coords[start:end]

            interpolated_neighborhood = self.neigh_manager.get(chunk_coords, torch_convention=True)
            
            # We crop the interpolated neighborhood to get a evenly-sized grid.
            # interpolated_neighborhood = interpolated_neighborhood[:, :, :-1, :-1, :-1]

            encoded_neighborhood = self.fodf_encoder(interpolated_neighborhood)
            placeholder[start:end] = encoded_neighborhood.reshape(end - start, -1)

        return placeholder

    def _compute_stopping_flags(
        self,
        streamlines: np.ndarray,
        stopping_criteria: Dict[StoppingFlags, Callable]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Checks which streamlines should stop and which ones should
        continue.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        stopping_criteria : dict of int->Callable
            List of functions that take as input streamlines, and output a
            boolean numpy array indicating which streamlines should stop

        Returns
        -------
        should_stop : `numpy.ndarray`
            Boolean array, True is tracking should stop
        flags : `numpy.ndarray`
            `StoppingFlags` that triggered stopping for each stopping
            streamline
        """
        idx = np.arange(len(streamlines))

        should_stop = np.zeros(len(idx), dtype=np.bool_)
        flags = np.zeros(len(idx), dtype=int)

        # For each possible flag, determine which streamline should stop and
        # keep track of the triggered flag
        for flag, stopping_criterion in stopping_criteria.items():
            stopped_by_criterion = stopping_criterion(streamlines)
            flags[stopped_by_criterion] |= flag.value
            should_stop[stopped_by_criterion] = True

        return should_stop, flags

    def _is_stopping():
        """ Check which streamlines should stop or not according to the
        predefined stopping criteria
        """
        pass

    def reset(self):
        """ Reset the environment to its initial state.
        """
        if self.compute_reward:
            self.reward_function.reset()

    def step():
        """
        Abstract method to be implemented by subclasses which defines
        the behavior of the environment when taking a step. This includes
        propagating the streamlines, computing the reward, and checking
        which streamlines should stop.
        """
        pass

    def transform_tractogram_to_mni(self, moving_sft):
        """
        Transform a tractogram to mni space using the current subject's
        transformation matrices which are available within the HDF5 file.
        """

        assert self.transformation is not None, \
            "Transformation matrix might be missing from the dataset."
        assert self.deformation is not None, \
            "Deformation volume might be missing from the dataset."
        assert self.extractor_target is not None, \
            "Can't transform to MNI space without a target file."
        assert moving_sft is not None, "Input moving_sft cannot be None"

        new_sft = transform_warp_sft(moving_sft, self.transformation,
                                    self.extractor_target,
                                    inverse=True,
                                    reverse_op=False,
                                    deformation_data=self.deformation,
                                    remove_invalid=True,
                                    cut_invalid=False)

        print("Transformed tractogram to MNI space.")

        transform_map_subj = {
            self.subject_id: {
                'transformation': self.transformation,
                'deformation': self.deformation,
                'reference': self.reference
            }
        }

        return new_sft, transform_map_subj
    
    def transform_tractogram_to_reference(self, moving_sft,
                                          reference=None, transformation=None,
                                          deformation=None):
        
        _reference = reference if reference is not None else self.reference
        _transformation = transformation if transformation is not None else self.transformation
        _deformation = deformation if deformation is not None else self.deformation

        assert _transformation is not None, \
            "Transformation matrix might be missing from the dataset."
        assert _deformation is not None, \
            "Deformation volume might be missing from the dataset."
        assert _reference is not None, \
            "Can't transform to MNI space without a target file."
        
        assert moving_sft is not None, "Input moving_sft cannot be None"

        new_sft = transform_warp_sft(moving_sft, _transformation,
                                     _reference,
                                     inverse=False,
                                     reverse_op=True,
                                     deformation_data=_deformation,
                                     remove_invalid=True,
                                     cut_invalid=False)

        print("Transformed tractogram to reference space.")
        
        return new_sft

    @property
    def can_transform_to_mni(self):
        return self.transformation is not None \
            and self.deformation is not None\
            and self.extractor_target is not None
    
    def save_anat_to(self, out_dir, filename):
        """
        Save the anat file with its associated affine to a directory.
        """
        nib.save(self.reference, os.path.join(out_dir, filename))

    def save_fa_to(self, out_dir, filename):
        """
        Save the FA file with its associated affine to a directory.
        """
        if self.fa is None:
            raise ValueError("FA file is not available. Make sure it is provided in the dataset.")
        nib.save(self.fa, os.path.join(out_dir, filename))
