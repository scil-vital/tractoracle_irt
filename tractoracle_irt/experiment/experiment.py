import nibabel as nib
import numpy as np

from argparse import ArgumentParser
from os.path import join as pjoin
from typing import Tuple

from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram
from dipy.io.streamline import save_tractogram

from nibabel.streamlines import Tractogram

from tractoracle_irt.environments.env import BaseEnv
from tractoracle_irt.environments.tracking_env import (
    TrackingEnvironment)
from tractoracle_irt.environments.noisy_tracking_env import (
    NoisyTrackingEnvironment)
from tractoracle_irt.environments.stopping_criteria import (
    is_flag_set, StoppingFlags)
from tractoracle_irt.utils.utils import LossHistory
from tractoracle_irt.utils.comet_monitor import CometMonitor


class Experiment(object):
    """ Base class for experiments
    """

    def run(self):
        """ Main method where data is loaded, classes are instanciated,
        everything is set up.
        """
        pass

    def setup_monitors(self):
        #  RL monitors
        self.train_reward_monitor = LossHistory(
            "Train Reward", "train_reward", self.hp.experiment_path)
        self.train_length_monitor = LossHistory(
            "Train Length", "length_reward", self.hp.experiment_path)
        self.train_ratio_monitor = LossHistory(
            "Train Log-Ratio", "log_ratio", self.hp.experiment_path)
        self.reward_monitor = LossHistory(
            "Reward - Alignment", "reward", self.hp.experiment_path)
        self.actor_loss_monitor = LossHistory(
            "Loss - Actor Policy Loss", "actor_loss", self.hp.experiment_path)
        self.critic_loss_monitor = LossHistory(
            "Loss - Critic MSE Loss", "critic_loss", self.hp.experiment_path)
        self.len_monitor = LossHistory(
            "Length", "length", self.hp.experiment_path)

        # Tractometer monitors
        # TODO: Infer the number of bundles from the GT
        if self.hp.tractometer_validator:
            self.vc_monitor = LossHistory(
                "Valid Connections", "vc", self.hp.experiment_path)
            self.ic_monitor = LossHistory(
                "Invalid Connections", "ic", self.hp.experiment_path)
            self.nc_monitor = LossHistory(
                "Non-Connections", "nc", self.hp.experiment_path)
            self.vb_monitor = LossHistory(
                "Valid Bundles", "VB", self.hp.experiment_path)
            self.ib_monitor = LossHistory(
                "Invalid Bundles", "IB", self.hp.experiment_path)
            self.ol_monitor = LossHistory(
                "Overlap monitor", "ol", self.hp.experiment_path)

        else:
            self.vc_monitor = None
            self.ic_monitor = None
            self.nc_monitor = None
            self.vb_monitor = None
            self.ib_monitor = None
            self.ol_monitor = None

        # Initialize monitors here as the first pass won't include losses
        self.actor_loss_monitor.update(0)
        self.actor_loss_monitor.end_epoch(0)
        self.critic_loss_monitor.update(0)
        self.critic_loss_monitor.end_epoch(0)

    def setup_comet(self, prefix=''):
        """ Setup comet environment
        """
        # The comet object that will handle monitors
        self.comet_monitor = CometMonitor(
            self.comet_experiment, self.hp.experiment_path,
            prefix, use_comet=self.hp.use_comet, offline=self.hp.offline)
        print(self.hp.to_dict())
        self.comet_monitor.log_parameters(self.hp.to_dict())

    def _get_env_dict_and_dto(
        self, noisy, npv=None
    ) -> Tuple[dict, dict]:
        """ Get the environment class and the environment DTO.

        Parameters
        ----------
        noisy: bool
            Whether to use the noisy environment or not.

        Returns
        -------
        class_dict: dict
            Dictionary of environment classes.
        env_dto: dict
            Dictionary of environment parameters.
        """

        env_dto = {
            'dataset_file': self.hp.dataset_file,
            'fa_map': self.fa_map,
            'n_dirs': self.hp.n_dirs,
            'step_size': self.hp.step_size,
            'theta': self.hp.theta,
            'min_length': self.hp.min_length,
            'max_length': self.hp.max_length,
            'noise': self.hp.noise,
            'npv': self.hp.npv if npv is None else npv,
            'rng': self.rng,
            'alignment_weighting': self.hp.alignment_weighting,
            'oracle_bonus': self.hp.oracle_bonus,
            'oracle_validator': self.hp.oracle_validator,
            'oracle_stopping_criterion': self.hp.oracle_stopping_criterion,
            'oracle_crit_checkpoint': self.hp.oracle_crit_checkpoint,
            'oracle_reward_checkpoint': self.hp.oracle_reward_checkpoint,
            'scoring_data': self.hp.scoring_data,
            'tractometer_validator': self.hp.tractometer_validator,
            'binary_stopping_threshold': self.hp.binary_stopping_threshold,
            'compute_reward': self.compute_reward,
            'device': self.device,
            'target_sh_order': self.hp.target_sh_order,
            'reward_with_gt': self.hp.reward_with_gt,
            'neighborhood_radius': self.hp.neighborhood_radius,
            'neighborhood_type': self.hp.neighborhood_type,
            'flatten_state': self.hp.flatten_state,
            'conv_state': self.hp.conv_state,
            'fodf_encoder_ckpt': self.hp.fodf_encoder_ckpt,
            'interpolation': self.hp.interpolation,
            'extractor_target': self.hp.extractor_target if hasattr(self.hp, 'extractor_target') else None,
            'exclude_direct_neigh': self.hp.exclude_direct_neigh
        }

        if noisy:
            class_dict = {
                'tracking_env': NoisyTrackingEnvironment
            }
        else:
            class_dict = {
                'tracking_env': TrackingEnvironment
            }
        return class_dict, env_dto

    def get_env(self) -> Tuple[BaseEnv, BaseEnv]:
        """ Build environments

        Returns:
        --------
        env: BaseEnv
            "Forward" environment only initialized with seeds
        """

        class_dict, env_dto = self._get_env_dict_and_dto(False)

        # Someone with better knowledge of design patterns could probably
        # clean this
        env = class_dict['tracking_env'].from_dataset(
            env_dto, 'training')

        return env

    def get_valid_env(self) -> Tuple[BaseEnv, BaseEnv]:
        """ Build environments

        Returns:
        --------
        env: BaseEnv
            "Forward" environment only initialized with seeds
        """

        class_dict, env_dto = self._get_env_dict_and_dto(True)

        # Someone with better knowledge of design patterns could probably
        # clean this
        self.valid_env = class_dict['tracking_env'].from_dataset(
            env_dto, 'training')

        return self.valid_env
    
    def get_rlhf_env(self, npv=None) -> Tuple[BaseEnv, BaseEnv]:
        """ Build environment to be able to track streamlines
        without computing the reward and without using the 
        oracle stopping criterion.

        It should be exactly the same as the validation environment
        but with no oracle stopping criterion and no reward computation.

        Returns:
        --------
        env: BaseEnv
            "Forward" environment only initialized with seeds
        """
        class_dict, env_dto = self._get_env_dict_and_dto(True, npv)

        env_dto.update({
            'compute_reward': False,
            'oracle_stopping_criterion': False,
        })

        rlhf_env = class_dict['tracking_env'].from_dataset(
            env_dto, 'training')
        
        return rlhf_env

    def get_tracking_env(self):
        """ Generate environments according to tracking parameters.

        Returns:
        --------
        env: BaseEnv
            "Forward" environment only initialized with seeds
        """

        class_dict, env_dto = self._get_env_dict_and_dto(True)

        # Update DTO to include indiv. files instead of hdf5
        env_dto.update({
            'in_odf': self.in_odf,
            'wm_file': self.wm_file,
            'in_seed': self.in_seed,
            'in_mask': self.in_mask,
            'sh_basis': self.sh_basis,
            'is_sh_basis_legacy': self.is_sh_basis_legacy,
            'input_wm': self.input_wm,
            'reference': self.hp.reference_file,
            'gm_mask': self.gm_mask if hasattr(self, 'gm_mask') else None,
            'in_fa': self.in_fa if hasattr(self, 'in_fa') else None,
            'in_peaks': self.in_peaks if hasattr(self, 'in_peaks') else None,
            # file instead of being passed directly.
        })

        # Someone with better knowledge of design patterns could probably
        # clean this
        env = class_dict['tracking_env'].from_files(env_dto)

        return env

    def stopping_stats(self, tractogram):
        """ Compute stopping statistics for a tractogram.

        Parameters
        ----------
        tractogram: Tractogram
            Tractogram to compute statistics on.

        Returns
        -------
        stats: dict
            Dictionary of stopping statistics.
        """
        # Compute stopping statistics
        if tractogram is None:
            return {}
        # Stopping statistics are stored in the data_per_streamline
        # dictionary
        flags = tractogram.data_per_streamline['flags']
        stats = {}
        # Compute the percentage of streamlines that have a given flag set
        # for each flag
        for f in StoppingFlags:
            if len(flags) > 0:
                set_pct = np.mean(is_flag_set(flags, f))
            else:
                set_pct = 0
            stats.update({f.name: set_pct})
        return stats

    def score_tractogram(self, filename, env):
        """ Score a tractogram using the tractometer or the oracle.

        Parameters
        ----------
        filename: str
            Filename of the tractogram to score.

        """
        # Dict of scores
        all_scores = {}

        # Compute scores for the tractogram according
        # to each validator.
        for scorer in self.validators:
            scores = scorer(filename, env)
            all_scores.update(scores)

        return all_scores
    
    def save_sft(self,
        sft,
        subject_id: str,
        save_dir: str = None,
        extension: str = 'trk'):

        # Save on the experiment path, or on a specific directory if provided.
        path_prefix = save_dir if save_dir else self.hp.experiment_path

        # Save tractogram so it can be looked at, used by the tractometer
        # and more
        filename = pjoin(
            path_prefix,
            "{}_{}_{}_tracking.{}".format(self.hp.experiment,
                                            self.hp.experiment_id,
                                            subject_id, extension))
        
        save_tractogram(sft, filename, bbox_valid_check=False)
        return filename

    def convert_to_rasmm_sft(
        self,
        tractogram,
        affine: np.ndarray,
        reference: nib.Nifti1Image,
        discard_dps: bool = False
    ) -> StatefulTractogram:
        """
        Converts a tractogram to RASMM space.
        """
        # Prune empty streamlines, keep only streamlines that have more
        # than the seed.
        indices = [i for (i, s) in enumerate(tractogram.streamlines)
                   if len(s) > 1]

        tractogram.apply_affine(affine)

        streamlines = tractogram.streamlines[indices]
        data_per_streamline = tractogram.data_per_streamline[indices]
        data_per_point = tractogram.data_per_point[indices]

        if discard_dps:
            sft = StatefulTractogram(
                streamlines,
                reference,
                Space.RASMM,
                origin=Origin.TRACKVIS)
        else:
            sft = StatefulTractogram(
                streamlines,
                reference,
                Space.RASMM,
                origin=Origin.TRACKVIS,
                data_per_streamline=data_per_streamline,
                data_per_point=data_per_point)

        sft.to_rasmm()
        return sft

    def save_rasmm_tractogram(
        self,
        tractogram,
        subject_id: str,
        affine: np.ndarray,
        reference: nib.Nifti1Image,
        save_dir: str = None,
        extension: str = 'trk'
    ) -> str:
        """
        Saves a non-stateful tractogram from the training/validation
        trackers.

        Parameters
        ----------
        tractogram: Tractogram
            Tractogram generated at validation time.

        Returns:
        --------
        filename: str
            Filename of the saved tractogram.
        """
        sft = self.convert_to_rasmm_sft(tractogram, affine, reference)
        return self.save_sft(sft, subject_id, save_dir, extension)
        

    def log(
        self,
        valid_tractogram: Tractogram,
        valid_reward: float = 0,
        i_episode: int = 0,
        scores: dict = None,
    ):
        """ Print training infos and log metrics to Comet, if
        activated.

        Parameters
        ----------
        valid_tractogram: Tractogram
            Tractogram generated at validation time.
        valid_reward: float
            Sum of rewards obtained during validation.
        i_episode: int
            ith training episode.
        scores: dict
            Scores as computed by the tractometer.
        """
        if valid_tractogram:
            lens = [len(s) for s in valid_tractogram.streamlines]
        else:
            lens = [0]
        avg_valid_reward = valid_reward / len(lens)
        avg_length = np.mean(lens)  # Euclidian length

        print('---------------------------------------------------')
        print(self.hp.experiment_path)
        print('Episode {} \t avg length: {} \t total reward: {}'.format(
            i_episode,
            avg_length,
            avg_valid_reward))
        print('---------------------------------------------------')

        if scores is not None:
            self.vc_monitor.update(scores['VC'])
            self.ic_monitor.update(scores['IC'])
            self.nc_monitor.update(scores['NC'])
            self.vb_monitor.update(scores['VB'])
            self.ib_monitor.update(scores['IB'])
            self.ol_monitor.update(scores['mean_OL'])

            self.vc_monitor.end_epoch(i_episode)
            self.ic_monitor.end_epoch(i_episode)
            self.nc_monitor.end_epoch(i_episode)
            self.vb_monitor.end_epoch(i_episode)
            self.ib_monitor.end_epoch(i_episode)
            self.ol_monitor.end_epoch(i_episode)

        # Update monitors
        self.len_monitor.update(avg_length)
        self.len_monitor.end_epoch(i_episode)

        self.reward_monitor.update(avg_valid_reward)
        self.reward_monitor.end_epoch(i_episode)

        if self.comet_experiment is not None:
            # Update comet
            self.comet_monitor.update(
                self.reward_monitor,
                self.len_monitor,
                self.vc_monitor,
                self.ic_monitor,
                self.nc_monitor,
                self.vb_monitor,
                self.ib_monitor,
                self.ol_monitor,
                i_episode=i_episode)


def add_experiment_args(parser: ArgumentParser):
    parser.add_argument('experiment_path', type=str,
                        help='Path to experiment')
    parser.add_argument('experiment',
                        help='Name of experiment.')
    parser.add_argument('experiment_id', type=str,
                        help='ID of experiment.')
    parser.add_argument('--workspace', type=str, default='tractoracle_irt',
                        help='Comet.ml workspace')
    parser.add_argument('--rng_seed', default=1337, type=int,
                        help='Seed to fix general randomness')
    parser.add_argument('--use_comet', action='store_true',
                        help='Use comet to display training or not')
    parser.add_argument("--backup_dir", type=str,
                        help="Directory where to save a backup of the experiment's path.\n"
                        "This will compress and archive the experiment's files and\n"
                        "save the archive at this specified location. To avoid \n"
                        "doing backups, omit this argument. The directory should exist.")
    parser.add_argument('--offline', action='store_true',
                        help='Run the experiment in offline mode. This will save the experiment to a local directory instead of using Comet.ml.')


def add_data_args(parser: ArgumentParser):
    parser.add_argument('dataset_file',
                        help='Path to preprocessed dataset file (.hdf5)')
    parser.add_argument('--target_sh_order', type=int, default=None)


def add_environment_args(parser: ArgumentParser):
    parser.add_argument('--n_dirs', default=4, type=int,
                        help='Last n steps taken')
    parser.add_argument(
        '--binary_stopping_threshold',
        type=float, default=0.1,
        help='Lower limit for interpolation of tracking mask value.\n'
             'Tracking will stop below this threshold.')


def add_reward_args(parser: ArgumentParser):
    parser.add_argument('--alignment_weighting', default=1, type=float,
                        help='Alignment weighting for reward')
    parser.add_argument('--reward_with_gt', action='store_true', default=False,
                        help='Use the ground truth to compute the reward instead of the oracle.')


def add_model_args(parser: ArgumentParser):
    parser.add_argument('--n_actor', default=4096, type=int,
                        help='Number of learners')
    parser.add_argument('--hidden_dims', default='1024-1024-1024', type=str,
                        help='Hidden layers of the model')
    
    # Arguments to enlarge the agent's state.
    parser.add_argument('--neighborhood_radius', type=int, default=1,
                        help='Radius of the neighborhood.')
    parser.add_argument('--neighborhood_type', type=str, choices=['axes', 'grid'],
                        default='axes', help='Type of neighborhood to use.')
    parser.add_argument('--interpolation', type=str, choices=['efficient', 'dwi_ml'],
                        default='dwi_ml', help='Type of interpolation to use.')
    
    conv_group = parser.add_mutually_exclusive_group(required=True)
    conv_group.add_argument('--flatten_state', action='store_true',
                            help='Whether to flatten the state representation.')
    conv_group.add_argument('--conv_state', action='store_true',
                            help='Whether to use a convolutional state representation.')
    conv_group.add_argument('--fodf_encoder_ckpt', type=str, default=None,
                            help='Path to the encoder checkpoint to use for FODF input.'
                            'If provided, the neighborhood will be used as a convolutional input to that encoder.')
    parser.add_argument('--exclude_direct_neigh', action='store_true',
                            help='Whether to exclude the direct neighborhood from the state representation.')


def add_tracking_args(parser: ArgumentParser):
    parser.add_argument('--npv', default=2, type=int,
                        help='Number of random seeds per seeding mask voxel.')
    parser.add_argument('--theta', default=30, type=int,
                        help='Max angle between segments for tracking.')
    parser.add_argument('--min_length', type=float, default=20.,
                        metavar='m',
                        help='Minimum length of a streamline in mm. '
                        '[%(default)s]')
    parser.add_argument('--max_length', type=float, default=200.,
                        metavar='M',
                        help='Maximum length of a streamline in mm. '
                        '[%(default)s]')
    parser.add_argument('--step_size', default=0.75, type=float,
                        help='Step size for tracking')
    parser.add_argument('--noise', default=0.0, type=float, metavar='sigma',
                        help='Add noise ~ N (0, `noise`) to the agent\'s\n'
                        'output to make tracking more probabilistic.\n'
                        'Should be between 0.0 and 0.1.'
                        '[%(default)s]')


def add_tractometer_args(parser: ArgumentParser):
    tractom = parser.add_argument_group('Tractometer')
    tractom.add_argument('--scoring_data', type=str, default=None,
                         help='Location of the tractometer scoring data.')
    tractom.add_argument('--tractometer_reference', type=str, default=None,
                         help='Reference anatomy for the Tractometer.')
    tractom.add_argument('--tractometer_validator', action='store_true',
                         help='Run tractometer during validation to monitor' +
                         ' how the training is doing w.r.t. ground truth.')
    tractom.add_argument('--tractometer_dilate', default=1, type=int,
                         help='Dilation factor for the ROIs of the '
                              'Tractometer.')

def add_extractor_args(parser: ArgumentParser):
    extractor = parser.add_argument_group('Extractor')
    extractor.add_argument('--extractor_validator', action='store_true',
                           help='Run extractor during validation to monitor' +
                           ' how the training is doing w.r.t. ground truth.')
    extractor.add_argument('--extractor_sif_img_path', type=str, default=None,
                           help='Path to the Extractor singularity (.sif) image to use for the filterer.'
                           ' If not provided, the filterer will use the Docker image.')
    extractor.add_argument('--extractor_target', type=str, default=None,
                           help='Target file for the extractor.')
    extractor.add_argument('--extractor_templates', type=str, default=None,
                           help='Directory containing the templates for the extractor.')

def add_verifyber_args(parser: ArgumentParser):
    verifyber = parser.add_argument_group('Verifyber Filterer')
    verifyber.add_argument('--verifyber_validator', action='store_true',
                           help='Run the Verifyber filterer during validation to monitor '
                           'how the training is doing.')
    verifyber.add_argument('--verifyber_sif_img_path', type=str, default=None,
                           help='Path to the Verifyber singularity (.sif) image to use for the filterer.'
                           ' If not provided, the filterer will use the Docker image.')

def add_rbx_args(parser: ArgumentParser):
    rbx = parser.add_argument_group('RBX Filterer')
    rbx.add_argument('--rbx_validator', action='store_true',
                     help='Run the RBX filterer during validation to monitor '
                     'how the training is doing.')
    rbx.add_argument('--rbx_sif_img_path', type=str, default=None,
                     help='Path to the RBX singularity (.sif) image to use for the filterer.'
                     ' If not provided, the filterer will use the Docker image.')
    rbx.add_argument('--atlas_directory', type=str, default=None,
                     help='Directory containing the atlas for the RBX filterer.')

def add_oracle_args(parser: ArgumentParser):
    oracle = parser.add_argument_group('Oracle')
    oracle.add_argument('--oracle_reward_checkpoint', type=str,
                        default=None,
                        help='Checkpoint file (.ckpt) of the Oracle used for '
                        'rewarding.')
    oracle.add_argument('--oracle_crit_checkpoint', type=str, default=None,
                        help='Checkpoint file (.ckpt) of the Oracle used for '
                        'stopping criterion.')
    oracle.add_argument('--oracle_validator', action='store_true',
                        help='Run a TractOracle model during validation to '
                        'monitor how the training is doing.')
    oracle.add_argument('--oracle_stopping_criterion', action='store_true',
                        help='Stop streamlines according to the Oracle.')
    oracle.add_argument('--oracle_bonus', default=10, type=float,
                        help='Sparse oracle weighting for reward.')
