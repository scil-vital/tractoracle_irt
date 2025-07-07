#!/usr/bin/env python3
import argparse
import json
import nibabel as nib
import numpy as np
import os
import random
import torch
from dataclasses import dataclass, fields

from argparse import RawTextHelpFormatter

from dipy.io.utils import get_reference_info, create_tractogram_header
from nibabel.streamlines import detect_format
from scilpy.io.utils import (add_overwrite_arg,
                             add_sh_basis_args,
                             parse_sh_basis_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             verify_compression_th)
from scilpy.tracking.utils import verify_streamline_length_options

from tractoracle_irt.algorithms.sac_auto import SACAuto
from tractoracle_irt.algorithms.cross_q import CrossQ
from tractoracle_irt.algorithms.dro_q import DroQ
from tractoracle_irt.datasets.utils import MRIDataVolume
from tractoracle_irt.experiment.experiment import Experiment
from tractoracle_irt.tracking.tracker import Tracker
from tractoracle_irt.utils.config.public_files import download_if_public_file, is_public_file
from tractoracle_irt.utils.torch_utils import get_device
from tractoracle_irt.utils.logging import get_logger
from tractoracle_irt.utils.utils import prettier_dict

LOGGER = get_logger(__name__)

@dataclass
class TrackConfig:
    in_odf: str
    in_mask: str
    in_seed: str
    input_wm: bool
    out_tractogram: str
    noise: float
    binary_stopping_threshold: float
    n_actor: int
    npv: int
    in_peaks: str
    min_length: int
    max_length: int
    save_seeds: bool
    agent_checkpoint: str
    rng_seed: int

    algorithm: str = None
    fa_map_file: str = None # Optional
    compress: float = 0.0

    def __post_init__(self):
        self.dataset_file = None
        self.subject_id = None
        self.tractometer_validator = False
        self.scoring_data = None
        self.compute_reward = False
        self.use_classic_reward = False
        self.render = False
        self.reward_with_gt = False

        self.reference_file = self.in_mask
        self.alignment_weighting = 0.0
        self.oracle_reward_checkpoint = None
        self.oracle_crit_checkpoint = None
        self.oracle_bonus = 0
        self.oracle_validator = False
        self.oracle_stopping_criterion = False
        self.exclude_direct_neigh = False

    @classmethod
    def from_dict(cls, config: dict, filter_extra_keys=True):
        if filter_extra_keys:
            valid_keys = {field.name for field in fields(cls) if field.init}
            filtered_config = {k: v for k, v in config.items() if k in valid_keys}
            extra_keys = set(config.keys()) - valid_keys
            if extra_keys:
                print(f"Warning: Ignoring unsupported parameters: {extra_keys}")
            return cls(**filtered_config)
        else:
            return cls(**config)
        
    def update_with_dict(self, config, overwrite=False):
        # This function iterates over the config file. If the key is already
        # present in the object, it won't do anything unless overwrite is True.
        # If the key is not present, it will add it to the object.
        for key, value in config.items():
            if hasattr(self, key):
                if overwrite:
                    setattr(self, key, value)
                elif key == 'algorithm':
                    # Special case for algorithm, we want to overwrite it
                    # only if its value if None.
                    if getattr(self, key) is None:
                        setattr(self, key, value)
            else:
                setattr(self, key, value)


class Track(Experiment):
    """ TractOracleIRT testing script. Should work on any model trained with a
    TractOracleIRT experiment
    """

    def __init__(
        self,
        track_dto,
    ):
        """
        """

        self.hp = self.hparams_class.from_dict(track_dto)
        self.in_odf = self.hp.in_odf
        self.wm_file = self.hp.in_mask

        self.in_seed = self.hp.in_seed
        self.in_mask = self.hp.in_mask
        self.input_wm = self.hp.input_wm
        self.in_fa = self.hp.fa_map_file
        self.in_peaks = self.hp.in_peaks

        self.noise = self.hp.noise
        self.binary_stopping_threshold = self.hp.binary_stopping_threshold
        self.n_actor = self.hp.n_actor
        self.npv = self.hp.npv
        self.min_length = self.hp.min_length
        self.max_length = self.hp.max_length
        self.compress = self.hp.compress
        (self.sh_basis, self.is_sh_basis_legacy) = parse_sh_basis_arg(argparse.Namespace(**track_dto))
        self.save_seeds = self.hp.save_seeds
        self.compute_reward = False

        self.device = get_device()

        self.fa_map = None
        if self.hp.fa_map_file is not None:
            fa_image = nib.load(self.hp.fa_map_file)
            self.fa_map = MRIDataVolume(
                data=fa_image.get_fdata(),
                affine_vox2rasmm=fa_image.affine)

        def load_hyperparameters(hparams_path):
            with open(hparams_path, 'r') as f:
                hparams = json.load(f)
            return hparams
        
        was_public_file = is_public_file(self.hp.agent_checkpoint)
        self.hp.agent_checkpoint = download_if_public_file(self.hp.agent_checkpoint)
        if was_public_file:
            # The previous method returns the path of the directory.
            # We need to get the checkpoint file from there.
            self.hp.agent_checkpoint = os.path.join(
                self.hp.agent_checkpoint, 'last_model_state.ckpt')

        checkpoint_dir = os.path.dirname(self.hp.agent_checkpoint)
        self.hparams = load_hyperparameters(os.path.join(
            checkpoint_dir, 'hyperparameters.json'))
        
        self.hp.update_with_dict(self.hparams, overwrite=False)

        torch.manual_seed(self.hp.rng_seed)
        np.random.seed(self.hp.rng_seed)
        random.seed(self.hp.rng_seed)
        self.rng = np.random.RandomState(seed=self.hp.rng_seed)

        self.comet_experiment = None
        self.discard_dps = True

    @property
    def hparams_class(self):
        return TrackConfig

    def run(self):
        """
        Main method where the magic happens
        """
        # Presume iso vox
        ref_img = nib.load(self.hp.reference_file)
        tracking_voxel_size = ref_img.header.get_zooms()[0]

        # # Set the voxel size so the agent traverses the same "quantity" of
        # # voxels per step as during training.
        step_size_mm = self.hp.step_size
        if abs(float(tracking_voxel_size) - float(self.hp.voxel_size)) >= 0.1:
            step_size_mm = (
                float(tracking_voxel_size) / float(self.hp.voxel_size)) * \
                self.hp.step_size

            print("Agent was trained on a voxel size of {}mm and a "
                  "step size of {}mm.".format(self.hp.voxel_size, self.hp.step_size))

            print("Subject has a voxel size of {}mm, setting step size to "
                  "{}mm.".format(tracking_voxel_size, step_size_mm))

        # Instanciate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline
        env = self.get_tracking_env()
        env.step_size_mm = step_size_mm

        # Get example state to define NN input size
        self.input_size = env.get_state_size()
        self.action_size = env.get_action_size()

        # Load agent
        algs = {'SACAuto': SACAuto, 'CrossQ': CrossQ, 'DroQ': DroQ}

        verify_algorithm_specified(self.hp)
        rl_alg = algs[self.hp.algorithm]
        print('Tracking with {} agent.'.format(self.hp.algorithm))
        # The RL training algorithm
        if rl_alg == CrossQ:
            alg = rl_alg(
                self.input_size,
                self.action_size,
                None,
                self.hp,
                rng=self.rng,
                device=self.device)
        else:
            alg = rl_alg(
                self.input_size,
                self.action_size,
                self.hp,
                rng=self.rng,
                device=self.device)

        # Load pretrained policies
        if self.hp.agent_checkpoint:
            # Load the bundled checkpoint file.
            alg.load_checkpoint(self.hp.agent_checkpoint)
        else:
            LOGGER.warning('No agent checkpoint provided. Exiting.')
            return

        # Run tracking
        env.load_subject()

        # Initialize Tracker, which will handle streamline generation
        tracker = Tracker(
            alg, self.hp.n_actor, compress=self.hp.compress,
            min_length=self.hp.min_length, max_length=self.hp.max_length,
            save_seeds=self.hp.save_seeds, prob=1.0)
        
        filetype = detect_format(self.hp.out_tractogram)
        
        tractogram = tracker.track(env, filetype)

        reference = get_reference_info(self.hp.reference_file)
        header = create_tractogram_header(filetype, *reference)

        # Use generator to save the streamlines on-the-fly
        nib.streamlines.save(tractogram, self.hp.out_tractogram, header=header)

def verify_algorithm_specified(config):
    # Make sure that the algorithm field is not None.
    if config.algorithm is None:
        raise ValueError("The 'algorithm' field in the hyperparameters must be specified. If not "
                         " it must be set to 'SACAuto', 'CrossQ' or 'DroQ' using the --algorithm argument.")

def add_mandatory_options_tracking(p):
    p.add_argument('in_odf',
                   help='File containing the orientation diffusion function \n'
                        'as spherical harmonics file (.nii.gz). Ex: ODF or '
                        'fODF.\nCan be of any order and basis (including "full'
                        '" bases for\nasymmetric ODFs). See also --sh_basis.')
    p.add_argument('in_seed',
                   help='Seeding mask (.nii.gz). Must be represent the WM/GM '
                        'interface.')
    p.add_argument('in_mask',
                   help='Tracking mask (.nii.gz).\n'
                        'Tracking will stop outside this mask.')
    p.add_argument('--out_tractogram', type=str, metavar='FILE',
                   default=os.environ.get('TRACTORACLE_IRT_OUTPUT_TRACTOGRAM', 'tractogram.trk'),
                   help='Tractogram output file (must be .trk or .tck).')
    p.add_argument('--input_wm', action='store_true',
                   help='If set, append the WM mask to the input signal. The '
                        'agent must have been trained accordingly.')


def add_out_options(p):
    out_g = p.add_argument_group('Output options')
    out_g.add_argument('--compress', type=float, metavar='thresh', default=0.0,
                       help='If set, will compress streamlines. The parameter '
                            'value is the \ndistance threshold. A rule of '
                            'thumb is to set it to 0.1mm for \ndeterministic '
                            'streamlines and 0.2mm for probabilitic '
                            'streamlines [%(default)s].')
    add_overwrite_arg(out_g)
    out_g.add_argument('--save_seeds', action='store_true',
                       help='If set, save the seeds used for the tracking \n '
                            'in the data_per_streamline property.\n'
                            'Hint: you can then use '
                            'scilpy_compute_seed_density_map.')
    return out_g


def add_track_args(parser):

    add_mandatory_options_tracking(parser)

    basis_group = parser.add_argument_group('Basis options')
    add_sh_basis_args(basis_group)
    parser.add_argument('--algorithm', choices=['SACAuto', 'CrossQ', 'DroQ'], help='The algorithm to use for tracking.\n'
                   'The algorithm should be already specified in the agent checkpoint hyperparameters, but this option will override it.')
    add_out_options(parser)
    agent_group = parser.add_argument_group('Tracking agent options')
    agent_group.add_argument('--agent_checkpoint', type=str, default="public://sac_irt_inferno", metavar='FILE',
                                        help='Path to the agent checkpoint FILE to load. There must be a hyperparameters.json file in the same directory.\n'
                                        'If the path is a public file, it will be downloaded automatically.')

    agent_group.add_argument('--n_actor', type=int, default=10000, metavar='N',
                             help='Number of streamlines to track simultaneous'
                             'ly.\nLimited by the size of your GPU and RAM. A '
                             'higher value\nwill speed up tracking up to a '
                             'point [%(default)s].')
    agent_group.add_argument('--big_neighborhood', action='store_true',
                                help='If set, the agent will consider a larger '
                                'neighborhood\naround the current position to '
                                'make decisions. This\nwill slow down tracking.')

    seed_group = parser.add_argument_group('Seeding options')
    seed_group.add_argument('--npv', type=int, default=1,
                            help='Number of seeds per voxel [%(default)s].')
    track_g = parser.add_argument_group('Tracking options')
    track_g.add_argument('--min_length', type=float, default=20.,
                         metavar='m',
                         help='Minimum length of a streamline in mm. '
                         '[%(default)s]')
    track_g.add_argument('--max_length', type=float, default=200.,
                         metavar='M',
                         help='Maximum length of a streamline in mm. '
                         '[%(default)s]')
    track_g.add_argument('--noise', default=0.0, type=float, metavar='sigma',
                         help='Add noise ~ N (0, `noise`) to the agent\'s\n'
                         'output to make tracking more probabilistic.\n'
                         'Should be between 0.0 and 0.1.'
                         '[%(default)s]')
    track_g.add_argument('--fa_map', type=str, default=None,
                         help='Scale the added noise (see `--noise`) according'
                         '\nto the provided FA map (.nii.gz). Optional.')
    track_g.add_argument('--in_peaks', type=str, default=None,
                         help='File containing the peaks volume. If not provided,'
                         'the peaks will be computed from the ODF (slightly longer at startup).')
    track_g.add_argument(
        '--binary_stopping_threshold',
        type=float, default=0.1,
        help='Lower limit for interpolation of tracking mask value.\n'
             'Tracking will stop below this threshold.')
    parser.add_argument('--rng_seed', default=1337, type=int,
                        help='Random number generator seed [%(default)s].')

def parse_args():
    """ Generate a tractogram from a trained model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_track_args(parser)
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_odf, args.in_seed, args.in_mask])
    assert_outputs_exist(parser, args, args.out_tractogram)
    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk or ' +
                     'tck): {0}'.format(args.out_tractogram))

    verify_streamline_length_options(parser, args)
    verify_compression_th(args.compress)

    return args


def main():
    """ Main tracking script """
    args = parse_args()

    experiment = Track(
        vars(args)
    )

    experiment.run()


if __name__ == '__main__':
    main()
