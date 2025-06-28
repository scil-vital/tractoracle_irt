#!/usr/bin/env python
import argparse
import numpy as np
import torch

from argparse import RawTextHelpFormatter
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamline import set_number_of_points
from tqdm import tqdm

from scilpy.io.utils import (
    assert_inputs_exist, assert_outputs_exist, add_overwrite_arg)

from tractoracle_irt.filterers.oracle.oracle_filterer import OracleFilterer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cast_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _build_arg_parser(parser):
    parser.add_argument('tractogram', type=str,
                        help='Tractogram file to score.')
    parser.add_argument('--reference', type=str, default='same',
                        help='Reference file for tractogram (.nii.gz).'
                             'For .trk, can be \'same\'. Default is '
                             '[%(default)s].')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint (.ckpt) containing hyperparameters '
                             'and weights of model. Default is '
                             '[%(default)s].')

def parse_args():
    """ Filter a tractogram. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    _build_arg_parser(parser)
    args = parser.parse_args()

    assert_inputs_exist(parser, args.tractogram)

    return parser, args


def main():
    parser, args = parse_args()

    f = OracleFilterer(args.checkpoint, device)
    nb_valid, nb_total = f(args.tractogram, args.reference)

    print(f"Number of valid streamlines: {nb_valid}")
    print(f"Total number of streamlines: {nb_total}")


if __name__ == "__main__":
    main()