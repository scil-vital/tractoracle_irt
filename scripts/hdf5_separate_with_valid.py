import argparse
import h5py
import numpy as np
from tractoracle_irt.trainers.oracle.streamline_dataset_manager import _copy_and_update_dataset, _copy_to_hdf5_target, _create_datasets_hdf5, _copy_to_hdf5_by_chunks
from tractoracle_irt.utils.utils import prettier_dict
from tractoracle_irt.utils.logging import get_logger, add_logging_args, setup_logging

LOGGER = get_logger(__name__)

"""
This script is just used to convert hdf5 files with minimally a 'train' set
into hdf5 files with 'train', 'valid' and 'test' sets. Each set, including
the initial 'train' set, should have the following structure:
- 'data': the streamlines of shape (nb_streamlines, nb_points, 3)
- 'scores': the scores of shape (nb_streamlines, 1)
"""

def _build_args():
    parser = argparse.ArgumentParser(description='Separate the dataset into training and validation sets.')
    parser.add_argument('dataset_file', type=str, help='Path to the hdf5 file containing the dataset.')
    parser.add_argument('out_file', type=str, help='Path to the directory where the new datasets will be saved.')

    add_logging_args(parser)

    return parser.parse_args()

def _get_nb_streamlines(dataset):
    nb_train = 0 
    nb_valid = 0
    nb_test = 0

    if 'train' in dataset:
        nb_train = dataset['train/scores'].shape[0]
    if 'valid' in dataset:
        nb_valid = dataset['valid/scores'].shape[0]
    if 'test' in dataset:
        nb_test = dataset['test/scores'].shape[0]

    return nb_train, nb_valid, nb_test
        

def main():
    args = _build_args()
    setup_logging(args)
    with h5py.File(args.dataset_file, 'r') as original:
        with h5py.File(args.out_file, 'w') as target:
            rng = np.random.RandomState(42)
            LOGGER.info('Copying and updating the dataset...')
            LOGGER.debug('test for debug')
            _copy_and_update_dataset(original, target, rng)

            (original_nb_train, original_nb_valid, original_nb_test) = \
                _get_nb_streamlines(original)

            (target_nb_train, target_nb_valid, target_nb_test) = \
                _get_nb_streamlines(target)
    
    numbers_as_dict = {
        'original_nb_train': original_nb_train,
        'original_nb_valid': original_nb_valid,
        'original_nb_test': original_nb_test,
        'target_nb_train': target_nb_train,
        'target_nb_valid': target_nb_valid,
        'target_nb_test': target_nb_test
    }

    print(prettier_dict(numbers_as_dict))


if __name__ == '__main__':
    main()