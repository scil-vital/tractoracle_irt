#!/usr/bin/env python
import argparse

import h5py
import json
import numpy as np

from argparse import RawTextHelpFormatter
from glob import glob
from os.path import expanduser
from tqdm import tqdm

from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import set_number_of_points
from nibabel.streamlines import load

"""
Script to process multiple subjects into a single .hdf5 file.
"""


def generate_dataset(
    config_file: str,
    dataset_file: str,
    nb_points: int = 128,
    max_streamline_subject: int = -1
) -> None:
    """ Generate a dataset from a configuration file and save it to disk.

    Parameters:
    -----------
    config_file: str
        Path to the configuration file containing the subjects and their
        streamlines.
    dataset_file: str
        Path to the output file where the dataset will be saved.
    nb_points: int
        Number of points to resample the streamlines to.
    max_streamline_subject: int, optional
        Maximum number of streamlines to use per subject. Default is -1,
        meaning all streamlines are used.
    """
    # Initialize database
    with h5py.File(dataset_file, 'w') as hdf_file:
        # Save version
        hdf_file.attrs['version'] = 1
        hdf_file.attrs['nb_points'] = nb_points

        with open(config_file, "r") as conf:
            config = json.load(conf)

            train_group = hdf_file.create_group('train')
            valid_group = hdf_file.create_group('valid')
            test_group = hdf_file.create_group('test')

            print(">>>>>>>>>>> Processing training set <<<<<<<<<<<")
            add_subjects_to_hdf5(
                config['train'], train_group, nb_points, max_streamline_subject)
            print(">>>>>>>>>>> Processing valid set <<<<<<<<<<<")
            add_subjects_to_hdf5(
                config['valid'], valid_group, nb_points, max_streamline_subject)
            print(">>>>>>>>>>> Processing test set <<<<<<<<<<<")
            add_subjects_to_hdf5(
                config['test'], test_group, nb_points, max_streamline_subject)

    print("Saved dataset : {}".format(dataset_file))


def add_subjects_to_hdf5(
    config, hdf_file, nb_points=128, max_streamline_subject=-1
):
    """ Process the subjects and add them to the hdf5 file.

    Parameters
    ----------
    config: dict
        Dictionary containing the subjects and their streamlines.
    hdf_file: h5py.File
        HDF5 file to save the dataset to.
    nb_points: int, optional
        Number of points to resample the streamlines to.
    max_streamline_subject: int, optional
        Maximum number of streamlines to use per subject. Default is -1,
        meaning all streamlines are used.
    """
    sub_files = []
    for subject_id in config:
        "Processing subject {}".format(subject_id),

        subject_config = config[subject_id]

        reference_anat = subject_config['reference']
        pos_streamlines_files_list = subject_config['pos_streamlines']
        neg_streamlines_files_list = subject_config['neg_streamlines']

        sub_files.append((reference_anat, pos_streamlines_files_list, neg_streamlines_files_list))

    process_subjects(sub_files, hdf_file, nb_points, max_streamline_subject)


def process_subjects(
    sub_files, hdf_subject, nb_points, max_streamline_subject
):
    """ Process the subjects and add them to the hdf5 file. First,
    the size of the dataset is computed, then the streamlines are
    loaded and assigned a random index so that they are spread
    across the dataset. Then, the streamlines are added to the
    dataset.

    Parameters
    ----------
    sub_files: list
        List of tuples containing the reference anatomy and the
        streamlines files for each subject.
    hdf_subject: h5py.File
        HDF5 file to save the dataset to.
    nb_points: int, optional
        Number of points to resample the streamlines to.
    max_streamline_subject: int, optional
        Maximum number of streamlines to use per subject. Default is -1,
        meaning all streamlines are used.
    """

    total = 0
    idx = 0
    max_strml = max_streamline_subject if max_streamline_subject > 0 else np.inf

    print('Computing size of dataset.')
    for anat, pos_strm_files, neg_strm_files in tqdm(sub_files):
        pos_streamlines_files = glob(expanduser(pos_strm_files[0]))
        neg_streamlines_files = glob(expanduser(neg_strm_files[0]))        
        anat_path = glob(expanduser(anat))[0]
        for pos_bundle, neg_bundle in zip(pos_streamlines_files, neg_streamlines_files):
            pos_len_p = load_streamlines(expanduser(pos_bundle), anat_path)
            neg_len_p = load_streamlines(expanduser(neg_bundle), anat_path)

            nb = min(max_strml, min(len(pos_len_p.streamlines), len(neg_len_p.streamlines)))
            total += nb*2

    print('Dataset will have {} streamlines'.format(total))
    print('Writing streamlines to dataset.')

    # Randomize the order of the streamlines
    idices = np.arange(total)
    np.random.shuffle(idices)

    # Add the streamlines to the dataset
    for anat, pos_strm_files, neg_strm_files in tqdm(sub_files):
        pos_streamlines_files = glob(expanduser(pos_strm_files[0]))
        neg_streamlines_files = glob(expanduser(neg_strm_files[0]))
        anat_path = glob(expanduser(anat))[0]
        for pos_bundle, neg_bundle in zip(pos_streamlines_files, neg_streamlines_files):
            # Load the streamlines
            pos_ps = load_streamlines(pos_bundle, anat_path, assign_score=1)
            neg_ps = load_streamlines(neg_bundle, anat_path, assign_score=0)

            # Make sure that there's the same number of positive and negative
            # streamlines
            nb_indices = min(max_strml, min(len(pos_ps.streamlines), len(neg_ps.streamlines)))
            pos_ps_indices = np.random.choice(len(pos_ps.streamlines), nb_indices, replace=False)
            neg_ps_indices = np.random.choice(len(neg_ps.streamlines), nb_indices, replace=False)

            print('Processing {} and {}'.format(pos_bundle, neg_bundle))
            # Get the indices to use
            idx = idices[:len(pos_ps_indices) + len(neg_ps_indices)]
            combined = pos_ps[pos_ps_indices] + neg_ps[neg_ps_indices]
            
            assert len(idx) == len(combined), "Indices and streamlines lengths don't match ({} and {})".format(len(idx), len(combined))

            # Add the streamlines to the dataset
            add_streamlines_to_hdf5(
                hdf_subject, combined, nb_points, total, idx)
            # Remove the indices that have been used
            idices = idices[len(pos_ps_indices) + len(neg_ps_indices):]

        # neg_streamlines_files = glob(expanduser(neg_strm_files[0]))
        # for bundle in neg_streamlines_files:
        #     # Load the streamlines
        #     ps = load_streamlines(bundle, anat_path, assign_score=0)
        #     # Randomize the order of the streamlines
        #     ps_idices = np.random.choice(len(ps), min(max_strml, len(ps)),
        #                                  replace=False)
        #     print('Processing {}'.format(bundle))
        #     # Get the indices to use
        #     idx = idices[:len(ps_idices)]
        #     # Add the streamlines to the dataset
        #     add_streamlines_to_hdf5(
        #         hdf_subject, ps[ps_idices], nb_points, total, idx)
        #     # Remove the indices that have been used
        #     idices = idices[len(ps_idices):]


def load_streamlines(
    streamlines_file: str,
    reference,
    assign_score: int = None
):
    """ Load the streamlines from a file and make sure they are in
    voxel space and aligned with the corner of the voxels.

    Parameters
    ----------
    streamlines_file: str
        Path to the file containing the streamlines.
    reference: str
        Path to the reference anatomy file.
    nb_points: int, optional
        Number of points to resample the streamlines to.
    """

    sft = load_tractogram(streamlines_file, reference, bbox_valid_check=False)
    sft.to_corner()
    sft.to_vox()

    assert assign_score == 0 or assign_score == 1 or assign_score is None
    if assign_score is not None and assign_score == 0:
        sft.data_per_streamline['score'] = np.zeros(len(sft.streamlines))
    elif assign_score is not None and assign_score == 1:
        sft.data_per_streamline['score'] = np.ones(len(sft.streamlines))

    return sft


def add_streamlines_to_hdf5(hdf_subject, sft, nb_points, total, idx):
    """ Add the streamlines to the hdf5 file.

    Parameters
    ----------
    hdf_subject: h5py.File
        HDF5 file to save the dataset to.
    sft: nib.streamlines.tractogram.Tractogram
        Streamlines to add to the dataset.
    nb_points: int, optional
        Number of points to resample the streamlines to
    total: int
        Total number of streamlines in the dataset
    idx: list
        List of positions to store the streamlines
    """

    # Get the scores and the streamlines
    scores = np.asarray(sft.data_per_streamline['score']).squeeze(-1)
    # Resample the streamlines
    streamlines = set_number_of_points(sft.streamlines, nb_points)
    streamlines = np.asarray(streamlines)

    # Create the dataset if it does not exist
    if 'data' not in hdf_subject:
        # Set the number of points
        streamlines = np.asarray(streamlines)
        # 'data' will contain the streamlines
        hdf_subject.create_dataset(
            'data', shape=(total, nb_points, streamlines.shape[-1]))
        # 'scores' will contain the scores
        hdf_subject.create_dataset('scores', shape=(total))

    data_group = hdf_subject['data']
    scores_group = hdf_subject['scores']

    for i, st, sc in zip(idx, streamlines, scores):
        data_group[i] = st
        scores_group[i] = sc


def parse_args():

    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)
    parser.add_argument('config_file', type=str,
                        help="Configuration file to load subjects and their"
                        " volumes.")
    parser.add_argument('output', type=str,
                        help="Output filename including path.")
    parser.add_argument('--nb_points', type=int, default=128,
                        help='Number of points to resample streamlines to.'
                             ' Default is [%(default)s].')
    parser.add_argument('--max_streamline_subject', type=int, default=-1,
                        help='Maximum number of streamlines per subject. '
                             'Default is -1, meaning all streamlines are '
                             'used.')

    arguments = parser.parse_args()

    return arguments


def main():
    """ Parse args, generate dataset and save it on disk """
    args = parse_args()

    generate_dataset(config_file=args.config_file,
                     dataset_file=args.output,
                     nb_points=args.nb_points,
                     max_streamline_subject=args.max_streamline_subject)


if __name__ == "__main__":
    main()
