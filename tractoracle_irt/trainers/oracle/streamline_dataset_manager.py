import os
import h5py
import numpy as np

from dipy.tracking.streamline import set_number_of_points
from dipy.io.stateful_tractogram import StatefulTractogram
from tqdm import tqdm
from dataclasses import dataclass, field

from tractoracle_irt.utils.logging import get_logger
from tractoracle_irt.utils.hdf5_utils import copy_by_batch

DEFAULT_DATASET_NAME = "new_dataset.hdf5"
DATA = 'data'
SCORES = 'scores'

LOGGER = get_logger(__name__)

"""
The StreamlineDatasetManager manages a HDF5 file containing the streamlines
and their respective scores used to train the Oracle model. Here we manage
the creation of the dataset if needed, and the addition of new streamlines
to the HDF5 file (the dataset).

The latter has a specific structure:
attrs:
    - version: the version of the dataset
    - nb_points: the number of points in the streamlines (e.g. 128)

dataset:
    - train (~90% of the data)
        - streamlines (N, 128, 3): the streamlines
        - scores (N,): the class/scores of the streamlines (0 or 1)
    - test (~10% of the data)
        - streamlines (N, 128, 3): the streamlines
        - scores (N,): the class/scores of the streamlines (0 or 1)
"""

@dataclass
class MaxDatasetSize:
    max_dataset_size: int
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    
    train: int = field(init=False)
    valid: int = field(init=False)
    test: int = field(init=False)

    def __post_init__(self):
        self.train_ratio = 1 - self.test_ratio - self.valid_ratio

        assert self.train_ratio > 0, "The training ratio must be greater " \
                "than 0 because you have a valid_ratio and a test_ratio of " \
                f"{self.valid_ratio} and {self.test_ratio} respectively."

        self.train = round(self.max_dataset_size * self.train_ratio)
        self.valid = round(self.max_dataset_size * self.valid_ratio)
        self.test = round(self.max_dataset_size * self.test_ratio)
        assert self.train + self.valid + self.test \
            == self.max_dataset_size, "Rounding issue with the max dataset sizes."

def _create_datasets_hdf5(group: h5py.File, size, nb_points, point_dim,
                          maxsize=None):
    data = group.create_dataset(
        DATA,
        shape=(size,
                nb_points,
                point_dim),
        maxshape=(maxsize, nb_points, point_dim))
    
    scores = group.create_dataset(
        SCORES,
        shape=(size,),
        maxshape=(maxsize,))
    
    LOGGER.debug(f"Created datasets with shape: {data.shape} and "
                 f"{scores.shape}")

    return data, scores

def _split_train_by_chunks(original, target, train_indices, valid_indices, test_indices, enable_pbar=True):
    """
    This function 'efficiently' copies the data from the original 'train' dataset
    and splits it into 'train', 'valid' and 'test' datasets in the target file.
    It loads each chunk of the 'train' dataset and splits it according to the
    provided indices.
    """
    
    # Quick sanity to make sure we don't overwrite any data.
    has_valid = 'valid' in original.keys()
    has_test = 'test' in original.keys()
    assert not has_valid or valid_indices.size == 0, \
        "The dataset already contains a 'valid' group."
    assert not has_test or test_indices.size == 0, \
        "The dataset already contains a 'test' group."
    assert not np.intersect1d(train_indices, valid_indices).any(), \
        "The train and valid indices overlap."
    assert not np.intersect1d(train_indices, test_indices).any(), \
        "The train and test indices overlap."
    assert not np.intersect1d(valid_indices, test_indices).any(), \
        "The valid and test indices overlap."
    
    from tractoracle_irt.utils.utils import SimpleTimer

    batch_size = 1000000
    total_size = original[f'train/{DATA}'].shape[0]
    num_batches = (total_size // batch_size) + (total_size % batch_size != 0)

    original_data = original[f'train/{DATA}']
    original_scores = original[f'train/{SCORES}']
    global_indices = np.arange(total_size)

    train_write_pos = 0 # Position in the file where we should 
    valid_write_pos = 0 # start writing the data.
    test_write_pos = 0

    for batch_start in tqdm(range(0, total_size, batch_size),
                            desc="Splitting/copying into new dataset",
                            total=num_batches,
                            leave=False, disable=not enable_pbar):
        with SimpleTimer() as t_read:
            batch_end = min(batch_start + batch_size, total_size)
            data = original_data[batch_start:batch_end]
            scores = original_scores[batch_start:batch_end]

            # Map chunk indices to global indices
            chunk_global_indices = global_indices[batch_start:batch_end]

            # Split batch into train, valid, and test
            train_mask = np.isin(chunk_global_indices, train_indices)
            valid_mask = np.isin(chunk_global_indices, valid_indices)
            test_mask = np.isin(chunk_global_indices, test_indices)

        with SimpleTimer() as t_write:
            # Write data to target datasets
            if train_mask.any():
                nb_new_train = train_mask.sum()
                train_write_slice = slice(
                    train_write_pos, train_write_pos + nb_new_train)
                target['train/data'][train_write_slice] = data[train_mask]
                target['train/scores'][train_write_slice] = scores[train_mask]
                train_write_pos += nb_new_train

            if valid_mask.any():
                nb_new_valid = valid_mask.sum()
                valid_write_slice = slice(
                    valid_write_pos, valid_write_pos + nb_new_valid)
                target['valid/data'][valid_write_slice] = data[valid_mask]
                target['valid/scores'][valid_write_slice] = scores[valid_mask]
                valid_write_pos += nb_new_valid

            if test_mask.any():
                nb_new_test = test_mask.sum()
                test_write_slice = slice(
                    test_write_pos, test_write_pos + nb_new_test)
                target['test/data'][test_write_slice] = data[test_mask]
                target['test/scores'][test_write_slice] = scores[test_mask]
                test_write_pos += nb_new_test

        LOGGER.debug(f"Read time: {t_read.interval}s | Write time: {t_write.interval}s")

    assert train_write_pos == train_indices.size, \
        "Not all training streamlines were copied."
    assert valid_write_pos == valid_indices.size, \
        "Not all validation streamlines were copied."
    assert test_write_pos == test_indices.size, \
        "Not all testing streamlines were copied."
        
def _hdf5_resample_and_copy(original, target, group, rng, indices=None):
    """
    This function is meant to be used almost identically to the
    h5py.copy() function, but with the added functionality of resampling
    the number of streamlines to the specified indices. This is useful
    mainly when we don't want our dataset to exceed a certain size upon
    copying it to the new file.
    """
    # Otherwise, resample the data and copy it to the target file.
    if group not in original.keys():
        raise ValueError(f"Group '{group}' not found in the original dataset.")

    # If no resampling needed, just use the faster method here.
    if indices is None:
        original.copy(group, target)
        return original[group][DATA].shape[0]
    
    indices = np.sort(indices) # Required by h5py
    
    original_group = original[group]
    original_data = original_group[DATA]
    original_scores = original_group[SCORES]

    # Resample the data
    # LOGGER.info(f"Resampling the '{group}' dataset.")
    # data = original_data[indices]
    # scores = original_scores[indices]

    # Copy the data to the target file
    LOGGER.info(f"Creating the '{group}' group in the target file.")
    target_group = target.create_group(group)
    ds_data, ds_scores = _create_datasets_hdf5(target_group, len(indices),
                                            original.attrs['nb_points'],
                                            original_data.shape[-1])
    
    LOGGER.info(f"Copying the '{group}' data to the target file.")
    # Copies data/scores from original to target efficiently.
    # It replaces the following:
    #   ds_data[:] = data
    #   ds_scores[:] = scores
    # Takes around ~1.5 minfor test set to be copied.
    total_copied = copy_by_batch([original_data, original_scores],
                                 [ds_data, ds_scores],
                                 indices, f"Copying {group} data/scores")
    
    return total_copied

def _copy_and_update_dataset(original, target, rng, max_sizes: MaxDatasetSize):
    """
    Copy the dataset to the target file and update the version number.

    However, the original dataset must at the very least contain the 'train'
    group. If it does not contain any of the 'valid' and 'test' groups, we
    will split the 'train' group into 'train', 'valid' and 'test' groups.
    """
    LOGGER.info("Max sizes to respect: Train: {}, Valid: {}, Test: {}".format(
        max_sizes.train, max_sizes.valid, max_sizes.test)
    )

    target.attrs['version'] = original.attrs['version']
    target.attrs['nb_points'] = original.attrs['nb_points']

    train_len, valid_len, test_len = None, None, None

    has_train = 'train' in original.keys()
    has_valid = 'valid' in original.keys()
    has_test = 'test' in original.keys()

    if not has_train:
        raise ValueError("The dataset does not contain the 'train' group.")

    # If 'valid' is there, just copy it while respecting the max dataset size.
    if has_valid:
        LOGGER.info("The dataset already contains the 'valid' group, copying it.")

        valid_resampling_indices = None
        # Resample if we exceed the maximum size
        if max_sizes.valid < original['valid'][DATA].shape[0]:
            LOGGER.info("Resampling the 'valid' dataset.")
            valid_resampling_indices = rng.choice(
                original['valid'][DATA].shape[0],
                max_sizes.valid, replace=False)

        valid_len = _hdf5_resample_and_copy(original, target, 'valid', rng,
                                            valid_resampling_indices)
    
    # If 'test' is there, just copy it while respecting the max dataset size.
    if has_test:
        LOGGER.info("The dataset already contains the 'test' group, copying it.")

        test_resampling_indices = None
        # Resample if we exceed the maximum size
        if max_sizes.test < original['test'][DATA].shape[0]:
            LOGGER.info(f"Resampling the 'test' dataset ({original['test'][DATA].shape[0]}->{max_sizes.test}).")
            test_resampling_indices = rng.choice(
                original['test'][DATA].shape[0],
                max_sizes.test, replace=False)
        
        test_len = _hdf5_resample_and_copy(original, target, 'test', rng,
                                           test_resampling_indices)
    
    # If both 'valid' and 'test' were copied, then we can just copy the 
    # 'train' as well.
    if has_train and has_valid and has_test:
        # Just copy the train dataset to the target file.
        LOGGER.info("The dataset already contains the 'train', 'valid' "
                        "and 'test' groups. Copying the train dataset.")
        
        train_resampling_indices = None
        # Resample if we exceed the maximum size
        if max_sizes.train < original['train'][DATA].shape[0]:
            LOGGER.info("Resampling the 'train' dataset.")
            train_resampling_indices = rng.choice(
                original['train'][DATA].shape[0],
                max_sizes.train, replace=False)
            
        train_len = _hdf5_resample_and_copy(original, target, 'train', rng,
                                            train_resampling_indices)
    else:
        # Split the 'train' dataset into 'train', 'valid' and 'test' datasets
        # for any or both 'valid' or 'test' groups missing.

        train_group = original['train']
        nb_train_streamlines = train_group[SCORES].shape[0]
        if nb_train_streamlines <= 0:
            raise ValueError("The training dataset is empty.") 

        LOGGER.info("Generating the indices for the training, validation and "
                    "test sets.")

        indices = np.arange(nb_train_streamlines)
        rng.shuffle(indices)

        # Split the training dataset into training and validation
        nb_valid_streamlines = 0 if has_valid else round(nb_train_streamlines * 0.1)
        nb_test_streamlines = 0 if has_test else round(nb_train_streamlines * 0.1)
        nb_train_streamlines = nb_train_streamlines - nb_valid_streamlines - nb_test_streamlines

        # NB: The number of indices has to respect our maximum sizes.
        nb_valid_streamlines = min(nb_valid_streamlines, max_sizes.valid)
        nb_test_streamlines = min(nb_test_streamlines, max_sizes.test)
        nb_train_streamlines = min(nb_train_streamlines, max_sizes.train)

        # Select randomly indices for the validation set
        train_indices = indices[:nb_train_streamlines]
        valid_indices = np.array([]) if has_valid else indices[nb_train_streamlines:nb_train_streamlines + nb_valid_streamlines]
        test_indices = np.array([]) if has_test else indices[nb_train_streamlines + nb_valid_streamlines:]

        LOGGER.info("New indices. Train: {}, Valid: {}, Test: {}".format(
            train_indices.shape,
            valid_indices.shape,
            test_indices.shape))

        # Need to sort to be able to index HDF5 files.
        train_indices.sort()
        valid_indices.sort()
        test_indices.sort()

        # Create TRAIN hdf5 dataset
        target_train_group = target.create_group('train')
        _create_datasets_hdf5(target_train_group, nb_train_streamlines,
                            original.attrs['nb_points'],
                            train_group[DATA].shape[-1])
        train_len = nb_train_streamlines

        # Create VALID hdf5 dataset
        if not has_valid:
            LOGGER.info("Creating the validation dataset.")
            target_valid_group = target.create_group('valid')
            _create_datasets_hdf5(target_valid_group, nb_valid_streamlines,
                                original.attrs['nb_points'],
                                train_group[DATA].shape[-1])
            valid_len = nb_valid_streamlines
        
        # Create TEST hdf5 dataset
        if not has_test:
            LOGGER.info("Creating the test dataset.")
            target_test_group = target.create_group('test')
            _create_datasets_hdf5(target_test_group, nb_test_streamlines,
                                    original.attrs['nb_points'],
                                    train_group[DATA].shape[-1])
            test_len = nb_test_streamlines

        # Copy the data to the target file
        _split_train_by_chunks(original, target, train_indices,
                                valid_indices, test_indices)

    assert train_len is not None
    assert valid_len is not None
    assert test_len is not None

    return train_len, valid_len, test_len

class StreamlineDatasetManager(object):
    def __init__(self,
                 saving_path: str,
                 dataset_to_augment_path: str = None,
                 augment_in_place: bool = False,
                 dataset_name: str = DEFAULT_DATASET_NAME,
                 nb_points: int = 128,
                 valid_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 add_batch_size: int = 1000,
                 max_dataset_size: int = 5_000_000,
                 rng_seed: int = -1):
        self.add_batch_size = add_batch_size
        self.nb_points = nb_points
        self.maxes = MaxDatasetSize(max_dataset_size, valid_ratio, test_ratio)
        self.rng = np.random.RandomState(rng_seed) if rng_seed >= 0 else np.random.RandomState()

        if not os.path.exists(saving_path) and not saving_path == "":
            raise FileExistsError(
                f"The saving path {saving_path} does not exist.")

        if dataset_to_augment_path is not None:
            LOGGER.info("Loading the dataset to augment: {}".format(dataset_to_augment_path))
            (self.current_train_nb_streamlines,
             self.current_valid_nb_streamlines,
             self.current_test_nb_streamlines) = \
                self._load_and_verify_streamline_dataset(
                dataset_to_augment_path)

            if not augment_in_place:
                # We don't want to modify the original dataset, so we copy it to the saving_path
                self.dataset_file_path = os.path.join(
                    saving_path, dataset_name)

                # Copy the dataset to the saving path
                with h5py.File(dataset_to_augment_path, 'r') as original:
                    with h5py.File(self.dataset_file_path, 'w') as target:
                        # Copy the dataset to the saving path.
                        # Also makes sure that:
                        # - The dataset is not exceeding the maximum size.
                        # - The dataset is split into train/valid/test datasets.
                        # If not, it will resample the dataset accordingly.
                        (self.current_train_nb_streamlines,
                         self.current_valid_nb_streamlines,
                         self.current_test_nb_streamlines) = \
                            _copy_and_update_dataset(original, target,
                                                     self.rng, self.maxes)
            else:
                # Let's just use the original dataset file.
                self.dataset_file_path = dataset_to_augment_path

                # It's not used for now, it will cause issues since we're not
                # initially resampling the dataset to respect the maximum sizes.
                raise NotImplementedError("Augmenting in place is not implemented yet.")

            self.file_is_created = True
        else:
            LOGGER.info("Creating a new dataset.")
            self.dataset_file_path = os.path.join(saving_path, dataset_name)
            self.current_train_nb_streamlines = 0
            self.current_valid_nb_streamlines = 0
            self.current_test_nb_streamlines = 0
            self.file_is_created = False

    def add_tractograms_to_dataset(self, filtered_tractograms: list[tuple[StatefulTractogram,
                                                                          StatefulTractogram]]):
        """ Gathers all the filtered tractograms and creates or appends a dataset for the
        reward model training. Outputs into a hdf5 file."""

        if len(filtered_tractograms) == 0:
            LOGGER.warning(
                "Called add_tractograms_to_dataset with an empty list of tractograms.")
            return 0
        elif np.all([len(sft_valid) == 0 and len(sft_invalid) == 0 for sft_valid, sft_invalid in filtered_tractograms]):
            LOGGER.warning(
                "Called add_tractograms_to_dataset with only empty tractograms.")
            return 0

        # For each sft, get the indices of streamlines for training or for testing.
        # We do that before adding anything to the dataset, because we want to know
        # the total number of streamlines to add for each set (train/test) to resize
        # the dataset accordingly.
        train_indices = []          # [(pos_indices, neg_indices), ...]
        valid_indices = []          # [(pos_indices, neg_indices), ...]
        test_indices = []           # [(pos_indices, neg_indices), ...]

        # Mainly for logging purposes below.
        train_total_nb_pos = 0
        train_total_nb_neg = 0
        valid_total_nb_pos = 0
        valid_total_nb_neg = 0
        test_total_nb_pos = 0
        test_total_nb_neg = 0

        train_nb_streamlines = 0    # train_nb_pos + train_nb_neg
        valid_nb_streamlines = 0    # valid_nb_pos + valid_nb_neg
        test_nb_streamlines = 0     # test_nb_pos + test_nb_neg

        # TODO: Split and resample the indices based on the scores of the
        #       streamlines, and not just the number of streamlines to get
        #       a balanced dataset.
        for sft_valid, sft_invalid in filtered_tractograms:
            nb_pos = len(sft_valid)
            nb_neg = len(sft_invalid)
            
            # Adjust the nb_pos/nb_neg to the maximum dataset size
            # but keep the ratio between pos/neg.
            nb_tot = nb_pos + nb_neg
            if nb_tot > self.maxes.max_dataset_size:
                excess = nb_tot - self.maxes.max_dataset_size
                nb_pos = round(nb_pos - (excess * nb_pos / nb_tot))
                nb_neg = round(nb_neg - (excess * nb_neg / nb_tot))

            # Positive
            nb_pos_train = round(nb_pos * self.maxes.train_ratio)
            nb_pos_valid = round(nb_pos * self.maxes.valid_ratio)
            nb_pos_test = nb_pos - nb_pos_train - nb_pos_valid

            pos_indices = self.rng.choice(len(sft_valid.streamlines), nb_pos, replace=False)
            pos_train_indices = pos_indices[:nb_pos_train]
            pos_valid_indices = pos_indices[nb_pos_train:nb_pos_train + nb_pos_valid]
            pos_test_indices = pos_indices[nb_pos_train + nb_pos_valid:]

            # Negative
            nb_neg_train = round(nb_neg * self.maxes.train_ratio)
            nb_neg_valid = round(nb_neg * self.maxes.valid_ratio)
            nb_neg_test = nb_neg - nb_neg_train - nb_neg_valid

            neg_indices = self.rng.choice(len(sft_invalid.streamlines), nb_neg, replace=False)
            neg_train_indices = neg_indices[:nb_neg_train]
            neg_valid_indices = neg_indices[nb_neg_train:nb_neg_train + nb_neg_valid]
            neg_test_indices = neg_indices[nb_neg_train + nb_neg_valid:]

            # Add to the list of indices
            train_indices.append((pos_train_indices, neg_train_indices))
            valid_indices.append((pos_valid_indices, neg_valid_indices))
            test_indices.append((pos_test_indices, neg_test_indices))

            # Mainly for stat purposes
            train_total_nb_pos += nb_pos_train
            train_total_nb_neg += nb_neg_train
            valid_total_nb_pos += nb_pos_valid
            valid_total_nb_neg += nb_neg_valid
            test_total_nb_pos += nb_pos_test
            test_total_nb_neg += nb_neg_test

            train_nb_streamlines += nb_pos_train + nb_neg_train
            valid_nb_streamlines += nb_pos_valid + nb_neg_valid
            test_nb_streamlines += nb_pos_test + nb_neg_test

        # Logging the stats of the streamlines to add
        LOGGER.info(
            f"Adding {train_nb_streamlines} ({train_total_nb_pos} val | {train_total_nb_neg} inv) training streamlines to the dataset.")
        LOGGER.info(
            f"Adding {valid_nb_streamlines} ({valid_total_nb_pos} val | {valid_total_nb_neg} inv) validation streamlines to the dataset.")
        LOGGER.info(
            f"Adding {test_nb_streamlines} ({test_total_nb_pos} val | {test_total_nb_neg} inv) testing streamlines to the dataset.")

        write_mode = 'w' if not self.file_is_created else 'a'
        with h5py.File(self.dataset_file_path, write_mode) as f:

            # Create the hdf5 file structure if not already done
            if not self.file_is_created:
                f.attrs['version'] = 1
                f.attrs['nb_points'] = self.nb_points
                direction_dimension = \
                    filtered_tractograms[0][0].streamlines[0].shape[-1] \
                    if len(filtered_tractograms[0][0].streamlines) > 0 \
                    else filtered_tractograms[0][1].streamlines[0].shape[-1]

                # Create the train/test groups
                train_group = f.create_group('train')
                valid_group = f.create_group('valid')
                test_group = f.create_group('test')

                # Create the TRAIN dataset (train/data & train/scores)                
                train_data, train_scores = \
                    _create_datasets_hdf5(train_group, train_nb_streamlines,
                                          self.nb_points,
                                          direction_dimension)
                
                # Create the VALIDATION dataset (validation/data & validation/scores)
                valid_data, valid_scores = \
                    _create_datasets_hdf5(valid_group, valid_nb_streamlines,
                                          self.nb_points,
                                          direction_dimension)

                # Create the TEST dataset (test/data & test/scores)
                test_data, test_scores = \
                    _create_datasets_hdf5(test_group, test_nb_streamlines,
                                          self.nb_points,
                                          direction_dimension)

                self.file_is_created = True
                do_resize = False

            # The dataset file is already created. Make sure it's
            # consistent with the current dataset.
            else:
                assert f.attrs['nb_points'] == self.nb_points, \
                    "The number of points in the dataset is different from the one in the manager."
                train_group = f['train']
                valid_group = f['valid']
                test_group = f['test']
                f.attrs['version'] += 1
                # We are appending new data, we need to resize the dataset.
                do_resize = True

            # Indices where to add the streamlines in the file (contiguous at the end of array).
            # TODO: This is where some tweaking needs to be done to make sure that
            #       we don't exceed the maximum dataset size, we can't always just
            #       add the streamlines at the end of the dataset if we want to
            #       respect a maximum size.
            new_train_nb_streamlines = self.current_train_nb_streamlines + train_nb_streamlines
            new_valid_nb_streamlines = self.current_valid_nb_streamlines + valid_nb_streamlines
            new_test_nb_streamlines = self.current_test_nb_streamlines + test_nb_streamlines

            (file_train_indices, file_valid_indices, file_test_indices,
             nb_new_train_indices, nb_new_valid_indices, nb_new_test_indices) = \
                self._get_file_indices(new_train_nb_streamlines,
                                       new_valid_nb_streamlines,
                                       new_test_nb_streamlines)
            
            # Resize the dataset to append the new streamlines.
            if do_resize:
                self._resize_datasets(
                    train_group, new_train_nb_streamlines,
                    valid_group, new_valid_nb_streamlines,
                    test_group, new_test_nb_streamlines,
                )

            # Actually add the streamlines to the dataset using the precalculated
            # indices.
            for i, (sft_train_indices, sft_valid_indices, sft_test_indices) in enumerate(tqdm(zip(train_indices, valid_indices, test_indices), desc="Adding tractograms to dataset", total=len(filtered_tractograms), leave=False)):
                # Unpack and setup
                pos_train_indices, neg_train_indices = sft_train_indices
                pos_valid_indices, neg_valid_indices = sft_valid_indices
                pos_test_indices, neg_test_indices = sft_test_indices

                valid_sft, invalid_sft = filtered_tractograms[i]
                valid_sft.to_vox()
                valid_sft.to_corner()
                invalid_sft.to_vox()
                invalid_sft.to_corner()

                # Add the training positive streamlines
                file_idx = file_train_indices[:len(pos_train_indices)]
                self._add_streamlines_to_hdf5(train_group,
                                              valid_sft[pos_train_indices],
                                              self.nb_points,
                                              file_idx,
                                              sub_pbar_desc="add train/pos streamlines",
                                              batch_size=self.add_batch_size)
                file_train_indices = file_train_indices[len(
                    pos_train_indices):]

                # Add the training negative streamlines
                file_idx = file_train_indices[:len(neg_train_indices)]
                self._add_streamlines_to_hdf5(train_group,
                                              invalid_sft[neg_train_indices],
                                              self.nb_points,
                                              file_idx,
                                              sub_pbar_desc="add train/neg streamlines",
                                              batch_size=self.add_batch_size)
                file_train_indices = file_train_indices[len(
                    neg_train_indices):]

                # Add the validation positive streamlines
                file_idx = file_valid_indices[:len(pos_valid_indices)]
                self._add_streamlines_to_hdf5(valid_group,
                                              valid_sft[pos_valid_indices],
                                              self.nb_points,
                                              file_idx,
                                              sub_pbar_desc="add valid/pos streamlines",
                                              batch_size=self.add_batch_size)
                file_valid_indices = file_valid_indices[len(
                    pos_valid_indices):]
                
                # Add the validation negative streamlines
                file_idx = file_valid_indices[:len(neg_valid_indices)]
                self._add_streamlines_to_hdf5(valid_group,
                                              invalid_sft[neg_valid_indices],
                                              self.nb_points,
                                              file_idx,
                                              sub_pbar_desc="add valid/neg streamlines",
                                              batch_size=self.add_batch_size)
                file_valid_indices = file_valid_indices[len(
                    neg_valid_indices):]

                # Add the testing positive streamlines
                file_idx = file_test_indices[:len(pos_test_indices)]
                self._add_streamlines_to_hdf5(test_group,
                                              valid_sft[pos_test_indices],
                                              self.nb_points,
                                              file_idx,
                                              sub_pbar_desc="add test/pos streamlines",
                                              batch_size=self.add_batch_size)
                file_test_indices = file_test_indices[len(pos_test_indices):]

                # Add the testing negative streamlines
                file_idx = file_test_indices[:len(neg_test_indices)]
                self._add_streamlines_to_hdf5(test_group,
                                              invalid_sft[neg_test_indices],
                                              self.nb_points,
                                              file_idx,
                                              sub_pbar_desc="add test/neg streamlines",
                                              batch_size=self.add_batch_size)
                file_test_indices = file_test_indices[len(neg_test_indices):]

            assert len(
                file_train_indices) == 0, "Not all training streamlines were added."
            assert len(
                file_valid_indices) == 0, "Not all validation streamlines were added."
            assert len(
                file_test_indices) == 0, "Not all testing streamlines were added."

            self.current_train_nb_streamlines += nb_new_train_indices
            self.current_valid_nb_streamlines += nb_new_valid_indices
            self.current_test_nb_streamlines += nb_new_test_indices

            assert self.current_train_nb_streamlines <= self.maxes.train, \
                "The training dataset is exceeding the maximum size."
            assert self.current_valid_nb_streamlines <= self.maxes.valid, \
                "The validation dataset is exceeding the maximum size."
            assert self.current_test_nb_streamlines <= self.maxes.test, \
                "The testing dataset is exceeding the maximum size."

            return train_nb_streamlines + valid_nb_streamlines + test_nb_streamlines

    def _get_file_indices(self, new_train_nb_streamlines,
                          new_valid_nb_streamlines, new_test_nb_streamlines):
        """
        This methods assigns indices where the streamlines should be added
        in the hdf5 file.

        If we don't have a maximum dataset size or we haven't reached the max
        size of the dataset yet, we will just produce contiguous indices at
        the end of the dataset.

        If we reach the maximum dataset size, we will choose indices at random
        to overwrite in the previous version of the dataset.
        For example with a max dataset of size 500:
            - We want to add 50 streamlines to the dataset with currently 496
              streamlines. We will add the first 4 streamlines at the end of
              the dataset (indices 496 to 500) and the next 46 streamlines
              will overwrite randomly 46 streamlines in the dataset.
            - We want to add 50 streamlines to the dataset with currently 500
              streamlines. We will overwrite randomly 50 streamlines in the
              dataset.
        """
        
        # We can just add the streamlines at the end of the dataset
        file_train_indices = np.arange(
            self.current_train_nb_streamlines, new_train_nb_streamlines)
        file_valid_indices = np.arange(
            self.current_valid_nb_streamlines, new_valid_nb_streamlines)
        file_test_indices = np.arange(
            self.current_test_nb_streamlines, new_test_nb_streamlines)

        nb_new_train_indices = len(file_train_indices)
        nb_new_valid_indices = len(file_valid_indices)
        nb_new_test_indices = len(file_test_indices)

        # In the following conditions, some indices are going out of bounds
        # with respect to the maximum dataset size. We need to overwrite those
        # with random indices taken from the current dataset to overwrite some
        # older streamlines.
        #
        # We compute max_pos which is the position in the list of indices
        # where we need to overwrite the streamlines. We overwrite the
        # indices from max_pos to the end of the list by randomly choosing
        # indices from the current dataset.
        # For example, if the max dataset size is 100, we have 95
        # streamlines and we want to add 7 more, we will have:
        #   file_indices = np.arange(95, 102)
        #   >>> [95, 96, 97, 98, 99, 100, 101]
        #   max_pos = 100 - 95
        #   >>> 5
        # Which corresponds to the indice of the indice 100.
        # In the case the dataset is already full, we will have:
        #   file_indices = np.arange(100, 107)
        #   >>> [100, 101, 102, 103, 104, 105, 106]
        #   max_pos = 100 - 100
        #   >>> 0
        # Which means we overwrite all indices in the file_indices list.

        if new_train_nb_streamlines > self.maxes.train:
            nb_new_train_indices = \
                self.maxes.train - self.current_train_nb_streamlines
            nb_to_overwrite = len(file_train_indices) - nb_new_train_indices
            file_train_indices[nb_new_train_indices:] = self.rng.choice(
                self.current_train_nb_streamlines,
                nb_to_overwrite, replace=False)

        if new_valid_nb_streamlines > self.maxes.valid:
            nb_new_valid_indices = \
                self.maxes.valid - self.current_valid_nb_streamlines
            nb_to_overwrite = len(file_valid_indices) - nb_new_valid_indices
            file_valid_indices[nb_new_valid_indices:] = self.rng.choice(
                self.current_valid_nb_streamlines,
                nb_to_overwrite, replace=False)
            
        if new_test_nb_streamlines > self.maxes.test:
            nb_new_test_indices = \
                self.maxes.test - self.current_test_nb_streamlines
            nb_to_overwrite = len(file_test_indices) - nb_new_test_indices
            file_test_indices[nb_new_test_indices:] = self.rng.choice(
                self.current_test_nb_streamlines,
                nb_to_overwrite, replace=False)

        self.rng.shuffle(file_train_indices)
        self.rng.shuffle(file_valid_indices)
        self.rng.shuffle(file_test_indices)

        return (file_train_indices, file_valid_indices, file_test_indices,
                nb_new_train_indices, nb_new_valid_indices,
                nb_new_test_indices)

    def _resize_datasets(self,
                         train_group, new_train_nb_streamlines,
                         valid_group, new_valid_nb_streamlines,
                         test_group, new_test_nb_streamlines):

        if self.current_train_nb_streamlines < self.maxes.train:
            LOGGER.debug("Resizing the training dataset.")
            train_group[DATA].resize(
                min(new_train_nb_streamlines, self.maxes.train), axis=0)
            train_group[SCORES].resize(
                min(new_train_nb_streamlines, self.maxes.train), axis=0)
        
        if self.current_valid_nb_streamlines < self.maxes.valid:
            LOGGER.debug("Resizing the validation dataset.")
            valid_group[DATA].resize(
                min(new_valid_nb_streamlines, self.maxes.valid), axis=0)
            valid_group[SCORES].resize(
                min(new_valid_nb_streamlines, self.maxes.valid), axis=0)

        if self.current_test_nb_streamlines < self.maxes.test:
            LOGGER.debug("Resizing the testing dataset.")
            test_group[DATA].resize(
                min(new_test_nb_streamlines, self.maxes.test), axis=0)
            test_group[SCORES].resize(
                min(new_test_nb_streamlines, self.maxes.test), axis=0)
                         

    def _add_streamlines_to_hdf5(self, f, sft, nb_points, idx, sub_pbar_desc="", batch_size=1000):
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
        scores = np.asarray(
            sft.data_per_streamline['score'], dtype=np.uint8).squeeze(-1)
        # Resample the streamlines
        streamlines = set_number_of_points(sft.streamlines, nb_points)
        idx = np.sort(idx)

        data_group = f[DATA]
        scores_group = f[SCORES]

        num_batches = (len(idx) // batch_size) + (len(idx) % batch_size != 0)

        for batch_start in tqdm(range(0, len(idx), batch_size), desc=sub_pbar_desc, total=num_batches, leave=False):
            batch_end = min(batch_start + batch_size, len(idx))
            batch_idx = idx[batch_start:batch_end]
            batch_streamlines = np.asarray(
                streamlines[batch_start:batch_end], dtype=np.float32)
            batch_scores = scores[batch_start:batch_end]

            data_group[batch_idx] = batch_streamlines
            scores_group[batch_idx] = batch_scores

    def _load_and_verify_streamline_dataset(self, dataset_to_augment_path: str):
        """ Verify the dataset in the hdf5 file."""
        def get_group_size(group):
            if DATA not in group:
                raise ValueError(
                    f"The dataset ({group}) does not contain the '{DATA}' group.")

            if SCORES not in group:
                raise ValueError(
                    f"The dataset ({group}) does not contain the '{SCORES}' group.")

            return group[SCORES].shape[0]

        with h5py.File(dataset_to_augment_path, 'r') as dataset:

            has_train = 'train' in dataset
            has_valid = 'valid' in dataset
            has_test = 'test' in dataset

            if not has_train and not has_test:
                raise ValueError(
                    "The dataset does not contain the 'train' or 'test' groups.")

            train_size = get_group_size(dataset['train']) if has_train else 0
            valid_size = get_group_size(dataset['valid']) if has_valid else 0
            test_size = get_group_size(dataset['test']) if has_test else 0

            return train_size, valid_size, test_size

    def fetch_dataset_stats(self):
        """ Get the current dataset statistics."""

        with h5py.File(self.dataset_file_path, 'r') as f:
            train_group = f['train']
            valid_group = f['valid']
            test_group = f['test']

            train_nb_pos = np.sum(train_group[SCORES])
            train_nb_neg = train_group[SCORES].shape[0] - train_nb_pos
            valid_nb_pos = np.sum(valid_group[SCORES])
            valid_nb_neg = valid_group[SCORES].shape[0] - valid_nb_pos
            test_nb_pos = np.sum(test_group[SCORES])
            test_nb_neg = test_group[SCORES].shape[0] - test_nb_pos


        assert train_nb_pos + train_nb_neg == \
            self.current_train_nb_streamlines, \
            "The number of positive and negative streamlines in the training" \
            "set does not match the total number of streamlines."
        
        assert valid_nb_pos + valid_nb_neg == \
            self.current_valid_nb_streamlines, \
            "The number of positive and negative streamlines in the validation" \
            "set does not match the total number of streamlines."
        
        assert test_nb_pos + test_nb_neg == \
            self.current_test_nb_streamlines, \
            "The number of positive and negative streamlines in the testing" \
            "set does not match the total number of streamlines."

        total_size = self.current_train_nb_streamlines + \
            self.current_valid_nb_streamlines + \
            self.current_test_nb_streamlines
        
        real_train_ratio = self.current_train_nb_streamlines / total_size
        real_valid_ratio = self.current_valid_nb_streamlines / total_size
        real_test_ratio = self.current_test_nb_streamlines / total_size

        stats = {
            'train': {
                'size': self.current_train_nb_streamlines,
                'nb_pos': train_nb_pos,
                'nb_neg': train_nb_neg,
                'ratio_pos': train_nb_pos / self.current_train_nb_streamlines,
                'ratio_neg': train_nb_neg / self.current_train_nb_streamlines
            },
            'valid': {
                'size': self.current_valid_nb_streamlines,
                'nb_pos': valid_nb_pos,
                'nb_neg': valid_nb_neg,
                'ratio_pos': valid_nb_pos / self.current_valid_nb_streamlines,
                'ratio_neg': valid_nb_neg / self.current_valid_nb_streamlines
            },
            'test': {
                'size': self.current_test_nb_streamlines,
                'nb_pos': test_nb_pos,
                'nb_neg': test_nb_neg,
                'ratio_pos': test_nb_pos / self.current_test_nb_streamlines,
                'ratio_neg': test_nb_neg / self.current_test_nb_streamlines
            },
            'train_ratio': real_train_ratio,
            'test_ratio': real_test_ratio,
            'total_size': total_size
        }

        return stats

# def _copy_to_hdf5_target(original, target, indices=None, enable_pbar=True):
#     """
#     Copy data indexed by indices from the original dataset to the target
#     dataset sequentially. We need to copy data with batches to track progress
#     and avoid freezes.
#     """
#     from tractoracle_irt.utils.utils import SimpleTimer
#     if indices is None:
#         indices = np.arange(original[DATA].shape[0])
#
#     batch_size = 1000
#     num_indices = len(indices)
#     num_batches = (num_indices // batch_size) + \
#         (num_indices % batch_size != 0)
    
#     mean_time_read = 0
#     mean_time_write = 0
#     nb_loops = 0

#     original_data = original[DATA]
#     original_scores = original[SCORES]

#     for batch_start in tqdm(range(0, num_indices, batch_size),
#                             desc="Copying data", total=num_batches,
#                             leave=False, disable=not enable_pbar):
#         batch_end = min(batch_start + batch_size, num_indices)
#         batch_indices = indices[batch_start:batch_end]
        
#         with SimpleTimer() as t_read:
#             # print("reading batch indices: ", batch_indices[:10])
#             data = original_data[batch_indices]
#             scores = original_scores[batch_indices]

#         with SimpleTimer() as t_write:
#             target[DATA][batch_start:batch_end] = data
#             target[SCORES][batch_start:batch_end] = scores

#         mean_time_read += t_read.interval
#         mean_time_write += t_write.interval
#         nb_loops += 1

#         print(f"Mean time read: {mean_time_read / nb_loops}s | "
#               f"Mean time write: {mean_time_write / nb_loops}s")
