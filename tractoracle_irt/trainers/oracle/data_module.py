import numpy as np
import torch
from torch.utils.data import (
    BatchSampler, DataLoader, SequentialSampler, Subset, Sampler)
from tractoracle_irt.trainers.oracle.StreamlineBatchDataset import StreamlineBatchDataset
from tractoracle_irt.utils.torch_utils import get_device_str


class StreamlineDataModule(object):
    """ Data module for the streamline dataset. This module is used to
    load the data and create the dataloaders for the training, validation
    and test sets.

    A custom sampler is used to shuffle the data in the training set
    while keeping the batches consistent.
    """

    def __init__(
        self,
        dataset_file: str,
        batch_size: int = 1024,
        num_workers: int = 20,
        nb_points: int = 128,
    ):
        """ Initialize the data module with the paths to the training,
        validation and test files. The batch size and number of workers
        for the dataloaders can also be set.

        Parameters:
        -----------
        dataset_file: str
            Path to the hdf5 file containing the dataset.
        batch_size: int, optional
            Size of the batches to use for the dataloaders
        num_workers: int, optional
            Number of workers to use for the dataloaders
        """

        super().__init__()
        self.dataset_file = dataset_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nb_points = nb_points

        self.data_loader_kwargs = {
            'num_workers': self.num_workers,
            'prefetch_factor': 8 if self.num_workers > 0 else None,
            'persistent_workers': False,
            'pin_memory': get_device_str() == 'cuda',
        }

        # Select a random distribution of indices for the training and validation sets.
        num_streamlines = len(StreamlineBatchDataset(
            self.dataset_file, stage="train", nb_points=nb_points))
        self.indices = np.arange(num_streamlines)
        np.random.shuffle(self.indices)

        # 80% of the training data is used for training
        self.train_indices = self.indices[:int(0.8 * num_streamlines)]
        # 20% of the training data is used for validation
        self.valid_indices = self.indices[int(0.8 * num_streamlines):]

        # Accessing elements in an HDF5 file requires indices
        # to be accessed in increasing order.
        self.train_indices = np.sort(self.train_indices)
        self.valid_indices = np.sort(self.valid_indices)

        assert len(self.train_indices) > 0 and \
            len(self.valid_indices) > 0, \
            "The dataset is too small to be split into train, validation and test sets." \
            f"Train: {len(self.train_indices)} Val: {len(self.valid_indices)} Test: {len(self.test_indices)}"

    def setup(self, stage: str, dense: bool = False, partial: bool = False):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            print("Setting up training and validation datasets with dense={} and partial={}".format(
                dense, partial))

            self.streamline_train = Subset(StreamlineBatchDataset(
                self.dataset_file, stage="train",
                dense=dense, partial=partial, nb_points=self.nb_points),
                self.train_indices)

            self.streamline_val = Subset(StreamlineBatchDataset(
                self.dataset_file, stage="train",
                dense=dense, partial=partial, nb_points=self.nb_points),
                self.valid_indices)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.streamline_test = StreamlineBatchDataset(
                self.dataset_file, noise=0.0, flip_p=0.0, stage="test",
                dense=dense, partial=partial, nb_points=self.nb_points)

    def train_dataloader(self):
        """ Create the dataloader for the training set
        """
        sampler = BatchSampler(WeakShuffleSampler(
            self.streamline_train, self.batch_size), self.batch_size,
            drop_last=False)

        return DataLoader(
            self.streamline_train,
            sampler=sampler,
            **self.data_loader_kwargs)

    def val_dataloader(self):
        """ Create the dataloader for the validation set
        """
        sampler = BatchSampler(SequentialSampler(
            self.streamline_val), self.batch_size,
            drop_last=False)
        return DataLoader(
            self.streamline_val,
            sampler=sampler,
            **self.data_loader_kwargs)

    def test_dataloader(self):
        """ Create the dataloader for the test set
        """
        sampler = BatchSampler(SequentialSampler(
            self.streamline_test), self.batch_size,
            drop_last=False)
        return DataLoader(
            self.streamline_test,
            batch_size=None,
            sampler=sampler,
            **self.data_loader_kwargs)

    def predict_dataloader(self):
        pass


class WeakShuffleSampler(Sampler):
    """ Weak shuffling inspired by https://towardsdatascience.com/reading-h5-files-faster-with-pytorch-datasets-3ff86938cc  # noqa E501

    Weakly shuffles by return batched ids in a random way, so that
    batches are not encountered in the same order every epoch. Adds
    randomness by adding a "starting index" which shifts the indices,
    so that every batch gets different data each epoch. "Neighboring"
    data may be put in the same batch still.

    Presumes that the dataset is already shuffled on disk.
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_length = len(dataset)

        self.n_batches = self.dataset_length // self.batch_size

    def __len__(self):
        return self.dataset_length

    def __iter__(self):

        # Get the shifting index
        starting_idx = np.random.randint(len(self))

        # Shuffle the batches
        batch_ids = np.random.permutation(int(self.n_batches))
        for id in batch_ids:
            # Batch slice beginning
            beg = (starting_idx + (id * self.batch_size)) % len(self)
            # Batch slice end
            end = (starting_idx + ((id + 1) * self.batch_size)) % len(self)
            # Indices are rolling over
            if beg > end:
                # Concatenate indices at the end of the dataset and the
                # beginning in the same batch
                idx = np.concatenate(
                    (np.arange(beg, len(self)), np.arange(end)))
            else:
                idx = range(beg, end)

            # Weirdly enough, precomputing the indices seems slower.
            for index in idx:
                yield int(index)
