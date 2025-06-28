import pytest

import h5py
import numpy as np
import nibabel as nib
from tempfile import TemporaryDirectory

from dipy.io.stateful_tractogram import StatefulTractogram, Space
from tractoracle_irt.trainers.oracle.streamline_dataset_manager \
    import StreamlineDatasetManager

NB_POINTS = 128
DIRECTION_DIM = 3
RNG_SEED = 42
fake_reference = nib.nifti1.Nifti1Image(np.zeros(1), affine=None)

def _create_dataset(path,
                    train_data, train_scores,
                    valid_data=None, valid_scores=None,
                    test_data=None, test_scores=None):

    with h5py.File(path, "w") as f:
        f.attrs['version'] = 1
        f.attrs['nb_points'] = NB_POINTS

        train_group = f.create_group("train")
        train_group.create_dataset("data", data=train_data,
                                   maxshape=(None, NB_POINTS, DIRECTION_DIM))
        train_group.create_dataset("scores", data=train_scores,
                                   maxshape=(None,))

        if valid_data is not None:
            valid_group = f.create_group("valid")
            valid_group.create_dataset("data", data=valid_data,
                                       maxshape=(None, NB_POINTS, DIRECTION_DIM))
            valid_group.create_dataset("scores", data=valid_scores,
                                       maxshape=(None,))
        if test_data is not None:
            test_group = f.create_group("test")
            test_group.create_dataset("data", data=test_data,
                                      maxshape=(None, NB_POINTS, DIRECTION_DIM))
            test_group.create_dataset("scores", data=test_scores,
                                      maxshape=(None,))

    return path

def _pack_into_sft(data, score):
    sft = StatefulTractogram(
        data,
        fake_reference,
        Space.VOX,
        data_per_streamline={"score": np.array([score] * len(data))},
    )
    return sft

def _has_expected_structure(hdf5_file, nb_train, nb_valid, nb_test):
    GROUP_KEYS = ["train", "valid", "test"]
    DATASET_KEYS = ["data", "scores"]

    for group_key in GROUP_KEYS:
        if group_key not in hdf5_file:
            print(f"{group_key} not found.")
            return False
        for dataset_key in DATASET_KEYS:
            if dataset_key not in hdf5_file[group_key]:
                print(f"{group_key}/{dataset_key} not found.")
                return False
            
            # Make sure the number of streamlines is as expected in each
            # dataset.
            if group_key == 'train' \
                and hdf5_file[group_key][dataset_key].shape[0] != nb_train:
                print(f"{group_key}/{dataset_key} has {hdf5_file[group_key][dataset_key].shape[0]} streamlines instead of {nb_train}.")
                return False
            elif group_key == 'valid' \
                and hdf5_file[group_key][dataset_key].shape[0] != nb_valid:
                print(f"{group_key}/{dataset_key} has {hdf5_file[group_key][dataset_key].shape[0]} streamlines instead of {nb_valid}.")
                return False
            elif group_key == 'test' \
                and hdf5_file[group_key][dataset_key].shape[0] != nb_test:
                print(f"{group_key}/{dataset_key} has {hdf5_file[group_key][dataset_key].shape[0]} streamlines instead of {nb_test}.")
                return False
            
    return True

def _data_has_zeros(hdf5_file):
    GROUP_KEYS = ["train", "valid", "test"]

    for group_key in GROUP_KEYS:
        if np.any(hdf5_file[group_key]["data"] == 0):
            return True
    return False

def _has_valid_scores(hdf5_file):
    GROUP_KEYS = ["train", "valid", "test"]

    for group_key in GROUP_KEYS:
        scores = hdf5_file[group_key]["scores"][...]
        if np.any(scores < 0):
            return False
        elif np.any(scores > 1):
            return False
    return True

@pytest.fixture
def setup_dataset_manager_empty_ds():
    with TemporaryDirectory() as temp_dir:
        dataset_manager = StreamlineDatasetManager(
            saving_path=temp_dir,
            dataset_to_augment_path=None,
            augment_in_place=False,
            dataset_name="test_dataset.hdf5",
            valid_ratio=0.1,
            test_ratio=0.1,
            max_dataset_size=100,
            rng_seed=RNG_SEED
        )
        yield dataset_manager

@pytest.fixture
def setup_dataset_manager_with_initial_ds():
    # Create a dataset with 100 streamlines.
    # Create a dataset manager with the dataset.
    with TemporaryDirectory() as temp_dir:
        rng = np.random.RandomState(42)
        train_data = rng.rand(80, NB_POINTS, DIRECTION_DIM)
        train_scores = rng.choice([0, 1], 80)
        valid_data = rng.rand(10, NB_POINTS, DIRECTION_DIM)
        valid_scores = rng.choice([0, 1], 10)
        test_data = rng.rand(10, NB_POINTS, DIRECTION_DIM)
        test_scores = rng.choice([0, 1], 10)

        path = f"{temp_dir}/test_dataset.hdf5"
        f = _create_dataset(path,
                            train_data, train_scores,
                            valid_data, valid_scores,
                            test_data, test_scores)

        dataset_manager = StreamlineDatasetManager(
            saving_path=temp_dir,
            dataset_to_augment_path=f,
            valid_ratio=0.1,
            test_ratio=0.1,
            max_dataset_size=120,
            rng_seed=RNG_SEED
        )

        yield dataset_manager

@pytest.fixture
def setup_manager_with_train_only_ds():
    # Create a dataset with 100 streamlines.
    # Create a dataset manager with the dataset.
    with TemporaryDirectory() as temp_dir:
        rng = np.random.RandomState(42)
        train_data = rng.rand(100, NB_POINTS, DIRECTION_DIM)
        train_scores = rng.choice([0, 1], 100)

        path = f"{temp_dir}/test_dataset.hdf5"
        f = _create_dataset(path,
                            train_data, train_scores)

        dataset_manager = StreamlineDatasetManager(
            saving_path=temp_dir,
            dataset_to_augment_path=f,
            valid_ratio=0.1,
            test_ratio=0.1,
            max_dataset_size=120,
            rng_seed=RNG_SEED
        )

        yield dataset_manager

@pytest.fixture
def setup_manager_with_full_ds():
    # Create a dataset with 100 streamlines.
    # Create a dataset manager with the dataset.
    with TemporaryDirectory() as temp_dir:
        rng = np.random.RandomState(42)
        train_data = rng.rand(80, NB_POINTS, DIRECTION_DIM)
        train_scores = rng.choice([0, 1], 80)
        valid_data = rng.rand(10, NB_POINTS, DIRECTION_DIM)
        valid_scores = rng.choice([0, 1], 10)
        test_data = rng.rand(10, NB_POINTS, DIRECTION_DIM)
        test_scores = rng.choice([0, 1], 10)

        path = f"{temp_dir}/test_dataset.hdf5"
        f = _create_dataset(path,
                            train_data, train_scores,
                            valid_data, valid_scores,
                            test_data, test_scores)

        dataset_manager = StreamlineDatasetManager(
            saving_path=temp_dir,
            dataset_to_augment_path=f,
            valid_ratio=0.1,
            test_ratio=0.1,
            max_dataset_size=100,
            rng_seed=RNG_SEED
        )

        yield dataset_manager
    

def test_add_normal_streamlines_to_empty(setup_dataset_manager_empty_ds):
    # We just expect that the streamlines are correctly added to the dataset.
    rng = np.random.RandomState(42)
    sft_valid, sft_invalid = \
        _pack_into_sft(rng.rand(5, NB_POINTS, DIRECTION_DIM), score=1), \
        _pack_into_sft(rng.rand(5, NB_POINTS, DIRECTION_DIM), score=0)

    total_added = setup_dataset_manager_empty_ds.add_tractograms_to_dataset(
        [(sft_valid, sft_invalid)]
    )

    assert total_added == 10

    sft_valid, sft_invalid = \
        _pack_into_sft(rng.rand(10, NB_POINTS, DIRECTION_DIM), score=1), \
        _pack_into_sft(rng.rand(10, NB_POINTS, DIRECTION_DIM), score=0)  
    
    total_added = setup_dataset_manager_empty_ds.add_tractograms_to_dataset(
        [(sft_valid, sft_invalid)]
    )

    assert total_added == 20

def test_add_normal_streamlines_to_init(setup_dataset_manager_with_initial_ds):
    # We just expect that the streamlines are correctly added to the dataset.
    rng = np.random.RandomState(42)
    sft_valid, sft_invalid = \
        _pack_into_sft(rng.rand(5, NB_POINTS, DIRECTION_DIM), score=1), \
        _pack_into_sft(rng.rand(5, NB_POINTS, DIRECTION_DIM), score=0)
    
    total_added = setup_dataset_manager_with_initial_ds.add_tractograms_to_dataset(
        [(sft_valid, sft_invalid)]
    )

    assert total_added == 10
    with h5py.File(setup_dataset_manager_with_initial_ds.dataset_file_path, "r") as f:
        # Imbalance between valid/test due to rounding floats.
        assert _has_expected_structure(f, 88, 10, 12)
        assert not _data_has_zeros(f)
        assert _has_valid_scores(f)

def test_add_more_than_max_size(setup_dataset_manager_empty_ds):
    # We expect only a subset of the streamlines to be added.
    rng = np.random.RandomState(42)
    sft_valid, sft_invalid = \
        _pack_into_sft(rng.rand(100, NB_POINTS, DIRECTION_DIM), score=1), \
        _pack_into_sft(rng.rand(100, NB_POINTS, DIRECTION_DIM), score=0)
    
    total_added = setup_dataset_manager_empty_ds.add_tractograms_to_dataset(
        [(sft_valid, sft_invalid)]
    )

    assert total_added == 100
    with h5py.File(setup_dataset_manager_empty_ds.dataset_file_path, "r") as f:
        assert _has_expected_structure(f, 80, 10, 10)
        assert not _data_has_zeros(f)
        assert _has_valid_scores(f)
        # TODO: Make sure the data is the one expected.

def test_train_only_dataset_conversion(setup_manager_with_train_only_ds):
    # We expect that the dataset manager will convert the train-only dataset
    # into a train/valid/test dataset.
    with h5py.File(setup_manager_with_train_only_ds.dataset_file_path, "r") as f:
        assert _has_expected_structure(f, 80, 10, 10)
        assert not _data_has_zeros(f)
        assert _has_valid_scores(f)
        # TODO: Make sure the data is the one expected.

def test_add_when_reaching_max_size(setup_dataset_manager_with_initial_ds):
    # We expect streamlines to be overwritten
    rng = np.random.RandomState(42)
    sft_valid, sft_invalid = \
        _pack_into_sft(rng.rand(40, NB_POINTS, DIRECTION_DIM), score=1), \
        _pack_into_sft(rng.rand(40, NB_POINTS, DIRECTION_DIM), score=0)
    
    total_added = setup_dataset_manager_with_initial_ds.add_tractograms_to_dataset(
        [(sft_valid, sft_invalid)]
    )

    assert total_added == 80
    with h5py.File(setup_dataset_manager_with_initial_ds.dataset_file_path, "r") as f:
        assert _has_expected_structure(f, 96, 12, 12)
        assert not _data_has_zeros(f)
        assert _has_valid_scores(f)
    

    # Add a second time to make sure the dataset is not growing.
    total_added = setup_dataset_manager_with_initial_ds.add_tractograms_to_dataset(
        [(sft_valid, sft_invalid)]
    )

    assert total_added == 80
    with h5py.File(setup_dataset_manager_with_initial_ds.dataset_file_path, "r") as f:
        assert _has_expected_structure(f, 96, 12, 12)
        assert not _data_has_zeros(f)
        assert _has_valid_scores(f)

def test_add_when_full(setup_manager_with_full_ds):
    # We expect streamlines to be overwritten
    rng = np.random.RandomState(42)
    sft_valid, sft_invalid = \
        _pack_into_sft(rng.rand(50, NB_POINTS, DIRECTION_DIM), score=1), \
        _pack_into_sft(rng.rand(50, NB_POINTS, DIRECTION_DIM), score=0)
    
    total_added = setup_manager_with_full_ds.add_tractograms_to_dataset(
        [(sft_valid, sft_invalid)]
    )

    assert total_added == 100
    with h5py.File(setup_manager_with_full_ds.dataset_file_path, "r") as f:
        assert _has_expected_structure(f, 80, 10, 10)
        assert not _data_has_zeros(f)
        assert _has_valid_scores(f)

def test_empty_sfts(setup_dataset_manager_empty_ds):
    sft_1 = _pack_into_sft(np.zeros((0, NB_POINTS, DIRECTION_DIM)), score=1)
    sft_2 = _pack_into_sft(np.zeros((0, NB_POINTS, DIRECTION_DIM)), score=0)
    total_added = setup_dataset_manager_empty_ds.add_tractograms_to_dataset(
        [(sft_1, sft_2)]
    )
    assert total_added == 0


def test_empty_list(setup_dataset_manager_empty_ds):
    # We expect nothing to happen.
    total_added = setup_dataset_manager_empty_ds.add_tractograms_to_dataset([])
    assert total_added == 0

def test_initial_dataset_too_big(setup_dataset_manager_with_initial_ds):
    # Create a dataset with 100 streamlines.
    # Create a dataset manager with the dataset.
    with TemporaryDirectory() as temp_dir:
        rng = np.random.RandomState(42)
        train_data = rng.rand(120, NB_POINTS, DIRECTION_DIM)
        train_scores = rng.choice([0, 1], 120)
        valid_data = rng.rand(20, NB_POINTS, DIRECTION_DIM)
        valid_scores = rng.choice([0, 1], 20)
        test_data = rng.rand(20, NB_POINTS, DIRECTION_DIM)
        test_scores = rng.choice([0, 1], 20)

        path = f"{temp_dir}/test_dataset.hdf5"
        f = _create_dataset(path,
                            train_data, train_scores,
                            valid_data, valid_scores,
                            test_data, test_scores)

        dataset_manager = StreamlineDatasetManager(
            saving_path=temp_dir,
            dataset_to_augment_path=f,
            valid_ratio=0.1,
            test_ratio=0.1,
            max_dataset_size=100, # 80-10-10 max
            rng_seed=RNG_SEED
        )

        with h5py.File(dataset_manager.dataset_file_path, "r") as f:
            assert _has_expected_structure(f, 80, 10, 10)
            assert not _data_has_zeros(f)
            assert _has_valid_scores(f)
