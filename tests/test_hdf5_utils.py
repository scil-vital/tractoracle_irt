import pytest
import numpy as np
import h5py

from tractoracle_irt.utils.utils import SimpleTimer
from tractoracle_irt.utils.hdf5_utils import (
    copy_by_batch,
    copy_by_batch_multiproc,
    read_hdf5_data_multiproc,
    copy_by_batch_efficient)
from tempfile import TemporaryDirectory

NB_POINTS = 128
DIR_DIM = 3

@pytest.fixture
def create_source_target():
    # TODO: Fix this to return h5py.Dataset objects instead of numpy arrays
    source = np.random.rand(1000, 10)
    target = np.zeros((1000, 10))
    return source, target


@pytest.fixture
def create_dummy_hdf5():
    with TemporaryDirectory() as tmpdir:
        with h5py.File(tmpdir + "/dummy.hdf5", "w") as f:
            f.create_dataset("data", data=np.random.rand(1000, NB_POINTS, DIR_DIM))
        yield tmpdir + "/dummy.hdf5"

@pytest.fixture
def create_huge_dummy_hdf5():
    with TemporaryDirectory() as tmpdir:
        with h5py.File(tmpdir + "/dummy.hdf5", "w") as f:
            print("Creating huge dummy hdf5")
            f.create_dataset("data", data=np.random.rand(100000, NB_POINTS, DIR_DIM))
        yield tmpdir + "/dummy.hdf5"

@pytest.fixture
def create_source_target_hdf5():
    NB_FAKE_DATA = 1000
    NB_DATA_TARGET = 500

    with TemporaryDirectory() as tmpdir:
        with h5py.File(tmpdir + "/source.hdf5", "w") as f:
            train_group = f.create_group("train")
            train_group.create_dataset("data", data=np.random.rand(NB_FAKE_DATA, NB_POINTS, DIR_DIM))
            train_group.create_dataset("scores", data=np.random.rand(NB_FAKE_DATA, 1))
        with h5py.File(tmpdir + "/target.hdf5", "w") as f:
            train_group = f.create_group("train")
            train_group.create_dataset("data", data=np.zeros((NB_DATA_TARGET, NB_POINTS, DIR_DIM)))
            train_group.create_dataset("scores", data=np.zeros((NB_DATA_TARGET, 1)))
        yield tmpdir + "/source.hdf5", tmpdir + "/target.hdf5", NB_FAKE_DATA, NB_DATA_TARGET

@pytest.fixture
def create_source_target_hdf5_huge():
    NB_FAKE_DATA = 500000
    NB_DATA_TARGET = 75000

    with TemporaryDirectory() as tmpdir:
        with h5py.File(tmpdir + "/source.hdf5", "w") as f:
            train_group = f.create_group("train")
            train_group.create_dataset("data", data=np.random.rand(NB_FAKE_DATA, NB_POINTS, DIR_DIM))
            train_group.create_dataset("scores", data=np.random.rand(NB_FAKE_DATA, 1))
        with h5py.File(tmpdir + "/target.hdf5", "w") as f:
            train_group = f.create_group("train")
            train_group.create_dataset("data", data=np.zeros((NB_DATA_TARGET, NB_POINTS, DIR_DIM)))
            train_group.create_dataset("scores", data=np.zeros((NB_DATA_TARGET, 1)))
        yield tmpdir + "/source.hdf5", tmpdir + "/target.hdf5", NB_FAKE_DATA, NB_DATA_TARGET

@pytest.fixture
def create_source_target_hdf5_enormous():
    NB_FAKE_DATA = 1000000
    NB_DATA_TARGET = 5000

    with TemporaryDirectory() as tmpdir:
        with h5py.File(tmpdir + "/source.hdf5", "w") as f:
            train_group = f.create_group("train")
            train_group.create_dataset("data", data=np.random.rand(NB_FAKE_DATA, NB_POINTS, DIR_DIM))
            train_group.create_dataset("scores", data=np.random.rand(NB_FAKE_DATA, 1))
        with h5py.File(tmpdir + "/target.hdf5", "w") as f:
            train_group = f.create_group("train")
            train_group.create_dataset("data", data=np.zeros((NB_DATA_TARGET, NB_POINTS, DIR_DIM)))
            train_group.create_dataset("scores", data=np.zeros((NB_DATA_TARGET, 1)))
        yield tmpdir + "/source.hdf5", tmpdir + "/target.hdf5", NB_FAKE_DATA, NB_DATA_TARGET

##########################################
# hdf5_utils.copy_by_batch(...)
##########################################

def test_copy_by_batch(create_source_target):
    source, target = create_source_target
    indices = np.random.choice(1000, 1000, replace=False)

    copy_by_batch(source, target, indices, batch_size=100)
    
    assert np.allclose(source[indices], target[:len(indices)])

##########################################
# hdf5_utils.read_hdf5_data_multiproc(...)
##########################################

def test_multiproc_read_single_worker(create_dummy_hdf5):
    with h5py.File(create_dummy_hdf5, "r") as f:
        indices = np.random.choice(1000, 500, replace=False)
        indices.sort()
        data = f["data"][indices]
        read_data = read_hdf5_data_multiproc(
            create_dummy_hdf5,
            None,
            "data",
            indices,
            num_workers=1
        )

        assert np.allclose(data, read_data)

def test_multiproc_read_multiple_worker(create_dummy_hdf5):
    with h5py.File(create_dummy_hdf5, "r") as f:
        indices = np.random.choice(1000, 500, replace=False)
        indices.sort()
        data = f["data"][indices]
        read_data = read_hdf5_data_multiproc(
            create_dummy_hdf5,
            None,
            "data",
            indices,
            num_workers=4
        )

        assert np.allclose(data, read_data)

def test_multiproc_read_faster(create_huge_dummy_hdf5):
    with h5py.File(create_huge_dummy_hdf5, "r") as f:
        print("Generating random indices")
        indices = np.random.choice(100000, 50000, replace=False)
        indices.sort()

        print("Reading data: normal")
        with SimpleTimer() as normal_timer:
            data = f["data"][indices]
        print("Reading data: multiproc")
        with SimpleTimer() as multiproc_timer:
            read_data = read_hdf5_data_multiproc(
                create_huge_dummy_hdf5,
                None,
                "data",
                indices,
                num_workers=None
            )

        print("Normal time:", normal_timer.interval)
        print("Multiproc time:", multiproc_timer.interval)
        assert normal_timer.interval > multiproc_timer.interval

        

def test_multiproc_not_sorted(create_dummy_hdf5):
    with pytest.raises(ValueError) as e_info:
        read_hdf5_data_multiproc(
            create_dummy_hdf5,
            None,
            "data",
            [0, 2, 4, 6, 5, 8])
        

##########################################
# hdf5_utils.copy_by_batch_multiproc(...)
##########################################

def test_copy_by_batch_multiproc_normal(create_source_target_hdf5):
    (source, target, nb_fake_data, nb_data_target) = \
        create_source_target_hdf5
    
    indices = np.random.choice(nb_fake_data, nb_data_target, replace=False)
    indices.sort()

    copy_by_batch_multiproc(source,
                            target,
                            'train',
                            'data',
                            indices,
                            num_readers=4)
    
    with h5py.File(source, "r") as f_src, h5py.File(target, "r") as f_target:
        # If there's too much data, that check might be slow.
        assert np.allclose(f_src['train/data'][indices],
                           f_target['train/data'][:])
        
def test_copy_by_batch_multiproc_multiple_ds(create_source_target_hdf5):
    (source, target, nb_fake_data, nb_data_target) = \
        create_source_target_hdf5
    
    indices = np.random.choice(nb_fake_data, nb_data_target, replace=False)
    indices.sort()

    copy_by_batch_multiproc(source,
                            target,
                            'train',
                            ['data', 'scores'],
                            indices,
                            num_readers=4)
    
    with h5py.File(source, "r") as f_src, h5py.File(target, "r") as f_target:
        # If there's too much data, that check might be slow.
        assert np.allclose(f_src['train/data'][indices],
                           f_target['train/data'][:])

def test_copy_by_batch_multiproc_faster_than_normal(create_source_target_hdf5_huge):
    (source, target, nb_fake_data, nb_data_target) = \
        create_source_target_hdf5_huge
    
    indices = np.random.choice(nb_fake_data, nb_data_target, replace=False)
    indices.sort()

    with SimpleTimer() as multiproc_timer:
        copy_by_batch_multiproc(source, target, 'train', 'data', indices,
                                num_readers=None)
    with SimpleTimer() as normal_timer:
        #Copy normally from source to target.
        with h5py.File(source, "r") as f_src, h5py.File(target, "a") as f_target:
            f_target["train/data"][:len(indices)] = f_src["train/data"][indices]

    # Here, simply make sure that the multiproc version is faster.
    print("Normal time:", normal_timer.interval)
    print("Multiproc time:", multiproc_timer.interval)

    assert normal_timer.interval > multiproc_timer.interval

# NB: This test is commented out because it's not always faster to use multiprocessing
#     depending on the type of machine this test is run on.
#     It's also not a good idea to run this test on a CI/CD pipeline.
#     Letting it here for future tests in case more optimizations are necessary.

# def test_copy_by_batch_multiproc_faster_than_batch(create_source_target_hdf5_enormous):
#     (source, target, nb_fake_data, nb_data_target) = \
#         create_source_target_hdf5_enormous
    
#     indices = np.random.choice(nb_fake_data, nb_data_target, replace=False)
#     indices.sort()

#     with SimpleTimer() as batch_timer:
#         with h5py.File(source, "r") as f_src, h5py.File(target, "a") as f_target:
#             source_ds = f_src["train/data"]
#             target_ds = f_target["train/data"]
#             copy_by_batch(source_ds, target_ds, indices, batch_size=1000)
#     with SimpleTimer() as multiproc_timer:
#         copy_by_batch_multiproc(source, target, 'train', 'data', indices,
#                                 num_readers=None)

#     # Here, simply make sure that the multiproc version is faster.
#     print("Batch time:", batch_timer.interval)
#     print("Multiproc time:", multiproc_timer.interval)

#     assert batch_timer.interval > multiproc_timer.interval


def test_copy_by_batch_efficient(create_source_target_hdf5, create_source_target_hdf5_huge):
    ###############################################################
    # Test the version with less data first. (uses the batch version)
    (source, target, nb_fake_data, nb_data_target) = \
        create_source_target_hdf5
    
    indices = np.random.choice(nb_fake_data, nb_data_target, replace=False)
    indices.sort()

    with SimpleTimer() as efficient_timer:
        copy_by_batch_efficient(source,
                                target,
                                'train',
                                ['data', 'scores'],
                                indices)
    print("Efficient with lots of data took:", efficient_timer.interval)

    with h5py.File(source, "r") as f_src, h5py.File(target, "r") as f_target:
        # If there's too much data, that check might be slow.
        assert np.allclose(f_src['train/data'][indices],
                           f_target['train/data'][:])

    ###############################################################
    # Test the version with more data. (uses the multiproc version)
    (source, target, nb_fake_data, nb_data_target) = \
        create_source_target_hdf5_huge
    
    indices = np.random.choice(nb_fake_data, nb_data_target, replace=False)
    indices.sort()

    with SimpleTimer() as efficient_timer:
        copy_by_batch_efficient(source,
                                target,
                                'train',
                                ['data', 'scores'],
                                indices)
        
    print("Efficient with lots of data took:", efficient_timer.interval)

    
    with h5py.File(source, "r") as f_src, h5py.File(target, "r") as f_target:
        # If there's too much data, that check might be slow.
        assert np.allclose(f_src['train/data'][indices],
                           f_target['train/data'][:])