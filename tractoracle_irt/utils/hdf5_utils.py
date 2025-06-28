import h5py
import numpy as np
import os
from tqdm import tqdm
from typing import Union
import multiprocessing as mp

from tractoracle_irt.utils.logging import get_logger

LOGGER = get_logger(__name__)

is_hdf5_file = lambda x: os.path.isfile(x) and (x.endswith(".hdf5") or x.endswith(".h5"))

SHARED_VARS = {}
SHARED_VARS_LOCK = mp.Lock()
def _init_shared(*pargs, **kwargs):
    global SHARED_VARS
    SHARED_VARS.update(kwargs)

def copy_by_batch(src: Union[h5py.Dataset, list[h5py.Dataset]],
                  target: Union[h5py.Dataset, list[h5py.Dataset]],
                  indices: Union[np.ndarray, list],
                  pbar_desc: str="Copying indices",
                  batch_size: int=1000):
    """
    As sometimes indexing a large dataset with h5py and non-sequential indices
    can be very very slow, we can copy the dataset by batch to accelerate and
    track the process.

    We allow src and target to be a list of datasets where each element
    within the src list will be copied to the corresponding element in the
    target list (i.e. src[i] -> target[i]).
    """

    # Quite a few checks to make sure that the inputs are correct.
    assert batch_size > 0, "The batch size should be a positive integer."
    assert isinstance(indices, (np.ndarray, list)), "The indices should be a numpy array or a list."

    if not isinstance(src, list):
        assert not isinstance(target, list), "The target and the source " \
            "should have the same number of elements."
        src = [src]
        target = [target]

    assert len(src) == len(target), "The source and target should have the same length."
    
    # Make sure all elements from source have the same shape.
    assert all(s.shape[0] == src[0].shape[0] for s in src), \
        "All elements in the source should have the same length."
    assert all(t.shape[0] == target[0].shape[0] for t in target), \
        "All elements in the target should have the same length."

    for i, (s, t) in enumerate(zip(src, target)):
        assert s.shape[1:] == t.shape[1:], \
            "The source and target should have the same common shape for " \
            f"element {i}"
        
        # Make sure the source is big enough compared to the indices to be
        # copied/indexed properly without errors.
        assert s.shape[0] >= indices.size and s.shape[0] >= indices.max(), \
            "The source should have at least the maximum index in the indices."
        
        # Make sure the target is big enough to store the data
        assert t.shape[0] >= indices.size, \
            "The target should have a length of at least the number of " \
            "indices."

    total_copied = 0

    # Check if the indices are sequential
    # If so, just use the h5py built-in method
    is_sequential = np.all(np.diff(indices) == 1)

        
    if is_sequential:
        for s, t in zip(src, target):
            t[:] = src[indices]
        total_copied = len(indices)
    else:
        for i in tqdm(range(0, len(indices), batch_size),
                      desc=pbar_desc,
                      leave=False):
            num_indices = min(batch_size, len(indices)-i)
            batch = indices[i:i+num_indices]

            for s, t in zip(src, target):
                t[i:i+num_indices] = s[batch]

            total_copied += num_indices

    # Just to avoid some errors down the line, make sure that the data copied
    # to the target matches with the target's shape, so we don't have to worry
    # about zeros.
    if total_copied != target[0].shape[0]:
        LOGGER.warning(f"Number of indices copied ({total_copied}) does not "
                       f"match the target shape ({target.shape[0]}).")

    return total_copied

def _validate_copy_by_batch_multiproc_inputs(
        src_file_path: str,
        target_file_path: str,
        group_name: str,
        dataset_names: list,
        indices: Union[np.ndarray, list]):
    
    if not os.path.exists(src_file_path):
        raise FileNotFoundError(f"The file {src_file_path} does not exist.")
    elif not is_hdf5_file(src_file_path):
        raise ValueError(f"{src_file_path} is not a HDF5 file.")
    elif not isinstance(indices, (np.ndarray, list)):
        raise ValueError("The indices should be a numpy array or a list.")
    elif not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("The indices should be integers.")
    elif not np.all(np.diff(indices) >= 0):
        raise ValueError("The indices should be sorted.")
    
    with h5py.File(src_file_path, 'r') as f_src, h5py.File(target_file_path, 'r') as f_target:
        for dataset_name in dataset_names:
            if group_name is not None:
                try:
                    dataset_location = f_src[group_name]
                except KeyError:
                    raise ValueError(f"The group {group_name} does not exist in the source file.")

                if dataset_name not in f_src[group_name]:
                    raise ValueError(f"The dataset {dataset_name} does not exist in the group {group_name} of the source file.")
                
                if group_name not in f_target:
                    raise ValueError(f"The group {group_name} does not exist in the target file.")
                if dataset_name not in f_target[group_name]:
                    raise ValueError(f"The dataset {dataset_name} does not exist in the group {group_name} of the target file.")
                if len(indices) < len(f_target[group_name][dataset_name]):
                    raise ValueError("The number of indices do not fit the target dataset (too much indices for the space available).")
            else:
                if dataset_name not in f_src:
                    raise ValueError(f"The dataset {dataset_name} does not exist in the source file.")
                dataset_location = f_src

                if dataset_name not in f_target:
                    raise ValueError(f"The dataset {dataset_name} does not exist in the target file.")
                if len(indices) < len(f_target[dataset_name]):
                    raise ValueError("The number of indices do not fit the target dataset (too much indices for the space available).")
        
        if indices[-1] >= len(dataset_location[dataset_name]):
            raise ValueError("The maximum index in the indices is greater than the dataset size.")
    
    # The target file must be already created with the strcture already setup.
    # That means that the groups and datasets must be already available.
    if not os.path.exists(target_file_path):
        raise FileNotFoundError(f"The file {target_file_path} does not exist.")
    elif not is_hdf5_file(target_file_path):
        raise ValueError(f"{target_file_path} is not a HDF5 file.")
    

def copy_by_batch_multiproc(src_file_path: str,
                            target_file_path: str,
                            group_name: Union[str, list],
                            dataset_names: Union[str, list],
                            src_indices: Union[np.ndarray, list],
                            num_readers: int=None):
    """
    Use this if the default method to transfer data between two HDF5 datasets
    is too slow (potentially because of non-sequential indices).

    This is to replace:
    target[indices] = src[indices]

    The implementation of this function is a bit weird since we're leveraging
    multiprocesses to copy the data. It was identified that the bottleneck of
    copying huge amounts of data between two HDF5 datasets is the indexing
    of the reading operation. The writing itself seemed pretty fast, especially
    if the indices where to write are sequential.

    We leverage a couple of processes that will read the data and one process
    that will write the data to the HDF5 file. Before starting the processes,
    we compute a certain number of splits to read the data.

    We will start the writer process which will open the target file in write
    mode (or append if it doesn't already exist) and will write the data that's
    been read by the reader processes in the shared memory queue. HOWEVER,
    we need to make sure that we write data sequentially in the correct order
    within the target file. 

    Also, if the readers are all closed and the writer is still waiting on new
    data, we need to make sure that the writer closes when it's not writing
    and there's no other reader alive.
    """

    if num_readers is None:
        num_readers = mp.cpu_count() - 1

    if not isinstance(dataset_names, list):
        dataset_names = [dataset_names]

    _validate_copy_by_batch_multiproc_inputs(
        src_file_path, target_file_path, group_name, dataset_names, src_indices)
    
    # I want my splits to be at maximum 5000 indices.
    # This is to avoid splitting the data too much and also
    # to avoid giving too much data to read to each reader process.
    # This is a bit arbitrary and might need to be adjusted.
    num_splits = max(1, len(src_indices) // 5000)

    # Reduce the number of readers if there's not enough splits.
    num_readers = min(num_splits, num_readers) 

    target_indices = np.arange(len(src_indices), dtype=np.int32)
    splits = np.array_split(target_indices, num_splits)

    if not isinstance(src_indices, np.ndarray):
        src_indices = np.array(src_indices)

    with SHARED_VARS_LOCK:
        _init_shared(
            src_file_path=src_file_path,
            target_file_path=target_file_path,
            group_name=group_name,
            dataset_names=dataset_names,
            src_indices=src_indices,
            src_indices_typecode=src_indices.dtype.char,
            total_nb_splits=len(splits))

        # Init the shared dict where the data will be stored based on the index of the elements.
        _init_shared(queue=mp.Queue(maxsize=10))

        # Start the writer. (not with pool)
        writer = mp.Process(target=copy_by_batch_writer)
        writer.start()

        # Start the readers.
        with mp.Pool(num_readers) as pool:
            pool.map(copy_by_batch_reader, splits)
            pool.close()
            pool.join()

        # Add sentinel to stop the writer once it's done writing.
        SHARED_VARS['queue'].put(None) 

        writer.join()

def copy_by_batch_reader(split):
    src_file_path = SHARED_VARS['src_file_path']
    group_name = SHARED_VARS['group_name']
    dataset_names = SHARED_VARS['dataset_names']
    src_indices = np.frombuffer(SHARED_VARS['src_indices'], dtype=SHARED_VARS['src_indices_typecode'])
    queue = SHARED_VARS['queue']

    with h5py.File(src_file_path, 'r') as f:
        split_indices = src_indices[split]
        all_datasets = []
        for dataset_name in dataset_names:
            if group_name is None:
                data = f[dataset_name][split_indices]
            else:
                data = f[group_name][dataset_name][split_indices]
            all_datasets.append(data)
        
        # Might halt if the queue is full.
        queue.put((split, all_datasets))

def copy_by_batch_writer():
    target_file_path = SHARED_VARS['target_file_path']
    group_name = SHARED_VARS['group_name']
    dataset_names = SHARED_VARS['dataset_names']
    queue = SHARED_VARS['queue']
    total_nb_splits = SHARED_VARS['total_nb_splits']

    with h5py.File(target_file_path, 'a') as f:
        processed_splits = 0
        while processed_splits < total_nb_splits:
            data_tuple = queue.get() # Blocks until there's data.
            if data_tuple is None:
                break # Sentinel to stop the writer.
            target_indices, data_for_all_datasets = data_tuple

            for dataset_name, data in zip(dataset_names, data_for_all_datasets):
                if group_name is None:
                    f[dataset_name][target_indices] = data
                else:
                    f[group_name][dataset_name][target_indices] = data
            processed_splits += 1

def _validate_inputs_read_multiproc(file_path: str,
              group_name: str,
              dataset_name: str,
              indices: np.ndarray):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not os.path.isfile(file_path):
        raise ValueError(f"{file_path} is not a file.")
    
    with h5py.File(file_path, 'r') as f:
        if group_name is not None:
            if group_name not in f:
                raise ValueError(f"The group {group_name} does not exist in the file.")
            if dataset_name not in f[group_name]:
                raise ValueError(f"The dataset {dataset_name} does not exist in the group {group_name}.")
        else:
            if dataset_name not in f:
                raise ValueError(f"The dataset {dataset_name} does not exist in the file.")
            
        is_sorted = (np.diff(indices) >= 0).all()
        if not is_sorted:
            raise ValueError("The indices should be sorted.")
        elif indices[-1] >= len(f[dataset_name]):
            raise ValueError("The maximum index in the indices is greater than the dataset size.")
        elif not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("The indices should be integers.")


def read_hdf5_data_multiproc(file_path: str,
                             group_name: str,
                             dataset_name: str,
                             indices: np.ndarray,
                             num_workers: int = 8):
    with SHARED_VARS_LOCK:
        if len(indices) == 0:
            return np.array([])

        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)

        if num_workers is None:
            num_workers = mp.cpu_count()
            print(f"num_workers not provided. Using cpu_count(): {num_workers}")
        
        _validate_inputs_read_multiproc(file_path, group_name, dataset_name,
                                        indices)

        nb_points = 128
        direction_dim = 3
        output_shape = (len(indices), nb_points, direction_dim)

        # Make sure the indices are ordered.
        _init_shared(file_path=file_path)
        _init_shared(group_name=group_name)
        _init_shared(dataset_name=dataset_name)
        _init_shared(all_indices=mp.Array(indices.dtype.char, indices.flat, lock=False))
        _init_shared(placeholder=mp.Array(np.dtype(float).char, int(np.prod(output_shape)), lock=False))
        _init_shared(indices_typecode=indices.dtype.char)
        _init_shared(placeholder_typecode=np.dtype(float).char)
        _init_shared(placeholder_shape=output_shape)
        pool = mp.Pool(num_workers, initializer=_init_shared, initargs=SHARED_VARS.items())

        # Those splits will be given to the workers to read the data.
        # Inside the worker, we can use these splits like:
        # split_indices = indices_to_index[split]
        # data = f[group_name][dataset_name][split_indices]
        # placeholder[split] = data
        splits = np.array_split(np.arange(len(indices)), num_workers)

        pool.map(_read_hdf5_data_worker, splits)
        output = np.frombuffer(SHARED_VARS['placeholder'], dtype=SHARED_VARS["placeholder_typecode"]).reshape(output_shape)
        SHARED_VARS.clear()
    
    return output

def _read_hdf5_data_worker(split):
    file_path = SHARED_VARS['file_path']
    group_name = SHARED_VARS['group_name']
    dataset_name = SHARED_VARS['dataset_name']
    placeholder_shape = SHARED_VARS['placeholder_shape']
    all_indices_shared = np.frombuffer(SHARED_VARS['all_indices'], dtype=SHARED_VARS['indices_typecode'])
    placeholder_shared = np.frombuffer(SHARED_VARS['placeholder'], dtype=SHARED_VARS['placeholder_typecode']).reshape(*placeholder_shape)
    with h5py.File(file_path, 'r') as f:
        split_indices = all_indices_shared[split]
        if group_name is None:
            data = f[dataset_name][split_indices]
        else:
            data = f[group_name][dataset_name][split_indices]
        placeholder_shared[split] = data

def copy_by_batch_efficient(src_file_path: str,
                            target_file_path: str,
                            group_name: str,
                            dataset_names: Union[str, list],
                            src_indices: Union[np.ndarray, list],
                            num_readers: int=None):
    """
    This function is a convenience wrapper around the `copy_by_batch_multiproc`
    and `copy_by_batch` functions. Sometimes, especially with less data, the
    overhead of creating processes and managing them can be more expensive than
    just copying the data using batches. This function will determine which
    method to use based on the number of indices to copy.    
    
    Use this if the default method to transfer data between two HDF5 datasets
    is too slow (potentially because of non-sequential indices). This is to
    replace (with indices non-sequential but *sorted*):
    `target[indices] = src[indices]`
    """
    THRESHOLD = 50000

    if not isinstance(dataset_names, list):
        dataset_names = [dataset_names]

    # Make sure all the dataset_names are within the group_name for the target and source files.
    
    _validate_copy_by_batch_multiproc_inputs(
        src_file_path, target_file_path, group_name, dataset_names,
        src_indices)

    do_multiproc = len(src_indices) >= THRESHOLD
    do_multiproc = do_multiproc and mp.cpu_count() > 8
    # Default to multiproc if num_readers is provided.
    do_multiproc = do_multiproc or num_readers is not None 

    if len(src_indices) < THRESHOLD and num_readers is None:
        with h5py.File(src_file_path, "r") as f_src, \
            h5py.File(target_file_path, "a") as f_target:
            source_datasets = []
            target_datasets = []
            for dataset_name in dataset_names:
                source_ds = f_src[group_name][dataset_name] \
                    if group_name is not None else f_src[dataset_name]
                target_ds = f_target[group_name][dataset_name] \
                    if group_name is not None else f_target[dataset_name]
                
                source_datasets.append(source_ds)
                target_datasets.append(target_ds)
            
            LOGGER.debug("Copying data by batch.")
            copy_by_batch(source_datasets, target_datasets, src_indices)
    else:
        LOGGER.debug("Copying data by batch with multiprocesses.")
        copy_by_batch_multiproc(src_file_path, target_file_path, group_name,
                                dataset_names, src_indices, num_readers)

