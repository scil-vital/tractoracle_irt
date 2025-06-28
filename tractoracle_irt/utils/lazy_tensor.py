import torch
import torch.multiprocessing as mp
import threading as th
import h5py as h5
import numpy as np
from queue import Empty
import tempfile
from pathlib import Path
import os
from time import sleep
from tqdm import tqdm

from tractoracle_irt.environments.state import State, StateShape, ConvState, ConvStateShape
from tractoracle_irt.utils.torch_utils import (torch_to_np, np_to_torch,
                                         is_torch_type, is_np_type)
from tractoracle_irt.utils.logging import get_logger
from tractoracle_irt.utils.utils import get_size_in_gb, assert_space_available
from enum import Enum

is_debug = False

LOGGER = get_logger(__name__)

class Sentinel(Enum):
    KILL = -1
    STOP_READING = None

def printw(*args, **kwargs):
    force_print = kwargs.get("force_print", False)
    if is_debug or force_print:
        print("[WORKER {}]".format(args[0]), *args[1:])

def get_size_in_mb(x):
    return x.nbytes / (1024**2)

def empty_queue(queue):
    try:
        while not queue.empty():
            queue.get(block=False)
    except:
        pass

def create_rb_file(shape, dtype=np.float32, file_path=None, permanent=False):
    """
    If only the filename is provided, the file will not be permanent and will
    be created within a temporary directory. If the filename is provided with
    a path, the file will be created at the specified location.

    For the current location, please specify "./filename.h5.
    """
    if file_path is None:
        file_path = "lazy_replay_buffer.h5"

    parent_dir_str = os.path.dirname(file_path)
    if len(parent_dir_str) == 0:
        # No parent dir specified. We should just create the file at this path.
        pass

    # Make sure the file we are trying to create will fit on disk.
    rb_file_size = get_size_in_gb([shape, shape], dtype)
    print("We are trying to create a rb_file of {} GB".format(rb_file_size))
    assert_space_available(rb_file_size)

    if not permanent:
        # If we don't want the file to be permanent, we should create a temporary file
        # That we will delete later.
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = Path(tmpdirname) / file_path
            with h5.File(file_path, 'w') as f:
                f.create_dataset("state", shape=shape, dtype=dtype)
                f.create_dataset("next_state", shape=shape, dtype=dtype)
            
            for _ in range(2):
                print("Yielding file path: ", file_path)
                yield file_path
    else:
        with h5.File(file_path, 'w') as f:
            f.create_dataset("state", shape=shape, dtype=dtype)
            f.create_dataset("next_state", shape=shape, dtype=dtype)
        
        for _ in range(2):
            print("Yielding file path: ", file_path)
            yield file_path

class NaiveLazyTensorManager(object):
    def __init__(self, max_size: int, state_dim: StateShape, batch_size, file_name=None, dtype = torch.float32, nb_prefetch=None, nb_readers=None):
        self.max_size = max_size
        self._shape = (max_size, *state_dim.neighborhood_common_shape)
        self._state_dim = state_dim
        self.current_read_size = 0
        self.batch_size = batch_size
        self._state_ds_name = "state"
        self._next_state_ds_name = "next_state"
        self._state_class = ConvState if isinstance(state_dim, ConvStateShape) else State

        self.torch_dtype = np_to_torch[dtype] if is_np_type(dtype) else dtype
        self.np_dtype = torch_to_np[dtype] if is_torch_type(dtype) else dtype
        
        self._file_gen = create_rb_file(self._shape, self.np_dtype, file_path=file_name)
        self._file = next(self._file_gen)

        self.s_prev_dirs = torch.zeros((self.max_size, self._state_dim.prev_dirs), dtype=self.torch_dtype)
        self.ns_prev_dirs = torch.zeros((self.max_size, self._state_dim.prev_dirs), dtype=self.torch_dtype)
        self.read_batch_size = None

    def enter_write_mode(self):
        pass

    def enter_read_mode(self, size):
        self.current_read_size = size

    def add(self, state: State, n_state: State, index):
        state_cpu = state.to('cpu')
        n_state_cpu = n_state.to('cpu')

        # Make sure the indices are sorted as it's required by h5py.
        if isinstance(index, torch.Tensor):
            index = index.cpu().numpy()
        
        index.sort()

        # Separate the state into neighborhood and previous directions
        state_conv = state_cpu._state_conv
        state_prev_dirs = state_cpu._previous_directions
        
        n_state_conv = n_state_cpu._state_conv
        n_state_prev_dirs = n_state_cpu._previous_directions

        # Write the data to the file
        with h5.File(self._file, 'r+') as f:
            f[self._state_ds_name][index] = state_conv.cpu().numpy()
            f[self._next_state_ds_name][index] = n_state_conv.cpu().numpy()

        # Add the previous dirs too
        self.s_prev_dirs[index] = state_prev_dirs
        self.ns_prev_dirs[index] = n_state_prev_dirs
        
    def get_next_batch(self):
        indices = np.random.choice(self.current_read_size, self.batch_size, replace=False)
        indices.sort()

        if self.read_batch_size is not None:
            o_state = np.zeros((self.batch_size, *self._state_dim.neighborhood_common_shape), dtype=self.np_dtype)
            o_next_state = np.zeros((self.batch_size, *self._state_dim.neighborhood_common_shape), dtype=self.np_dtype)

        with h5.File(self._file, 'r') as f:
            if self.read_batch_size is not None:
                # Read the data at the indices in batches as it's sometimes faster than in one go.
                for start in range(0, len(indices), self.read_batch_size):
                    end = min(start + self.read_batch_size, len(indices))

                    batch_indices = indices[start:end]

                    o_state[start:end] = f["state"][batch_indices]
                    o_next_state[start:end] = f["next_state"][batch_indices]
            else:
                # Read the data in one go from the file.
                o_state = f["state"][indices]
                o_next_state = f["next_state"][indices]

        # Convert to tensors.
        o_state = torch.from_numpy(o_state)
        o_next_state = torch.from_numpy(o_next_state)

        # Reconstruct the State object so we can have the previous directions.
        o_state = self._state_class(o_state, self.s_prev_dirs[indices])
        o_next_state = self._state_class(o_next_state, self.ns_prev_dirs[indices])

        return o_state, o_next_state, indices

class LazyTensorManager(object):
    def __init__(self, max_size, state_dim: StateShape, batch_size, dtype = torch.float32, nb_prefetch = 3,
                 nb_readers = 3, file_name=None):

        print("Creating LazyTensorManager")
        # TODO: parameterize these
        self.max_size = max_size
        self._state_dim = state_dim
        self._shape = (max_size, *state_dim.neighborhood_common_shape) # (max_size, *shape)
        self._placeholder_shape = (batch_size, *state_dim.neighborhood_common_shape) # (batch_size, *shape)
        self._read_batch_size = 2 # TODO: Why is it that the read batch size can't be higher than 5? If it is, the child process is terminated because of a segmentation fault.
        self._nb_prefetch = nb_prefetch
        self._nb_readers = nb_readers
        self._state_ds_name = "state"
        self._next_state_ds_name = "next_state"
        self._state_class = ConvState if isinstance(state_dim, ConvStateShape) else State

        self.torch_dtype = np_to_torch[dtype] if is_np_type(dtype) else dtype
        self.np_dtype = torch_to_np[dtype] if is_torch_type(dtype) else dtype

        print("Creating queues")
        self._reset_queues()

        self.current_read_size = None

        self.rng = np.random.default_rng()
        self.is_writing_mode = False
        self.workers_can_work = mp.Event()
        self.all_stopped_condition = mp.Condition()
        self.stopped_counter = mp.Value('i', 0)
        #self._manager = mp.Manager()
        self._reading_workers = []

        # Create the file
        # print("Creating the file")
        self._file_gen = create_rb_file(self._shape, self.np_dtype, file_path=file_name)
        # print("File created")
        self._file = next(self._file_gen)

        # Since the states we are storing are State objects, we need to store the previous directions as well.
        # However, it's light enough to store them in memory directly.
        self.s_prev_dirs = torch.zeros((self.max_size, self._state_dim.prev_dirs), dtype=self.torch_dtype)
        self.ns_prev_dirs = torch.zeros((self.max_size, self._state_dim.prev_dirs), dtype=self.torch_dtype)

        # Create the monitor to check if the workers are still alive.
        self.kill_monitor_event = th.Event()
        self.monitor_thread = MonitorThread(self._reading_workers, self.kill_monitor_event)
        self.kill_monitor_event.clear()
        self.monitor_thread.start()

    def __del__(self):
        LOGGER.debug("LazyTensorManager object deleted. Cleaning up.")
        self.kill_monitor_event.set()
        self.stop_workers(kill=True, force=False)
        LOGGER.debug("LazyTensorManager object deleted. Cleaned up.")

    def add(self, state: State, n_state: State, index):
        if not self.is_writing_mode:
            raise RuntimeError("The manager is not in writing mode. Call enter_write_mode first.")

        state_cpu = state.to('cpu')
        n_state_cpu = n_state.to('cpu')

        # Make sure the indices are sorted as it's required by h5py.
        if isinstance(index, torch.Tensor):
            index = index.cpu().numpy()
        
        index.sort()

        # Separate the state into neighborhood and previous directions
        state_conv = state_cpu._state_conv
        state_prev_dirs = state_cpu._previous_directions
        
        n_state_conv = n_state_cpu._state_conv
        n_state_prev_dirs = n_state_cpu._previous_directions

        # Write the data to the file
        with h5.File(self._file, 'r+') as f:
            f[self._state_ds_name][index] = state_conv.cpu().numpy()
            f[self._next_state_ds_name][index] = n_state_conv.cpu().numpy()

        # Add the previous dirs too
        self.s_prev_dirs[index] = state_prev_dirs
        self.ns_prev_dirs[index] = n_state_prev_dirs

    def enter_write_mode(self):
        """
        This function should block all the readers from reading the data.
        """
        if self.is_writing_mode:
            raise RuntimeError("The manager is already in writing mode.")
        
        self.stop_workers(kill=False, force=False)

        self.current_read_size = None
        self.is_writing_mode = True

    def enter_read_mode(self, size):
        if not self.is_writing_mode:
            raise RuntimeError("The manager is already in reading mode.")
        
        self.is_writing_mode = False
        self.current_read_size = size

        self._assert_queues_are_init(suffix="[before enter_read_mode]")

        for _ in range(self._nb_readers):
            self.add_indices()
        
        self.workers_can_work.set()

        # Spawn the reading workers
        self._spawn_and_repair_reading_workers()

        assert len(self._reading_workers) == self._nb_readers

    def _spawn_and_repair_reading_workers(self):
        """
        This function manages the spawning/starting of the reading workers.
        If no workers are currently running, we might need to respawn
        them.

        If the workers are still running and haven't exited, do not do anything.
        """
        assert self._nb_readers > 0

        if len(self._reading_workers) == 0:
            for i in range(self._nb_readers):
                reader = self._spawn_one_reading_worker(i)
                reader.start()
                self._reading_workers.append(reader)            
        else:
            """
            There are workers that were already spawned.
            Make sure every worker haven't exited. If it did, clean it up
            and spawn a new one.
            """
            for i, reader in enumerate(self._reading_workers):
                if not reader.is_alive():
                    LOGGER.info("Reader worker {} is dead. Terminating it and creating a new one.".format(i))
                    reader.terminate()
                    reader.join()

                    self._reading_workers[i] = self._spawn_one_reading_worker(i)
                    self._reading_workers[i].start()

    def _spawn_one_reading_worker(self, i):
        LOGGER.debug("Spawning reading worker {}".format(i))
#        reader = mp.Process(
#            target=_reading_worker,
#            args=(i, self._file, self.placeholders_and_indices,
#                self.placeholders_and_indices_ready_to_consume,
#                self._read_batch_size, self._state_ds_name,
#                self._next_state_ds_name, self.workers_can_work,
#                self.all_stopped_condition, self.stopped_counter,
#                self._nb_readers, self.s_placeholders, self.ns_placeholders))
        
        reader = ReadingWorker(i, self._file, self.placeholders_and_indices,
                                self.placeholders_and_indices_ready_to_consume,
                                self._read_batch_size, self._state_ds_name,
                                self._next_state_ds_name, self.workers_can_work,
                                self.all_stopped_condition, self.stopped_counter,
                                self._nb_readers, self.s_placeholders, self.ns_placeholders)

        return reader

    def get_next_batch(self, check_worker_states=False):
        """
        This function should fetch the next indices that are ready to be consumed.
        """
        if check_worker_states:
            readers_checked = []
            got_something = False
            while not got_something:
                try:
                    indices, state, next_state = self.placeholders_and_indices_ready_to_consume.get(timeout=1)
                    got_something = True
                except Empty:
                    got_something = False

                    # Check if the workers are still alive at this point.
                    for reader in self._reading_workers:
                        if not reader.is_alive() and reader.pid not in readers_checked:
                            print("Reader worker {} is dead. Exit code: {}".format(reader.pid, reader.exitcode))
                            readers_checked.append(reader.pid)
                            break
        else:
            indices, state, next_state = self.placeholders_and_indices_ready_to_consume.get()

        # Clone the tensors out of shared memory.
        o_state = state.clone()
        o_next_state = next_state.clone()

        # Reuse the buffers that are in shared memory.
        self.s_placeholders.put(state)
        self.ns_placeholders.put(next_state)

        self.add_indices()

        # Reconstruct the State object so we can have the previous directions.
        o_state = self._state_class(o_state, self.s_prev_dirs[indices])
        o_next_state = self._state_class(o_next_state, self.ns_prev_dirs[indices])

        return o_state, o_next_state, indices

    def add_indices(self):
        """ Be careful, this method can block.
        """
        random_indices = self.rng.choice(self.current_read_size, self._placeholder_shape[0], replace=False)

        # Reserve free placeholders
        reserved_state_placeholder = self.s_placeholders.get()
        reserved_next_state_placeholder = self.ns_placeholders.get()

        # Associate the random indices with the reserved placeholder index
        indices_and_placeholder_index = (random_indices, reserved_state_placeholder, reserved_next_state_placeholder)
        self.placeholders_and_indices.put(indices_and_placeholder_index)
    
    def _assert_queues_are_init(self, suffix=""):
        is_indices_empty = self.placeholders_and_indices.empty()
        is_ready_empty = self.placeholders_and_indices_ready_to_consume.empty()
        s_placeholders_size = self.s_placeholders.qsize()
        ns_placeholders_size = self.ns_placeholders.qsize()

        assert is_indices_empty, "There are indices left in the queue. {} {}".format(suffix, self._get_queues_states_str())
        assert is_ready_empty, "There is still stuff to be consumed in the queue. {} {}".format(suffix, self._get_queues_states_str())
        #assert s_placeholders_size == self._nb_prefetch, "There are still placeholders in the queue ({} != {}). {} {}".format(s_placeholders_size, self._nb_prefetch, suffix, self._get_queues_states_str())
        #assert ns_placeholders_size == self._nb_prefetch, "There are still placeholders in the queue ({} != {}). {} {}".format(ns_placeholders_size, self._nb_prefetch, suffix, self._get_queues_states_str())

        if s_placeholders_size != self._nb_prefetch:
            self._assert_no_sentinels_in_queue(self.s_placeholders)
            assert s_placeholders_size == self._nb_prefetch, "There are still placeholders in the queue ({} != {}). {} {}".format(s_placeholders_size, self._nb_prefetch, suffix, self._get_queues_states_str())
        if ns_placeholders_size != self._nb_prefetch:
            self._assert_no_sentinels_in_queue(self.ns_placeholders)
            assert ns_placeholders_size == self._nb_prefetch, "There are still placeholders in the queue ({} != {}). {} {}".format(ns_placeholders_size, self._nb_prefetch, suffix, self._get_queues_states_str())

    def _assert_no_sentinels_in_queue(self, queue):
        nb_elements = queue.qsize()
        stop_sentinel_count = 0
        kill_sentinel_count = 0
        tensor_count = 0
        other_count = 0
        for _ in range(nb_elements):
            elem = queue.get()
            
            if isinstance(elem, Sentinel):
                if elem == Sentinel.STOP_READING:
                    stop_sentinel_count += 1
                elif elem == Sentinel.KILL:
                    kill_sentinel_count += 1
            elif isinstance(elem, torch.Tensor):
                tensor_count += 1
            else:
                other_count += 1

            queue.put(elem)

        assert stop_sentinel_count == 0, "There are still STOP_READING sentinels in the queue. ({})".format(stop_sentinel_count)
        assert kill_sentinel_count == 0, "There are still KILL sentinels in the queue. ({})".format(kill_sentinel_count)
        assert other_count == 0, "There are still other elements in the queue. ({})".format(other_count)

        print("[stop_sentinel_count={}] [kill_sentinel_count={}] [tensor_count={}] [other_count={}]".format(stop_sentinel_count, kill_sentinel_count, tensor_count, other_count))


    def stop_workers(self, kill=False, force=False):
        sentinel = Sentinel.STOP_READING

        if len(self._reading_workers) == 0:
            return

        # This should block the workers.        
        self.workers_can_work.clear()

        # Provide Sentinel to stop the workers.
        for _ in range(self._nb_readers):
            self.placeholders_and_indices_ready_to_consume.put(sentinel)
            self.placeholders_and_indices.put(sentinel)

        # Wait for all the readers to block on the lock
        with self.all_stopped_condition:
            self.all_stopped_condition.wait_for(lambda: self.stopped_counter.value == self._nb_readers)

        # We need to empty the queues and make sure we recycle the placeholders.
        while not self.placeholders_and_indices.empty():
            p = self.placeholders_and_indices.get()
            if not isinstance(p, Sentinel):
                self.s_placeholders.put(p[1])
                self.ns_placeholders.put(p[2])
        
        while not self.placeholders_and_indices_ready_to_consume.empty():
            p = self.placeholders_and_indices_ready_to_consume.get()
            if not isinstance(p, Sentinel):
                self.s_placeholders.put(p[1])
                self.ns_placeholders.put(p[2])

        self._assert_queues_are_init(suffix="[after stop_workers]")

        # If we kill the workers, we should unlock them and wait for them to finish.
        # They should finish once they read the sentinel value.
        if kill:
            for reader in self._reading_workers:
                self.placeholders_and_indices.put(Sentinel.KILL)

            self.workers_can_work.set()

            LOGGER.info("Killing workers gracefully. Waiting for them to finish.")

            for reader in self._reading_workers:
                reader.join()
                if reader.exitcode != 0:
                    print("Reader worker {} exited with exit code {}".format(
                        reader.pid, reader.exitcode))
            self._reading_workers = []
            LOGGER.info("All workers have been stopped.")
        elif force:
            for reader in self._reading_workers:
                reader.kill()
                reader.join()

            self._reading_workers = []

    def _get_queues_states_str(self):
        state = ""
        state += "s_placeholders: {}\n".format(self.s_placeholders.qsize())
        state += "ns_placeholders: {}\n".format(self.ns_placeholders.qsize())
        state += "placeholders_and_indices: {}\n".format(self.placeholders_and_indices.qsize())
        state += "placeholders_and_indices_ready_to_consume: {}\n".format(self.placeholders_and_indices_ready_to_consume.qsize())
        return state

    def _reset_queues(self):
        self.s_placeholders = mp.Queue() # Free placeholders
        self.ns_placeholders = mp.Queue() # Free placeholders
        self.placeholders_and_indices = mp.Queue() # Holds (<handle>, <indices>). This is filled when entering reading mode.
        self.placeholders_and_indices_ready_to_consume = mp.Queue() # Queue with the handles ready to be consumed (handles associated with the placeholders and the indices)
        print("Creating placeholders")

        size_of_placeholders = get_size_in_gb([self._placeholder_shape, self._placeholder_shape], self.np_dtype)
        print("Size of placeholders: ", size_of_placeholders)
        print("Total size for all placeholders: ", size_of_placeholders * self._nb_prefetch)

        for i in range(self._nb_prefetch):
            # print("Started creating placeholder number: ", i)
            s_placeholder = torch.zeros(self._placeholder_shape, dtype=self.torch_dtype) #.share_memory_()
            ns_placeholder = torch.zeros(self._placeholder_shape, dtype=self.torch_dtype) #.share_memory_()
            # print("Adding placeholder number: ", i)
            self.s_placeholders.put(s_placeholder)
            self.ns_placeholders.put(ns_placeholder)
            # print("Finished creating placeholder number: ", i)

        print("Finished resetting queues. [s_placeholders {}] [ns_placeholders {}] [placeholders_and_indices {}] [placeholders_and_indices_ready_to_consume {}]".format(self.s_placeholders.qsize(), self.ns_placeholders.qsize(), self.placeholders_and_indices.qsize(), self.placeholders_and_indices_ready_to_consume.qsize()))

    def clear(self):
        pass

class WorkerManager(mp.Process):
    def __init__(self, *args, **kwargs):
        super(WorkerManager, self).__init__(*args, **kwargs)

    def run(self):
        pass

class ReadingWorker(mp.Process):
    """
    Reading workers are spawned when we enter reading mode.
    """
    def __init__(self, reader_id, file_path, placeholders_and_indices,
                 placeholders_and_indices_ready_to_consume, read_batch_size,
                 state_ds_name, next_state_ds_name, workers_can_work,
                 all_stopped_condition, stopped_counter, total_nb_readers,
                 s_placeholders, ns_placeholders):
        super(ReadingWorker, self).__init__()

        # Make sure this process is terminated when the main process ends.
        # As they should stay alive for the rest of the main process' life
        # once they are spawned.
        self.daemon = True

        self.reader_id = reader_id
        self.file_path = file_path
        self.placeholders_and_indices = placeholders_and_indices
        self.placeholders_and_indices_ready_to_consume = placeholders_and_indices_ready_to_consume
        self.read_batch_size = read_batch_size
        self.state_ds_name = state_ds_name
        self.next_state_ds_name = next_state_ds_name
        self.workers_can_work = workers_can_work
        self.all_stopped_condition = all_stopped_condition
        self.stopped_counter = stopped_counter
        self.total_nb_readers = total_nb_readers
        self.s_placeholders = s_placeholders
        self.ns_placeholders = ns_placeholders
        
    def run(self):
        kill = False
        while not kill:
            if not self.workers_can_work.is_set():
                # Here we have to make sure that we signal the main process
                # when all the workers are done.
                with self.all_stopped_condition:
                    self.stopped_counter.value += 1
                    if self.stopped_counter.value == self.total_nb_readers:
                        self.all_stopped_condition.notify_all() # Notify the main process that all the workers have been stopped.

                self.workers_can_work.wait() # Wait here until the workers can work again.

                with self.all_stopped_condition:
                    self.stopped_counter.value -= 1

            # Read the data from the file
            printw(self.reader_id, "Waiting for data (q currently having {} elems).".format(self.placeholders_and_indices.qsize()))
            ps_and_i = self.placeholders_and_indices.get()
            printw(self.reader_id, "Got data from queue.")
            if ps_and_i == Sentinel.STOP_READING:
                printw(self.reader_id, "Read a STOP_READING sentinel. Pausing the worker.")
                kill = False
                continue
            elif ps_and_i == Sentinel.KILL:
                printw(self.reader_id, "Read a KILL sentinel. Stopping the worker")
                kill = True
                continue

            # Get the placeholders in which to store the values.
            indices, state_placeholder, n_state_placeholder = ps_and_i

            printw(self.reader_id, "Found placeholders and indexes.")

            indices.sort()

            num_indices = len(indices)
            with h5.File(self.file_path, 'r') as f:
                printw(self.reader_id, "Reading data")
                state_dataset = f[self.state_ds_name]
                next_state_dataset = f[self.next_state_ds_name]

                was_stopped_early = False # To make sure we don't add the placeholders multiple times.

                for i in tqdm(range(0, len(indices), self.read_batch_size), desc="[WORKER {}] Reading data".format(self.reader_id), disable=not is_debug):
                    if not self.workers_can_work.is_set():
                        # Let's stop all work and put the placeholders back in the queue.
                        # This is to speed up the stopping process of the workers.
                        printw(self.reader_id, "Putting placeholders back in the queue.", force_print=True)
                        self.s_placeholders.put(state_placeholder)
                        self.ns_placeholders.put(n_state_placeholder)
                        was_stopped_early = True
                        break

                    #printw(self.reader_id, "Slicing indices")
                    batch_slice = slice(i, min(i+self.read_batch_size, num_indices))
                    batch_indices = indices[batch_slice]

                    printw(self.reader_id, "Reading data from file")
                    state = state_dataset[batch_indices]
                    n_state = next_state_dataset[batch_indices]

                    # printw(self.reader_id, "state type: {} state shape: {} state dtype: {}".format(type(state), state.shape, state.dtype))
                    # printw(self.reader_id, "n_state type: {} n_state shape: {} n_state dtype: {}".format(type(n_state), n_state.shape, n_state.dtype))

                    #printw(self.reader_id, "Converting data to torch")
                    state_tensor = torch.from_numpy(state)
                    n_state_tensor = torch.from_numpy(n_state)

                    #printw(self.reader_id, "Copying data to placeholders with indices of shape {} and data of shape {} and slice of {}".format(batch_indices.shape, state_tensor.shape, batch_slice))
                    #printw(self.reader_id, "state_placeholder shape: {} and state_tensor shape: {}".format(state_placeholder[batch_slice].shape, state_tensor.shape))
                    state_placeholder[batch_slice] = state_tensor
                    #printw(self.reader_id, "n_state_placeholder shape: {} and n_state_tensor shape: {}".format(n_state_placeholder.shape, n_state_tensor.shape))
                    n_state_placeholder[batch_slice] = n_state_tensor
                    #printw(self.reader_id, "Done loading batch.")

                # Signal that the placeholder is ready to be consumed
                # only if the worker has finished reading/filling the placeholders.
                if not was_stopped_early:
                    printw(self.reader_id, "Signal that the placeholder is ready to be consumed.")
                    self.placeholders_and_indices_ready_to_consume.put((indices, state_placeholder, n_state_placeholder))

        printw(self.reader_id, "Killed.")

class MonitorThread(th.Thread):
    """
    The monitor thread is used to check the status of the workers. Sometimes
    if the process is killed expectedly (e.g. SIGINT, SEG FAULT, etc.) no
    feedback is given. This thread is used to detect such unexpected behavior.

    We could also potentially extend this thread to perform the following tasks:
    - Check the memory usage of the workers
    - Revive/respawn the workers if they are killed unexpectedly.
    
    Note: We are using Python's threading lib instead of the multiprocessing lib
          because this is not an intensive task that should only be executed
          periodically. If we were to use the multiprocessing lib, we encounter
          some issues as we can only access the Process.is_alive() method from the
          process launching the worker processes (aka the main process). Using a
          thread alleviates this problem.
    """
    def __init__(self, workers: list, kill_monitor_event: th.Event, yield_time: int = 1):
        super(MonitorThread, self).__init__()
        self.workers = workers
        self.kill_monitor_event = kill_monitor_event
        self.yield_time = yield_time  # seconds
        self.deaths_signaled = {}  # pid -> exitcode
        self.debug = False

    def run(self):
        while not self.kill_monitor_event.wait(self.yield_time):
            for worker in self.workers:
                # Make sure each worker is alive. If not, print a message.
                if not worker.is_alive():
                    pid = worker.pid
                    exitcode = worker.exitcode

                    if pid not in self.deaths_signaled:
                        self.deaths_signaled[pid] = exitcode
                        LOGGER.error("Worker {} is dead (exitcode: {}).".format(pid, worker.exitcode))
                elif self.debug:
                    LOGGER.info("Worker {} is alive.".format(worker))
            

"""
def _reading_worker(reader_id, file_path, placeholders_and_indices,
                    placeholders_and_indices_ready_to_consume,
                    read_batch_size, state_ds_name, next_state_ds_name,
                    workers_can_work, all_stopped_condition, stopped_counter,
                    total_nb_readers, s_placeholders, ns_placeholders):
    # Reading workers are spawned when we enter reading mode.
    kill = False
    while not kill:
        if not workers_can_work.is_set():
            # Here we have to make sure that we signal the main process
            # when all the workers are done.
            with all_stopped_condition:
                stopped_counter.value += 1
                if stopped_counter.value == total_nb_readers:
                    all_stopped_condition.notify_all() # Notify the main process that all the workers have been stopped.

            workers_can_work.wait() # Wait here until the workers can work again.

            with all_stopped_condition:
                stopped_counter.value -= 1

        # Read the data from the file
        printw(reader_id, "Waiting for data (q currently having {} elems).".format(placeholders_and_indices.qsize()))
        ps_and_i = placeholders_and_indices.get()
        printw(reader_id, "Got data from queue.")
        if ps_and_i == Sentinel.STOP_READING:
            printw(reader_id, "Read a STOP_READING sentinel. Pausing the worker.")
            kill = False
            continue
        elif ps_and_i == Sentinel.KILL:
            printw(reader_id, "Read a KILL sentinel. Stopping the worker")
            kill = True
            continue

        # Get the placeholders in which to store the values.
        indices, state_placeholder, n_state_placeholder = ps_and_i

        printw(reader_id, "Found placeholders and indexes.")

        indices.sort()

        num_indices = len(indices)
        with h5.File(file_path, 'r') as f:

            printw(reader_id, "Reading data")
            state_dataset = f[state_ds_name]
            next_state_dataset = f[next_state_ds_name]

            for i in tqdm(range(0, len(indices), read_batch_size), desc="[WORKER {}] Reading data".format(reader_id), disable=not is_debug):
                if not workers_can_work.is_set():
                    # Let's stop all work and put the placeholders back in the queue.
                    # This is speed up the stopping process of the workers.
                    s_placeholders.put(state_placeholder)
                    ns_placeholders.put(n_state_placeholder)
                    break

                printw(reader_id, "Slicing indices")
                batch_slice = slice(i, min(i+read_batch_size, num_indices))
                batch_indices = indices[batch_slice]

                printw(reader_id, "Reading data from file")
                state = state_dataset[batch_indices]
                n_state = next_state_dataset[batch_indices]

                # printw(reader_id, "state type: {} state shape: {} state dtype: {}".format(type(state), state.shape, state.dtype))
                # printw(reader_id, "n_state type: {} n_state shape: {} n_state dtype: {}".format(type(n_state), n_state.shape, n_state.dtype))

                printw(reader_id, "Converting data to torch")
                state_tensor = torch.from_numpy(state)
                n_state_tensor = torch.from_numpy(n_state)

                printw(reader_id, "Copying data to placeholders with indices of shape {} and data of shape {} and slice of {}".format(batch_indices.shape, state_tensor.shape, batch_slice))
                printw(reader_id, "state_placeholder shape: {} and state_tensor shape: {}".format(state_placeholder[batch_slice].shape, state_tensor.shape))
                state_placeholder[batch_slice] = state_tensor
                printw(reader_id, "n_state_placeholder shape: {} and n_state_tensor shape: {}".format(n_state_placeholder.shape, n_state_tensor.shape))
                n_state_placeholder[batch_slice] = n_state_tensor
                printw(reader_id, "Done loading batch.")

            # Signal that the placeholder is ready to be consumed
            printw(reader_id, "Signal that the placeholder is ready to be consumed.")
            placeholders_and_indices_ready_to_consume.put((indices, state_placeholder, n_state_placeholder))

    printw(reader_id, "Killed.")
"""
