import pytest

from tractoracle_irt.environments.state import ConvStateShape, ConvState
from tractoracle_irt.utils.lazy_tensor import (LazyTensorManager,
                                         NaiveLazyTensorManager,
                                         MonitorThread)
import numpy as np
import torch
import threading as th
import multiprocessing as mp
from tqdm import tqdm

common_shape = (28, 5, 5, 5)

def wrap_into_conv_state(tensor):
    return ConvState(tensor, torch.tensor([[]]))

def fill_file(manager, nb_elements):
    batch_size = 10000
    state = torch.ones((batch_size, *common_shape))
    next_state = torch.fill_(torch.zeros((batch_size, *common_shape)), 2)

    # The LazyTensorManager is made to also include the previous directions
    # since we're not always testing with that enabled, we need to wrap
    # the tensor in a ConvState object, even if the previous directions
    # are empty.
    state = wrap_into_conv_state(state)
    next_state = wrap_into_conv_state(next_state)

    for i in tqdm(range(0, nb_elements, batch_size), disable=False):
        manager.add(state, next_state, np.arange(i, i + batch_size))

@pytest.fixture
def init_lazy_tensor_manager():
    conv_shape = ConvStateShape(1000000, *common_shape, prev_dirs=0)

    batch_size = 4096
    nb_prefetch = 20
    nb_readers = 10
    m = LazyTensorManager(conv_shape.nb_streamlines, conv_shape, batch_size, nb_prefetch=nb_prefetch, nb_readers=nb_readers)

    # Write data to the file.
    m.enter_write_mode()

    fill_file(m, conv_shape.nb_streamlines)

    yield m, conv_shape

@pytest.fixture
def init_naive_lazy_tensor_manager():
    conv_shape = ConvStateShape(1000000, *common_shape, prev_dirs=0)
    batch_size = 4096
    m = NaiveLazyTensorManager(conv_shape.nb_streamlines, conv_shape, batch_size)

    # Write data to the file.
    m.enter_write_mode()

    fill_file(m, conv_shape.nb_streamlines)

    yield m, conv_shape

@pytest.fixture
def init_mock_workers():
    workers = []
    stop_event = mp.Event()

    def mock_worker(id, stop_event):
        while stop_event.wait(1) is False:
            print("Worker {} is running...".format(id))

    for i in range(4):
        worker = mp.Process(target=mock_worker, args=(i, stop_event))
        worker.start()
        workers.append(worker)
    
    yield workers, stop_event

def test_read_into_write_mode(init_lazy_tensor_manager):
    m, conv_shape = init_lazy_tensor_manager
    
    m.stop_workers() # Should be fine since there are no workers running

    # Start reading
    m.enter_read_mode(conv_shape.nb_streamlines)
    nb_batches = 50

    for i in range(nb_batches):
        batch = m.get_next_batch()
        print("batch loaded: {} with max val of {} and min val of ".format(i, batch[0]._state_conv.max(), batch[0]._state_conv.min()))

    #print("Process exitcode: ", m._reading_workers[0].exitcode)
    m.enter_write_mode()

    def add_to_file():
        batch_size = 10000
        state = torch.ones((batch_size, *common_shape))
        next_state = torch.fill_(torch.zeros((batch_size, *common_shape)), 2)

        state = wrap_into_conv_state(state)
        next_state = wrap_into_conv_state(next_state)

        for i in range(30000, conv_shape.nb_streamlines, batch_size):
            m.add(state, next_state, np.arange(i, i + batch_size))

    add_to_file()

    m.enter_read_mode(conv_shape.nb_streamlines)

    for i in range(nb_batches):
        batch = m.get_next_batch()
        print("batch loaded: {} with max val of {} and min val of {}".format(i, batch[0]._state_conv.max(), batch[0]._state_conv.min()))

    m.enter_write_mode()
    m.stop_workers(kill=True)

from tractoracle_irt.utils.utils import SimpleTimer

def test_parallel_speed_read(init_lazy_tensor_manager, init_naive_lazy_tensor_manager):
    m, conv_shape = init_lazy_tensor_manager
    m_naive, naive_conv_shape = init_naive_lazy_tensor_manager

    assert shape == naive_shape

    # Define the task to benchmark
    def task(manager: LazyTensorManager):
        with SimpleTimer() as t:            
            manager.enter_read_mode(conv_shape.nb_streamlines)
            nb_batches = 100

            for i in tqdm(range(nb_batches), desc="Task for {}".format(manager.__class__.__name__)):
                batch = manager.get_next_batch()
        return t.interval

    # Run the task for both managers and compare the time
    time_lazy = task(m)
    time_naive = task(m_naive)

    m.enter_write_mode()

    assert time_lazy < time_naive, "LazyTensorManager should be faster than NaiveLazyTensorManager" \
        "but got {} (parallel) and {} (naive)".format(time_lazy, time_naive)

def test_monitor_shut_down(init_mock_workers):
    workers, stop_event = init_mock_workers
    kill_monitor_event = th.Event()
    monitor = MonitorThread(workers, kill_monitor_event)

    monitor.start()
    stop_event.set() # Stop the workers
    kill_monitor_event.set() # Stop the monitor
    monitor.join(2)

    assert monitor.is_alive() == False, "Monitor should have shut down after 2 seconds."
    
    # Just to be sure, make sure all workers have shut down.
    for worker in workers:
        worker.join(1)
        assert worker.is_alive() == False, "Worker should have shut down (worker: {})".format(worker.pid)
