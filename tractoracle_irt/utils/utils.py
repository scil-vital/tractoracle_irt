from typing import Iterable
import math
import os
import sys
import hashlib
from pathlib import Path

from dipy.core.geometry import sphere2cart
from os.path import join as pjoin
from time import time, sleep
import cProfile
import pstats
import io
import numpy as np
import torch
from functools import wraps
from typing import Optional, Type
from types import TracebackType
from tractoracle_irt.utils.logging import get_logger
import threading
import atexit

LOGGER = get_logger(__name__)

COLOR_CODES = {
    'black': '\u001b[30m',
    'red': '\u001b[31m',
    'green': '\u001b[32m',
    'yellow': '\u001b[33m',
    'blue': '\u001b[34m',
    'magenta': '\u001b[35m',
    'cyan': '\u001b[36m',
    'white': '\u001b[37m',
    'reset': '\u001b[0m'
}

# Global registry of all LossHistory instances
THREAD_LOCK = threading.Lock()
THREAD = None

LOSS_HISTORY_REGISTRY = []
LOSS_HISTORY_LOCK = threading.Lock()

def register_loss_history(instance):
    with LOSS_HISTORY_LOCK:
        LOSS_HISTORY_REGISTRY.append(instance)

def background_saver(interval=20):
    while True:
        sleep(interval)
        LOGGER.debug("Saving all LossHistory instances in the background...")
        with LOSS_HISTORY_LOCK:
            for instance in LOSS_HISTORY_REGISTRY:
                try:
                    instance.write(only_if_changed=True)
                except Exception as e:
                    LOGGER.warning(f"Failed to save LossHistory {instance.name}: {e}")

def write_on_exit():
    with LOSS_HISTORY_LOCK:
        for instance in LOSS_HISTORY_REGISTRY:
            try:
                instance.write()
            except Exception as e:
                LOGGER.warning(f"Failed to save LossHistory {instance.name} on exit: {e}")

atexit.register(write_on_exit)

class LossHistory(object):
    """ History of the loss during training.
    Usage:
        monitor = LossHistory()
        ...
        # Call update at each iteration
        monitor.update(2.3)
        ...
        monitor.avg  # returns the average loss
        ...
        monitor.end_epoch()  # call at epoch end
        ...
        monitor.epochs  # returns the loss curve as a list
    """

    def __init__(self, name, filename, path, log_each_step=True, handle_out_dir=True):
        self.name = name
        self.history = []
        self.epochs = []
        self.sum = 0.0
        self.count = 0
        self._avg = 0.0
        self.num_iter = 0
        self.num_epochs = 0
        self.HISTORY_LOCK = threading.Lock()
        self.HISTORY_LAST_WRITTEN_LENGTH = -1

        # Create the directory if it doesn't exist and
        # set the file path for saving the history
        self.filename = filename
        if handle_out_dir:
            directory = pjoin(path, 'plots')
        else:
            directory = path
        os.makedirs(directory, exist_ok=True)
        self.file_path = pjoin(directory, f'{self.filename}.npy')

        self.log_each_step = log_each_step
        self.handle_out_dir = handle_out_dir

        LOGGER.debug(f"Creating new monitor for {name} at {self.file_path}")

        with THREAD_LOCK:
            global THREAD
            if THREAD is None or not THREAD.is_alive():
                THREAD = threading.Thread(target=background_saver, daemon=True)
                THREAD.start()
            register_loss_history(self)

    def __len__(self):
        return len(self.history)

    def update(self, value, step=None, epoch=None):
        if np.isinf(value):
            return

        self.sum += value
        self.count += 1

        time_point = (self.count, value)
        self.history.append(time_point)
        self._avg = self.sum / self.count
        self.num_iter += 1

    @property
    def avg(self):
        return self._avg

    def end_epoch(self, epoch):
        if self.num_iter == 0:
            return

        if len(self.epochs) > 1 \
            and not self.log_each_step \
            and epoch == self.epochs[-1][0]:
            LOGGER.warning("Epoch {} already exists in the history.")
            return

        self.epochs.append((epoch, self._avg))
        self.sum = 0.0
        # self.count = 0
        self._avg = 0.0
        self.num_epochs += 1

    def write(self, only_if_changed=False):
        do_write = True
        if only_if_changed:
            with self.HISTORY_LOCK:
                if len(self.history) == self.HISTORY_LAST_WRITTEN_LENGTH:
                    do_write = False
                else:
                    self.HISTORY_LAST_WRITTEN_LENGTH = len(self.history)

        if do_write:
            with open(self.file_path, 'wb') as f:
                if self.log_each_step:
                    np.save(f, self.history)
                else:
                    np.save(f, self.epochs)


class SimpleTimer:
    def __init__(self):
        self.start = None
        self.end = None
        self.interval = None
    
    def __enter__(self):
        self.start = time()
        return self
    
    def __exit__(self, exctype, excinst, tb):
        self.end = time()
        self.interval = self.end - self.start

class Timer:
    """ Times code within a `with` statement, optionally adding color. """

    def __init__(self, txt, newline=False, color=None):
        try:
            prepend = (COLOR_CODES[color] if color else '')
            append = (COLOR_CODES['reset'] if color else '')
        except KeyError:
            prepend = ''
            append = ''

        self.txt = prepend + txt + append
        self.newline = newline

    def __enter__(self):
        self.start = time()
        if not self.newline:
            print(self.txt + "... ", end="")
            sys.stdout.flush()
        else:
            print(self.txt + "... ")

    def __exit__(self, type, value, tb):
        if self.newline:
            print(self.txt + " done in ", end="")

        print("{:.10f} sec.".format(time() - self.start))

    @classmethod
    def decorator(cls, enabled=True, color=None):
        def _decorator(func):
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                with cls(f"Running {func.__name__}", newline=True,
                         color=color):
                    result = func(*args, **kwargs)
                return result
            return wrapper
        
        _identity = lambda func: func

        return _decorator if enabled else _identity

class ManualProfiler:
    """
    Since a profiler often gives way too much information at one, this class
    aims to be able to manually profile certain parts of the code instead of
    using time.time() several times manually. This class is meant to be used
    as a context manager for a similar purpose.

    Usage:
    with ManualProfiler("Some title") as profiler:
        # Code to profile 1
        profiler.point("Code 1")
        # Code to profile 2
        profiler.point("Code 2")
        # Code to profile 3
        profiler.point("Code 3")
    
    # On exit, the profiler will print the time taken between each point with
    # their associated text. It will also compute the percentage of time spent
    # between each point.
        
    """
    def __init__(self, title=None, print_enabled=True):
        self.title = title
        self.points_txt = []
        self.point_times = []
        self.print_enabled = print_enabled

        try:
            self.error_color = COLOR_CODES['red']
            self.reset_color = COLOR_CODES['reset']
        except KeyError:
            self.error_color = ''
            self.reset_color = ''

    def __enter__(self):
        self.start = time()
        self.points_txt.append("start")
        self.point_times.append(0)
        return self
    
    def __exit__(self, exctype: Optional[Type[BaseException]],
                 excinst: Optional[BaseException],
                 tb: Optional[TracebackType]) -> bool:
        self.end = time()
        self.points_txt.append("end")
        self.point_times.append(self.end - self.start)

        self.total_time = self.end - self.start
        
        if not self.print_enabled:
            return False

        if self.title:
            print("==============================================")
            print(f"ManualProfiler: {self.title}")
        
        print("==============================================")

        if exctype is not None:
            print(self.error_color + "An exception has occurred: "
                  f"{exctype.__name__}. The results may be incomplete."
                  + self.reset_color)

        print("Time between each point:")

        for i in range(1, len(self.points_txt)):
            duration = self.point_times[i] - self.point_times[i - 1]
            percentage = (duration / self.total_time) * 100

            millisecond_str = ""
            if duration < 1:
                millisecond_str = f"{(duration*1000):.1f} ms - "

            print(f"{self.points_txt[i - 1]} => {self.points_txt[i]}: {duration:.4f} s ({millisecond_str}{percentage:.2f}%)")

        print("====================================")


    def point(self, text):
        point_time = time()
        point_duration = point_time - self.start
        self.points_txt.append(text)
        self.point_times.append(point_duration)
    

def from_sphere(actions, sphere, norm=1.):
    vertices = sphere.vertices[actions]
    return vertices * norm


def normalize_vectors(v, norm=1.):
    # v = (v / np.sqrt(np.sum(v ** 2, axis=-1, keepdims=True))) * norm
    v = (v / np.sqrt(np.einsum('...i,...i', v, v))[..., None]) * norm
    # assert np.all(np.isnan(v) == False), (v, np.argwhere(np.isnan(v)))
    return v


def from_polar(actions, radius=1.):

    radii = np.ones((actions.shape[0])) * radius
    theta = ((actions[..., 0] + 1) / 2.) * (math.pi)
    phi = ((actions[..., 1] + 1) / 2.) * (2 * math.pi)

    X, Y, Z = sphere2cart(radii, theta, phi)
    cart_directions = np.stack((X, Y, Z), axis=-1)
    return cart_directions


def prettier_metrics(metrics, as_line: bool = False, title: str = None):
    """ Pretty print metrics """
    if as_line:
        return " | ".join(["{}: {:.4f}".format(k, v) for k, v in metrics.items()])

    # Build a string of the metrics to eventually print
    # The string should be the representation of a table
    # with each metrics on a row with the following format:
    # ===================================
    # Test results
    # ===================================
    # | metric_name     |   metric_value |
    # | metric_name     |   metric_value |
    # ===================================

    # Get the length of the longest metric value and name
    max_key_len = max([len(k) for k in metrics.keys()])
    max_val_len = max([len(str(round(v, 4))) for v in metrics.values()])

    # Create the header
    header = "=" * (max_key_len + max_val_len + 7)
    if title is None:
        header = header + "\nTest results\n"
    else:
        header = header + "\n" + title + "\n"
    header = header + "=" * (max_key_len + max_val_len + 7)

    # Create the table
    table = ""
    for k, v in sorted(metrics.items()):
        table = table + \
            "\n| {:{}} | {:.4f} |".format(
                k, max_key_len, round(v, 4), max_val_len)

    # Create the footer
    footer = "=" * (max_key_len + max_val_len + 7)

    return header + table + "\n" + footer

def prettier_dict(d: dict, title: str = None):
    # Build a string of the metrics to eventually print
    # The string should be the representation of a table
    # with each metrics on a row with the following format:
    # ===================================
    # <Print the title here>
    # ===================================
    # | train
    # |   ↳ other_dict
    # |     ↳ value1 : 0.1
    # |     ↳ value2 : 0.2
    # |   ↳ value3 : 0.3
    # | test
    # |   ↳ other_dict
    # |     ↳ value1 : 0.1
    # |     ↳ value2 : 0.2
    # |   ↳ value4 : 0.4
    # ===================================

    # We want to recursively print the dictionary in a pretty way
    # with increasing indentation for each level of the dictionary.
    # At level 0, there should only be | and one whitespace before the key.
    # At level 1, there should be |, 2 whitespaces, ↳ and one whitespace before the key.
    # At level 2, there should be |, 4 whitespaces, ↳ and one whitespace before the key.
    # etc.
    left_padding = 1        # Padding after |
    nb_indent_spaces = 2    # Whitespace after left padding and before ↳
    right_padding = 4       # Number of extra "=" characters for the header/footer

    def pretty(d, level=0):
        table = ""
        indentation = " " * nb_indent_spaces * level
        for k, v in d.items():
            if isinstance(v, dict):
                if level > 0:
                    table = table + \
                        "\n|" + " " * left_padding + indentation + "↳ " + k + pretty(v, level + 1)
                else:
                    table = table + \
                        "\n|" + " " * left_padding + indentation + k + pretty(v, level + 1)
            else:
                if level > 0:
                    table = table + \
                        "\n|" + " " * left_padding + indentation + "↳ " + k + " : " + str(v)
                else:
                    table = table + \
                        "\n|" + " " * left_padding + indentation + k + " : " + str(v)
        return table
    
    table = pretty(d)

    # Create the header. The header length should be as long as the longest line in the table.
    max_line_len = max([len(line) for line in table.split("\n")])
    if title is not None:
        header = "=" * (max_line_len + right_padding) + "\n"
        header = header + title + "\n"
        header = header + "=" * (max_line_len + right_padding)
    else:
        header = "=" * (max_line_len + right_padding)

    # Create the footer
    footer = "=" * (max_line_len + right_padding)

    return header + table + "\n" + footer


class TTLProfiler:
    def __init__(self, enabled: bool = True, throw_at_stop: bool = True, out_file: str = None) -> None:
        self.pr = None
        self.enabled = enabled
        self.throw_at_stop = throw_at_stop
        self.out_file = out_file

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exctype, excinst, tb):
        self.stop()

    def start(self):
        if not self.enabled:
            return

        if self.pr is not None:
            import warnings
            warnings.warn(
                "Profiler already started. Stop it before starting a new one.")
            return

        self.pr = cProfile.Profile()
        self.pr.enable()

    def stop(self):
        if not self.enabled:
            return

        if self.pr is None:
            import warnings
            warnings.warn("Profiler not started, but stop() was called.")
            return

        self.pr.disable()
        if self.out_file is not None:
            self.pr.dump_stats(self.out_file)
        else:
            s = io.StringIO()
            ps = pstats.Stats(self.pr, stream=s).sort_stats('cumulative')
            ps.print_stats()
            print(s.getvalue())
        
        self.pr = None

        if self.throw_at_stop:
            raise RuntimeError("Profiling stopped.")


def break_if_grads_have_nans(named_params: Iterable):
    for name, param in named_params:
        if param.grad is not None and torch.isnan(param.grad).any():
            breakpoint()
            indexes = get_index_where_nans(param.grad)
            print(f"Gradient of parameter {name} has NaNs.")
            raise ValueError('Gradient has NaNs')


def break_if_params_have_nans(params: Iterable):
    for p in params:
        if torch.isnan(p).any():
            breakpoint()
            indexes = get_index_where_nans(p)
            print("Parameter has NaNs.")
            raise ValueError('Parameter has NaNs')
        elif torch.isinf(p).any():
            breakpoint()
            indexes = torch.isinf(p).nonzero()
            print("Parameter has Infs.")
            raise ValueError('Parameter has Infs')


def break_if_found_nans(t: torch.Tensor):
    if isinstance(t, torch.Tensor) and torch.numel(t) != 0:
        # Check if there's a NaN
        if torch.isnan(t).any():
            breakpoint()
            indexes = get_index_where_nans(t)
            print("Tensor has NaNs.")
            raise ValueError('Tensor has NaNs')
        # Check if there's any infinity
        elif torch.isinf(t).any():
            breakpoint()
            indexes = torch.isinf(t).nonzero()
            print("Tensor has Infs.")
            raise ValueError('Tensor has Infs')


def break_if_found_nans_args(*args):
    for arg in args:
        break_if_found_nans(arg)


def get_index_where_nans(t: torch.Tensor):
    if torch.numel(t) != 0:
        if torch.isnan(t.max()) or torch.isnan(t.min()):
            return torch.isnan(t).nonzero()
    return torch.tensor([], dtype=torch.int32)

def get_unique_experiment_name(experiment_name, exp_id):
    full_string = ''.join([experiment_name, exp_id, str(time())])
    h = hashlib.sha1(full_string.encode('ascii'))
    digestible_hash = h.hexdigest()
    
    unique_name = digestible_hash[:8] + experiment_name
    return unique_name

def assert_space_available(size_in_gb, perc_threshold=0.8):
    assert perc_threshold > 0 and perc_threshold < 1, \
        "The threshold should be between 0 and 1."

    # Get the disk space available
    statvfs = os.statvfs("/")
    free_space = statvfs.f_frsize * statvfs.f_bavail
    free_space_in_gb = free_space / (1024**3)

    # Make sure that new file doesn't exceed 80% of the disk space.
    if size_in_gb > perc_threshold * free_space_in_gb:
        raise RuntimeError("The file you're trying to create will probably "
                           "fill up (or almost) the disk space available "
                           "(size: {} GB, available: {} GB).".format(
                               size_in_gb, free_space_in_gb))

def get_size_in_gb(shapes, dtype=np.float32):
    """
    This function takes a list of shapes and a dtype and computes the total
    size of the file that would be created if we were to store the data all
    at once (in GB).

    This could be used to compare with the remaining disk space to make sure
    that we don't fill up the disk.

    Parameters
    ----------
    shapes: tuple
        An array of tuples of shapes to be stored in the file. We will need to add 
        the size of the state and next_state arrays.
    dtype: np.dtype
        The data type of the arrays to be stored in the file. Used to compute the
        number of bytes each element requires.
    """

    if not isinstance(shapes, list):
        shapes = [shapes]

    def get_gb_for_shape(shape):
        total_size_in_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        total_size_in_gb = total_size_in_bytes / (1024**3)
        return total_size_in_gb

    total_file_size_in_gb = sum(map(get_gb_for_shape, shapes))

    return total_file_size_in_gb

def assert_same_weights(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.all(torch.eq(p1, p2))

def count_parameters(model):
    # Parameters should have requires_grad=True to be counted properly.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_project_root_dir(as_str: bool = False):
    if as_str:
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    else:
        return Path(__file__).parents[2]
    
def is_running_on_slurm():
    """
    Check if the script is running on a SLURM cluster.
    Returns True if running on SLURM, False otherwise.
    """
    return 'SLURM_JOB_ID' in os.environ or 'SLURM_JOB_NAME' in os.environ or 'SLURM_ARRAY_TASK_ID' in os.environ

class LoadingThread:
    def __init__(self, message: str):
        self.message = message
        self.thread = None
        self.done = False
        self.t0 = None

    def __enter__(self):
        self.start()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        if exc_type is not None:
            print(f"\nError occurred: {exc_value}")

    def start(self):
        self.thread = threading.Thread(target=self.loading)
        self.t0 = time()
        self.thread.start()

    def stop(self):
        self.done = True
        self.thread.join()

    def loading(self):
        while True:
            if self.done:
                break
            sys.stdout.write(f'\r{self.message} {time() - self.t0:.1f}s')
            sys.stdout.flush()
            sleep(0.1)
        sys.stdout.write(f'\r{self.message} {time() - self.t0:.1f}s. Done.\n')