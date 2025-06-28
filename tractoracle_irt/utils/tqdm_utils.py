import contextlib
import inspect
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

"""
As printing to the console can break the progress bars, we often need to
redirect printing and logging to tqdm.write.
"""

@contextlib.contextmanager
def print_redirect_tqdm():
    """
    This is a context manager that redirects the print function to tqdm.write.
    This ONLY works for the builtin print function.
    """
    # Store builtin print
    old_print = print
    def new_print(*args, **kwargs):
        # If tqdm.tqdm.write raises error, use builtin print
        try:
            tqdm.write(*args, **kwargs)
        except Exception as e:
            old_print("Error in tqdm.write: ", e)
            old_print(*args, ** kwargs)

    try:
        # Globaly replace print with new_print
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print

@contextlib.contextmanager
def tqdm_redirect_context():
    with print_redirect_tqdm(), logging_redirect_tqdm():
        yield

class tqdm_redirect_class(tqdm):
    """
    This class is made to be a drop-in replacement for tqdm, that redirects
    all printings to tqdm.write. This is useful when you have a lot of print
    statements in your code and you don't want to replace them all with
    tqdm.write calls.

    It is meant for the following usage:
    1 - Iterating on the object itself
        e.g.: for x in tqdm_redirect_class(range(10), desc="1st loop"):
                  ...

    2 - Using it as a context manager
        e.g.: with tqdm_redirect_class(total=10, desc="my loop") as pbar:
                  for i in range(10):
                      pbar.update(1)

    The following usage is NOT supported (will have no effect):
    1 - AVOID instantiating it as a normal object (no with statement).
            pbar = tqdm_redirect_class()

        If you really need to use it as a normal object, you can do instead:
            with tqdm_redirect_context():
                pbar = tqdm(...)
    """
    def __enter__(self):
        self.print_redirector = print_redirect_tqdm()
        self.logging_redirector = logging_redirect_tqdm()
        self.print_redirector.__enter__()
        self.logging_redirector.__enter__()
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.print_redirector.__exit__(exc_type, exc_value, traceback)
        self.logging_redirector.__exit__(exc_type, exc_value, traceback)
        return super().__exit__(exc_type, exc_value, traceback)
    
    def __iter__(self):
        with print_redirect_tqdm(), logging_redirect_tqdm():
            for x in super().__iter__():
                yield x
