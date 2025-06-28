# -*- coding: utf-8 -*-

import os

if __name__ == "__main__":
    import argparse
    argparse = argparse.ArgumentParser(description="Verify the BER post-processing results.")
    argparse.add_argument('tractogram', type=str, help='Path to the tractogram file containing all streamlines.')
    argparse.add_argument('--invalid_streamlines', type=str, default=None, help='Path to the file containing indices of invalid streamlines.')
    argparse.add_argument('--invalid_output', type=str, default=None, help='Path to the output file for invalid streamlines (trk file).')
    argparse.add_argument('--valid_streamlines', type=str, default=None, help='Path to the file containing indices of valid streamlines.')
    argparse.add_argument('--valid_output', type=str, default=None, help='Path to the output file for valid streamlines (trk file).')
    ARGS = argparse.parse_args()

    # Check arguments
    if ARGS.tractogram is None or not os.path.exists(ARGS.tractogram):
        raise ValueError("Tractogram file must be provided and must exist.")
    elif ARGS.invalid_streamlines is None and ARGS.valid_streamlines is None:
        raise ValueError("At least one of --invalid_streamlines or --valid_streamlines must be provided.")
    elif ARGS.invalid_streamlines is not None and not os.path.exists(ARGS.invalid_streamlines):
        raise ValueError(f"Invalid streamlines file not found: {ARGS.invalid_streamlines}")
    elif ARGS.valid_streamlines is not None and not os.path.exists(ARGS.valid_streamlines):
        raise ValueError(f"Valid streamlines file not found: {ARGS.valid_streamlines}")
    elif ARGS.invalid_output is None and ARGS.invalid_streamlines is not None:
        raise ValueError("Missing '--invalid_output'. Output path for invalid streamlines must be provided if invalid streamlines are specified.")
    elif ARGS.invalid_output is not None and ARGS.invalid_output.endswith('.trk') is False:
        raise ValueError("Invalid output path for invalid streamlines. It should end with .trk.")
    elif ARGS.valid_output is None and ARGS.valid_streamlines is not None:
        raise ValueError("Missing '--valid_output'. Output path for valid streamlines must be provided if valid streamlines are specified.")
    elif ARGS.valid_output is not None and ARGS.valid_output.endswith('.trk') is False:
        raise ValueError("Invalid output path for valid streamlines. It should end with .trk.")

# Imports
import nibabel as nib
import numpy as np
from dipy.io.streamline import load_tractogram, save_tractogram
from time import time, sleep
import threading
import sys

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

def load_indexes(file_path: str) -> np.ndarray:
    """
    Load streamline indices from a text file.
    Each line in the file should contain a single index.
    """
    if file_path is None:
        return None

    t0 = time()
    with open(file_path, 'r') as f:
        indexes = np.array([int(line.strip()) for line in f if line.strip().isdigit()])
    
    assert indexes.ndim == 1, "Indexes should be a 1D array."
    assert len(indexes) > 0, "No valid indexes found in the file."
    assert np.all(indexes >= 0), "All indexes should be non-negative."

    return indexes

def fetch_and_save_streamlines(tractogram, valid_indices: np.ndarray = None, invalid_indices: np.ndarray = None):
    with LoadingThread(" - Performing sanity checks..."):
        if valid_indices is None and invalid_indices is None:
            raise ValueError("Provide at least one of valid_indices or invalid_indices.")

        lazy_trk = nib.streamlines.load(tractogram, lazy_load=True)
        nb_streamlines = lazy_trk.header['nb_streamlines']

        if valid_indices is not None:
            if valid_indices.max() >= nb_streamlines or valid_indices.min() < 0:
                raise ValueError(f"Valid streamline index out of bounds: max index {valid_indices.max()}, "
                                f"number of streamlines in tractogram: {nb_streamlines}")

        if invalid_indices is not None:
            if invalid_indices.max() >= nb_streamlines or invalid_indices.min() < 0:
                raise ValueError(f"Invalid streamline index out of bounds: max index {invalid_indices.max()}, "
                                f"number of streamlines in tractogram: {nb_streamlines}")

    with LoadingThread(f" - Loading all streamlines..."):
        sft = load_tractogram(tractogram, reference='same', bbox_valid_check=False)

    if valid_indices is not None:
        with LoadingThread(f" - Fetching valid streamlines..."):
            valid_streamlines = sft.streamlines[valid_indices]
            valid_sft = sft.from_sft(valid_streamlines, sft)

        with LoadingThread(" - Saving valid streamlines..."):
            save_tractogram(valid_sft, ARGS.valid_output, bbox_valid_check=False)

    
    if invalid_indices is not None:
        with LoadingThread(f" - Fetching invalid streamlines..."):
            invalid_streamlines = sft.streamlines[invalid_indices]
            invalid_sft = sft.from_sft(invalid_streamlines, sft)

        with LoadingThread(" - Saving invalid streamlines..."):
            save_tractogram(invalid_sft, ARGS.invalid_output, bbox_valid_check=False)

def print_header():
    print("=========================================================")
    print("Gathering Verifyber results")
    print(" ↳ tractogram: {}".format(ARGS.tractogram))
    if ARGS.valid_streamlines:
        print(" ↳ valid streamlines: {}".format(ARGS.valid_streamlines))
        print(" ↳ valid output: {}".format(ARGS.valid_output))
    if ARGS.invalid_streamlines:
        print(" ↳ invalid streamlines: {}".format(ARGS.invalid_streamlines))
        print(" ↳ invalid output: {}".format(ARGS.invalid_output))
    print("=========================================================")

def print_footer():
    print("Post-processing verification completed successfully.")
    print("=========================================================") 


def main(args):
    print_header()
    valid_indexes = load_indexes(args.valid_streamlines)
    invalid_indexes = load_indexes(args.invalid_streamlines)
    fetch_and_save_streamlines(args.tractogram, valid_indexes, invalid_indexes)
    print_footer()


if __name__ == "__main__":
    main(ARGS)