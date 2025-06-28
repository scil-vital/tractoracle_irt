import numpy as np
import nibabel as nib

from dipy.io.streamline import load_tractogram
from dipy.io.streamline import save_tractogram
from pathlib import Path
from glob import glob

from tractoracle_irt.utils.logging import get_logger
from tractoracle_irt.utils.utils import LoadingThread

LOGGER = get_logger(__name__)

class VerifyberPostProcessor:
    def __init__(self):
        pass

    def __call__(self, in_directory, results_dir):
        valid_paths, invalid_paths, subject_ids = [], [], []

        if not isinstance(in_directory, Path):
            in_directory = Path(in_directory)
        if not isinstance(results_dir, Path):
            results_dir = Path(results_dir)

        for subject_dir in results_dir.iterdir():
            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name
            subject_ids.append(subject_id)

            # Fetch the tractogram file for the subject.
            tractogram_glob = str(in_directory / subject_id / "*.trk")
            tractogram_files = glob(tractogram_glob)
            if not tractogram_files:
                raise ValueError(f"No tractogram files found in {tractogram_glob}. Please check the directory structure.")
            elif len(tractogram_files) > 1:
                raise ValueError(f"More than one tractogram file found in {tractogram_glob}. Please ensure only one tractogram per subject.")
            tractogram = tractogram_files[0] if tractogram_files else None

            # Define paths for valid and invalid streamlines and their outputs.
            valid_streamlines = subject_dir / "idxs_plausible.txt"
            invalid_streamlines = subject_dir / "idxs_non-plausible.txt"
            valid_output = subject_dir / "valid.trk"
            invalid_output = subject_dir / "invalid.trk"
            
            # Start processing
            self.print_header(subject_id, tractogram, valid_streamlines, valid_output, invalid_streamlines, invalid_output)

            valid_indexes = self.load_indexes(valid_streamlines)
            invalid_indexes = self.load_indexes(invalid_streamlines)

            valid_path, invalid_path = self.fetch_and_save_streamlines(tractogram, valid_indexes, invalid_indexes, valid_output, invalid_output)
            valid_paths.append(valid_path)
            invalid_paths.append(invalid_path)

            self.print_footer(subject_id)

        return valid_paths, invalid_paths, subject_ids

    def load_indexes(self, file_path: str) -> np.ndarray:
        """
        Load streamline indices from a text file.
        Each line in the file should contain a single index.
        """
        if file_path is None:
            return None

        with open(file_path, 'r') as f:
            indexes = np.array([int(line.strip()) for line in f if line.strip().isdigit()])

        assert indexes.ndim == 1, "Indexes should be a 1D array."
        assert len(indexes) > 0, "No valid indexes found in the file."
        assert np.all(indexes >= 0), "All indexes should be non-negative."

        return indexes

    def fetch_and_save_streamlines(self, tractogram, valid_indices: np.ndarray = None,
                                   invalid_indices: np.ndarray = None,
                                   valid_output: str = None, invalid_output: str = None):
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

            with LoadingThread(" - Assigning scores to valid streamlines..."):
                valid_sft.data_per_streamline['score'] = np.ones(len(valid_sft.streamlines), dtype=np.float32)

            with LoadingThread(" - Saving valid streamlines..."):
                save_tractogram(valid_sft, str(valid_output), bbox_valid_check=False)


        if invalid_indices is not None:
            with LoadingThread(f" - Fetching invalid streamlines..."):
                invalid_streamlines = sft.streamlines[invalid_indices]
                invalid_sft = sft.from_sft(invalid_streamlines, sft)

            with LoadingThread(" - Assigning scores to invalid streamlines..."):
                invalid_sft.data_per_streamline['score'] = np.zeros(len(invalid_sft.streamlines), dtype=np.float32)

            with LoadingThread(" - Saving invalid streamlines..."):
                save_tractogram(invalid_sft, str(invalid_output), bbox_valid_check=False)
        
        return str(valid_output), str(invalid_output)

    def print_header(self, subject, tractogram, valid_streamlines, valid_output, invalid_streamlines, invalid_output):
        print("=========================================================")
        print(f"Gathering Verifyber results for {subject}")
        print(" ↳ tractogram: {}".format(tractogram))
        print(" ↳ valid streamlines: {}".format(valid_streamlines))
        print(" ↳ valid output: {}".format(valid_output))
        print(" ↳ invalid streamlines: {}".format(invalid_streamlines))
        print(" ↳ invalid output: {}".format(invalid_output))
        print("=========================================================")

    def print_footer(self, subject):
        print(f"Post-processing verification completed successfully for {subject}.")
        print("=========================================================")