import os
import nextflow

from dipy.io.streamline import load_tractogram
from dipy.io.streamline import save_tractogram
import os
from pathlib import Path
import numpy as np

from tractoracle_irt.filterers.filterer import Filterer
from tractoracle_irt.filterers.nextflow import build_pipeline_command
from tractoracle_irt.utils.logging import get_logger

LOGGER = get_logger(__name__)

# TODO: Add the streamline sampler.
class ExtractorFilterer(Filterer):
        
    def __init__(self,
                 templates_dir: str,
                 use_apptainer: bool = False,
                 end_space: str = "orig",
                 keep_intermediate_steps: bool = False,
                 quick_registration: bool = True,
                 pipeline_path: str = "levje/nf-extractor -r main"):
        super(ExtractorFilterer, self).__init__()
        self.templates_dir = templates_dir
        if self.templates_dir is None:
            raise ValueError("Templates directory must be specified.")
        elif not os.path.exists(self.templates_dir):
            raise ValueError(f"Templates directory does not exist: {self.templates_dir}")

        self.pipeline_command = build_pipeline_command(pipeline_path,
                                                       use_apptainer)

        self.profiles = []
        if use_apptainer:
            self.profiles.append('apptainer')
        else:
            self.profiles.append('docker')
        

        self.keep_intermediate_steps = keep_intermediate_steps
        self.quick_registration = quick_registration
        self.end_space = end_space
        if end_space == "mni":
            self.space_directory = "mni_space"
        elif end_space == "orig":
            self.space_directory = "orig_space"
        else:
            raise ValueError(f"Space {end_space} is not supported.")
        
        self.extra_params = {
            "quick_registration": verify_param_type_as_str(quick_registration, bool),
            "keep_intermediate_steps": verify_param_type_as_str(keep_intermediate_steps, bool),
            "orig": verify_param_type_as_str(self.ends_up_in_orig_space, bool)
        }

    @property
    def ends_up_in_orig_space(self):
        return self.end_space == "orig"

    def _filter(self, tractogram, out_dir, scored_extension="trk"):
        pass

    def __call__(self, in_directory, tractograms, out_dir):

        # TODO: We need to copy the T1w file to the in_directory corresponding to the subject.
        # TODO: And, to improve speed, we should manually register the tractograms.
        #       Maybe the transformation matrices should be provided as instead in the HDF5.
        #       We should have a method within the RLHF class that uses the environment to
        #       register the tractograms since the environment holds either the T1w or the
        #       transformation matrices.
        #
        # TODO: Add a check to see if the T1w file is in the in_directory, otherwise raise an error.
        assert os.path.exists(in_directory), f"In directory does not exist: {in_directory}"
        assert os.path.exists(out_dir), f"Output directory does not exist: {out_dir}"
        assert verify_root_structure(in_directory, requires_t1w=self.ends_up_in_orig_space)
        params = {
            "input": in_directory,
            "templates_dir": self.templates_dir,
            **self.extra_params
        }
        
        results_dir = self._run_pipeline(params, out_dir)
        valid_paths, invalid_paths, subject_ids = self._get_valid_invalid_paths(results_dir)
        self._set_all_tractograms_scores(valid_paths, invalid_paths)

        return valid_paths, invalid_paths, subject_ids
    
    def _run_pipeline(self, params, run_path):
        for execution in nextflow.run_and_poll(sleep=5,
                    pipeline_path=self.pipeline_command,
                    run_path=run_path,
                    params=params,
                    profiles=self.profiles):
            LOGGER.info("Running Extractor pipeline. ")
            LOGGER.info(execution.stdout)
        
        if execution.return_code == '0':
            LOGGER.info("Extractor pipeline executed successfully. "
                        "Duration {}.".format(execution.duration))
        else:
            LOGGER.error(execution.stdout)
            LOGGER.error(execution.stderr)
            raise ValueError("Extractor pipeline failed to execute "
                             "successfully. Duration {} ".format(execution.duration) +
                             "Return code: {}.".format(execution.return_code))
        
        # Results are exported 
        results_dir = Path(run_path) / "results_extractorflow" / "final_outputs"
        assert results_dir.exists(), f"Results directory {results_dir} does not exist."
        LOGGER.debug(f"Extractor pipeline results are in {results_dir}.")

        return str(results_dir)

    def _get_valid_invalid_paths(self, results_dir: str):
        """
        Extractor-flow organizes the results in the results_dir the following way:
        results_dir/ (i.e. final_outputs/)
        ├── <subid_1>/
        │   └── orig_space/
        |       ├── ...
        │       ├── <subid_1>__plausible*_orig_space.trk
        │       └── <subid_1>__unplausible*_orig_space.trk
        │   └── mni_space/
        |       ├── ...
        │       ├── <subid_1>__plausible*_mni_space.trk
        │       └── <subid_1>__unplausible*_mni_space.trk
        ├── <subid_2>/
        │   └── orig_space/
        |       ├── ...
        │       ├── <subid_2>__plausible*_orig_space.trk
        │       └── <subid_2>__unplausible*_orig_space.trk
        │   └── mni_space/
        |       ├── ...
        │       ├── <subid_2>__plausible*_mni_space.trk
        │       └── <subid_2>__unplausible*_mni_space.trk
        ├── ...
    
        This function returns the paths to the plausible/unplausible tractograms for each subject
        and makes sure they exist.
        """

        valid = []
        invalid = []
        subject_ids = []

        for subject_dir in Path(results_dir).iterdir():
            if not subject_dir.is_dir():
                continue

            results_space_dir = subject_dir / self.space_directory
            if not results_space_dir.exists():
                LOGGER.warning(f"Subject directory {results_space_dir} does not exist.")
                continue

            plausible_pattern = f"{subject_dir.name}__*plausible*_{self.space_directory}.trk"
            plausible = get_single_file_glob(results_space_dir, plausible_pattern, exclude="unplausible")
            if plausible is None:
                LOGGER.error(f"No plausible tractograms found for {subject_dir.name} in {plausible_pattern}.")
            else:
                valid.append(str(plausible))

            unplausible_pattern = f"{subject_dir.name}__*unplausible*_{self.space_directory}.trk"
            unplausible = get_single_file_glob(results_space_dir, unplausible_pattern)
            if unplausible is None:
                LOGGER.error(f"No unplausible tractograms found for {subject_dir.name} in {unplausible_pattern}.")
            else:
                invalid.append(str(unplausible))

            subject_ids.append(subject_dir.name)

        return valid, invalid, subject_ids
    
    def _set_all_tractograms_scores(self, valids, invalids):
        for tractogram in valids:
            self._set_tractograms_scores(tractogram, score=1)

        for tractogram in invalids:
            self._set_tractograms_scores(tractogram, score=0)

        return valids, invalids

    def _set_tractograms_scores(self, tractogram_file, score):
        assert score == 1 or score == 0

        # Load the tractogram
        tractogram = load_tractogram(tractogram_file, "same", bbox_valid_check=False)

        nb_streamlines = len(tractogram.streamlines)
        scores = np.ones(nb_streamlines) if score == 1 else np.zeros(nb_streamlines)
        tractogram.data_per_streamline['score'] = scores

        save_tractogram(tractogram, tractogram_file, bbox_valid_check=False)

def get_single_file_glob(path, pattern, exclude=None):
    # This function takes in a path and a pattern and finds a single file matching the pattern.
    if not isinstance(path, (Path)):
        path = Path(path)

    files = path.glob(pattern)

    if exclude is not None:
        files = [f for f in files if exclude not in f.name]
    else:
        files = list(files)

    if len(files) == 1:
        return files[0]
    elif len(files) == 0:
        return None
    else:
        raise ValueError(f"Multiple files found for pattern {pattern} in {path}.")

def verify_root_structure(root_dir, requires_t1w=False):
    # We need to make sure that the root_dir as the following structure:
    # root_dir
    # ├── subject1
    # │   ├── *.trk
    # │   └── *_t1.nii.gz
    # ├── subject2
    # │   ├── *.trk
    # │   └── *_t1.nii.gz
    # ├── ...
    # └── subjectN
    #     ├── *.trk
    #     └── *_t1.nii.gz
    print("Verifying root structure...")
    for subject in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, subject)
        if os.path.isdir(subject_path):
            trk_files = list(Path(subject_path).glob("*.trk"))
            nii_files = list(Path(subject_path).glob("*_t1.nii.gz"))
            if requires_t1w and not nii_files:
                print(f"Warning: Subject {subject} is missing the T1w file.")
                return False
            if not trk_files:
                print(f"Warning: Subject {subject} is missing a *.trk file.")
                return False
        else:
            print(f"Warning: {subject_path} is not a directory.")
            return False
    print("Root structure is valid.")
    return True

def verify_param_type_as_str(value, exp_type):
    if not isinstance(value, exp_type):
        raise TypeError(f"Expected {exp_type} but got {type(value)} for value: {value}")
    return str(value).lower()
