from tractoracle_irt.filterers.filterer import Filterer

import numpy as np
import nibabel as nib
import nextflow
import json

from dipy.io.streamline import load_tractogram
from dipy.io.streamline import save_tractogram
import os
from pathlib import Path

from tractoracle_irt.utils.logging import get_logger
from tractoracle_irt.filterers.nextflow import build_pipeline_command

LOGGER = get_logger(__name__)

RESULTS_INDICES_DIRNAME = "COMPILE_REPORTS"
RESULTS_INDICES_FILENAME = ""

# TODO: Add the streamline sampler.
class RbxFilterer(Filterer):

    def __init__(self, atlas_directory: str, use_apptainer: bool = False, pipeline_path: str = "levje/nf-rbx -r main"):
        super(RbxFilterer, self).__init__()

        self.pipeline_command = build_pipeline_command(pipeline_path,
                                                       use_apptainer)
        self.profiles = ['essential', 'compile_reports']
        if use_apptainer:
            self.profiles.append('apptainer')
        else:
            self.profiles.append('docker')

        self.flow_configs = []
        self.atlas_directory = atlas_directory
        
        if atlas_directory is None:
            raise ValueError("Atlas directory must be provided.")
        elif not os.path.exists(atlas_directory):
            raise ValueError(f"Atlas directory {atlas_directory} does not exist.")

    @property
    def ends_up_in_orig_space(self):
        return True 

    def _filter(self, tractogram, out_dir, scored_extension="trk"):
        pass

    def __call__(self, in_directory, tractograms, out_dir):

        # TODO: We need to copy the T1w file to the in_directory corresponding to the subject.
        # TODO: And, to improve speed, we should manually register the tractograms.
        #       Maybe the transformation matrices should be provided as instead in the HDF5.
        #       We should have a method within the RLHF class that uses the environment to
        #       register the tractograms since the environment holds either the T1w or the
        #       transformation matrices.
        assert os.path.exists(in_directory), f"In directory does not exist: {in_directory}"
        assert os.path.exists(out_dir), f"Output directory does not exist: {out_dir}"
        assert verify_root_structure(in_directory)
        params = {
            "input": in_directory,
            "atlas_directory": self.atlas_directory,
        }
        subjs_nb_streamlines = self._count_nb_streamlines(in_directory)
        results_dir = self._run_pipeline(params, out_dir)
        valid_indices, invalid_indices = self._get_valid_invalid_indices(results_dir, subjs_nb_streamlines)
        valid_paths, invalid_paths, subject_ids = self._compile_results(in_directory, valid_indices, invalid_indices)
        # valid_paths, invalid_paths, subject_ids = self._get_valid_invalid_paths(results_dir)
        # self._set_all_tractograms_scores(valid_paths, invalid_paths)

        return valid_paths, invalid_paths, subject_ids
    
    def _run_pipeline(self, params, run_path):
        for execution in nextflow.run_and_poll(sleep=30,
                    pipeline_path=self.pipeline_command,
                    run_path=run_path,
                    configs=self.flow_configs,
                    params=params,
                    profiles=self.profiles):
            LOGGER.info("Running RBX pipeline...")
            # LOGGER.info(execution.stdout)
            pass
        
        if execution.return_code == '0':
            LOGGER.info("RBX flow executed successfully. "
                        "Duration {}.".format(execution.duration))
        else:
            LOGGER.error(execution.stdout)
            LOGGER.error(execution.stderr)
            raise ValueError("RBX pipeline failed to execute "
                             "successfully. Duration {} ".format(execution.duration) +
                             "Return code: {}.".format(execution.return_code))
        
        # Results are exported 
        results_dir = Path(run_path) / "results_rbx"
        assert results_dir.exists(), f"Results directory {results_dir} does not exist."
        LOGGER.debug(f"RBX pipeline results are in {results_dir}.")

        return str(results_dir)

    def _count_nb_streamlines(self, in_directory: str):
        subjs_nb_streamlines = {}
        for subject in os.listdir(in_directory):
            subject_path = Path(os.path.join(in_directory, subject))
            tractogram = next(subject_path.glob("*.trk"))
            nb_streamlines = nib.streamlines.load(tractogram, lazy_load=True).header['nb_streamlines']
            subjs_nb_streamlines[subject] = nb_streamlines
        return subjs_nb_streamlines

    def _get_valid_invalid_indices(self, results_dir: str, subjs_nb_streamlines: dict):
        """
        RBX flow organizes the results in the results_dir the following way:
        ├── <subid_1>/
        │   ├── COMPILE_REPORTS/
        |   |   ├── ...
        │   |   └── <subid_1>__report_clean.json
        │   ├── CLEAN_BUNDLES/
        │   ├── RECOGNIZE_BUNDLES/
        │   └── REGISTER_ANAT/
        ├── <subid_2>/
        │   ├── COMPILE_REPORTS/
        |   |   ├── ...
        │   |   └── <subid_2>__report_clean.json
        │   ├── CLEAN_BUNDLES/
        │   ├── RECOGNIZE_BUNDLES/
        │   └── REGISTER_ANAT/
        ├── ...

        The *__report_clean.json file contains the indices of the streamlines
        that were recognized to be associated to each bundle. Some duplicates
        are possible.
        """

        valid_indices = {} # List of np.ndarray for each subject
        invalid_indices = {} # List of np.ndarray for each subject

        for subject_dir in Path(results_dir).iterdir():
            if not subject_dir.is_dir():
                continue

            subid = subject_dir.name

            results_indices_dir = subject_dir / RESULTS_INDICES_DIRNAME
            if not results_indices_dir.exists():
                LOGGER.warning(f"Subject directory {results_indices_dir} does not exist.")
                continue

            results_indices = results_indices_dir / f"{subject_dir.name}__report_clean.json"
            if not results_indices.exists():
                LOGGER.warning(f"Results indices {results_indices} does not exist.")
                continue

            with open(results_indices, "r") as f:
                res = json.load(f)

            nb_streamlines = subjs_nb_streamlines[subid]

            # The recognized indices are spread across the bundles
            sub_recognized_indices = set()
            for bundle in res.keys():
                rec = res[bundle]
                sub_recognized_indices.update(rec)

            sub_all_indices = np.arange(nb_streamlines)
            sub_recognized_indices = np.array(list(sub_recognized_indices), dtype=int)
            sub_unrecognized_indices = np.setdiff1d(sub_all_indices, sub_recognized_indices)
            
            print("Recognized ({}): {}".format(subid, len(sub_recognized_indices)))
            print("Unrecognized ({}): {}".format(subid, len(sub_unrecognized_indices)))

            assert sub_recognized_indices.size + sub_unrecognized_indices.size == nb_streamlines

            valid_indices[subid] = sub_recognized_indices
            invalid_indices[subid] = sub_unrecognized_indices

        return valid_indices, invalid_indices
    
    def _compile_results(self, in_directory: str, valid_indices, invalid_indices):
        valid_paths = []
        invalid_paths = []
        subject_ids = []

        for subject_dir in Path(in_directory).iterdir():
            subid = subject_dir.name

            LOGGER.info(f"Compiling results for subject {subid}.")

            tractogram = next(subject_dir.glob("*.trk"))

            sft = load_tractogram(str(tractogram), "same", bbox_valid_check=False)
            
            valid_indices_subj = valid_indices[subid]
            invalid_indices_subj = invalid_indices[subid]

            valid_sft = sft[valid_indices_subj]
            invalid_sft = sft[invalid_indices_subj]

            # Set the score for each streamline
            valid_scores = np.ones(len(valid_sft.streamlines))
            invalid_scores = np.zeros(len(invalid_sft.streamlines))
            
            valid_sft.data_per_streamline['score'] = valid_scores
            invalid_sft.data_per_streamline['score'] = invalid_scores

            # Save the tractograms
            valid_path = subject_dir / f"{subid}__recognized.trk"
            save_tractogram(valid_sft, str(valid_path), bbox_valid_check=False)

            invalid_path = subject_dir / f"{subid}__unrecognized.trk"
            save_tractogram(invalid_sft, str(invalid_path), bbox_valid_check=False)

            valid_paths.append(str(valid_path))
            invalid_paths.append(str(invalid_path))
            subject_ids.append(subid)

        return valid_paths, invalid_paths, subject_ids

    def _get_valid_invalid_paths(self, results_dir: str):
        """
        Extractor-flow organizes the results in the results_dir the following way:
        results_dir/ (i.e. final_outputs/)
        ├── <subid_1>/
        │   └── mni_space/
        |       ├── ...
        │       ├── <subid_1>__plausible_mni_space.trk
        │       └── <subid_1>__unplausible_mni_space.trk
        ├── <subid_2>/
        │   └── mni_space/
        |       ├── ...
        │       ├── <subid_2>__plausible_mni_space.trk
        │       └── <subid_2>__unplausible_mni_space.trk
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

            mni_space_dir = subject_dir / self.space_directory
            if not mni_space_dir.exists():
                LOGGER.warning(f"Subject directory {mni_space_dir} does not exist.")
                continue

            plausible = mni_space_dir / f"{subject_dir.name}__plausible_{self.space_directory}.trk"
            if not plausible.exists():
                LOGGER.warning(f"Plausible tractogram {plausible} does not exist.")
            else:
                valid.append(str(plausible))

            unplausible = mni_space_dir / f"{subject_dir.name}__unplausible_{self.space_directory}.trk"
            if not unplausible.exists():
                LOGGER.warning(f"Unplausible tractogram {unplausible} does not exist.")
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

def verify_root_structure(root_dir):
    # We need to make sure that the root_dir as the following structure:
    # root_dir
    # ├── subject1
    # │   ├── *.trk
    # │   └── *_fa.nii.gz
    # ├── subject2
    # │   ├── *.trk
    # │   └── *_fa.nii.gz
    # ├── ...
    # └── subjectN
    #     ├── *.trk
    #     └── *_fa.nii.gz
    print("Verifying root structure...")
    for subject in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, subject)
        if os.path.isdir(subject_path):
            trk_files = list(Path(subject_path).glob("*.trk"))
            nii_files = list(Path(subject_path).glob("*_fa.nii.gz"))
            if not nii_files:
                print(f"Warning: Subject {subject} is missing the *_fa.nii.gz file.")
                return False
            if not trk_files:
                print(f"Warning: Subject {subject} is missing a *.trk file.")
                return False
        else:
            print(f"Warning: {subject_path} is not a directory.")
            return False
    print("Root structure is valid.")
    return True
