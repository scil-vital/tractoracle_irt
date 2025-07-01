from tractoracle_irt.filterers.filterer import Filterer
from dipy.io.stateful_tractogram import StatefulTractogram

import argparse
import tempfile
import numpy as np
import subprocess
import nibabel as nib
import nextflow

from dipy.io.streamline import load_tractogram
from dipy.io.streamline import save_tractogram
import os
from pathlib import Path
from typing import Union

from tractoracle_irt.filterers.nextflow import build_pipeline_command
from tractoracle_irt.utils.logging import get_logger
from tractoracle_irt.utils.utils import get_project_root_dir, is_running_on_slurm

LOGGER = get_logger(__name__)

# TODO: Add the streamline sampler.
class ExtractorFilterer(Filterer):
        
    def __init__(self, end_space="mni", keep_intermediate_steps=True, quick_registration=True, sif_img_path: str = None, pipeline_path: str = "levje/extractor_flow -r dev2023"):
        super(ExtractorFilterer, self).__init__()

        self.pipeline_command = build_pipeline_command(pipeline_path, use_docker=sif_img_path is None, img_path=sif_img_path)
        self.flow_configs = [ str(get_project_root_dir() / "configs/nextflow/extractor.config") ] # TODO
        
        self.keep_intermediate_steps = keep_intermediate_steps
        self.quick_registration = quick_registration
        self.end_space = end_space
        if end_space == "mni":
            self.space_directory = "mni_space"
        elif end_space == "orig":
            self.space_directory = "orig_space"
        else:
            raise ValueError(f"Space {end_space} is not supported.")

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
            "quick_registration": str(self.quick_registration).lower(),
            "orig": str(self.ends_up_in_orig_space).lower(), 
            "keep_intermediate_steps": str(self.keep_intermediate_steps).lower()
        }
        
        results_dir = self._run_pipeline(params, out_dir)
        valid_paths, invalid_paths, subject_ids = self._get_valid_invalid_paths(results_dir)
        self._set_all_tractograms_scores(valid_paths, invalid_paths)

        return valid_paths, invalid_paths, subject_ids
    
    def _run_pipeline(self, params, run_path):
        for execution in nextflow.run_and_poll(sleep=5,
                    pipeline_path=self.pipeline_command,
                    run_path=run_path,
                    configs=self.flow_configs,
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
