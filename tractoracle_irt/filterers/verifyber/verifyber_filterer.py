from tractoracle_irt.filterers.filterer import Filterer

import os
from pathlib import Path

from tractoracle_irt.filterers.verifyber.config_generator import VerifyberConfigGenerator
from tractoracle_irt.filterers.verifyber.runners import (VerifyberApptainerRunner, VerifyberDockerRunner)
from tractoracle_irt.filterers.verifyber.post_processor import VerifyberPostProcessor
from tractoracle_irt.utils.logging import get_logger

LOGGER = get_logger(__name__)

DOCKER_IMAGE = "mrzarfir/verifyber:latest"

# TODO: Add the streamline sampler.
class VerifyberFilterer(Filterer):
    def __init__(self, sif_img_path: str = None):
        super(VerifyberFilterer, self).__init__()

        self.config_generator = VerifyberConfigGenerator(
            t1_mandatory=True, fa_mandatory=False, return_trk=False,
            model="sdec_extractor")
        
        if sif_img_path is not None:
            self.runner = VerifyberApptainerRunner(
                sif_img_path=sif_img_path, use_slurm_tmpdir=True)
        else:
            self.runner = VerifyberDockerRunner(docker_image=DOCKER_IMAGE)


        self.post_processor = VerifyberPostProcessor()

    @property
    def ends_up_in_orig_space(self):
        return True

    def _filter(self, tractogram, out_dir, scored_extension="trk"):
        pass

    def __call__(self, in_directory, tractograms, out_dir):
        # Make sure the input directory exists and is structured correctly.
        assert os.path.exists(in_directory), f"In directory does not exist: {in_directory}"
        assert os.path.exists(out_dir), f"Output directory does not exist: {out_dir}"
        assert verify_root_structure(in_directory, requires_t1w=self.config_generator.t1_mandatory)

        # Generate the configuration for the Verifyber pipeline.
        configs_paths, subjects = self.config_generator.generate_configs(
            in_directory, os.path.join(out_dir, "configs"), trk_glob="*.trk",
            t1_glob="*_t1.nii.gz", fa_glob="*_fa.nii.gz")
        
        if not configs_paths:
            raise ValueError("No valid configurations generated. Please check the input directory and the glob patterns.")
        
        results_dir = os.path.join(out_dir, "results")
        for config_path, subject in zip(configs_paths, subjects):
            self.runner.run(
                config_path=config_path,
                output_dir=os.path.join(results_dir, subject)
            )
        
        valid_paths, invalid_paths, subject_ids = self.post_processor(Path(in_directory), Path(results_dir))

        return valid_paths, invalid_paths, subject_ids

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
    LOGGER.info("Verifying root structure...")
    for subject in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, subject)
        if os.path.isdir(subject_path):
            trk_files = list(Path(subject_path).glob("*.trk"))
            nii_files = list(Path(subject_path).glob("*_t1.nii.gz"))
            if requires_t1w and not nii_files:
                LOGGER.warning(f"Warning: Subject {subject} is missing the T1w file.")
                return False
            if not trk_files:
                LOGGER.warning(f"Warning: Subject {subject} is missing a *.trk file.")
                return False
        else:
            LOGGER.warning(f"Warning: {subject_path} is not a directory.")
            return False
    LOGGER.info("Root structure is valid.")
    return True
