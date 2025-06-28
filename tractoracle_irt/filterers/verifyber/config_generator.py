import os
import json
from glob import glob

from tractoracle_irt.utils.logging import get_logger

LOGGER = get_logger(__name__)

class VerifyberConfigGenerator():
    def __init__(self, t1_mandatory=True, fa_mandatory=False, return_trk=True, model="sdec_extractor"):
        if not (t1_mandatory or fa_mandatory):
            raise ValueError("At least one of T1 or FA must be mandatory.")
        
        self.t1_mandatory = t1_mandatory
        self.fa_mandatory = fa_mandatory
        self.return_trk = return_trk
        self.model = model

    def generate_configs(self, tracking_dir, output_dir,
                        trk_subdir=None, t1_subdir=None, fa_subdir=None,
                        trk_glob="*.trk", t1_glob="*_t1.nii.gz", fa_glob="*_fa.nii.gz"):
        configs_paths = []
        subjects = []
        tracking_dir = os.path.abspath(tracking_dir)
        output_dir = os.path.abspath(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        for subdir in os.listdir(tracking_dir):
            sub_path = os.path.join(tracking_dir, subdir)

            if not os.path.isdir(sub_path):
                LOGGER.warning(f"Skipping {sub_path}, not a directory.")
                continue
            
            # Fetch TRK files.
            if trk_subdir is not None:
                trk_files = glob(os.path.join(sub_path, trk_subdir, trk_glob))
            else:
                trk_files = glob(os.path.join(sub_path, trk_glob))
            if not trk_files:
                LOGGER.warning(f"No tractograms found in {sub_path} with glob {trk_glob}.")
                continue
            elif len(trk_files) > 1:
                LOGGER.warning(f"More than one tractogram found in {sub_path} with glob {trk_glob}. Using the first one.")
            trk_path = os.path.abspath(trk_files[0])

            # Fetch T1-weighted files.
            if t1_subdir is not None:
                t1_files = glob(os.path.join(sub_path, t1_subdir, t1_glob))
            else:
                t1_files = glob(os.path.join(sub_path, t1_glob))
            if not t1_files and self.t1_mandatory:
                LOGGER.warning(f"No T1-weighted images found in {sub_path} with glob {t1_glob}.")
                continue
            elif len(t1_files) > 1:
                LOGGER.warning(f"More than one T1-weighted image found in {sub_path} with glob {t1_glob}. Using the first one.")
            t1_path = os.path.abspath(t1_files[0]) if t1_files else ""

            # Fetch FA files.
            if fa_subdir is not None:
                fa_files = glob(os.path.join(sub_path, fa_subdir, fa_glob))
            else:
                fa_files = glob(os.path.join(sub_path, fa_glob))
            if not fa_files and self.fa_mandatory:
                LOGGER.warning(f"No FA images found in {sub_path} with glob {fa_glob}.")
                continue
            elif len(fa_files) > 1:
                LOGGER.warning(f"More than one FA image found in {sub_path} with glob {fa_glob}. Using the first one.")
            fa_path = os.path.abspath(fa_files[0]) if fa_files else ""

            config = {
                "trk": trk_path,
                "t1": t1_path,
                "fa": fa_path,
                "resample_points": True,
                "task": "classification",
                "return_trk": self.return_trk,
                "warp": "fast",
                "model": self.model
            }

            subject_dir = subdir
            config_filename = f'{subject_dir}_config.json'
            config_path = os.path.join(output_dir, config_filename)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            LOGGER.info(f"Generated config saved to {config_path}")
            configs_paths.append(config_path)
            subjects.append(subdir)

        return configs_paths, subjects