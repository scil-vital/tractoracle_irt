import os
import json
import argparse
import glob
from pathlib import Path

from tractoracle_irt.utils.utils import ArgparseFullFormatter
from tractoracle_irt.utils.logging import get_logger, add_logging_args, setup_logging

LOGGER = get_logger(__name__)

LONG_DESCRIPTION = """
This script creates a configuration file in order to train a
tracking model (possibly RL-based tracking agent) on your own
dataset. This serves as a convenience script to simply generate
a JSON file that contains the paths to the FODF, peaks, tracking masks,
seeding images, anatomical images, and gray matter masks for each subject
in the dataset. Once the configuration file is generated, you can use it
to build the full HDF5 dataset using the
`tractoracle_irt.datasets.create_tracking_dataset.py` script.

The script assumes that the dataset is organized in a
specific way, where your subjects should already be categorized into
trainset, validset and  testset directories and each subject has their
own corresponding directory containing all the required files. The required
files are the following:
- FODF image
- Peaks image
- Tracking mask
- Seeding image
- Anatomical image
- Fractional anisotropy (FA) image
- Gray matter mask

By default, the script will assume that your dataset is organized in a
predefined way, but you can customize the glob patterns and subdirectory names
using the appropriate command line arguments. By default, the structure
of the dataset is as follows:
```
example_directory/
├── trainset
│   ├── 100610
│   │   ├── fodfs/
│   │   │   ├── 100610__fodf.nii.gz
│   │   │   └── 100610__peaks.nii.gz
│   │   ├── masks/
│   │   │   ├── 100610__mask_gm.nii.gz
│   │   │   └── 100610__mask_wm.nii.gz
│   │   ├── maps/
│   │   │   └── 100610__interface.nii.gz
│   │   ├── dti/
│   │   │   └── 100610__fa.nii.gz
│   │   └── anat/
│   │       └── 100610_T1.nii.gz
│   └── 100711
│       ├── fodfs/
│       ...
├── validset
│   ├── 136126
│   │   ├── fodfs/
│   │   │   ├── 136126__fodf.nii.gz
│   │   │   └── 136126__peaks.nii.gz
│   │   ├── masks/
│   │   │   ├── 136126__mask_gm.nii.gz
│   │   │   └── 136126__mask_wm.nii.gz
│   │   ├── maps/
│   │   │   └── 136126__interface.nii.gz
│   │   ├── dti/
│   │   │   └── 136126__fa.nii.gz
│   │   └── anat/
│   │       └── 136126_T1.nii.gz
│   └── 136227
│       ├── fodfs/
│       ...
└── testset
    ├── 139435
    │   ├── fodfs/
    │   │   ├── 139435__fodf.nii.gz
    │   │   └── 139435__peaks.nii.gz
    │   ├── masks/
    │   │   ├── 139435__mask_gm.nii.gz
    │   │   └── 139435__mask_wm.nii.gz
    │   ├── maps/
    │   │   └── 139435__interface.nii.gz
    │   ├── dti/
    │   │   └── 139435__fa.nii.gz
    │   └── anat/
    │       └── 139435_T1.nii.gz
    └── 140117
        ├── fodfs/
        ...
```

The script will generate a JSON file with the following structure:
```
{
    "training": {
        "100610": {
            "inputs": ["example_directory/trainset/100610/fodfs/100610__fodf.nii.gz"],
            "peaks": "example_directory/trainset/100610/fodfs/100610__peaks.nii.gz",
            "tracking": "example_directory/trainset/100610/masks/100610__mask_wm.nii.gz",
            "seeding": "example_directory/trainset/100610/maps/100610__interface.nii.gz",
            "fa": "example_directory/trainset/100610/dti/100610__fa.nii.gz",
            "anat": "example_directory/trainset/100610/anat/100610_T1.nii.gz",
            "gm": "example_directory/trainset/100610/masks/100610__mask_gm.nii.gz"
        },
        "100711": {...}
    },
    "validation": {
        "136126": {
            "inputs": ["example_directory/validset/136126/fodfs/136126__fodf.nii.gz"],
            "peaks": "example_directory/validset/136126/fodfs/136126__peaks.nii.gz",
            "tracking": "example_directory/validset/136126/masks/136126__mask_wm.nii.gz",
            "seeding": "example_directory/validset/136126/maps/136126__interface.nii.gz",
            "fa": "example_directory/validset/136126/dti/136126__fa.nii.gz",
            "anat": "example_directory/validset/136126/anat/136126_T1.nii.gz",
            "gm": "example_directory/validset/136126/masks/136126__mask_gm.nii.gz"
        },
        "136227": {...}
    },
    "testing": {
        "139435": {
            "inputs": ["example_directory/testset/139435/fodfs/139435__fodf.nii.gz"],
            "peaks": "example_directory/testset/139435/fodfs/139435__peaks.nii.gz",
            "tracking": "example_directory/testset/139435/masks/139435__mask_wm.nii.gz",
            "seeding": "example_directory/testset/139435/maps/139435__interface.nii.gz",
            "fa": "example_directory/testset/139435/dti/139435__fa.nii.gz",
            "anat": "example_directory/testset/139435/anat/139435_T1.nii.gz",
            "gm": "example_directory/testset/139435/masks/139435__mask_gm.nii.gz"
        },
        "140117": {...}
    }
```
"""

def parse_args():
    parser = argparse.ArgumentParser(description=LONG_DESCRIPTION, formatter_class=ArgparseFullFormatter)
    parser.add_argument("base_dir", type=str, help="Base directory for HCP datasets")
    parser.add_argument("output_file", type=str, default="tracking_config.json", help="Output JSON file name")
    parser.add_argument("--train_dirname", type=str, default="trainset", help="Directory name for training set. This should be the path from the base directory.")
    parser.add_argument("--valid_dirname", type=str, default="validset", help="Directory name for validation set. This should be the path from the base directory.")
    parser.add_argument("--test_dirname", type=str, default="testset", help="Directory name for test set. This should be the path from the base directory.")

    parser.add_argument("--fodf_glob", type=str, default="*__fodf.nii.gz", help="Glob pattern for FODF images")
    parser.add_argument("--peaks_glob", type=str, default="*__peaks.nii.gz", help="Glob pattern for peaks images")
    parser.add_argument("--tracking_glob", type=str, default="*__mask_wm.nii.gz", help="Glob pattern for tracking masks")
    parser.add_argument("--seeding_glob", type=str, default="*__interface.nii.gz", help="Glob pattern for seeding images")
    parser.add_argument("--fa_glob", type=str, default="*__fa.nii.gz", help="Glob pattern for fractional anisotropy images")
    parser.add_argument("--anat_glob", type=str, default="*_T1.nii.gz", help="Glob pattern for anatomical images")
    parser.add_argument("--gm_glob", type=str, default="*__mask_gm.nii.gz", help="Glob pattern for gray matter masks")

    parser.add_argument("--fodfs_subdir", type=str, default="fodfs", help="Subdirectory where the FODF images are stored within each subject directory.")
    parser.add_argument("--peaks_subdir", type=str, default="fodfs", help="Subdirectory where the peaks images are stored within each subject directory.")
    parser.add_argument("--tracking_subdir", type=str, default="masks", help="Subdirectory where the tracking masks are stored within each subject directory.")
    parser.add_argument("--seeding_subdir", type=str, default="maps", help="Subdirectory where the seeding images are stored within each subject directory.")
    parser.add_argument("--fa_subdir", type=str, default="dti", help="Subdirectory where the fractional anisotropy images are stored within each subject directory.")
    parser.add_argument("--anat_subdir", type=str, default="anat", help="Subdirectory where the anatomical images are stored within each subject directory.")
    parser.add_argument("--gm_subdir", type=str, default="masks", help="Subdirectory where the gray matter masks are stored within each subject directory.")

    add_logging_args(parser)
    return parser.parse_args()

def get_full_path(subject_path, subdir, pattern):
    """Constructs the full file path based on the subject path and ID."""
    if isinstance(subject_path, str):
        subject_path = Path(subject_path)
    return str(subject_path / subdir / pattern)

def resolve_globs(pattern, get_single=False):
    """Expand user and system glob patterns into absolute paths."""
    LOGGER.debug("Resolving glob pattern:", pattern)
    expanded_paths = []
    user_resolved = os.path.expanduser(pattern)
    for path in glob.glob(user_resolved):
        # Resolve the path to an absolute path
        absolute_path = os.path.abspath(path)

        # Need to make sure the path is a file and exists
        if os.path.exists(absolute_path) and os.path.isfile(absolute_path):
            expanded_paths.append(absolute_path)
    
    if get_single:
        if len(expanded_paths) != 1:
            raise ValueError(f"Expected a single path for pattern '{pattern}', but found {len(expanded_paths)}.")
        return expanded_paths[0]
    else:
        return expanded_paths


def get_subject_data(subject_path, dataset, fodf_glob, peaks_glob,
                     tracking_glob, seeding_glob, fa_glob,
                     anat_glob, gm_glob,
                     fodfs_subdir, peaks_subdir,
                     tracking_subdir, seeding_subdir, fa_subdir,
                     anat_subdir, gm_subdir):
    """Extracts file paths for a given subject."""
    subject_id = subject_path.name
    LOGGER.info(f"Processing subject: {subject_id} ({dataset})")
    subject_data = {
        "inputs": [ subject_path / fodfs_subdir / fodf_glob],
        "peaks":    subject_path / peaks_subdir / peaks_glob,
        "tracking": subject_path / tracking_subdir / tracking_glob,
        "seeding":  subject_path / seeding_subdir / seeding_glob,
        "fa":       subject_path / fa_subdir / fa_glob,
        "anat":     subject_path / anat_subdir / anat_glob,
        "gm":       subject_path / gm_subdir / gm_glob,
    }

    # Resolve globs to absolute paths
    for key, value in subject_data.items():
        if isinstance(value, list):
            subject_data[key] = [resolve_globs(v, get_single=True) for v in value]
        else:
            subject_data[key] = resolve_globs(value, get_single=True)

    return subject_data

def get_all_subjects(input_dir):
    subjects = []
    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            subjects.append(dir_name)
        break # We only want the top-level directories
    return sorted(subjects)

def main(args):
    setup_logging(args)

    BASE_DIR = args.base_dir
   
    TRAIN_DATASET = "training"
    VALID_DATASET = "validation"
    TEST_DATASET = "testing"

    DATASETS_DIRNAME = {
        TRAIN_DATASET: args.train_dirname,
        VALID_DATASET: args.valid_dirname,
        TEST_DATASET: args.test_dirname
    }

    config = {}
    for dataset, dataset_dirname in DATASETS_DIRNAME.items():
        dataset_path = Path(BASE_DIR) / dataset_dirname
        if not dataset_path.exists() or not dataset_path.is_dir():
            LOGGER.error(f"The directory {dataset_path} does not exist or is not a directory.")
            continue

        # Get all subjects in the dataset directory
        subjects = get_all_subjects(dataset_path)

        # Create a dictionary for the dataset
        config[dataset] = {}
        for sub in subjects:
            subject_path = dataset_path / sub
            config[dataset][str(sub)] = \
                get_subject_data(subject_path, dataset,
                                 args.fodf_glob, args.peaks_glob,
                                 args.tracking_glob, args.seeding_glob,
                                 args.fa_glob, args.anat_glob, args.gm_glob,
                                 args.fodfs_subdir, args.peaks_subdir,
                                 args.tracking_subdir, args.seeding_subdir,
                                 args.fa_subdir, args.anat_subdir,
                                 args.gm_subdir)


    with open(args.output_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"JSON file saved at {args.output_file}")


if __name__ == "__main__":
    main(parse_args())
