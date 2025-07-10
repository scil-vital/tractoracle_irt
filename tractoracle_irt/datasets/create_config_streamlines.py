import os
import json
import argparse
import glob
import sys
from argparse import RawTextHelpFormatter

from tractoracle_irt.utils.logging import get_logger, add_logging_args, setup_logging
from tractoracle_irt.utils.utils import ArgparseFullFormatter

LOGGER = get_logger(__name__)

LONG_DESCRIPTION = """
This script creates a configuration file in order to pretrain the oracle
to predict your own streamlines. The output of this script is a JSON
file that should be inputted to the
`tractoracle_irt.datasets.create_dataset_streamlines.py` script.

Your streamlines should all be located within a root directory. They should
already be split into train, validation and test sets, corresponding to the
directories trainset, validset and testset respectively. Each subject should
contain at least one recognized, one unrecognized and one reference image file.
By default, the script will look for files matching the following glob patterns
within each subject directory (at the top level):
- Reference image: `*__fa.nii.gz`
- Recognized streamlines: `*__recognized.trk`
- Unrecognized streamlines: `*__unrecognized.trk`

example_directory/
├── trainset
│   ├── 100610
│   │   ├── 100610__fa.nii.gz
│   │   ├── 100610__recognized.trk
│   │   └── 100610__unrecognized.trk
│   └── 101006
│       ├── 101006__fa.nii.gz
│       ├── 101006__recognized.trk
│       └── 101006__unrecognized.trk
├── validset
│   ├── 136126
│   │   ├── 136126__fa.nii.gz
│   │   ├── 136126__recognized.trk
│   │   └── 136126__unrecognized.trk
│   └── 136227
│       ├── 136227__fa.nii.gz
│       ├── 136227__recognized.trk
│       └── 136227__unrecognized.trk
└── testset
    ├── 139435
    │   ├── 139435__fa.nii.gz
    │   ├── 139435__recognized.trk
    │   └── 139435__unrecognized.trk
    └── 140117
        ├── 140117__fa.nii.gz
        ├── 140117__recognized.trk
        └── 140117__unrecognized.trk

If your image files are named differently, you can specify the glob patterns
using the `--reference_glob`, `--recognized_glob` and `--unrecognized_glob`
arguments.

If your files are not located directly in the subject directories but
in nested directories within each subject's directory, you can specify
the nested directories name using the `--nested_directories` argument.
This will make the script look for the streamlines and reference images
in the specified nested directories in addition to the subject's directory.

For example, if your directory structure looks like the following, the script
should be supplied with the `--nested_directories streamlines` argument, pointing
to the directory where the streamlines are located:
example_directory/
├── trainset
│   ├── 100610
│   │   ├── 100610__fa.nii.gz
│   │   └── streamlines
│   │       ├── 100610__recognized.trk
│   │       └── 100610__unrecognized.trk
│   └── 101006
│       ├── 101006__fa.nii.gz
│       └── streamlines
│           ├── 101006__recognized.trk
│           └── 101006__unrecognized.trk
├── validset
│   ├── 136126
│   │   ├── 136126__fa.nii.gz
│   │   └── streamlines
│   │       ├── 136126__recognized.trk
│   │       └── 136126__unrecognized.trk
│   └── 136227
│       ├── 136227__fa.nii.gz
│       └── streamlines
│           ├── 136227__recognized.trk
│           └── 136227__unrecognized.trk
└── testset
    ├── 139435
    │   ├── 139435__fa.nii.gz
    │   └── streamlines
    │       ├── 139435__recognized.trk
    │       └── 139435__unrecognized.trk
    └── 140117
        ├── 140117__fa.nii.gz
        └── streamlines
            ├── 140117__recognized.trk
            └── 140117__unrecognized.trk

Multiple nested directories can be specified by separating them with spaces.
"""

def parse_args():
    parser = argparse.ArgumentParser(description=LONG_DESCRIPTION, formatter_class=ArgparseFullFormatter)
    parser.add_argument("base_dir", type=str, help="Base directory for streamlines datasets")
    parser.add_argument("output_file", type=str, default="tracts_config.json", help="Output JSON file name")
    parser.add_argument("--train_dirname", type=str, default="trainset", help="Directory name for training set. This should be the path from the base directory.")
    parser.add_argument("--valid_dirname", type=str, default="validset", help="Directory name for validation set. This should be the path from the base directory.")
    parser.add_argument("--test_dirname", type=str, default="testset", help="Directory name for test set. This should be the path from the base directory.")
    parser.add_argument("--reference_glob", type=str, default="*__fa.nii.gz", help="Glob pattern for reference image")
    parser.add_argument("--recognized_glob", type=str, default="*__recognized.trk", help="Glob pattern for recognized streamlines")
    parser.add_argument("--unrecognized_glob", type=str, default="*__unrecognized.trk", help="Glob pattern for unrecognized streamlines")
    parser.add_argument("--nested_directories", type=str, default=None, nargs='*', help="From the subjects directory, the nested directories to look for streamlines or the reference image. If not provided, it will look in the base directory. If the streamlines are in deeper levels of the directory structure, provide the path like 'subdir1/subdir2/subdir3.")
    add_logging_args(parser)

    args = parser.parse_args()
    return args

def get_all_subjects(input_dir):
    subjects = []
    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            subjects.append(dir_name)
        break # We only want the top-level directories
    return sorted(subjects)

def resolve_globs(pattern):
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
    
    return expanded_paths

def find_streamline_paths(base_dir, subject, reference_glob, recognized_glob, unrecognized_glob, nested_directories=None):
    """Find recognized and unrecognized streamline paths for a subject."""

    if nested_directories is None:
        nested_directories = []
    
    recognized_paths = []
    unrecognized_paths = []
    reference_paths = []
    
    for directory in nested_directories:
        recognized_paths.extend(resolve_globs(os.path.join(base_dir, subject, directory, recognized_glob)))
        unrecognized_paths.extend(resolve_globs(os.path.join(base_dir, subject, directory, unrecognized_glob)))
        reference_paths.extend(resolve_globs(os.path.join(base_dir, subject, directory, reference_glob)))
    
    # Also search in the base directory for any files that might not be in nested directories
    recognized_paths.extend(resolve_globs(os.path.join(base_dir, subject, recognized_glob)))
    unrecognized_paths.extend(resolve_globs(os.path.join(base_dir, subject, unrecognized_glob)))
    reference_paths.extend(resolve_globs(os.path.join(base_dir, subject, reference_glob)))

    # Ensure we have at least one recognized path and one unrecognized path
    if not recognized_paths:
        raise ValueError(f"No recognized streamlines found for subject {subject} in {base_dir}")
    if not unrecognized_paths:
        raise ValueError(f"No unrecognized streamlines found for subject {subject} in {base_dir}")
    
    if not reference_paths:
        raise ValueError(f"No reference image found for subject {subject} in {base_dir}")
    elif len(reference_paths) > 1:
        LOGGER.warning(f"Multiple reference paths found for subject {subject}. Using the first one: {reference_paths[0]}")
    reference_path = reference_paths[0]

    return recognized_paths, unrecognized_paths, reference_path

def main():
    args = parse_args()
    setup_logging(args)

    BASE_DIR = args.base_dir
   
    TRAIN_DATASET = "train"
    VALID_DATASET = "valid"
    TEST_DATASET = "test"

    DATASETS_DIRNAME = {
        TRAIN_DATASET: args.train_dirname,
        VALID_DATASET: args.valid_dirname,
        TEST_DATASET: args.test_dirname
    }

    config = {}

    # Function to generate paths based on the subject ID
    def generate_subject_paths(dataset_dirname, subject):
        DATASET_BASE_DIR = os.path.join(BASE_DIR, dataset_dirname)
        pos_streamlines, neg_streamlines, reference_path = find_streamline_paths(
            DATASET_BASE_DIR, subject, args.reference_glob, args.recognized_glob, args.unrecognized_glob, args.nested_directories
        )
        sub_config = {
            "pos_streamlines": pos_streamlines,
            "neg_streamlines": neg_streamlines,
            "reference": reference_path
        }

        assert os.path.exists(sub_config["pos_streamlines"][0]), f"Positive streamlines not found: {sub_config['pos_streamlines'][0]}"
        assert os.path.exists(sub_config["neg_streamlines"][0]), f"Negative streamlines not found: {sub_config['neg_streamlines'][0]}"
        assert os.path.exists(sub_config["reference"]), f"Reference file not found: {sub_config['reference']}"

        return sub_config

    for dataset_name, dataset_dirname in DATASETS_DIRNAME.items():
        dataset_path = os.path.join(BASE_DIR, dataset_dirname)
        subjects = get_all_subjects(dataset_path)
        LOGGER.debug("Found subjects:", subjects)

        # Make sure we have subjects in the dataset.
        if not subjects:
            raise ValueError(f"No subjects found in {dataset_path}. Please check the directory structure.")

        LOGGER.info(f"Found {len(subjects)} subjects for '{dataset_name}' in {dataset_path}")

        config[dataset_name] = {subject: generate_subject_paths(dataset_dirname, subject) for subject in subjects}

    # Save to JSON file
    with open(args.output_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Configuration file saved as {args.output_file}")

if __name__ == "__main__":
    main()
