import os
import json
from pathlib import Path
import argparse

def get_subject_data(subject_path):
    """Extracts file paths for a given subject."""
    subject_id = subject_path.name
    return {
        "inputs": [str(subject_path / f"fodfs/{subject_id}__fodf.nii.gz")],
        "peaks": str(subject_path / f"fodfs/{subject_id}__peaks.nii.gz"),
        "tracking": str(subject_path / f"masks/{subject_id}__mask_wm.nii.gz"),
        "seeding": str(subject_path / f"maps/{subject_id}__interface.nii.gz"),
        "anat": str(subject_path / f"anat/{subject_id}_T1.nii.gz"),
        "gm": str(subject_path / f"masks/{subject_id}__mask_gm.nii.gz"),
    }

def categorize_subjects(subjects):
    """Categorizes subjects into training, validation, and testing."""
    subjects = sorted(subjects)  # Sort for consistency
    split_1 = int(len(subjects) * 0.6)  # 60% training
    split_2 = int(len(subjects) * 0.8)  # 20% validation, 20% testing
    return {
        "training": {str(sub): get_subject_data(sub) for sub in subjects[:split_1]},
        "validation": {str(sub): get_subject_data(sub) for sub in subjects[split_1:split_2]},
        "testing": {str(sub): get_subject_data(sub) for sub in subjects[split_2:]},
    }

def generate_json(directory):
    """Generates a JSON file from the subject directory structure."""
    base_path = Path(directory)
    subjects = [p for p in base_path.iterdir() if p.is_dir() and p.name.startswith("sub-")]
    data = categorize_subjects(subjects)
    
    output_file = base_path / "dataset.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON file saved at {output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate a JSON file for the dataset.")
    parser.add_argument("dataset_directory", help="Path to the dataset directory.")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_directory):
        raise ValueError(f"The provided dataset directory '{args.dataset_directory}' does not exist.")
    
    generate_json(args.dataset_directory)
