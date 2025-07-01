import os

def check_file_exists(file_path, can_be_none=False):
    """
    Check if the file exists and raise an error if it doesn't.
    """
    if can_be_none and file_path is None:
        return file_path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    return file_path

def is_false(value):
    return isinstance(value, bool) and value is False

def get_dataset_type(dataset_path):
    """Determine dataset type based on file extension."""
    if dataset_path.endswith(".tar.gz") or dataset_path.endswith(".tar"):
        return "tar"
    elif dataset_path.endswith(".hdf5"):
        return "hdf5"
    else:
        raise ValueError(f"Unknown dataset format: {dataset_path}")
