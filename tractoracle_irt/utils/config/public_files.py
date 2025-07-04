import os
import yaml
import requests
from dataclasses import dataclass
from tqdm import tqdm
import zipfile

from typing import Union
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

from tractoracle_irt.utils.utils import get_project_root_dir
from tractoracle_irt.utils.logging import get_logger

LOGGER = get_logger(__name__)

CONFIG_FILE = get_project_root_dir() / "configs/defaults/public_files.yml"
OUTPUT_DIR = get_project_root_dir() / ".data" / "public"

class FileData:
    class FileDataType:
        FILE = "file"
        DIR = "dir"

    def __init__(self, url: Union[list, str], name, out_type):
        self.urls = url if isinstance(url, list) else [url]
        self.name = name
        self.out_type = out_type

        # If we want to write to a directory, make sure that the name doesn't have an extension.
        if self.out_type == self.FileDataType.DIR and '.' in self.name:
            raise ValueError(f"Invalid name for directory: {self.name}. Directories should not have an extension.") 
    
        self.path = str(OUTPUT_DIR / self.name)

        if self.out_type == self.FileDataType.DIR:
            os.makedirs(self.path, exist_ok=True)

    def __repr__(self):
        return f"FileData(name={self.name}, urls={self.urls}, out_type={self.out_type}, path={self.path})"

    def __str__(self):
        return f"FileData(name={self.name}, urls={self.urls}, out_type={self.out_type}, path={self.path})"

def load_public_files():
    raw_config = yaml.safe_load(CONFIG_FILE.read_text())
    public_files = {}
    for file_name, file_info in raw_config.items():
        if not isinstance(file_info, dict):
            raise ValueError(f"Invalid format for {file_name}: expected a dictionary, got {type(file_info)}")
        if 'url' not in file_info or 'name' not in file_info:
            raise ValueError(f"Missing 'url' or 'name' in {file_name}")
        
        file_data = FileData(url=file_info['url'], name=file_info['name'], out_type=file_info.get('type', FileData.FileDataType.FILE))
        public_files[file_name] = file_data

    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return public_files


# Make sure that we do not have duplicate entries (by name)
def check_duplicate_files(public_files):
    count = {}
    for file_data in public_files.values():
        count[file_data.name] = count.get(file_data.name, 0) + 1

    if any(c > 1 for c in count.values()):
        duplicates = [name for name, c in count.items() if c > 1]
        raise ValueError(f"Duplicate file names found in PUBLIC_FILES: {duplicates}")

#############################################
# PUBLIC_FILES global variable
# 
# This variable is loaded from the public_files.yml file and contains
# the URLs and names of public files that should be downloaded.
# It is used to ensure that the files are available for the experiments.
# The files are downloaded to the .data/public directory.
#############################################
PUBLIC_FILES = load_public_files()
check_duplicate_files(PUBLIC_FILES)
#############################################

def is_public_file(path: str) -> bool:
    """
    Check if the given path is a public file.
    A public file is defined as a file that starts with "public://".
    """
    return path.startswith("public://")

def download_if_public_file(path: str) -> str:
    if is_public_file(path):
        # Remove the "public://" prefix
        path = path[9:]
        public_file_data = PUBLIC_FILES.get(path, None)
        if public_file_data is not None:
            download_file_data(public_file_data)
            path = public_file_data.path # Update the path to the downloaded file
    return path

def download_file_data(file_data: FileData):
    if not isinstance(file_data, FileData):
        raise TypeError(f"Expected FileData, got {type(file_data)}")
    
    for url in file_data.urls:
        download_file(url, file_data.path, except_on_error=True)
    return True

def download_file(url, path, skip_if_exists=True, except_on_error=True):
    # If path is a directory, add the filename from the URL
    if os.path.isdir(path):
        path_dir = path
        os.makedirs(path, exist_ok=True)
        url_filename = os.path.basename(urlparse(url).path)
        path = os.path.join(path, url_filename)
    else:
        path_dir = os.path.dirname(path)

    if skip_if_exists and os.path.exists(path):
        LOGGER.debug(f"File {path} already exists. Skipping download.")
        return True

    filename = os.path.basename(path)

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024

            with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading {}".format(filename), leave=True) as t:
                with open(path, 'wb') as f:
                    for data in r.iter_content(block_size):
                        if data:
                            t.update(len(data))
                            f.write(data)

    except Exception as e:
        if except_on_error:
            raise e
        else:
            print(f"Error downloading {filename}: {e}")
        return False

    if path.endswith('.zip'):
        uncompress_files_into_directory(path, output_dir=path_dir)

    return True

def uncompress_files_into_directory(*zip_files, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    for zip_file in zip_files:
        if not zip_file.endswith('.zip'):
            raise ValueError(f"Expected a .zip file, got {zip_file}")
        
        zip_path = os.path.join(output_dir, zip_file)
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip file {zip_path} does not exist.")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    return str(output_dir)
