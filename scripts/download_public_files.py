import argparse

parser = argparse.ArgumentParser(description="Utility script to download one of the project's public files.")
parser.add_argument(
    "files",
    type=str,
    nargs="+",
    help="The names of the public files to download. They should be specified in the public_files.yml file.",
)
parser.add_argument("--remove_zip", action="store_true", help="Remove the zip file after downloading. Default: False")

args = parser.parse_args()

from tractoracle_irt.utils.config.public_files import PUBLIC_FILES, download_file_data
from tractoracle_irt.utils.logging import get_logger

LOGGER = get_logger(__name__)

if __name__ == "__main__":
    LOGGER.info("Downloading public files: %s", args.files)
    for file_name in args.files:
        if file_name not in PUBLIC_FILES:
            raise ValueError(f"File {file_name} is not a valid public file. Available files: {list(PUBLIC_FILES.keys())}")
        file_data = PUBLIC_FILES[file_name]
        download_file_data(file_data, remove_archive=args.remove_zip)
        LOGGER.info("Downloaded %s to %s", file_data.name, file_data.path)