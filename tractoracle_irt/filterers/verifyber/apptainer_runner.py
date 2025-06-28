import os
import json
import subprocess
import tempfile
from tractoracle_irt.utils.utils import is_running_on_slurm
from tractoracle_irt.utils.logging import get_logger

LOGGER = get_logger(__name__)

class GetTempDir():
    def __init__(self, use_slurm_tmpdir=False):
        self.using_slurm = False
        if use_slurm_tmpdir and is_running_on_slurm():
            SLURM_TMPDIR = os.environ.get("SLURM_TMPDIR")
            if SLURM_TMPDIR:
                self.using_slurm = True
                self.temp_dir = os.path.join(SLURM_TMPDIR, "tmp_verifyber")
                os.makedirs(self.temp_dir, exist_ok=True)  # Create a temporary directory within SLURM_TMPDIR
            else:
                LOGGER.warning("SLURM_TMPDIR is not set. Please ensure you are running this on a SLURM cluster with a valid temporary directory.")
                self.temp_dir = tempfile.TemporaryDirectory()
        else:
            self.temp_dir = tempfile.TemporaryDirectory()

    def __enter__(self):
        if self.using_slurm:
            return self.temp_dir
        else:
            return self.temp_dir.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.using_slurm:
            self.temp_dir.cleanup()


class VerifyberApptainerRunner():
    def __init__(self, apptainer_image_path, use_slurm_tmpdir=True):
        self.apptainer_image_path = apptainer_image_path
        self.use_slurm_tmpdir = use_slurm_tmpdir
        if not os.path.exists(self.apptainer_image_path):
            raise FileNotFoundError(f"Apptainer image not found: {self.apptainer_image_path}")

    def run(self, config_path, output_dir):
        trk_file, t1_file, fa_file = self._parse_config(config_path)

        with GetTempDir(use_slurm_tmpdir=self.use_slurm_tmpdir) as tmp_dir:
            LOGGER.info("==========================================")
            LOGGER.info(f"Running Verifyber with config: {config_path}")
            LOGGER.info(f" ↳ Container: {self.apptainer_image_path}")
            LOGGER.info(f" ↳ Config file: {config_path}")
            LOGGER.info(f" ↳ Output directory: {output_dir}")
            LOGGER.info(f" ↳ TRK file: {trk_file}")
            LOGGER.info(f" ↳ T1 file: {t1_file}")
            LOGGER.info(f" ↳ FA file: {fa_file}")
            LOGGER.info(f" ↳ tmp: {tmp_dir}")
            LOGGER.info("==========================================")

            command = self._build_command(config_path, output_dir, tmp_dir)
            os.makedirs(output_dir, exist_ok=True)

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                LOGGER.error(f"Error running Verifyber: {e}")
                raise RuntimeError(f"Failed to run Verifyber with command: {' '.join(command)}") from e

            LOGGER.info("==========================================")
            LOGGER.info("Done. Results are in:")
            LOGGER.info(f" ↳ {output_dir}")
            LOGGER.info("==========================================")

    def _build_command(self, config_path, output_dir, tmp_dir):
        # Prepare the command to run the Apptainer container.
        command = [
            "apptainer", "run", "--nv",
            "--bind", f"{output_dir}:/app/output",
            "--bind", f"{tmp_dir}:/app/verifyber_tmp",  # Required so the container can write temporary files.
            self.apptainer_image_path,
            "-config", config_path,
        ]
        return command

    def _parse_config(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

        trk_file = config.get("trk")
        t1_file = config.get("t1", "")
        fa_file = config.get("fa", "")

        if not trk_file:
            raise ValueError("TRK file is mandatory in the configuration.")
        
        if not os.path.exists(trk_file):
            raise FileNotFoundError(f"TRK file not found: {trk_file}")

        if t1_file and not os.path.exists(t1_file):
            raise FileNotFoundError(f"T1 file not found: {t1_file}")

        if fa_file and not os.path.exists(fa_file):
            raise FileNotFoundError(f"FA file not found: {fa_file}")
        
        if not t1_file and not fa_file:
            raise ValueError("At least one of T1 or FA files must be provided in the configuration.")

        return trk_file, t1_file, fa_file