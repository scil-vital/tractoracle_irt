import os
import time
import tarfile
from pathlib import Path

from tractoracle_irt.utils.utils import get_unique_experiment_name
from tractoracle_irt.utils.logging import get_logger

LOGGER = get_logger(__name__)

class Backuper(object):
    """
    This allows to save the experiment's path files into a compressed
    archive in a specific directory.
    
    This is useful for running long jobs (that might get shutdown on
    HPC).
    """
    def __init__(self, exp_path, exp, exp_name, backup_dir: str = None,
                 min_interval: int = 60, remove_old_backups=True, start_at_step=200,
                 each_n_steps=100):
        self.exp_path = Path(exp_path)
        if not self.exp_path.exists() or not self.exp_path.is_dir():
            raise FileNotFoundError("The provided experiment directory"
                                    "doesn't exist or is not a directory "
                                    "({})".format(str(self.exp_path)))
        
        self.backup_dir = Path(backup_dir) if backup_dir else None
        if self.backup_dir \
            and (not self.backup_dir.exists()\
            or not self.backup_dir.is_dir()):
            raise FileNotFoundError("The provided backup directory doesn't "
                                    "exist or is not a directory ({})".format(
                                        str(self.backup_dir)))
        
        self.start_at_step = start_at_step
        self.min_interval = min_interval # In seconds.
        self.each_n_steps = each_n_steps
        self.last_timestamp = 0 # Used to make sure we don't save periodically too often.

        self.stamp = get_unique_experiment_name(exp, exp_name)
        self.previous_backup_path: Path = None
        self.previous_step = 0
        self.ext = ".tar.gz"
        self.remove_old_backups = remove_old_backups

        if self.backup_dir is not None:
            LOGGER.info("Backup directory specified. Backups will be saved at "
                        "{}".format(self.backup_dir))

    def backup(self, step: int = None):

        if self.backup_dir is None \
            or step < self.start_at_step \
            or step <= self.previous_step + self.each_n_steps:
            return ""
        
        # Check the interval since last backup
        # to make sure we're not saving too often.
        interval_since_last_save = time.time() - self.last_timestamp
        if interval_since_last_save < self.min_interval:
            LOGGER.warning("The interval between two backups should be at "
                           "least {:.2f}s (not {:.2f}s). Skipping.".format(
                               self.min_interval, interval_since_last_save))
            return ""


        # Save new file
        step_str = str(step) if step else ""
        new_filename = '-'.join([self.stamp, step_str]) + self.ext
        new_backup_file = self.backup_dir / new_filename

        self._archive_experiment(new_backup_file)

        # Delete old file
        if self.remove_old_backups and self.previous_backup_path is not None:
            if self.previous_backup_path.exists():
                os.remove(self.previous_backup_path)
            else:
                LOGGER.error("Couldn't remove old backup (file not found: "
                             "{})".format(str(self.previous_backup_path)))

        self.previous_backup_path = new_backup_file
        self.previous_step = step
        return str(new_backup_file)
    
    def disable(self):
        self.backup_dir = None
        LOGGER.debug("Backups disabled.")
        
    def _archive_experiment(self, out_file):
        LOGGER.info("Archiving new backup.")
        with tarfile.open(out_file, "w:gz") as tar:
            tar.add(self.exp_path, arcname=self.exp_path.stem)
        LOGGER.info("Experiment backup saved at {}".format(out_file))

    def to_dict(self):
        return {
            'exp_path': str(self.exp_path),
            'backup_dir': str(self.backup_dir),
            'min_interval': self.min_interval,
            'remove_old_backups': self.remove_old_backups,
            'start_at_step': self.start_at_step,
            'each_n_steps': self.each_n_steps
        }