import numpy as np

import os
from os.path import join as pjoin
from comet_ml import Experiment

from tractoracle_irt.utils.logging import get_logger
from tractoracle_irt.utils.utils import LossHistory

LOGGER = get_logger(__name__)

def _get_prefix_with_delim(prefix):
    delims = ['_', '-', '/']

    if prefix[-1] not in delims:
        out_prefix = f"{prefix}/"
    elif out_prefix is None:
        out_prefix = ""
    else:
        out_prefix = prefix

    return out_prefix

def _force_prefix_to_slash(prefix):
    """ Force the prefix to end with a slash. """
    if prefix is None or len(prefix) == 0:
        LOGGER.warning("Prefix is None or empty.")
        return None

    # First remove any trailing delimiters
    delims = ['_', '-', '/']
    for delim in delims:
        if prefix[-1] == delim:
            prefix = prefix[:-1]
    # Then add a trailing slash
    if prefix[-1] != '/':
        return f"{prefix}/"
    else:
        return prefix


class CometMonitor():
    """ Wrapper class to track information using Comet.ml
    """

    def __init__(
        self,
        experiment: Experiment,
        experiment_path: str,
        prefix: str,
        render: bool = False,
        use_comet: bool = False,
        offline: bool = False
    ):
        """
        Parameters:
        -----------
            experiment: str
                Name of experiment. Will contain many interations
                of experiment based on different parameters
            experiment_path: str
                Experiment path used to fetch images or other stuff
            prefix: str
                Prefix for metrics
            use_comet: bool
                Whether to actually use comet or not. Useful when
                Comet access is limited
        """
        # IMPORTANT
        # This presumes that your API key is in your home folder or at
        # the project root.
        self.experiment_path = experiment_path
        self.e = experiment

        self.offline_monitor = OfflineMonitor(
            experiment_path=experiment_path,
            prefix=prefix,
            enabled=offline
        )

        self.prefix = _get_prefix_with_delim(prefix)
        self.render = render
        self.use_comet = use_comet

    def log_parameters(self, hyperparameters: dict):
        print("Logging hyperparameters to Comet...")
        if not self.use_comet:
            return

        self.e.log_parameters(hyperparameters)

    def update(
        self,
        reward_monitor,
        len_monitor,
        vc_monitor=None,
        ic_monitor=None,
        nc_monitor=None,
        vb_monitor=None,
        ib_monitor=None,
        ol_monitor=None,
        i_episode=0
    ):
        if not self.use_comet:
            return

        reward_x, reward_y = zip(*reward_monitor.epochs)
        len_x, len_y = zip(*len_monitor.epochs)

        self.e.log_metrics(
            {
                self.prefix + "Reward": reward_y[-1],
                self.prefix + "Length": len_y[-1],
            },
            step=i_episode
        )

        if vc_monitor is not None and len(vc_monitor) > 0:
            vc_x, vc_y = zip(*vc_monitor.epochs)
            nc_x, nc_y = zip(*nc_monitor.epochs)
            ic_x, ic_y = zip(*ic_monitor.epochs)
            vb_x, vb_y = zip(*vb_monitor.epochs)
            ib_x, ib_y = zip(*ib_monitor.epochs)
            ol_x, ol_y = zip(*ol_monitor.epochs)

            self.e.log_metrics(
                {
                    self.prefix + "VC": vc_y[-1],
                    self.prefix + "NC": nc_y[-1],
                    self.prefix + "IC": ic_y[-1],
                    self.prefix + "VB": vb_y[-1],
                    self.prefix + "IB": ib_y[-1],
                    self.prefix + "OL": ol_y[-1],
                },
                step=i_episode
            )

        if self.render:
            self.e.log_image(
                pjoin(self.experiment_path, 'render',
                      '{}.png'.format(i_episode)),
                step=i_episode)

    def log_losses(self, loss_dict, i):
        if not self.use_comet:
            return

        for k, v in loss_dict.items():
            if type(v) is np.ndarray:
                self.e.log_histogram_3d(v, name=self.prefix + k, step=i)
            else:
                self.e.log_metric(self.prefix + k, v, step=i)

        self.offline_monitor.log_metrics(loss_dict, step=None, epoch=i)

    def update_train(
        self,
        monitor,
        i_episode,
    ):
        if not self.use_comet:
            return

        x, y = zip(*monitor.epochs)

        metrics = {
            self.prefix + monitor.name: y[-1],
        }

        self.e.log_metrics(
            metrics,
            step=i_episode
        )

        self.offline_monitor.log_metrics(metrics, step=i_episode)


class OracleMonitor(object):

    def __init__(
        self,
        experiment: Experiment,
        experiment_path: str,
        use_comet: bool = False,
        metrics_prefix: str = None,
        offline: bool = False
    ):
        self.experiment = experiment

        # This monitor will automatically log each metric to a
        # file in the experiment path if we're running in offline mode.
        self.offline_monitor = OfflineMonitor(
            experiment_path=experiment_path,
            prefix=metrics_prefix,
            log_each_step=True,  # Log each step for Oracle training
            enabled=offline
        )

        self.metrics_prefix = _get_prefix_with_delim(metrics_prefix) if metrics_prefix else None

        self.use_comet = use_comet
        if not self.use_comet:
            LOGGER.warning(
                "Comet is not being used. No metrics will be logged for the "
                "Oracle training.")

    def log_parameters(self, hyperparameters: dict):
        if not self.use_comet:
            return
        
        if self.metrics_prefix:
            prefix = self.metrics_prefix
        else:
            prefix = None
        
        self.experiment.log_parameters(hyperparameters, prefix=prefix)

    def log_metrics(self, metrics_dict, step: int, epoch: int):
        if not self.use_comet:
            return

        for k, v in metrics_dict.items():
            assert isinstance(v, (int, float, np.int64, np.float64,
                              np.float32, np.int32)), "Metrics must be numerical."
            
            if self.metrics_prefix:
                k = f"{self.metrics_prefix}{k}"

            self.experiment.log_metric(k, v, step=step, epoch=epoch)

        self.offline_monitor.log_metrics(metrics_dict, step, epoch)


class OfflineMonitor(object):
    """ Monitor that automatically logs corresponding metrics to a file if enabled.
    Using LossHistory.
    """

    def __init__(self, experiment_path: str, prefix: str, log_each_step=True, enabled: bool = True):
        self.monitors = {}
        self.enabled = enabled
        self.log_each_step = log_each_step

        experiment_path = experiment_path
        prefix = _force_prefix_to_slash(prefix)

        self.path = pjoin(experiment_path, "offline_plots", prefix)
        if self.enabled:
            os.makedirs(self.path, exist_ok=True)
            print(f"Offline monitor initialized at: {self.path}")
            

    def log_metrics(self, metrics_dict, step: int, epoch: int):
        """ Log metrics to the monitors.
        
        Parameters:
        -----------
            metrics_dict: dict
                Dictionary of metrics to log.
            step: int
                Current step in training.
            epoch: int
                Current epoch in training.
        """
        if not self.enabled:
            return

        for k, v in metrics_dict.items():
            if k not in self.monitors:
                name=k
                filename=k
                self.monitors[k] = LossHistory(name, filename, self.path,
                                               log_each_step=self.log_each_step,
                                               handle_out_dir=False)
                
            # Make sure that the value is not a list or array
            if isinstance(v, (list, np.ndarray)):
                # TODO
                raise NotImplementedError("Logging lists or arrays is not supported yet.")
            else:
                self.monitors[k].update(v, step, epoch)
                self.monitors[k].end_epoch(epoch)
