#!/usr/bin/env python
import argparse
import comet_ml  # noqa: F401 ugh
import torch

from argparse import RawTextHelpFormatter
from comet_ml import Experiment as CometExperiment

from tractoracle_irt.algorithms.sac import SAC, SACHParams
from tractoracle_irt.trainers.train import Training, add_training_args
from tractoracle_irt.utils.torch_utils import get_device, assert_accelerator
from argparse import RawTextHelpFormatter

device = get_device()
assert_accelerator()

class SACTraining(Training):
    """
    Train a RL tracking agent using SAC.
    """

    def __init__(
        self,
        sac_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        sac_train_dto: dict
            SAC training parameters
        comet_experiment: CometExperiment
            Allows for logging and experiment management.
        """

        super().__init__(
            sac_train_dto,
            comet_experiment,
        )

    @property
    def hparams_class(self):
        return SACHParams

    def get_alg(self, max_nb_steps: int, neighborhood_manager):
        alg = SAC(
            self.input_size,
            self.action_size,
            self.hp,
            self.rng,
            device)
        return alg


def add_sac_args(parser):
    parser.add_argument('--alpha', default=0.2, type=float,
                        help='Initial temperature parameter')
    parser.add_argument('--batch_size', default=2**12, type=int,
                        help='How many tuples to sample from the replay '
                        'buffer.')
    parser.add_argument('--replay_size', default=1e6, type=int,
                        help='How many tuples to store in the replay buffer.')
    parser.add_argument('--utd', default=1, type=int,
                        help='Update to data ratio. How many times to update '
                        'the model per data sample.')
    parser.add_argument('--save_replay_buffer', default=False, action='store_true',
                        help='Save the replay buffer within the checkpoint.')


def parse_args():
    """ Generate a tractogram from a trained model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_training_args(parser)
    add_sac_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """

    args = parse_args()
    print(args)

    experiment = CometExperiment(project_name=args.experiment,
                                 workspace=args.workspace, parse_args=False,
                                 auto_metric_logging=False,
                                 disabled=not args.use_comet)

    sac_experiment = SACTraining(
        vars(args),
        experiment
    )
    sac_experiment.run()


if __name__ == '__main__':
    main()
