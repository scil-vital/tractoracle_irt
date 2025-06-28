#!/usr/bin/env python
import comet_ml  # noqa: F401 ugh
import torch

from tractoracle_irt.trainers.cross_q_train import (
    parse_args,
    CrossQTraining)
from tractoracle_irt.utils.torch_utils import get_device, assert_accelerator
device = get_device()
assert_accelerator()


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)
    from comet_ml import Optimizer

    # We only need to specify the algorithm and hyperparameters to use:
    config = {
        # We pick the Bayes algorithm:
        "algorithm": "grid",

        # Declare your hyperparameters in the Vizier-inspired format:
        "parameters": {
            "lr": {
                "type": "discrete",
                "values": [1e-5, 5e-5, 1e-4, 5e-4, 5e-3, 1e-3]},
            "gamma": {
                "type": "discrete",
                "values": [0.99, 0.95, 0.90]},
        },

        # Declare what we will be optimizing, and how:
        "spec": {
            "metric": "VC",
            "objective": "maximize",
            "seed": args.rng_seed,
            "retryAssignLimit": 3,
        },
    }

    # Next, create an optimizer, passing in the config:
    opt = Optimizer(config, project_name=args.experiment)

    for experiment in opt.get_experiments():
        experiment.auto_metric_logging = False
        experiment.workspace = args.workspace
        experiment.parse_args = False
        experiment.disabled = not args.use_comet

        lr = experiment.get_parameter("lr")
        gamma = experiment.get_parameter("gamma")

        arguments = vars(args)
        arguments.update({
            'lr': lr,
            'gamma': gamma
        })

        sac_experiment = CrossQTraining(
            arguments,
            experiment
        )
        sac_experiment.run()


if __name__ == '__main__':
    main()
