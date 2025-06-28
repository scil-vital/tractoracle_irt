#!/usr/bin/env python
import comet_ml  # noqa: F401 ugh
import torch
import os

from tractoracle_irt.trainers.rlhf_train import (
    RlhfTraining,
    parse_args,
    get_trainer_cls_and_args)
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
        "algorithm": "bayes",

        # Declare your hyperparameters in the Vizier-inspired format:
        "parameters": {
            "lr": {
                "type": "discrete",
                "values": [1e-5]
            },
            "init_critic_to_oracle": {
                "type": "discrete",
                "values": [0]
            },
            "disable_oracle_training": {
                "type": "discrete",
                "values": [1]
            },
            "kl_penalty_coeff": {
                "type": "discrete",
                "values": [0.0]
            },
        },

        # Declare what we will be optimizing, and how:
        "spec": {
            "metric": "VC",
            "objective": "maximize",
            "seed": args.rng_seed,
            "retryLimit": 3,
            "retryAssignLimit": 3,
        },
    }

    # Next, create an optimizer, passing in the config:
    opt = Optimizer(config)

    for experiment in opt.get_experiments(project_name=args.experiment):
        experiment.auto_metric_logging = False
        experiment.workspace = args.workspace
        experiment.parse_args = False
        experiment.disabled = not args.use_comet

        lr = experiment.get_parameter("lr")
        is_critic_init_to_oracle = True if experiment.get_parameter("init_critic_to_oracle") > 0 else None
        is_oracle_training_disabled = True if experiment.get_parameter("disable_oracle_training") > 0 else None

        arguments = vars(args)
        path_suffix = f"lr_{str(lr).replace('.', '')}_Crit2Oracle_{is_critic_init_to_oracle}_OracleTrain_{not is_oracle_training_disabled}"
        arguments.update({
            'path': os.path.join(args.path, path_suffix),
            'lr': lr,
            'init_critic_to_oracle': is_critic_init_to_oracle,
            'disable_oracle_training': is_oracle_training_disabled,
        })

        trainer_cls = get_trainer_cls_and_args(args.alg)
        sac_experiment = RlhfTraining(
            arguments,
            trainer_cls
            # experiment
        )
        sac_experiment.run()


if __name__ == '__main__':
    main()
