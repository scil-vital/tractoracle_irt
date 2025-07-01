import json
import os
import random
from os.path import join as pjoin
from pathlib import Path
import numpy as np
import torch
import time
import shutil

from tractoracle_irt.algorithms.rl import RLAlgorithm
from tractoracle_irt.algorithms.shared.utils import old_mean_losses as mean_losses, mean_rewards
from tractoracle_irt.environments.env import BaseEnv
from tractoracle_irt.experiment.experiment import (add_data_args,
                                                add_environment_args,
                                                add_experiment_args,
                                                add_model_args,
                                                add_oracle_args,
                                                add_reward_args,
                                                add_tracking_args,
                                                add_tractometer_args,
                                                add_extractor_args,
                                                add_verifyber_args,
                                                add_rbx_args)
from tractoracle_irt.experiment.oracle_validator import OracleValidator
from tractoracle_irt.experiment.tractometer_validator import TractometerValidator
from tractoracle_irt.experiment.experiment import Experiment
from tractoracle_irt.tracking.tracker import Tracker
from tractoracle_irt.utils.torch_utils import get_device, assert_accelerator
from tractoracle_irt.utils.hooks import HooksManager, RlHookEvent
from tractoracle_irt.utils.backuper import Backuper
from tractoracle_irt.utils.utils import TTLProfiler
from tractoracle_irt.algorithms.shared.hyperparameters import HParams


class Training(Experiment):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        config: dict,
        comet_experiment=None,
    ):
        """
        Parameters
        ----------
        config: dict
            Dictionnary containing the training parameters.
            Put into a dictionnary to prevent parameter errors if modified.
        """
        self.init_hyperparameters(config)

        self.comet_experiment = comet_experiment
        self.comet_experiment.set_name(self.hp.experiment_id)
        self.best_epoch_metric = -np.inf
        self._hooks_manager = HooksManager(RlHookEvent)

        # Setup validators, which will handle validation and scoring
        # of the generated streamlines
        self.validators = []
        if self.hp.tractometer_validator:
            tractometer_validator = TractometerValidator(
                self.hp.scoring_data, self.hp.tractometer_reference,
                dilate_endpoints=self.hp.tractometer_dilate,
                min_length=self.hp.min_length, max_length=self.hp.max_length,
                oracle_model=self.hp.oracle_crit_checkpoint)
            self.validators.append(tractometer_validator)
        if self.hp.oracle_validator:  # TODO: This is problematic if we call rl_train multiple times
            self.validators.append(OracleValidator(
                self.hp.oracle_crit_checkpoint, self.device))

    @property
    def hparams_class(self):
        return HParams

    def init_hyperparameters(self, config: dict):
        # Load hyperparameters
        self.hp = self.hparams_class.from_dict(config)

        self.comet_monitor_was_setup = False
        self.compute_reward = True  # Always compute reward during training
        self.fa_map = None
        self.last_episode = 0
        self.device = get_device()

        # Directories
        self.model_dir = os.path.join(self.hp.experiment_path, "model")
        self.model_saving_dirs = [self.model_dir]

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # RNG
        torch.manual_seed(self.hp.rng_seed)
        np.random.seed(self.hp.rng_seed)
        self.rng = np.random.RandomState(seed=self.hp.rng_seed)
        random.seed(self.hp.rng_seed)

        # Setup Backuper
        self.backuper = Backuper(self.hp.experiment_path, self.hp.experiment,
                                    self.hp.experiment_id, self.hp.backup_dir)
        

    def save_hyperparameters(self, filename: str = "hyperparameters.json"):
        """ Save hyperparameters to json file
        """
        # Add input and action size to hyperparameters
        # These are added here because they are not known before
        hparams_dict = self.hp.to_dict()
        hparams_dict.update(self.backuper.to_dict())
        hparams_dict.update({
            'input_size': self.input_size.to_dict(),
            'action_size': self.action_size,
            'voxel_size': str(self.voxel_size)})

        for saving_dir in self.model_saving_dirs:
            with open(
                pjoin(saving_dir, filename),
                'w'
            ) as json_file:
                json_file.write(
                    json.dumps(
                        hparams_dict,
                        indent=4,
                        separators=(',', ': ')))

    def save_model(self, alg, save_model_dir=None, is_best_model=False,
                   scores_info: dict = {}):
        """ Save the model state to disk
        """

        directory = self.model_dir if save_model_dir is None else save_model_dir
        if not os.path.exists(directory):
            os.makedirs(directory)

        if is_best_model:
            ckpt_path = os.path.join(directory, "best_model_state.ckpt")
        else:
            ckpt_path = os.path.join(directory, "last_model_state.ckpt")    
        alg.save_checkpoint(ckpt_path, **scores_info)

        return ckpt_path

    def rl_train(
        self,
        alg: RLAlgorithm,
        env: BaseEnv,
        valid_env: BaseEnv,
        max_ep: int = 1000,
        starting_ep: int = 0,
        save_model_dir: str = None,
        test_before_training: bool = True
    ):
        """ Train the RL algorithm for N epochs. An epoch here corresponds to
        running tracking on the training set until all streamlines are done.
        This loop should be algorithm-agnostic. Between epochs, report stats
        so they can be monitored during training

        Parameters:
        -----------
            alg: RLAlgorithm
                The RL algorithm, either TD3, PPO or any others
            env: BaseEnv
                The tracking environment
            valid_env: BaseEnv
                The validation tracking environment (forward).
            """

        # Current epoch
        i_episode = starting_ep
        upper_bound = i_episode + max_ep
        # Transition counter
        t = 0

        # Trigger start hooks
        self._hooks_manager.trigger_hooks(RlHookEvent.ON_RL_TRAIN_START)

        # Initialize Trackers, which will handle streamline generation and
        # trainnig
        train_tracker = Tracker(
            alg, self.hp.n_actor, prob=0.0, compress=0.0)

        valid_tracker = Tracker(
            alg, self.hp.n_actor,
            prob=1.0, compress=0.0)

        # Run tracking before training to see what an untrained network does
        if test_before_training:
            valid_env.load_subject()
            valid_tractogram, valid_reward = valid_tracker.track_and_validate(
                valid_env, enable_pbar=True)
            stopping_stats = self.stopping_stats(valid_tractogram)
            print(stopping_stats)
            if valid_tractogram:
                self.comet_monitor.log_losses(stopping_stats, i_episode)

                filename = self.save_rasmm_tractogram(valid_tractogram,
                                                    valid_env.subject_id,
                                                    valid_env.affine_vox2rasmm,
                                                    valid_env.reference,
                                                    save_dir=save_model_dir)
                scores = self.score_tractogram(filename, valid_env)
                print(scores)

                self.comet_monitor.log_losses(scores, i_episode)
            self.save_model(alg, save_model_dir)

            # Display the results of the untrained network
            self.log(
                valid_tractogram, valid_reward, i_episode)

        # Main training loop
        with TTLProfiler(out_file="profiling.prof", enabled=False) as profiler:
            while i_episode < upper_bound:
                # Train for an episode
                self._train_iter(env, train_tracker, i_episode, t)

                i_episode += 1

                # Time to do a valid run and display stats
                if i_episode % self.hp.log_interval == 0 \
                    or i_episode == upper_bound:
                    self._valid_iter(valid_env, valid_tracker, alg, i_episode, save_model_dir)
                
                # Backup to that directory after each validation run.
                # This can take a while.
                self.backuper.backup(step=i_episode)

        # Trigger end hooks
        self._hooks_manager.trigger_hooks(RlHookEvent.ON_RL_TRAIN_END)

    def _train_iter(
        self,
        env: BaseEnv,
        train_tracker: Tracker,
        i_episode: int,
        t: int
    ):
        # Last episode/epoch. Was initially for resuming experiments but
        # since they take so little time I just restart them from scratch
        # Not sure what to do with this
        self.last_episode = i_episode

        # Train for an episode
        env.load_subject()

        tractogram, losses, reward, reward_factors, mean_ratio = \
            train_tracker.track_and_train(env)

        # Compute average streamline length
        lengths = [len(s) for s in tractogram]
        avg_length = np.mean(lengths)  # Nb. of steps

        # Keep track of how many transitions were gathered
        t += sum(lengths)

        # Compute average reward per streamline
        # Should I use the mean or the sum ?
        avg_reward = reward / self.hp.n_actor

        print(
            f"Episode Num: {i_episode+1} "
            f"Avg len: {avg_length:.3f} Avg. reward: "
            f"{avg_reward:.3f} sub: {env.subject_id}"
            f"Avg. log-ratio: {mean_ratio:.3f}")

        # Update monitors
        # self.train_reward_monitor.update(avg_reward)
        # self.train_reward_monitor.end_epoch(i_episode)
        # self.train_length_monitor.update(avg_length)
        # self.train_length_monitor.end_epoch(i_episode)
        # self.train_ratio_monitor.update(mean_ratio)
        # self.train_ratio_monitor.end_epoch(i_episode)

        # Update comet logs
        if self.comet_experiment is not None:
            # self.comet_monitor.update_train(
            #     self.train_reward_monitor, i_episode)
            # self.comet_monitor.update_train(
            #     self.train_length_monitor, i_episode)
            # self.comet_monitor.update_train(
            #     self.train_ratio_monitor, i_episode)
            self.comet_monitor.log_losses({
                "train_reward": avg_reward,
                "train_length": avg_length,
                "train_ratio": mean_ratio,
            }, i_episode)

            mean_ep_reward_factors = mean_rewards(reward_factors)
            self.comet_monitor.log_losses(
                mean_ep_reward_factors, i_episode)
            mean_ep_losses = mean_losses(losses)
            self.comet_monitor.log_losses(mean_ep_losses, i_episode)

    def _valid_iter(self, valid_env, valid_tracker, alg, i_episode, save_model_dir):
        print("Validation run!")
        print("Loading subject...", end="")
        start = time.time()
        valid_env.load_subject()
        print(f" in {time.time() - start} seconds")

        start = time.time()
        print("Tracking and validating...", end="")
        valid_tractogram, valid_reward = \
            valid_tracker.track_and_validate(valid_env, enable_pbar=True)
        print(f" in {time.time() - start} seconds")

        print("Computing stopping stats...", end="")
        start = time.time()
        stopping_stats = self.stopping_stats(valid_tractogram)
        print(f" in {time.time() - start} seconds")

        print(stopping_stats)  # DO NOT REMOVE

        print("Logging losses", end="")
        start = time.time()
        self.comet_monitor.log_losses(stopping_stats, i_episode)
        print(f" in {time.time() - start} seconds")

        print("Saving tractogram...", end="")
        start = time.time()
        filename = self.save_rasmm_tractogram(
            valid_tractogram, valid_env.subject_id,
            valid_env.affine_vox2rasmm, valid_env.reference)
        print(f" in {time.time() - start} seconds")

        print("Scoring tractogram...", end="")
        start = time.time()
        scores = self.score_tractogram(
            filename, valid_env)

        print(f" in {time.time() - start} seconds")

        print(scores)

        # Display what the network is capable-of "now"
        self.log(
            valid_tractogram, valid_reward, i_episode)
        self.comet_monitor.log_losses(scores, i_episode)
        ckpt_path = self.save_model(alg, save_model_dir=save_model_dir)

        # Save best_epoch separately

        metric = scores["VC"] if "VC" in scores else valid_reward
        is_best_agent = metric > self.best_epoch_metric
        if is_best_agent:
            self.best_epoch_metric = metric
            self._hooks_manager.trigger_hooks(
                RlHookEvent.ON_RL_BEST_VC)

            # Instead of having to pack and serialize the model again,
            # as this takes time, just copy the file.
            # This aims to do the following:
            # self.save_model(alg, save_model_dir=save_model_dir,
            #                 is_best_model=True)
            ckpt_path = Path(ckpt_path)
            best_ckpt_path = ckpt_path.parent / "best_model_state.ckpt"
            shutil.copyfile(ckpt_path, best_ckpt_path)

    def setup_logging(self):
        # Save hyperparameters
        self.save_hyperparameters()

        # Setup monitors to monitor training as it goes along
        self.setup_monitors()

        # Setup comet monitors to monitor experiment as it goes along
        if not self.comet_monitor_was_setup:
            self.setup_comet("agent")
            self.comet_monitor_was_setup = True

    def setup_environment_and_info(self):
        # Instantiate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline
        # internally
        env = self.get_env()

        # Get example state to define NN input size
        self.input_size = env.get_state_size()
        self.action_size = env.get_action_size()

        # Voxel size
        self.voxel_size = env.get_voxel_size()

        # SH Order (used for tracking afterwards)
        self.hp.target_sh_order = env.target_sh_order

        return env

    def run(self):
        """ Prepare the environment, algorithm and trackers and run the
        training loop
        """

        assert_accelerator(), \
            "Training is only supported with hardware accelerated devices."

        env = self.setup_environment_and_info()
        valid_env = self.get_valid_env()

        max_traj_length = env.max_nb_steps

        # The RL training algorithm
        alg = self.get_alg(max_traj_length, env.neigh_manager)

        self.setup_logging()

        # Start training !
        self.rl_train(alg, env, valid_env, self.hp.max_ep, test_before_training=False)


def add_rl_args(parser):
    # Add RL training arguments.
    parser.add_argument('--max_ep', default=1000, type=int,
                        help='Number of episodes to run the training '
                        'algorithm')
    parser.add_argument('--log_interval', default=50, type=int,
                        help='Log statistics, update comet, save the model '
                        'and hyperparameters at n steps')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='Learning rate')
    parser.add_argument('--gamma', default=0.95, type=float,
                        help='Gamma param for reward discounting')

    add_reward_args(parser)


def add_training_args(parser):
    # Add all training arguments here. Less prone to error than
    # in every training script.

    add_experiment_args(parser)
    add_data_args(parser)
    add_environment_args(parser)
    add_model_args(parser)
    add_rl_args(parser)
    add_tracking_args(parser)
    add_oracle_args(parser)
    add_tractometer_args(parser)
    add_extractor_args(parser)
    add_rbx_args(parser)
    add_verifyber_args(parser)
