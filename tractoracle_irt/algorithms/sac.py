import copy
import numpy as np
import torch

from typing import Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from tractoracle_irt.algorithms.rl import RLAlgorithm
from tractoracle_irt.algorithms.shared.hyperparameters import HParams
from tractoracle_irt.environments.env import BaseEnv
from tractoracle_irt.algorithms.shared.offpolicy import SACActorCritic
from tractoracle_irt.algorithms.shared.replay import OffPolicyReplayBuffer
from tractoracle_irt.algorithms.shared.utils import add_item_to_means
from tractoracle_irt.utils.torch_utils import get_device
from tractoracle_irt.utils.logging import get_logger

LOGGER = get_logger(__name__)

@dataclass
class SACHParams(HParams):
    algorithm: str = field(default="SAC", init=False, repr=False)
    alpha: float
    batch_size: int
    replay_size: int
    utd: int
    save_replay_buffer: bool

class SAC(RLAlgorithm):
    """
    The sample-gathering and training algorithm.
    Based on

        Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018, July). Soft
        actor-critic: Off-policy maximum entropy deep reinforcement learning
        with a stochastic actor. In International conference on machine
        learning (pp. 1861-1870). PMLR.

    Implementation is based on Spinning Up's and rlkit

    See https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py
    See https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py  # noqa E501

    Some alterations have been made to the algorithms so it could be
    fitted to the tractography problem.

    """

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hparams: SACHParams,
        rng: np.random.RandomState = None,
        device: torch.device = get_device(),
    ):
        """ Initialize the algorithm. This includes the replay buffer,
        the policy and the target policy.

        Parameters
        ----------
        input_size: int
            Input size for the model
        action_size: int
            Output size for the actor
        hidden_dims: str
            Dimensions of the hidden layers
        lr: float
            Learning rate for the optimizer(s)
        gamma: float
            Discount factor
        alpha: float
            Entropy regularization coefficient
        n_actors: int
            Number of actors to use
        batch_size: int
            Batch size for the update
        replay_size: int
            Size of the replay buffer
        rng: np.random.RandomState
            Random number generator
        device: torch.device
            Device to train on. Should always be cuda:0
        """
        self.hp = hparams
        self.max_action = 1.
        self.t = 1

        self.action_size = action_size
        self.lr = self.hp.lr
        self.gamma = self.hp.gamma
        self.device = device
        self.n_actors = self.hp.n_actor

        self.rng = rng

        # Initialize main policy
        self.agent = SACActorCritic(
            input_size, action_size, self.hp.hidden_dims, device,
        )

        # Initialize target policy to provide baseline
        self.target = copy.deepcopy(self.agent)

        # SAC requires a different model for actors and critics
        # Optimizer for actor
        self.actor_optimizer = torch.optim.Adam(
            self.agent.actor.parameters(), lr=self.hp.lr)

        # Optimizer for critic
        self.critic_optimizer = torch.optim.Adam(
            self.agent.critic.parameters(), lr=self.hp.lr)

        # Temperature
        self.alpha = self.hp.alpha

        # SAC-specific parameters
        self.max_action = 1.
        self.on_policy = False

        self.start_timesteps = 1000
        self.total_it = 0
        self.tau = 0.005

        self.batch_size = self.hp.batch_size
        self.replay_size = self.hp.replay_size

        # Replay buffer
        self.replay_buffer = OffPolicyReplayBuffer(
            input_size, action_size, max_size=self.hp.replay_size)

        self.rng = rng
        self.start_update_log_was_printed = False

    def load_checkpoint(self, checkpoint_file: str):
        """
        Load a checkpoint into the algorithm.

        Parameters
        ----------
        checkpoint: dict
            Dictionary containing the checkpoint to load.
        """
        checkpoint = torch.load(checkpoint_file, weights_only=False)

        self.agent.load_checkpoint(checkpoint['agent'])
        self.target.load_checkpoint(checkpoint['target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if checkpoint.get('replay_buffer', None) is not None:
            self.replay_buffer.load_state_dict(checkpoint['replay_buffer'])

    def save_checkpoint(self, checkpoint_file: str, **extra_info):
        """
        Save the current state of the algorithm into a checkpoint.

        Parameters
        ----------
        checkpoint_file: str
            File to save the checkpoint into.
        """
        checkpoint = {
            'agent': self.agent.state_dict(as_dict=True),
            'target': self.target.state_dict(as_dict=True),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            **extra_info
        }

        if self.hp.save_replay_buffer:
            checkpoint['replay_buffer'] = self.replay_buffer.state_dict()

        torch.save(checkpoint, checkpoint_file)

    def sample_action(
        self,
        state: torch.Tensor
    ) -> np.ndarray:
        """ Sample an action according to the algorithm.
        """

        # Select action according to policy + noise for exploration
        action = self.agent.select_action(state, probabilistic=1.0)

        return action

    def _episode(
        self,
        initial_state: np.ndarray,
        env: BaseEnv,
    ) -> Tuple[float, float, float, int]:
        """
        Main loop for the algorithm
        From a starting state, run the model until the env. says its done
        Gather transitions and train on them according to the RL algorithm's
        rules.

        Parameters
        ----------
        initial_state: np.ndarray
            Initial state of the environment
        env: BaseEnv
            The environment actions are applied on. Provides the state fed to
            the RL algorithm

        Returns
        -------
        running_reward: float
            Sum of rewards gathered during the episode
        running_losses: dict
            Dict. containing losses and training-related metrics.
        episode_length: int
            Length of the episode
        running_reward_factors: dict
            Dict. containing the factors that contributed to the reward
        """

        running_reward = 0
        state = initial_state
        done = False
        running_losses = defaultdict(list)
        running_reward_factors = defaultdict(list)

        episode_length = 0

        self.replay_buffer.enter_write_mode()

        while not np.all(done):

            # Select action according to policy + noise for exploration
            with torch.no_grad():
                action = self.sample_action(state)

            # Perform action
            next_state, _, reward, done, info = env.step(
                action.to(device='cpu', copy=True).numpy())
            done_bool = done

            running_reward_factors = add_item_to_means(
                running_reward_factors, info['reward_info'])

            # Store data in replay buffer
            # WARNING: This is a bit of a trick and I'm not entirely sure this
            # is legal. This is effectively adding to the replay buffer as if
            # I had n agents gathering transitions instead of a single one.
            # This is not mentionned in the TD3 paper. PPO2 does use multiple
            # learners, though.
            # I'm keeping it since since it reaaaally speeds up training with
            # no visible costs
            self.replay_buffer.add(
                state.to('cpu', copy=True),
                action.to('cpu', copy=True),
                next_state.to('cpu', copy=True),
                torch.as_tensor(reward[..., None], dtype=torch.float32),
                torch.as_tensor(done_bool[..., None], dtype=torch.float32))

            running_reward += sum(reward)

            # Train agent after collecting sufficient data
            if self.t >= self.start_timesteps:
                if not self.start_update_log_was_printed:
                    self.start_update_log_was_printed = True
                    LOGGER.info("Acquired enough data to start updating the agent!")
                    print("Acquired enough data to start updating the agent!")

                # Update several times
                self.replay_buffer.enter_read_mode()
                for _ in range(self.hp.utd):
                    batch = self.replay_buffer.sample(self.batch_size)
                    losses = self.update(
                        batch)
            
                    running_losses = add_item_to_means(running_losses, losses)

            self.t += action.shape[0]

            # "Harvesting" here means removing "done" trajectories
            # from state as well as removing the associated streamlines
            # This line also set the next_state as the state
            state, _, _ = env.harvest()

            # Keeping track of episode length
            episode_length += 1
        return (
            running_reward,
            running_losses,
            episode_length,
            running_reward_factors,
            0)

    def update(
        self,
        batch,
    ) -> Tuple[float, float]:
        """

        SAC improves over DDPG by introducing an entropy regularization term
        in the actor loss. This encourages the policy to be more stochastic,
        which improves exploration. Additionally, SAC uses the minimum of two
        Q-functions in the value loss, rather than just one Q-function as in
        DDPG. This helps mitigate positive value biases and makes learning more
        stable.

        Parameters
        ----------
        batch: tuple
            Tuple containing the batch of data to train on, including
            state, action, next_state, reward, not_done.

        Returns
        -------
        losses: dict
            Dictionary containing the losses for the actor and critic and
            various other metrics.
        """
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = \
            batch

        pi, logp_pi = self.agent.act(state)
        alpha = self.alpha

        q1, q2 = self.agent.critic(state, pi)
        q_pi = torch.min(q1, q2)

        # Entropy-regularized policy loss
        actor_loss = (alpha * logp_pi - q_pi).mean()

        with torch.no_grad():
            # Target actions come from *current* policy
            next_action, logp_next_action = self.agent.act(next_state)

            # Compute the target Q value
            target_Q1, target_Q2 = self.target.critic(
                next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            backup = reward + self.gamma * not_done * \
                (target_Q - alpha * logp_next_action)

        # Get current Q estimates
        current_Q1, current_Q2 = self.agent.critic(
            state, action)

        # MSE loss against Bellman backup
        loss_q1 = ((current_Q1 - backup)**2).mean()
        loss_q2 = ((current_Q2 - backup)**2).mean()
        critic_loss = loss_q1 + loss_q2

        losses = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'loss_q1': loss_q1.item(),
            'loss_q2': loss_q2.item(),
            'Q1': current_Q1.mean().item(),
            'Q2': current_Q2.mean().item(),
            'backup': backup.mean().item(),
        }

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(
            self.agent.critic.parameters(),
            self.target.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(
            self.agent.actor.parameters(),
            self.target.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return losses
