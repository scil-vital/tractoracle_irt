import copy
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Tuple

from tractoracle_irt.algorithms.sac import SAC, SACHParams
from tractoracle_irt.algorithms.shared.offpolicy import SACActorCritic
from tractoracle_irt.algorithms.shared.replay import OffPolicyReplayBuffer, OffPolicyLazyReplayBuffer
from tractoracle_irt.utils.torch_utils import get_device, gradients_norm
from tractoracle_irt.environments.state import StateShape
from tractoracle_irt.algorithms.shared.hyperparameters import HParams

LOG_STD_MAX = 2
LOG_STD_MIN = -20

@dataclass
class SACAutoHParams(SACHParams):
    algorithm: str = field(default="SACAuto", init=False, repr=False)

class SACAuto(SAC):
    """
    The sample-gathering and training algorithm.
    Based on

        Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., ...
        & Levine, S. (2018). Soft actor-critic algorithms and applications.
        arXiv preprint arXiv:1812.05905.

    Implementation is based on Spinning Up's and rlkit

    See https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py
    See https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py  # noqa E501

    Some alterations have been made to the algorithms so it could be
    fitted to the tractography problem.

    """

    def __init__(
        self,
        input_shape: StateShape,
        action_size: int,
        hparams: SACAutoHParams,
        rng: np.random.RandomState = None,
        device: torch.device = get_device,
    ):
        """
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
            Initial entropy coefficient (temperature).
        n_actors: int
            Number of actors to use
        batch_size: int
            Batch size to sample the memory
        replay_size: int
            Size of the replay buffer
        rng: np.random.RandomState
            Random number generator
        device: torch.device
            Device to use for the algorithm. Should be either "cuda:0"
        """
        self.hp = hparams
        
        # TO REMOVE
        self.batch_size = self.hp.batch_size
        self.gamma = self.hp.gamma
        self.alpha = self.hp.alpha
        self.n_actors = self.hp.n_actor
        self.replay_size = self.hp.replay_size

        self.max_action = 1.
        self.t = 1
        self.nb_updates_per_sample = 5

        self.action_size = action_size
        self.device = device

        self.rng = rng

        # Initialize main agent
        self.agent = SACActorCritic(
            input_shape, action_size, self.hp.hidden_dims, device,
        )

        # Auto-temperature adjustment
        # SAC automatically adjusts the temperature to maximize entropy and
        # thus exploration, but reduces it over time to converge to a
        # somewhat deterministic policy.
        starting_temperature = np.log(self.hp.alpha)  # Found empirically
        self.target_entropy = -np.prod(action_size).item()
        self.log_alpha = torch.full(
            (1,), starting_temperature, requires_grad=True, device=device)
        # Optimizer for alpha
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.hp.lr)

        # Initialize target agent to provide baseline
        self.target = copy.deepcopy(self.agent)

        # SAC requires a different model for actors and critics
        # Optimizer for actor
        self.actor_optimizer = torch.optim.Adam(
            self.agent.actor.parameters(), lr=self.hp.lr)

        # Optimizer for critic
        self.critic_optimizer = torch.optim.Adam(
            self.agent.critic.parameters(), lr=self.hp.lr)

        # SAC-specific parameters
        self.max_action = 1.
        self.on_agent = False

        self.start_timesteps = 80000
        self.total_it = 0
        self.tau = 0.005
        self.agent_freq = 1

        # Replay buffer
        self.replay_buffer = OffPolicyReplayBuffer(
            input_shape, action_size, max_size=self.hp.replay_size)

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
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        if checkpoint.get('replay_buffer', None) is not None:
            self.replay_buffer.load_state_dict(checkpoint['replay_buffer'])
        if checkpoint.get('log_alpha', None) is not None:
            self.log_alpha = checkpoint['log_alpha']

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
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            **extra_info
        }

        if self.hp.save_replay_buffer:
            checkpoint['replay_buffer'] = self.replay_buffer.state_dict()

        torch.save(checkpoint, checkpoint_file)

    def update(
        self,
        batch,
    ) -> Tuple[float, float]:
        """

        SAC Auto improves upon SAC by automatically adjusting the temperature
        parameter alpha. This is done by optimizing the temperature parameter
        alpha to maximize the entropy of the policy. This is done by
        maximizing the following objective:
            J_alpha = E_pi [log pi(a|s) + alpha H(pi(.|s))]
        where H(pi(.|s)) is the entropy of the policy.


        Parameters
        ----------
        batch: Tuple containing the batch of data to train on.

        Returns
        -------
        losses: dict
            Dictionary containing the losses of the algorithm and various
            other metrics.
        """
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = \
            batch
        # Compute \pi_\theta(s_t) and log \pi_\theta(s_t)
        pi, logp_pi = self.agent.act(
            state, probabilistic=1.0)
        # Compute the temperature loss and the temperature
        alpha_loss = -(self.log_alpha * (
            logp_pi + self.target_entropy).detach()).mean()
        alpha = self.log_alpha.exp()

        # Compute the Q values and the minimum Q value
        q1, q2 = self.agent.critic(state, pi)
        q_pi = torch.min(q1, q2)

        # Entropy-regularized agent loss
        actor_loss = (alpha * logp_pi - q_pi).mean()

        with torch.no_grad():
            # Target actions come from *current* agent
            next_action, logp_next_action = self.agent.act(
                next_state, probabilistic=1.0)

            # Compute the next Q values using the target agent
            target_Q1, target_Q2 = self.target.critic(
                next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            # Compute the backup which is the Q-learning "target"
            backup = reward + self.gamma * not_done * \
                (target_Q - alpha * logp_next_action)

        # Get current Q estimates
        current_Q1, current_Q2 = self.agent.critic(
            state, action)

        # MSE loss against Bellman backup
        loss_q1 = F.mse_loss(current_Q1, backup.detach()).mean()
        loss_q2 = F.mse_loss(current_Q2, backup.detach()).mean()
        # Total critic loss
        critic_loss = loss_q1 + loss_q2

        # Optimize the temperature
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        nn.utils.clip_grad_norm_(self.log_alpha, 0.5)
        self.alpha_optimizer.step()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 0.5)
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
                self.tau * param.data + (1 - self.tau) * target_param.data)

        # Compute the norm of the gradients to plot.
        alpha_norm = self.log_alpha.grad.norm(2).cpu().detach().numpy()
        critic_norm = gradients_norm(self.agent.critic)
        actor_norm = gradients_norm(self.agent.actor)

        # print("alpha_norm: ", type(alpha_norm))
        # print("critic_norm: ", type(critic_norm))
        # print("actor_norm: ", type(actor_norm))

        losses = {
            # 'actor_loss': actor_loss.detach(),
            # 'alpha_loss': alpha_loss.detach(),
            # 'critic_loss': critic_loss.detach(),
            # 'loss_q1': loss_q1.detach(),
            # 'loss_q2': loss_q2.detach(),
            # 'entropy': alpha.detach(),
            # 'Q1': current_Q1.mean().detach(),
            # 'Q2': current_Q2.mean().detach(),
            # 'backup': backup.mean().detach(),
            "alpha_norm": alpha_norm,
            "critic_norm": critic_norm,
            "actor_norm": actor_norm,
        }

        return losses
