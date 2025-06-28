import numpy as np
import torch

import torch.nn.functional as F

from os.path import join as pjoin
from torch import nn
from torch.distributions.normal import Normal

from tractoracle_irt.algorithms.shared.fodf_encoder import FodfEncoder, encoding_layers
from tractoracle_irt.environments.state import State, StateShape
from tractoracle_irt.algorithms.shared.utils import (
    format_widths, make_fc_network, make_conv_network)


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
    """ Actor module that takes in a state and outputs an action.
    Its policy is the neural network layers
    """

    def __init__(
        self,
        state_dim: StateShape,
        action_dim: int,
        hidden_dims: str,
        output_activation=nn.Tanh
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: str
                String representing layer widths

        """
        super(Actor, self).__init__()

        self.action_dim = action_dim

        conv_state_shape = (state_dim.nb_sh_coefs, state_dim.depth,
                            state_dim.height, state_dim.width)
        conv_output_size = 256

        self.conv_layers = make_conv_network(input_size=conv_state_shape,
            output_size=conv_output_size)
        full_fc_state_dim = conv_output_size + state_dim.prev_dirs_size

        self.hidden_layers = format_widths(hidden_dims)
        self.layers = make_fc_network(
            self.hidden_layers, full_fc_state_dim, action_dim)

        self.output_activation = output_activation()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """
        p = self.layers(state)
        p = self.output_activation(p)

        return p


class MaxEntropyActor(nn.Module):
    """ Actor module that takes in a state and outputs an action.
    Its policy is the neural network layers
    """

    def __init__(
        self,
        state_dim: StateShape,
        action_dim: int,
        hidden_dims: str,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: str
                String representing layer widths

        """
        super(MaxEntropyActor, self).__init__()

        # Setup the encoding part of the actor if required.
        # e.g. when we are requiring a large neighborhood
        # that requires convolutions.
        self.state_is_flat = state_dim.is_flat
        if self.state_is_flat:
            flat_neigh_size = state_dim.neighborhood_common_shape[0]
        else:
            # Large neighborhood will be encoded by some CNN layers.
            conv_state_shape = (state_dim.nb_sh_coefs, state_dim.depth,
                                state_dim.height, state_dim.width)

            self.encoder, flat_neigh_size = encoding_layers(conv_state_shape[0])

        self.action_dim = action_dim
        self.hidden_layers = format_widths(hidden_dims)

        # Setup the core of the policy
        total_fc_input_size = flat_neigh_size + state_dim.prev_dirs_size
        self.layers = make_fc_network(
            self.hidden_layers, total_fc_input_size, action_dim * 2)
        
        self.output_activation = nn.Tanh()

    def forward(
        self,
        state: State,
        probabilistic: float,
    ) -> torch.Tensor:
        """ Forward propagation of the actor. Log probability is computed
        from the Gaussian distribution of the action and correction
        for the Tanh squashing is applied.

        Parameters:
        -----------
        state: torch.Tensor
            Current state of the environment
        probabilistic: float
            Factor to multiply the standard deviation by when sampling.
            0 means a deterministic policy, 1 means a fully stochastic.
        """
        # Encode the state if needed.
        if self.state_is_flat:
            flat_neighborhood = state.neighborhood
        else:
            flat_neighborhood = self.encoder(state.neighborhood)

        # Concatenate the encoded neighborhood with the previous directions")
        state = torch.cat([flat_neighborhood, state.prev_dirs], dim=1)
        
        # Compute mean and log_std from neural network. Instead of
        # have two separate outputs, we have one output of size
        # action_dim * 2. The first action_dim are the means, and
        # the last action_dim are the log_stds.
        p = self.layers(state)
        mu = p[:, :self.action_dim]
        log_std = p[:, self.action_dim:]
        # Constrain log_std inside [LOG_STD_MIN, LOG_STD_MAX]
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        # Compute std from log_std
        std = torch.exp(log_std) * probabilistic
        # Sample from Gaussian distribution using reparametrization trick
        pi_distribution = Normal(mu, std, validate_args=False)
        pi_action = pi_distribution.rsample()

        # Trick from Spinning Up's implementation:
        # Compute logprob from Gaussian, and then apply correction for Tanh
        # squashing. NOTE: The correction formula is a little bit magic. To
        # get an understanding of where it comes from, check out the
        # original SAC paper (arXiv 1801.01290) and look in appendix C.
        # This is a more numerically-stable equivalent to Eq 21.
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        # Squash correction
        logp_pi -= (2*(np.log(2) - pi_action -
                       F.softplus(-2*pi_action))).sum(axis=1)

        # Run actions through tanh to get -1, 1 range
        pi_action = self.output_activation(pi_action)
        # Return action and logprob
        return pi_action, logp_pi


class Critic(nn.Module):
    """ Critic module that takes in a pair of state-action and outputs its
    q-value according to the network's q function.
    """

    def __init__(
        self,
        state_dim: StateShape,
        action_dim: int,
        hidden_dims: str,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: int
                String representing layer widths

        """
        super(Critic, self).__init__()


        # conv_state_shape = (state_dim.nb_sh_coefs, state_dim.depth,
        #                     state_dim.height, state_dim.width)
        # conv_output_size = 256

        # self.conv_layers = make_conv_network(input_size=conv_state_shape,
        #     output_size=conv_output_size)

        # full_fc_state_dim = + state_dim.prev_dirs_size

        # self.hidden_layers = format_widths(hidden_dims)
        # self.q1 = make_fc_network(
        #     self.hidden_layers, full_fc_state_dim + action_dim, 1)

    def forward(self, state: State, action) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs a q estimate from both critics
        """
        q1_input = torch.cat([state.neighborhood, state.prev_dirs, action], -1)

        q1 = self.q1(q1_input).squeeze(-1)

        return q1


class DoubleCritic(Critic):
    """ Critic module that takes in a pair of state-action and outputs its
5   q-value according to the network's q function. TD3 uses two critics
    and takes the lowest value of the two during backprop.
    """

    def __init__(
        self,
        state_dim: StateShape,
        action_dim: int,
        hidden_dims: str,
        critic_size_factor=1,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: str
                String representing layer widths

        """
        super(DoubleCritic, self).__init__(
            state_dim, action_dim, hidden_dims)

        self.state_is_flat = state_dim.is_flat
        
        if self.state_is_flat:
            flat_neigh_size = state_dim.neighborhood_common_shape[0]
        else:
            conv_state_shape = (state_dim.nb_sh_coefs, state_dim.depth,
                        state_dim.height, state_dim.width)
            self.q1_neighbor_encoder, flat_neigh_size = encoding_layers(conv_state_shape[0])
            self.q2_neighbor_encoder, flat_neigh_size = encoding_layers(conv_state_shape[0])
            
            # flat_neigh_size = self.q1_neighbor_encoder.flat_output_size
        
        full_fc_state_dim = flat_neigh_size + state_dim.prev_dirs_size
        self.hidden_layers = format_widths(
            hidden_dims) * critic_size_factor

        self.q1 = make_fc_network(
            self.hidden_layers, full_fc_state_dim + action_dim, 1)
        self.q2 = make_fc_network(
            self.hidden_layers, full_fc_state_dim + action_dim, 1)

    def forward(self, state: State, action) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs a q estimate from both critics
        """
        # assert isinstance(state.neighborhood, torch.Tensor), "state.neighborhood must be a tensor"
        # assert state.neighborhood.requires_grad, "state.neighborhood must have requires_grad=True"
        if self.state_is_flat:
            encoded_neighborhood_1 = state.neighborhood
            encoded_neighborhood_2 = state.neighborhood
        else:
            encoded_neighborhood_1 = self.q1_neighbor_encoder(state.neighborhood)
            encoded_neighborhood_2 = self.q2_neighbor_encoder(state.neighborhood)

        q1_input = torch.cat([encoded_neighborhood_1, state.prev_dirs, action], -1)
        q2_input = torch.cat([encoded_neighborhood_2, state.prev_dirs, action], -1)

        q1 = self.q1(q1_input).squeeze(-1)
        q2 = self.q2(q2_input).squeeze(-1)

        return q1, q2

    def Q1(self, state: State, action) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs a q estimate from first critic
        """
        q1_input = torch.cat([state, action], -1)

        q1 = self.q1(q1_input).squeeze(-1)

        return q1


class ActorCritic(object):
    """ Module that handles the actor and the critic
    """

    def __init__(
        self,
        state_dim: StateShape,
        action_dim: int,
        hidden_dims: str,
        device: torch.device,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: int
                String representing layer widths

        """
        self.device = device
        self.actor = Actor(
            state_dim, action_dim, hidden_dims
        ).to(device)

        self.critic = Critic(
            state_dim, action_dim, hidden_dims,
        ).to(device)

    def act(self, state: State) -> torch.Tensor:
        """ Select action according to actor

        Parameters:
        -----------
            state: torch.Tensor
                Current state of the environment

        Returns:
        --------
            action: torch.Tensor
                Action selected by the policy
        """
        return self.actor(state)

    def select_action(self, state: State, probabilistic=0.0) -> np.ndarray:
        """ Move state to torch tensor, select action and
        move it back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment
            probabilistic: float
                Unused as TD3 does not use probabilistic actions.

        Returns:
        --------
            action: np.array
                Action selected by the policy
        """
        # if state is not batched, expand it to "batch of 1"
        if len(state.shape) < 2:
            state = state[None, :]
        action = self.act(state)

        return action

    def parameters(self):
        """ Access parameters for grad clipping
        """
        return self.actor.parameters()

    def load_state_dict(self, state_dict):
        """ Load parameters into actor and critic
        """
        actor_state_dict, critic_state_dict = state_dict
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)

    def state_dict(self, as_dict=False):
        """ Returns state dicts so they can be loaded into another policy
        """
        if as_dict:
            return {
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict()
            }
        else:
            return self.actor.state_dict(), self.critic.state_dict()

    def save(self, path: str, filename: str):
        """ Save policy at specified path and filename
        Parameters:
        -----------
            path: string
                Path to folder that will contain saved state dicts
            filename: string
                Name of saved models. Suffixes for actors and critics
                will be appended
        """
        torch.save(
            self.critic.state_dict(), pjoin(path, filename + "_critic.pth"))
        torch.save(
            self.actor.state_dict(), pjoin(path, filename + "_actor.pth"))

    def load(self, path: str, filename: str):
        """ Load policy from specified path and filename
        Parameters:
        -----------
            path: string
                Path to folder containing saved state dicts
            filename: string
                Name of saved models. Suffixes for actors and critics
                will be appended
        """
        self.critic.load_state_dict(
            torch.load(pjoin(path, filename + '_critic.pth'),
                       map_location=self.device), weights_only=False)
        self.actor.load_state_dict(
            torch.load(pjoin(path, filename + '_actor.pth'),
                       map_location=self.device), weights_only=False)
        
    def load_checkpoint(self, agent_checkpoint: dict):
        """
        This function should get passed a dictionary containing the state
        of the actor and the critic. It should have two keys: 'actor' and
        'critic' both associated with their respective state_dicts.
        """
        self.actor.load_state_dict(agent_checkpoint['actor'])
        self.critic.load_state_dict(agent_checkpoint['critic'])

    def eval(self):
        """ Switch actors and critics to eval mode
        """
        self.actor.eval()
        self.critic.eval()

    def train(self):
        """ Switch actors and critics to train mode
        """
        self.actor.train()
        self.critic.train()


class TD3ActorCritic(ActorCritic):
    """ Module that handles the actor and the critic for TD3
    The actor is the same as the DDPG actor, but the critic is different.

    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        device: torch.device,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: str
                String representing layer widths
            device: torch.device

        """
        self.device = device
        self.actor = Actor(
            state_dim, action_dim, hidden_dims,
        ).to(device)

        self.critic = DoubleCritic(
            state_dim, action_dim, hidden_dims,
        ).to(device)


class SACActorCritic(ActorCritic):
    """ Module that handles the actor and the critic
    """

    def __init__(
        self,
        state_dim: StateShape,
        action_dim: int,
        hidden_dims: str,
        device: torch.device,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: str
                String representing layer widths
            device: torch.device

        """
        self.device = device
        self.actor = MaxEntropyActor(
            state_dim, action_dim, hidden_dims
        ).to(device)

        self.critic = DoubleCritic(
            state_dim, action_dim, hidden_dims
        ).to(device)

    def act(self, state: State, probabilistic=1.0) -> torch.Tensor:
        """ Select action according to actor

        Parameters:
        -----------
            state: torch.Tensor
                Current state of the environment
            probabilistic: float
                Factor to multiply the standard deviation by when sampling
                actions.

        Returns:
        --------
            action: torch.Tensor
                Action selected by the policy
            logprob: torch.Tensor
                Log probability of the action
        """
        action, logprob = self.actor(state, probabilistic)
        return action, logprob

    def select_action(self, state: State, probabilistic=1.0) -> np.ndarray:
        """ Act on a state and return an action.

        Parameters:
        -----------
            state: np.array
                State of the environment
            probabilistic: float
                Factor to multiply the standard deviation by when sampling
                actions.

        Returns:
        --------
            action: np.array
                Action selected by the policy
        """
        # if state is not batched, expand it to "batch of 1"
        # if len(state.shape) < 2:
        #     state = state[None, :]

        action, _ = self.act(state, probabilistic)

        return action
