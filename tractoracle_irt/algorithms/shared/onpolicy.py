import numpy as np
import torch
import contextlib

from os.path import join as pjoin
from torch import nn
from torch.distributions.normal import Normal
from typing import Tuple
import torch.nn.functional as F

from tractoracle_irt.algorithms.shared.utils import (
    format_widths, make_fc_network)
from tractoracle_irt.oracles.transformer_oracle import TransformerOracle
from tractoracle_irt.utils.torch_utils import get_device_str, get_device
from tractoracle_irt.utils.utils import break_if_found_nans

autocast_context = torch.cuda.amp.autocast if torch.cuda.is_available() else contextlib.nullcontext

class HybridMaxEntropyActor(nn.Module):
    """ Actor module that takes in a state and outputs an action.
    Its policy is the neural network layers
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: list,
        device: torch.device,
        action_std: float = 0.0,
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

        """
        NB: This is a modified Actor. We want to be able to use PPO with the same actor as SAC.
        Classically, in PPO, the STD is state-independent, but in SAC, it is state-dependent.
        We modified the PPO's actor to have a state-dependent STD, as in SAC.
        """
        super(HybridMaxEntropyActor, self).__init__()

        self.action_dim = action_dim
        self.hidden_layers = hidden_layers

        self.output_activation = output_activation()

        self.layers = make_fc_network(
            self.hidden_layers, state_dim, action_dim * 2)


    def forward(
        self,
        state: torch.Tensor,
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

        LOG_STD_MAX = 2
        LOG_STD_MIN = -20

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
        pre_pi_action = pi_distribution.rsample()

        break_if_found_nans(pre_pi_action) # REMOVE
        
        # Trick from Spinning Up's implementation:
        # Compute logprob from Gaussian, and then apply correction for Tanh
        # squashing. NOTE: The correction formula is a little bit magic. To
        # get an understanding of where it comes from, check out the
        # original SAC paper (arXiv 1801.01290) and look in appendix C.
        # This is a more numerically-stable equivalent to Eq 21.
        logp_pi = pi_distribution.log_prob(pre_pi_action).sum(axis=-1)
        # Squash correction
        logp_pi -= (2*(np.log(2) - pre_pi_action -
                       F.softplus(-2*pre_pi_action))).sum(axis=1)

        # Run actions through tanh to get -1, 1 range
        pi_action = self.output_activation(pre_pi_action)
        # Return action and logprob


        return pi_action, logp_pi, pi_distribution.entropy()

class Actor(nn.Module):
    """ Actor module that takes in a state and outputs an action.
    Its policy is the neural network layers
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: list,
        device: torch.device,
        action_std: float = 0.0,
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

        self.layers = make_fc_network(
            hidden_layers, state_dim, action_dim, activation=nn.Tanh)

        # State-independent STD, as opposed to SAC which uses a
        # state-dependent STD.
        # See https://spinningup.openai.com/en/latest/algorithms/sac.html
        # in the "You Should Know" box
        log_std = -action_std * np.ones(action_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))

    def _distribution(self, state: torch.Tensor):
        mu = self.layers(state)
        std = torch.exp(self.log_std)
        try:
            dist = Normal(mu, std)
        except ValueError as e:
            print(mu, std)
            raise e

        return dist

    def forward(self, state: torch.Tensor, probabilistic: float = 1.0) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """
        return self._distribution(state)


class PolicyGradient(nn.Module):
    """ PolicyGradient module that handles actions
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        device: torch.device,
        action_std: float = 0.0,
        actor_cls: nn.Module = Actor,
    ):
        super(PolicyGradient, self).__init__()
        self.device = device
        self.action_dim = action_dim

        self.hidden_layers = format_widths(hidden_dims)

        self.actor = actor_cls(
            state_dim, action_dim, self.hidden_layers, action_std,
        ).to(device)

    def act(
        self, state: torch.Tensor, probabilistic: float = 1.0,
    ) -> torch.Tensor:
        """ Select noisy action according to actor
        """
        pi = self.actor.forward(state, probabilistic)
        if probabilistic > 0.0:
            action = pi.sample()
        else:
            action = pi.mean
            
        break_if_found_nans(action) # REMOVE
        return action

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor, probabilistic: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Get output of value function for the actions, as well as
        logprob of actions and entropy of policy for loss
        """

        pi = self.actor(state)
        mu, std = pi.mean, pi.stddev
        action_logprob = pi.log_prob(action).sum(axis=-1)
        entropy = pi.entropy()

        return action_logprob, entropy, mu, std

    def select_action(
        self, state: np.array, probabilistic: float = 1.0,
    ) -> np.ndarray:
        """ Move state to torch tensor, select action and
        move it back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment

        Returns:
        --------
            action: np.array
                Action selected by the policy
        """

        if len(state.shape) < 2:
            state = state[None, :]

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = self.act(state, probabilistic)

        return action

    def get_evaluation(
        self, state: np.array, action: np.array, probabilistic: float = 1.0
    ) -> Tuple[np.array, np.array, np.array]:
        """ Move state and action to torch tensor,
        get value estimates for states, probabilities of actions
        and entropy for action distribution, then move everything
        back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment
            action: np.array
                Actions taken by the policy

        Returns:
        --------
            v: np.array
                Value estimates for state
            prob: np.array
                Probabilities of actions
            entropy: np.array
                Entropy of policy
        """

        if len(state.shape) < 2:
            state = state[None, :]
        if len(action.shape) < 2:
            action = action[None, :]

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(
            action, dtype=torch.float32, device=self.device)

        prob, entropy, mu, std = self.evaluate(state, action)

        # REINFORCE does not use a critic
        values = np.zeros((state.size()[0]))

        return (
            values,
            prob.cpu().data.numpy(),
            entropy.cpu().data.numpy(),
            mu.cpu().data.numpy(),
            std.cpu().data.numpy())

    def load_state_dict(self, state_dict):
        """ Load parameters into actor and critic
        """
        actor_state_dict = state_dict
        self.actor.load_state_dict(actor_state_dict)

    def state_dict(self):
        """ Returns state dicts so they can be loaded into another policy
        """
        return self.actor.state_dict()

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
        self.actor.load_state_dict(
            torch.load(pjoin(path, filename + '_actor.pth'),
                       map_location=self.device), weights_only=False)

    def eval(self):
        """ Switch actors and critics to eval mode
        """
        self.actor.eval()

    def train(self):
        """ Switch actors and critics to train mode
        """
        self.actor.train()


class Critic(nn.Module):
    """ Critic module that takes in a pair of state-action and outputs its
    q-value according to the network's q function. TD3 uses two critics
    and takes the lowest value of the two during backprop.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: list,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dim: int
                Width of network. Presumes all intermediary
                layers are of same size for simplicity

        """
        super(Critic, self).__init__()

        self.layers = make_fc_network(
            hidden_layers, state_dim, 1, activation=nn.Tanh)

    def forward(self, state) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs a q estimate from first critic
        """

        return self.layers(state)
    
class OracleBasedCritic(nn.Module):
    def __init__(
        self,
        checkpoint: dict,
        batch_size = 4096,
        device = get_device()
    ):
        super(OracleBasedCritic, self).__init__()
        self.model = TransformerOracle.load_from_checkpoint(checkpoint)
        self.device = device
        self.batch_size = batch_size

    def forward(self, streamlines) -> torch.Tensor:
        # This is copied from OracleSingleton!!

        # Total number of predictions to return
        N = len(streamlines)
        # Placeholders for input and output data
        placeholder = torch.zeros(
            (self.batch_size, 127, 3), pin_memory=get_device_str() == "cuda")
        result = torch.zeros((N), dtype=torch.float, device=self.device)

        # Get the first batch
        batch = streamlines[:self.batch_size]
        N_batch = len(batch)
        # Resample streamlines to fixed number of point to set all
        # sequences to same length
        with torch.no_grad():
            if isinstance(batch, torch.Tensor):
                data = batch
            else:
                data = torch.tensor(batch, dtype=torch.float32, device=self.device)
            # Compute streamline features as the directions between points
            dirs = torch.diff(data, dim=1)
        # Send the directions to pinned memory
        placeholder[:N_batch] = dirs
        # Send the pinned memory to GPU asynchronously
        input_data = placeholder[:N_batch].to(
            self.device, non_blocking=True, dtype=torch.float)
        i = 0

        while i <= N // self.batch_size:
            start = (i+1) * self.batch_size
            end = min(start + self.batch_size, N)
            # Prefetch the next batch
            if start < end:
                batch = streamlines[start:end]
                # Resample streamlines to fixed number of point to set all
                # sequences to same length
                data = batch #set_number_of_points(batch, 128)
                # Compute streamline features as the directions between points
                dirs = np.diff(data, axis=1)
                # Put the directions in pinned memory
                placeholder[:end-start] = torch.from_numpy(dirs)

            with autocast_context():
                with torch.no_grad():
                    predictions = self.model(input_data)
                    result[
                        i * self.batch_size:
                        (i * self.batch_size) + self.batch_size] = predictions
            i += 1
            if i >= N // self.batch_size:
                break
            # Send the pinned memory to GPU asynchronously
            input_data = placeholder[:end-start].to(
                self.device, non_blocking=True, dtype=torch.float)
        return result


class ActorCritic(PolicyGradient):
    """ Actor-Critic module that handles both actions and values
    Actors and critics here don't share a body but do share a loss
    function. Therefore they are both in the same module
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        device: torch.device,
        action_std: float = 0.0,
        actor_cls: nn.Module = Actor,
        critic_checkpoint: dict = None
    ):
        super(ActorCritic, self).__init__(
            state_dim,
            action_dim,
            hidden_dims,
            device,
            action_std,
            actor_cls
        )

        if critic_checkpoint is not None:
            self.critic = OracleBasedCritic(critic_checkpoint).to(self.device)
        else:
            self.critic = Critic(
                state_dim, action_dim, self.hidden_layers,
            ).to(self.device)

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor, probabilistic: float, streamlines: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Get output of value function for the actions, as well as
        logprob of actions and entropy of policy for loss
        """
        pi = self.actor.forward(state, probabilistic)
        # mu, std = pi.mean, pi.stddev
        action_logprob = pi.log_prob(action).sum(axis=-1)
        entropy = pi.entropy()
        if isinstance(self.critic, OracleBasedCritic):
            values = self.critic(streamlines)
        else:
            values = self.critic(state)

        if values.dim() > 1:
            values = values.squeeze(-1)

        return values, action_logprob, entropy

    def get_evaluation(
        self, state: np.array, action: np.array, probabilistic: float, streamlines: np.array = None
    ) -> Tuple[np.array, np.array, np.array]:
        """ Move state and action to torch tensor,
        get value estimates for states, probabilities of actions
        and entropy for action distribution, then move everything
        back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment
            action: np.array
                Actions taken by the policy

        Returns:
        --------
            v: np.array
                Value estimates for state
            prob: np.array
                Probabilities of actions
            entropy: np.array
                Entropy of policy
        """

        if len(state.shape) < 2:
            state = state[None, :]
        if len(action.shape) < 2:
            action = action[None, :]

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(
            action, dtype=torch.float32, device=self.device)

        v, prob, entropy = self.evaluate(state, action, probabilistic, streamlines)

        return (
            v,
            prob,
            entropy)

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
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict()
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

class PPOActorCritic(ActorCritic):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: str,
                 device: torch.device,
                 action_std: float = 0,
                 actor_cls=HybridMaxEntropyActor,
                 critic_checkpoint: dict = None):
        super().__init__(state_dim,
                         action_dim,
                         hidden_dims,
                         device,
                         action_std,
                         actor_cls=actor_cls,
                         critic_checkpoint=critic_checkpoint)

    def load_policy(self, path: str, filename: str):
        self.actor.load_state_dict(
            torch.load(pjoin(path, filename + '_actor.pth'),
                       map_location=self.device), weights_only=False)
