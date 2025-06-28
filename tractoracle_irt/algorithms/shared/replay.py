import numpy as np
import torch
import scipy.signal
from typing import Tuple
#from tractoracle_irt.algorithms.shared.disc_cumsum import disc_cumsum
from tractoracle_irt.utils.utils import break_if_found_nans, break_if_found_nans_args
from tractoracle_irt.environments.state import State, StateShape
from tractoracle_irt.utils.lazy_tensor import LazyTensorManager, NaiveLazyTensorManager
from tractoracle_irt.environments.neighborhood_manager import NeighborhoodManager

from tractoracle_irt.utils.torch_utils import get_device, get_device_str

device = get_device()
rb_type = torch.float32

class OffPolicySemiLazyReplayBuffer(object):
    def __init__(self, state_dim: StateShape, action_dim: int,
                 neighborhood_manager: NeighborhoodManager, max_size=int(1e6)):
        print("Creating replay buffer with shape: ", (max_size, *state_dim.neighborhood_common_shape))
        self.size = 0
        self.ptr = 0
        self.device = device
        self.max_size = int(max_size)
        self.neigh_manager = neighborhood_manager

        self.state_coords = torch.zeros((max_size, 3), dtype=rb_type) # Store the coordinates of the states
        self.state_prev_dirs = torch.zeros((max_size, state_dim.prev_dirs_size), dtype=rb_type) # Store the previous directions of the states

        self.n_state_coords = torch.zeros((max_size, 3), dtype=rb_type) # Store the coordinates of the next states
        self.n_state_prev_dirs = torch.zeros((max_size, state_dim.prev_dirs_size), dtype=rb_type) # Store the previous directions of the next states

        self.action = torch.zeros((max_size, action_dim), dtype=rb_type)
        self.reward = torch.zeros((max_size, 1), dtype=rb_type)
        self.not_done = torch.zeros((max_size, 1), dtype=rb_type)

    def add(self, state: State, action, next_state: State, reward, done):
        indices = (np.arange(0, len(state)) + self.ptr) % self.max_size

        self.state_coords[indices] = state.coords # Here we only want to keep the coordinates in memory, the rest of the state should be discarded
        self.n_state_coords[indices] = next_state.coords
        self.action[indices] = action
        self.reward[indices] = reward
        self.not_done[indices] = 1. - done

        self.ptr = (self.ptr + len(indices)) % self.max_size
        self.size = min(self.size + len(indices), self.max_size)

    def __len__(self):
        return self.size

    def sample(self, batch_size=4096):
        ind = np.random.choice(self.size, batch_size, replace=False)
        ind = torch.from_numpy(ind)

        # Interpolate the state and next state coordinates.
        state_coords = self.state_coords.index_select(0, ind)
        state_coords = state_coords.to(device=self.device, non_blocking=True)
        n_state_coords = self.n_state_coords.index_select(0, ind)
        n_state_coords = n_state_coords.to(device=self.device, non_blocking=True)

        s_neigh = self.neigh_manager.get(state_coords, torch_convention=True)
        s_prev_dirs = self.state_prev_dirs.index_select(0, ind).to(device=self.device)
        s = State(s_neigh, s_prev_dirs, state_coords, device=self.device)

        ns = self.neigh_manager.get(n_state_coords, torch_convention=True)
        ns_prev_dirs = self.n_state_prev_dirs.index_select(0, ind).to(device=self.device)
        ns = State(ns, ns_prev_dirs, n_state_coords, device=self.device)

        a = self.action.index_select(0, ind)
        r = self.reward.index_select(0, ind).squeeze(-1)
        d = self.not_done.index_select(0, ind).to(
            dtype=torch.float32).squeeze(-1)
        
        if get_device_str() == "cuda":
            # s = s.pin_memory()
            a = a.pin_memory()
            # ns = ns.pin_memory()
            r = r.pin_memory()
            d = d.pin_memory()

        # Return tensors on the same device as the buffer in pinned memory
        return (s.to(device=self.device), # they are already on the device after interpolation from neighborhood manager
                a.to(device=self.device, non_blocking=True),
                ns.to(device=self.device),
                r.to(device=self.device, non_blocking=True),
                d.to(device=self.device, non_blocking=True))

    def enter_read_mode(self):
        pass

    def enter_write_mode(self):
        pass

    def clear_memory(self):
        """ Reset the buffer
        """
        self.state.clear()
        self.next_state.clear()
        self.ptr = 0
        self.size = 0

    def save_to_file(self, path):
        """ TODO for imitation learning
        """
        pass

    def load_from_file(self, path):
        """ TODO for imitation learning
        """
        pass

    def state_dict(self):
        pass
    
    def load_state_dict(self, state_dict):
        pass

class OffPolicyLazyReplayBuffer(object):
    """ Replay buffer to store transitions. Implemented in a "ring-buffer"
    fashion.

    If the data to store in the replay buffer gets very large, a simple replay
    buffer will have a limited size to be able to fit in memory. This class
    allows to have a big replay buffer by storing the data on disk and only
    loading it when needed.

    There are some optimizations behind the scenes such as pre-fetching the
    next data that will be sampled to mitigate the disk I/O bottleneck.
    """

    def __init__(self, state_dim: StateShape, action_dim: int,
                 max_size=int(1e6)):
        print("Creating replay buffer with shape: ", (max_size, *state_dim.neighborhood_common_shape))
        self.size = 0
        self.ptr = 0
        self.device = device
        self.max_size = int(max_size)

        nb_readers = nb_prefetch = 10
        self.state_manager = NaiveLazyTensorManager(max_size,
                                       state_dim,
                                       batch_size=4096,
                                       nb_prefetch=nb_prefetch,
                                       nb_readers=nb_readers)

        self.action = torch.zeros((max_size, action_dim), dtype=rb_type)
        self.reward = torch.zeros((max_size, 1), dtype=rb_type)
        self.not_done = torch.zeros((max_size, 1), dtype=rb_type)

        self.is_in_writing_mode = True
        self.state_manager.enter_write_mode()

    def enter_read_mode(self):
        # Upon repeated calls, just ignore.
        if self.is_in_writing_mode:
            self.is_in_writing_mode = False
            self.state_manager.enter_read_mode(self.size)

    def enter_write_mode(self):
        # Upon repeated calls, just ignore.
        if not self.is_in_writing_mode:
            self.is_in_writing_mode = True
            self.state_manager.enter_write_mode()

    def add(self, state: State, action, next_state: State, reward, done):
        if not self.is_in_writing_mode:
            raise RuntimeError("The buffer is not in writing mode. Call writing_mode first.")

        indices = (np.arange(0, len(state)) + self.ptr) % self.max_size

        self.action[indices] = action
        self.reward[indices] = reward
        self.not_done[indices] = 1. - done

        # This might be a bit slower.
        self.state_manager.add(state, next_state, indices)

        self.ptr = (self.ptr + len(indices)) % self.max_size
        self.size = min(self.size + len(indices), self.max_size)

    def __len__(self):
        return self.size

    def sample(self, batch_size=4096):
        if self.is_in_writing_mode:
            raise RuntimeError("The buffer is in writing mode. Call reading_mode first.")

        s, ns, ind = self.state_manager.get_next_batch() # This will block until the data is ready

        ind = torch.from_numpy(ind)

        a = self.action.index_select(0, ind)
        r = self.reward.index_select(0, ind).squeeze(-1)
        d = self.not_done.index_select(0, ind).to(
            dtype=torch.float32).squeeze(-1)
        
        if get_device_str() == "cuda":
            s = s.pin_memory()
            a = a.pin_memory()
            ns = ns.pin_memory()
            r = r.pin_memory()
            d = d.pin_memory()

        # Return tensors on the same device as the buffer in pinned memory
        return (s.to(device=self.device, non_blocking=True),
                a.to(device=self.device, non_blocking=True),
                ns.to(device=self.device, non_blocking=True),
                r.to(device=self.device, non_blocking=True),
                d.to(device=self.device, non_blocking=True))

    def clear_memory(self):
        """ Reset the buffer
        """
        self.state.clear()
        self.next_state.clear()
        self.ptr = 0
        self.size = 0

    def save_to_file(self, path):
        """ TODO for imitation learning
        """
        pass

    def load_from_file(self, path):
        """ TODO for imitation learning
        """
        pass

    def state_dict(self):
        pass
    
    def load_state_dict(self, state_dict):
        pass

class OffPolicyReplayBuffer(object):
    """ Replay buffer to store transitions. Implemented in a "ring-buffer"
    fashion. Efficiency could probably be improved

    TODO: Add possibility to save and load to disk for imitation learning
    """

    def __init__(
        self, state_dim: StateShape, action_dim: int, max_size=int(1e6)
    ):
        """
        Parameters:
        -----------
        state_dim: int
            Size of states
        action_dim: int
            Size of actions
        max_size: int
            Number of transitions to store
        """
        self.device = device
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        # Buffers "filled with zeros"
        self.state = State.zeros(
            (self.max_size, *state_dim.neighborhood_common_shape), prev_dirs_size=state_dim.prev_dirs_size, dtype=rb_type)
        self.action = torch.zeros(
            (self.max_size, action_dim), dtype=rb_type)
        self.next_state = State.zeros(
            (self.max_size, *state_dim.neighborhood_common_shape), prev_dirs_size=state_dim.prev_dirs_size, dtype=rb_type)
        self.reward = torch.zeros(
            (self.max_size, 1), dtype=rb_type)
        self.not_done = torch.zeros(
            (self.max_size, 1), dtype=rb_type)
    
    def _pin_to_memory(self):
        if get_device_str() == "cuda":
            self.state = self.state.pin_memory()
            self.action = self.action.pin_memory()
            self.next_state = self.next_state.pin_memory()
            self.reward = self.reward.pin_memory()
            self.not_done = self.not_done.pin_memory()

    def enter_write_mode(self, *args, **kwargs):
        pass

    def enter_read_mode(self, *args, **kwargs):
        pass

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray
    ):
        """ Add new transitions to buffer in a "ring buffer" way

        Parameters:
        -----------
        state: np.ndarray
            Batch of states to be added to buffer
        action: np.ndarray
            Batch of actions to be added to buffer
        next_state: np.ndarray
            Batch of next-states to be added to buffer
        reward: np.ndarray
            Batch of rewards obtained for this transition
        done: np.ndarray
            Batch of "done" flags for this batch of transitions
        """

        ind = (np.arange(0, len(state)) + self.ptr) % self.max_size

        self.state[ind] = state
        self.action[ind] = action
        self.next_state[ind] = next_state
        self.reward[ind] = reward
        self.not_done[ind] = 1. - done

        self.ptr = (self.ptr + len(ind)) % self.max_size
        self.size = min(self.size + len(ind), self.max_size)

    def __len__(self):
        return self.size

    def sample(
        self,
        batch_size=4096
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """ Off-policy sampling. Will sample min(batch_size, self.size)
        transitions in an unordered way.

        Parameters:
        -----------
        batch_size: int
            Number of transitions to sample

        Returns:
        --------
        s: torch.Tensor
            Sampled states
        a: torch.Tensor
            Sampled actions
        ns: torch.Tensor
            Sampled s'
        r: torch.Tensor
            Sampled non-discounted rewards
        d: torch.Tensor
            Sampled 1-done flags
        """
        ind = torch.randperm(self.size, dtype=torch.long)[
            :min(self.size, batch_size)]

        s = self.state.index_select(0, ind)
        a = self.action.index_select(0, ind)
        ns = self.next_state.index_select(0, ind)
        r = self.reward.index_select(0, ind).squeeze(-1)
        d = self.not_done.index_select(0, ind).to(
            dtype=torch.float32).squeeze(-1)
        
        if get_device_str() == "cuda":
            s = s.pin_memory()
            a = a.pin_memory()
            ns = ns.pin_memory()
            r = r.pin_memory()
            d = d.pin_memory()

        # Return tensors on the same device as the buffer in pinned memory
        return (s.to(device=self.device, non_blocking=True),
                a.to(device=self.device, non_blocking=True),
                ns.to(device=self.device, non_blocking=True),
                r.to(device=self.device, non_blocking=True),
                d.to(device=self.device, non_blocking=True))

    def clear_memory(self):
        """ Reset the buffer
        """
        self.ptr = 0
        self.size = 0

    def save_to_file(self, path):
        """ TODO for imitation learning
        """
        pass

    def load_from_file(self, path):
        """ TODO for imitation learning
        """
        pass

    def state_dict(self):
        size = self.size
        return {
            "state": self.state[:size],
            "action": self.action[:size],
            "next_state": self.next_state[:size],
            "reward": self.reward[:size],
            "not_done": self.not_done[:size],
            "ptr": self.ptr,
            "size": self.size
        }
    
    def load_state_dict(self, state_dict):
        self.size = state_dict["size"]
        self.ptr = state_dict["ptr"]

        self.state[:self.size] = state_dict["state"]
        self.action[:self.size] = state_dict["action"]
        self.next_state[:self.size] = state_dict["next_state"]
        self.reward[:self.size] = state_dict["reward"]
        self.not_done[:self.size] = state_dict["not_done"]

class OnPolicyReplayBuffer(object):
    """ Replay buffer to store transitions. Efficiency could probably be
    improved.

    While it is called a ReplayBuffer, it is not actually one as no "Replay"
    is performed. As it is used by on-policy algorithms, the buffer should
    be cleared every time it is sampled.

    TODO: Add possibility to save and load to disk for imitation learning
    """

    def __init__(
        self, state_dim: int, action_dim: int, n_trajectories: int,
        max_traj_length: int, gamma: float, lmbda: float = 0.95
    ):
        """
        Parameters:
        -----------
        state_dim: int
            Size of states
        action_dim: int
            Size of actions
        n_trajectories: int
            Number of learned accumulating transitions
        max_traj_length: int
            Maximum length of trajectories
        gamma: float
            Discount factor.
        lmbda: float
            GAE factor.
        """
        self.ptr = 0

        self.n_trajectories = n_trajectories
        self.max_traj_length = max_traj_length
        self.device = device
        self.lens = np.zeros((n_trajectories), dtype=np.int32)
        self.gamma = gamma
        self.lmbda = lmbda
        self.state_dim = state_dim
        self.action_dim = action_dim

        # RL Buffers "filled with zeros"
        self.state = np.zeros((
            self.n_trajectories, self.max_traj_length, self.state_dim))
        self.action = np.zeros((
            self.n_trajectories, self.max_traj_length, self.action_dim))
        self.next_state = np.zeros((
            self.n_trajectories, self.max_traj_length, self.state_dim))
        self.reward = np.zeros((self.n_trajectories, self.max_traj_length))
        self.not_done = np.zeros((self.n_trajectories, self.max_traj_length))
        self.values = np.zeros((self.n_trajectories, self.max_traj_length))
        self.next_values = np.zeros(
            (self.n_trajectories, self.max_traj_length))
        self.probs = np.zeros((self.n_trajectories, self.max_traj_length))

        # GAE buffers
        self.ret = np.zeros((self.n_trajectories, self.max_traj_length))
        self.adv = np.zeros((self.n_trajectories, self.max_traj_length))

    def add(
        self,
        ind: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        probs: np.ndarray
    ):
        """ Add new transitions to buffer in a "ring buffer" way

        Parameters:
        -----------
        state: np.ndarray
            Batch of states to be added to buffer
        action: np.ndarray
            Batch of actions to be added to buffer
        next_state: np.ndarray
            Batch of next-states to be added to buffer
        reward: np.ndarray
            Batch of rewards obtained for this transition
        done: np.ndarray
            Batch of "done" flags for this batch of transitions
        values: np.ndarray
            Batch of "old" value estimates for this batch of transitions
        next_values : np.ndarray
            Batch of "old" value-primes for this batch of transitions
        probs: np.ndarray
            Batch of "old" log-probs for this batch of transitions

        """
        self.state[ind, self.ptr] = state
        self.action[ind, self.ptr] = action

        # These are actually not needed
        self.next_state[ind, self.ptr] = next_state
        self.reward[ind, self.ptr] = reward
        self.not_done[ind, self.ptr] = (1. - done)

        # Values for losses
        self.values[ind, self.ptr] = values
        self.next_values[ind, self.ptr] = next_values
        self.probs[ind, self.ptr] = probs

        self.lens[ind] += 1

        for j in range(len(ind)):
            i = ind[j]

            if done[j]:
                # Calculate the expected returns: the value function target
                rew = self.reward[i, :self.ptr]
                # rew = (rew - rew.mean()) / (rew.std() + 1.e-8)
                self.ret[i, :self.ptr] = \
                    self.discount_cumsum(
                        rew, self.gamma)

                # Calculate GAE-Lambda with this trick
                # https://stackoverflow.com/a/47971187
                # TODO: make sure that this is actually correct
                # TODO?: do it the usual way with a backwards loop
                deltas = rew + \
                    (self.gamma * self.next_values[i, :self.ptr] *
                     self.not_done[i, :self.ptr]) - \
                    self.values[i, :self.ptr]

                if self.lmbda == 0:
                    self.adv[i, :self.ptr] = self.ret[i, :self.ptr] - \
                        self.values[i, :self.ptr]
                else:
                    self.adv[i, :self.ptr] = \
                        self.discount_cumsum(deltas, self.gamma * self.lmbda)

        self.ptr += 1

    def discount_cumsum(self, x, discount):
        """
        # Taken from spinup implementation
        magic from rllab for computing discounted cumulative sums of vectors.
        input:
                vector x,
                [x0,
                 x1,
                 x2]
        output:
                [x0 + discount * x1 + discount^2 * x2,
                 x1 + discount * x2,
                 x2]
        """
        return scipy.signal.lfilter(
            [1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def sample(
        self,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """ Sample all transitions.

        Parameters:
        -----------

        Returns:
        --------
        s: torch.Tensor
            Sampled states
        a: torch.Tensor
            Sampled actions
        ret: torch.Tensor
            Sampled return estimate, target for V
        adv: torch.Tensor
            Sampled advantges, factor for policy update
        probs: torch.Tensor
            Sampled old action probabilities
        """
        # TODO?: Not sample whole buffer ? Have M <= N*T ?

        # Generate indices
        row, col = zip(*((i, le)
                         for i in range(len(self.lens))
                         for le in range(self.lens[i])))

        s, a, ret, adv, probs, vals = (
            self.state[row, col], self.action[row, col], self.ret[row, col],
            self.adv[row, col], self.probs[row, col], self.values[row, col])

        # Normalize advantage. Needed ?
        # Trick used by OpenAI in their PPO impl
        # adv = (adv - adv.mean()) / (adv.std() + 1.e-8)

        shuf_ind = np.arange(s.shape[0])

        # Shuffling makes the learner unable to track in "two directions".
        # Why ?
        # np.random.shuffle(shuf_ind)

        self.clear_memory()

        return (s[shuf_ind], a[shuf_ind], ret[shuf_ind], adv[shuf_ind],
                probs[shuf_ind], vals[shuf_ind])

    def clear_memory(self):
        """ Reset the buffer
        """

        self.lens = np.zeros((self.n_trajectories), dtype=np.int32)
        self.ptr = 0

        # RL Buffers "filled with zeros"
        # TODO: Is that actually needed ? Can't just set self.ptr to 0 ?
        self.state = np.zeros((
            self.n_trajectories, self.max_traj_length, self.state_dim))
        self.action = np.zeros((
            self.n_trajectories, self.max_traj_length, self.action_dim))
        self.next_state = np.zeros((
            self.n_trajectories, self.max_traj_length, self.state_dim))
        self.reward = np.zeros((self.n_trajectories, self.max_traj_length))
        self.not_done = np.zeros((self.n_trajectories, self.max_traj_length))
        self.values = np.zeros((self.n_trajectories, self.max_traj_length))
        self.next_values = np.zeros(
            (self.n_trajectories, self.max_traj_length))
        self.probs = np.zeros((self.n_trajectories, self.max_traj_length))

        # GAE buffers
        self.ret = np.zeros((self.n_trajectories, self.max_traj_length))
        self.adv = np.zeros((self.n_trajectories, self.max_traj_length))

    def __len__(self):
        return np.sum(self.lens)

    def save_to_file(self, path):
        """ TODO for imitation learning
        """
        pass

    def load_from_file(self, path):
        """ TODO for imitation learning
        """
        pass

class RlhfReplayBuffer(object):
    """ Replay buffer to store transitions. Efficiency could probably be
    improved.

    While it is called a ReplayBuffer, it is not actually one as no "Replay"
    is performed. As it is used by on-policy algorithms, the buffer should
    be cleared every time it is sampled.
    """

    def __init__(
        self, state_dim: int, action_dim: int, n_trajectories: int,
        max_traj_length: int, gamma: float, lmbda: float = 0.95
    ):
        """
        Parameters:
        -----------
        state_dim: int
            Size of states
        action_dim: int
            Size of actions
        n_trajectories: int
            Number of learned accumulating transitions
        max_traj_length: int
            Maximum length of trajectories
        gamma: float
            Discount factor.
        lmbda: float
            GAE factor.
        """
        self.ptr = 0

        self.n_trajectories = n_trajectories
        self.max_traj_length = max_traj_length
        self.device = device
        self.storing_device = 'cpu'
        self.gamma = gamma
        self.lmbda = lmbda
        self.state_dim = state_dim
        self.action_dim = action_dim

        # # RL Buffers "filled with zeros"
        # self.state = np.zeros((
        #     self.n_trajectories, self.max_traj_length, self.state_dim))
        # self.streamlines = np.zeros((
        #     self.n_trajectories, self.max_traj_length, 128, 3))
        # self.action = np.zeros((
        #     self.n_trajectories, self.max_traj_length, self.action_dim))
        # self.next_state = np.zeros((
        #     self.n_trajectories, self.max_traj_length, self.state_dim))
        # self.reward = np.zeros((self.n_trajectories, self.max_traj_length))
        # self.not_done = np.zeros((self.n_trajectories, self.max_traj_length))
        # self.values = np.zeros((self.n_trajectories, self.max_traj_length))
        # self.next_values = np.zeros(
        #     (self.n_trajectories, self.max_traj_length))
        # self.probs = np.zeros((self.n_trajectories, self.max_traj_length))
        # self.lens = np.zeros((self.n_trajectories,), dtype=np.int32)

        # # # GAE buffers
        # self.ret = np.zeros((self.n_trajectories, self.max_traj_length))
        # self.adv = np.zeros((self.n_trajectories, self.max_traj_length))

        # RL Buffers "filled with zeros"
        self.state = torch.zeros((self.n_trajectories, self.max_traj_length, self.state_dim), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.streamlines = torch.zeros((self.n_trajectories, self.max_traj_length, 128, 3), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.action = torch.zeros((self.n_trajectories, self.max_traj_length, self.action_dim), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.next_state = torch.zeros((self.n_trajectories, self.max_traj_length, self.state_dim), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.reward = torch.zeros((self.n_trajectories, self.max_traj_length), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.not_done = torch.zeros((self.n_trajectories, self.max_traj_length), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.values = torch.zeros((self.n_trajectories, self.max_traj_length), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.next_values = torch.zeros((self.n_trajectories, self.max_traj_length), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.probs = torch.zeros((self.n_trajectories, self.max_traj_length), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.lens = torch.zeros((self.n_trajectories,), dtype=torch.int32, device=self.storing_device, requires_grad=False)

        # # GAE buffers
        self.ret = torch.zeros((self.n_trajectories, self.max_traj_length), dtype=torch.float32, device=self.storing_device)
        self.adv = torch.zeros((self.n_trajectories, self.max_traj_length), dtype=torch.float32, device=self.storing_device)
        
        if get_device_str() == "cuda":
            self.state = self.state.pin_memory()
            self.streamlines = self.streamlines.pin_memory()
            self.action = self.action.pin_memory()
            self.next_state = self.next_state.pin_memory()
            self.reward = self.reward.pin_memory()
            self.not_done = self.not_done.pin_memory()
            self.values = self.values.pin_memory()
            self.next_values = self.next_values.pin_memory()
            self.probs = self.probs.pin_memory()
            self.lens = self.lens.pin_memory()

    def add(
        self,
        ind: np.ndarray,
        state: np.ndarray,
        streamlines: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        probs: np.ndarray
    ):
        """ Add new transitions to buffer in a "ring buffer" way

        Parameters:
        -----------
        state: np.ndarray
            Batch of states to be added to buffer
        streamlines: np.ndarray
            Batch of streamlines to be added to buffer
        action: np.ndarray
            Batch of actions to be added to buffer
        next_state: np.ndarray
            Batch of next-states to be added to buffer
        reward: np.ndarray
            Batch of rewards obtained for this transition
        done: np.ndarray
            Batch of "done" flags for this batch of transitions
        values: np.ndarray
            Batch of "old" value estimates for this batch of transitions
        next_values : np.ndarray
            Batch of "old" value-primes for this batch of transitions
        probs: np.ndarray
            Batch of "old" log-probs for this batch of transitions

        """

        break_if_found_nans_args(state, streamlines, action, next_state, reward, done, values, next_values, probs)

        self.state[ind, self.ptr] = state
        self.streamlines[ind, self.ptr] = streamlines
        self.action[ind, self.ptr] = action

        # These are actually not needed
        self.next_state[ind, self.ptr] = next_state
        self.reward[ind, self.ptr] = reward
        self.not_done[ind, self.ptr] = (1. - done)

        # Values for losses
        self.values[ind, self.ptr] = values
        self.next_values[ind, self.ptr] = next_values
        self.probs[ind, self.ptr] = probs

        self.lens[ind] += 1

        self._compute_adv_rets(ind, done)
        break_if_found_nans_args(self.ret, self.adv)

        self.ptr += 1

    def _compute_adv_rets(self, ind, done):
        for j in range(len(ind)):

            i = ind[j]
            if done[j]:
                # Calculate the expected returns: the value function target
                rew = self.reward[i, :self.ptr]
                self.ret[i, :self.ptr] = \
                    self.discount_cumsum(
                        rew, self.gamma)

                # Calculate GAE-Lambda with this trick
                # https://stackoverflow.com/a/47971187
                # TODO: make sure that this is actually correct
                # TODO?: do it the usual way with a backwards loop
                deltas = rew + \
                    (self.gamma * self.next_values[i, :self.ptr] * self.not_done[i, :self.ptr]) - self.values[i, :self.ptr]

                if self.lmbda == 0:
                    self.adv[i, :self.ptr] = self.ret[i, :self.ptr] - self.values[i, :self.ptr]
                else:
                    self.adv[i, :self.ptr] = self.discount_cumsum(deltas, self.gamma * self.lmbda)

    def discount_cumsum(self, x, discount):
        """
        # Taken from spinup implementation
        magic from rllab for computing discounted cumulative sums of vectors.
        input:
                vector x,
                [x0,
                 x1,
                 x2]
        output:
                [x0 + discount * x1 + discount^2 * x2,
                 x1 + discount * x2,
                 x2]
        """
        #if isinstance(x, torch.Tensor):
        #    return disc_cumsum(x, discount)
        #else:
        return scipy.signal.lfilter(
            [1], [1, float(-discount)], x[::-1], axis=0)[::-1]
            

    def sample(
        self,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """ Sample all transitions.

        Parameters:
        -----------

        Returns:
        --------
        s: torch.Tensor
            Sampled states
        a: torch.Tensor
            Sampled actions
        ret: torch.Tensor
            Sampled return estimate, target for V
        adv: torch.Tensor
            Sampled advantges, factor for policy update
        probs: torch.Tensor
            Sampled old action probabilities
        """
        # TODO?: Not sample whole buffer ? Have M <= N*T ?

        # Generate indices
        row, col = zip(*((i, le)
                         for i in range(len(self.lens))
                         for le in range(self.lens[i])))
        
        s, st, a, ret, adv, probs, vals = \
            (self.state[row, col],
             self.streamlines[row, col],
             self.action[row, col],
             self.ret[row, col],
             self.adv[row, col],
             self.probs[row, col],
             self.values[row, col])

        if get_device_str() == "cuda":
            s = s.pin_memory()
            st = st.pin_memory()
            a = a.pin_memory()
            ret = ret.pin_memory()
            adv = adv.pin_memory()
            probs = probs.pin_memory()
            vals = vals.pin_memory()

        # Essential for on-policy algorithms.
        self.clear_memory()

        return (s.to(device=self.device, non_blocking=True),
                st.to(device=self.device, non_blocking=True),
                a.to(device=self.device, non_blocking=True),
                ret.to(device=self.device, non_blocking=True),
                adv.to(device=self.device, non_blocking=True),
                probs.to(device=self.device, non_blocking=True),
                vals.to(device=self.device, non_blocking=True))

    def clear_memory(self):
        """ Reset the buffer
        """

        self.ptr = 0

        # RL Buffers "filled with zeros"
        # TODO: Is that actually needed ? Can't just set self.ptr to 0 ?
        self.state = torch.zeros((
            self.n_trajectories, self.max_traj_length, self.state_dim), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.streamlines = torch.zeros((
            self.n_trajectories, self.max_traj_length, 128, 3), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.action = torch.zeros((
            self.n_trajectories, self.max_traj_length, self.action_dim), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.next_state = torch.zeros((
            self.n_trajectories, self.max_traj_length, self.state_dim), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.reward = torch.zeros((
            self.n_trajectories, self.max_traj_length), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.not_done = torch.zeros((
            self.n_trajectories, self.max_traj_length), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.values = torch.zeros((
            self.n_trajectories, self.max_traj_length), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.next_values = torch.zeros(
            (self.n_trajectories, self.max_traj_length), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.probs = torch.zeros((
            self.n_trajectories, self.max_traj_length), dtype=torch.float32, device=self.storing_device, requires_grad=False)
        self.lens = torch.zeros((
            self.n_trajectories), dtype=torch.int32, device=self.storing_device, requires_grad=False)

        # GAE buffers
        self.ret = torch.zeros((
            self.n_trajectories, self.max_traj_length), dtype=torch.float32, device=self.storing_device)
        self.adv = torch.zeros((
            self.n_trajectories, self.max_traj_length), dtype=torch.float32, device=self.storing_device)

    def __len__(self):
        return torch.sum(self.lens)

    def save_to_file(self, path):
        """ TODO for imitation learning
        """
        pass

    def load_from_file(self, path):
        """ TODO for imitation learning
        """
        pass

