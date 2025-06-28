import numpy as np
import torch
from tqdm import tqdm

from tractoracle_irt.environments.env import BaseEnv
from tractoracle_irt.utils.torch_utils import get_device
from tractoracle_irt.utils.utils import prettier_dict

class RLAlgorithm(object):
    """
    Abstract sample-gathering and training algorithm.
    """

    def __init__(
        self,
        input_size: int,
        action_size: int = 3,
        hidden_size: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 10000,
        rng: np.random.RandomState = None,
        device: torch.device = get_device(),
    ):
        """
        Parameters
        ----------
        input_size: int
            Input size for the model
        action_size: int
            Output size for the actor
        hidden_size: int
            Width of the NN
        action_std: float
            Starting standard deviation on actions for exploration
        lr: float
            Learning rate for optimizer
        gamma: float
            Gamma parameter future reward discounting
        batch_size: int
            Batch size for replay buffer sampling
        rng: np.random.RandomState
            rng for randomness. Should be fixed with a seed
        device: torch.device,
            Device to use for processing (CPU or GPU)
            Should always on GPU
        """

        self.max_action = 1.
        self.t = 1

        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size

        self.rng = rng

    def validation_episode(
        self,
        initial_state,
        env: BaseEnv,
        prob: float = 1.,
        enable_pbar: bool = True,
        compute_reward=True
    ):
        """
        Main loop for the algorithm
        From a starting state, run the model until the env. says its done

        Parameters
        ----------
        initial_state: np.ndarray
            Initial state of the environment
        env: BaseEnv
            The environment actions are applied on. Provides the state fed to
            the RL algorithm

        Returns
        -------
        tractogram: Tractogram
            Tractogram containing the tracked streamline
        running_reward: float
            Cummulative training steps reward
        """

        running_reward = 0
        state = initial_state
        n_streamlines = state.shape.nb_streamlines
        done = False
        current_nb_dones = 0
        pbar = tqdm(total=n_streamlines, desc='Val episode',
                    leave=False, disable=not enable_pbar)
        if hasattr(env, 'rollout_env') and env.rollout_env:
            env.rollout_env.rollout_stats.reset()
        step = 0
        while not np.all(done):
            # Select action according to policy + noise to make tracking
            # probabilistic
            with torch.no_grad():
                action = self.agent.select_action(state, probabilistic=prob)

            # Perform action
            if isinstance(action, torch.Tensor):
                action = action.to(device='cpu', copy=True).numpy()

            next_state, _, reward, done, *_ = env.step(
                action)
            
            new_nb_dones = np.sum(done)
            
            # Make sure we don't have more done streamlines than we started
            # with. That should never happen and could happen if the indices
            # for continuing streamlines are not correctly handled in the 
            # environment. That would lead the tqdm progress bar to break and
            # "overflow" to higher than 100%.
            current_nb_dones += new_nb_dones
            assert current_nb_dones <= n_streamlines

            pbar.update(new_nb_dones)

            if hasattr(env, 'rollout_env') and env.rollout_env:
                stats = env.rollout_env.rollout_stats.get_stats(reduce='mean')
                # Make sure that each value of the stats is exactly two decimals long
                # stats = {k: f'{v:.2f}' for k, v in stats.items()}
                for k, v in stats.items():
                    if isinstance(v, float) and not v.is_integer():
                        stats[k] = f'{v:.2f}'
                stats.update({'step': step})
                pbar.set_postfix(stats)
            else:
                pbar.set_postfix({'step': step})

            # Keep track of reward
            if compute_reward:
                running_reward += sum(reward)

            # "Harvesting" here means removing "done" trajectories
            # from state. This line also set the next_state as the
            # state
            state, _, _ = env.harvest()
            step += 1

        # env.render()

        if hasattr(env.rollout_env, 'utility_tracker') \
            and env.rollout_env.utility_tracker:

            stats = env.rollout_env.utility_tracker.get_stats()
            print(prettier_dict(stats, 'Rollout utility stats'))
            env.rollout_env.utility_tracker.reset()

        return running_reward
