import numpy as np

from collections import defaultdict
from nibabel.streamlines import TrkFile
from tqdm import tqdm
from typing import Tuple

from dipy.tracking.streamlinespeed import compress_streamlines, length
from nibabel.streamlines import Tractogram
from nibabel.streamlines.tractogram import LazyTractogram
from nibabel.streamlines.tractogram import TractogramItem

from tractoracle_irt.algorithms.rl import RLAlgorithm
from tractoracle_irt.algorithms.shared.utils import add_to_means
from tractoracle_irt.environments.env import BaseEnv


class Tracker(object):
    """ Tracking class similar to scilpy's or dwi_ml's. This class is
    responsible for generating streamlines, as well as giving back training
    or RL-associated metrics if applicable.
    """

    def __init__(
        self,
        alg: RLAlgorithm,
        n_actor: int,
        prob: float = 0.,
        compress: float = 0.0,
        min_length: float = 20,
        max_length: float = 200,
        save_seeds: bool = False
    ):
        """

        Parameters
        ----------
        alg: RLAlgorithm
            Tracking agent.
        n_actor: int
            Number of actors to track at once.
        prob: float
            Factor to influence the output of the agent.
        compress: float
            Compression factor when saving streamlines.
        min_length: float
            Minimum length of a streamline.
        max_length: float
            Maximum length of a streamline.
        save_seeds: bool
            Save seeds in the tractogram.
        """

        self.alg = alg
        self.n_actor = n_actor
        self.prob = prob
        self.compress = compress
        self.min_length = min_length
        self.max_length = max_length
        self.save_seeds = save_seeds

    def track(
        self,
        env: BaseEnv,
        tracts_format
    ):
        """ Actual tracking function. Use this if you just want streamlines.

        Track with a generator to save streamlines to file
        as they are tracked. Used at tracking (test) time. No
        reward should be computed.

        Arguments
        ---------
        env : BaseEnv
            Environment to track in.
        tracts_format : TrkFile or TckFile
            Tractogram format.

        Returns:
        --------
        tractogram: Tractogram
            Tractogram in a generator format.

        """

        batch_size = self.n_actor

        self.alg.agent.eval()
        affine = env.affine_vox2rasmm

        # Shuffle seeds so that massive tractograms wont load "sequentially"
        # when partially displayed
        np.random.shuffle(env.seeds)

        def tracking_generator():
            # Presume iso vox
            vox_size = np.mean(
                np.abs(affine)[np.diag_indices(4)][:3])
            scaled_min_length = self.min_length / vox_size
            scaled_max_length = self.max_length / vox_size

            compress_th_vox = self.compress / vox_size

            # Track for every seed in the environment
            for start in tqdm(range(0, len(env.seeds), batch_size), desc="Tracking"):
                # Last batch might not be "full"
                end = min(start + batch_size, len(env.seeds))

                state, _ = env.reset(start, end)

                # Track forward
                self.alg.validation_episode(
                    state, env, self.prob, compute_reward=False)

                batch_tractogram = env.get_streamlines()

                for item in batch_tractogram:
                    streamline = item.streamline
                    if scaled_min_length <= length(streamline) \
                            <= scaled_max_length:

                        if self.compress:
                            streamline = compress_streamlines(
                                streamline, compress_th_vox)

                        if tracts_format is TrkFile:
                            streamline += 0.5
                            streamline *= vox_size
                        else:
                            # Streamlines are dumped in true world space with
                            # origin center as expected by .tck files.
                            streamline = np.dot(
                                streamline,
                                affine[:3, :3]) + \
                                affine[:3, 3]

                        flag = item.data_for_streamline['flags']
                        data_for_streamline = {
                            'flags': flag,
                        }

                        if self.save_seeds:
                            seed = item.data_for_streamline['seeds']
                            data_for_streamline['seeds'] = seed - 0.5

                        yield TractogramItem(
                            streamline, data_for_streamline, {})

        tractogram = LazyTractogram.from_data_func(tracking_generator)
        tractogram.affine_to_rasmm = affine

        return tractogram

    def track_and_train(
        self,
        env: BaseEnv,
    ) -> Tuple[Tractogram, float, float, float]:
        """
        Call the main training loop forward then backward.
        This can be considered an "epoch". Note that N=self.n_actor
        streamlines will be tracked instead of one streamline per seed.

        Parameters
        ----------
        env: BaseEnv
            Environment to track in.

        Returns
        -------
        train_tractogram: Tractogram
            Tractogram generated during training.
        mean_losses: dict
            Mean losses during training.
        reward: float
            Total reward obtained during training.
        mean_reward_factors: dict
            Reward separated into its components.
        """

        self.alg.agent.train()

        mean_losses = defaultdict(list)
        mean_reward_factors = defaultdict(list)

        # Fetch n=n_actor seeds
        state, _ = env.nreset(self.n_actor)

        # Track and train forward
        reward, losses, length, reward_factors, mean_ratio = \
            self.alg._episode(state, env)
        # Get the streamlines generated from forward training
        train_tractogram = env.get_streamlines()

        if len(losses.keys()) > 0:
            mean_losses = add_to_means(mean_losses, losses)
        # if len(reward_factors.keys()) > 0:
        #     mean_reward_factors = add_to_means(
        #         mean_reward_factors, reward_factors)

        return (
            train_tractogram,
            mean_losses,
            reward,
            mean_reward_factors,
            mean_ratio)

    def track_and_validate(
        self,
        env: BaseEnv,
        enable_pbar=False
    ) -> Tuple[Tractogram, float, dict]:
        """
        Run the tracking algorithm without training to see how it performs, but
        still compute the reward.

        Parameters
        ----------
        env: BaseEnv
            Environment to track in.

        Returns:
        --------
        tractogram: Tractogram
            Validation tractogram.
        cummulative_reward: float
            Total reward obtained during validation.
        """
        # Switch policy to eval mode so no gradients are computed
        self.alg.agent.eval()

        # Initialize tractogram
        tractogram = None

        # Reward gotten during validation
        cummulative_reward = 0

        def _generate_streamlines_and_rewards():

            # Track for every seed in the environment
            for i, start in enumerate(
                    range(0, len(env.seeds), self.n_actor)):

                # Last batch might not be "full"
                end = min(start + self.n_actor, len(env.seeds))

                state, _ = env.reset(start, end)

                # Track forward
                reward = self.alg.validation_episode(
                    state, env, self.prob, enable_pbar=False)

                batch_tractogram = env.get_streamlines()

                yield batch_tractogram, reward

        with tqdm(total=len(env.seeds), desc="Validation tracking", disable=not enable_pbar) as pbar:
            for t, r in _generate_streamlines_and_rewards():
                if tractogram is None and len(t) > 0:
                    tractogram = t
                elif len(t) > 0:
                    tractogram += t
                cummulative_reward += r
                pbar.update(len(t))

        return tractogram, cummulative_reward
