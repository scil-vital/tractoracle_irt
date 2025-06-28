import nibabel as nib
import numpy as np

from dipy.io.stateful_tractogram import StatefulTractogram, Space, Tractogram

from tractoracle_irt.environments.reward import Reward
from tractoracle_irt.environments.utils import fix_streamlines_length
from tractoracle_irt.oracles.oracle import OracleSingleton


class OracleReward(Reward):

    """ Reward streamlines based on the predicted scores of an "Oracle".
    A binary reward is given by the oracle at the end of tracking.
    """

    def __init__(
        self,
        checkpoint: str,
        min_nb_steps: int,
        reference: nib.Nifti1Image,
        affine_vox2rasmm: np.ndarray,
        device: str,
        proportional_reward: bool = False,
        reward_valid_threshold: float = 0.5,
    ):
        # Name for stats
        self.name = 'oracle_reward'
        # Minimum number of steps before giving reward
        # Only useful for 'sparse' reward
        self.min_nb_steps = min_nb_steps
        self.proportional_reward = proportional_reward
        self.reward_valid_threshold = reward_valid_threshold
        # Checkpoint of the oracle, which contains weights and hyperparams.
        if checkpoint:
            self.checkpoint = checkpoint
            # The oracle is declared as a singleton to prevent loading the
            # weights in memory multiple times.
            self.model = OracleSingleton(checkpoint, device)
        else:
            self.checkpoint = None

        self.reference = reference
        self.affine_vox2rasmm = affine_vox2rasmm

        # Reference anat
        self.device = device

    def change_subject(
        self,
        subject_id: str,
        min_nb_steps: int,
        reference: nib.Nifti1Image,
        affine_vox2rasmm: np.ndarray,
        peaks
    ):
        """
        Change the subject of the oracle.
        """
        self.subject_id = subject_id
        self.reference = reference
        self.affine_vox2rasmm = affine_vox2rasmm
        self.min_nb_steps = min_nb_steps

    def reward(self, streamlines, dones):
        """
        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space

        Returns
        -------
        rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
            Array containing the reward
        """
        if not self.checkpoint:
            return None
        N = dones.shape[0]
        reward = np.zeros((N))
        #print("reward streamlines shape: ", streamlines.shape)
        predictions = self.model.predict(streamlines)
        # Double indexing to get the indexes. Don't forget you
        # can't assign using double indexes as the first indexing
        # will return a copy of the array.
        idx = np.arange(N)[dones]
        # Assign the reward using the precomputed double indexes.
        if self.proportional_reward:
            reward[idx] = predictions
        else:
            idx = idx[predictions > self.reward_valid_threshold]
            reward[idx] = 1.0
        return reward

    def __call__(
        self,
        streamlines: np.ndarray,
        dones: np.ndarray,
    ):

        N, L, P = streamlines.shape
        if L > self.min_nb_steps and sum(dones.astype(int)) > 0:

            # Change ref of streamlines. This is weird on the ISMRM2015
            # dataset as the diff and anat are not in the same space,
            # but it should be fine on other datasets.
            done_streamlines = streamlines.copy()[dones]
            tractogram = Tractogram(
                streamlines=done_streamlines)
            tractogram.apply_affine(self.affine_vox2rasmm)
            sft = StatefulTractogram(
                streamlines=tractogram.streamlines,
                reference=self.reference,
                space=Space.RASMM)
            sft.to_vox()
            sft.to_corner()

            return self.reward(sft.streamlines, dones)
        return np.zeros((N))
