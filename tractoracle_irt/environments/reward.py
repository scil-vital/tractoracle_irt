import numpy as np

class Reward(object):

    """ Abstract function that all "rewards" must implement.
    """

    def __call__(
        self,
        streamlines: np.ndarray,
        dones: np.ndarray
    ):
        self.name = "Undefined"

        assert False, "Not implemented"

    def reset(self):
        """ Most reward factors do not need to be reset.
        """
        pass

    def change_subject(
        self,
        subject_id: str,
        min_nb_steps: int,
        reference: str,
        affine_vox2rasmm: np.ndarray,
        peaks
    ):
        """ Most reward factors do not need to change subject.
        """
        raise NotImplementedError(
            "This reward factor does not support changing subject."
        )


class RewardFunction():

    """ Compute the reward function as the sum of its weighted factors.
    Each factor may reward streamlines "densely" (i.e. at every step) or
    "sparsely" (i.e. once per streamline).

    """

    def __init__(
        self,
        factors,
        weights,
    ):
        """
        """
        assert len(factors) == len(weights)

        self.factors = factors
        self.weights = weights

        self.F = len(self.factors)

    def change_subject(
        self,
        subject_id: str,
        min_nb_steps: int,
        reference: str,
        affine_vox2rasmm: np.ndarray,
        peaks
    ):
        """
        Change the subject of the oracle.
        """
        for f in self.factors:
            f.change_subject(subject_id, min_nb_steps, reference, affine_vox2rasmm, peaks)

    def __call__(self, streamlines, dones):
        """
        Each reward component is weighted according to a
        coefficient and then summed.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        dones: `numpy.ndarray` of shape (n_streamlines)
            Whether tracking is over for each streamline or not.
        penalty: float
            Penalty to apply to the reward. This is useful in the case of RLHF using PPO for example
            where the reward is modified to penalize with a KL_penalty. This parameter should be the
            final value of the penalty (positive value = penalize, negative value = increase reward).

        Returns
        -------
        rewards: np.ndarray of floats
            Reward components weighted by their coefficients as well
            as the penalites
        """

        N = len(streamlines)

        rewards_factors = np.zeros((self.F, N))

        for i, (w, f) in enumerate(zip(self.weights, self.factors)):
            if w > 0:
                rewards_factors[i] = w * f(streamlines, dones)

        info = {}
        for i, f in enumerate(self.factors):
            info[f.name] = np.mean(rewards_factors[i])

        reward = np.sum(rewards_factors, axis=0)

        return reward, info

    def reset(self):
        """
        """

        for f in self.factors:
            f.reset()
