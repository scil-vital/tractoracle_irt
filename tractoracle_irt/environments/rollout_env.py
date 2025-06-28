import numpy as np
from typing import Callable

import torch
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
from dipy.tracking.streamline import set_number_of_points
from tractoracle_irt.algorithms.shared.offpolicy import ActorCritic
from tractoracle_irt.environments.stopping_criteria import (StoppingFlags, count_flags, is_flag_set)
from tractoracle_irt.oracles.transformer_oracle import LightningLikeModule
from fury import actor, window
from dipy.io.stateful_tractogram import Space, StatefulTractogram, Tractogram
from dipy.io.streamline import save_tractogram, load_tractogram

class RolloutStats(object):
    """
    The objective of this class is to store an average of different
    interesting metrics that can be used to monitor the performance
    of the rollout procedure.
    """
    def __init__(self):
        self.n_rollouts = None
        self.reset()

    def reset(self):
        self.stats = {
            "nr": self.n_rollouts,
            "perc_saved": [],
            "perc_reach_gm": [],
            "perc_not_saved": [],
            "rdist": []
        }

    def set_nb_rollouts(self, n_rollouts):
        self.n_rollouts = n_rollouts
        self.stats["nr"] = n_rollouts

    def update_step(self, **kwargs):
        self.stats["perc_saved"].append(kwargs.get("perc_saved", None))
        self.stats["perc_reach_gm"].append(kwargs.get("perc_reach_gm", None))
        self.stats["perc_not_saved"].append(kwargs.get("perc_not_saved", None))
        self.stats["rdist"].append(kwargs.get("rdist", None))
        self.stats["action_variance"] = kwargs.get("action_variance", None)

    def get_stats(self, reduce='mean'):
        # Round everything to 2 decimals
        if reduce == 'mean':
            reduced_stats = {k: np.round(np.mean(v), 6) for k, v in self.stats.items()}
        elif reduce == 'sum':
            reduced_stats = {k: np.round(np.sum(v), 6) for k, v in self.stats.items()}
        else:
            raise ValueError("Invalid reduce argument. Please use 'mean' or 'sum'.")

        return reduced_stats
    
class RolloutUtilityTracker(object):
    """
    The objective of this class is to track what's happening to streamlines
    that were backtracked. If we save a streamline, what happens to it in the
    following steps? Does it reach the target? Does it stop again? If so, why
    and how many times do we have to save this streamline?
    """
    def __init__(self, nb_actors: int):
        # This holds the number of times a streamlines was backtracked
        self.n_backtrack_streamline = np.zeros((nb_actors,), dtype=np.uint32)

        # This holds the last flag that stopped the streamline (if any)
        self.last_flag = np.zeros((nb_actors,), dtype=np.uint16)

    def reset(self):
        self.n_backtrack_streamline.fill(0)
        self.last_flag.fill(0)

    def update(self,
               backtracked_idx: np.ndarray,
               not_backtracked_idx: np.ndarray,
               flags: np.ndarray,
               ):
        """
        At the end of a rollout, we increment the number of times a streamline
        was rolled out according to the streamline_idx provided. We also
        update the flag associated to that streamline at the end of the
        rollout.

        Parameters
        ----------
        streamline_idx : np.ndarray
            Index of all the streamlines that were backtracked.
        flags : np.ndarray
            Flags associated to the streamlines that were backtracked.
        """
        self.n_backtrack_streamline[backtracked_idx] += 1
        self.last_flag[backtracked_idx] = flags

        # Streamlines that were not backtracked are streamlines
        # that reached the target. We set the flag to STOPPING_TARGET
        # only if the corresponding streamline was backtracked at least once
        backtracked_once_mask = \
            self.n_backtrack_streamline[not_backtracked_idx] > 0
        
        self.last_flag[not_backtracked_idx[backtracked_once_mask]] |= \
            StoppingFlags.STOPPING_TARGET.value

    def get_stats(self):
        """
        This function will compile the stats of the utility tracker and
        return them as a dictionary.

        Returns
        -------
        dict
            Dictionary containing the stats of the utility tracker.
        """
        backtracked_once_mask = self.n_backtrack_streamline > 0

        tot_n_backtracked = np.sum(backtracked_once_mask)
        tot_n_never_backtracked = np.sum(~backtracked_once_mask)
        assert tot_n_backtracked + tot_n_never_backtracked == self.n_backtrack_streamline.size
        
        tot_n_reached_gm_after = count_flags(self.last_flag, StoppingFlags.STOPPING_TARGET)
        
        avg_freq_of_backtracking = np.mean(self.n_backtrack_streamline[self.n_backtrack_streamline > 0])
        
        nb_streamlines_cant_be_saved = count_flags(self.last_flag[backtracked_once_mask],
                                                   StoppingFlags.STOPPING_TARGET,
                                                   equal=False)
        assert nb_streamlines_cant_be_saved + tot_n_reached_gm_after == np.sum(backtracked_once_mask)

        return {
            "n_streamlines": self.n_backtrack_streamline.size,
            "tot_n_backtracked": tot_n_backtracked,
            "tot_perc_backtracked": tot_n_backtracked / self.n_backtrack_streamline.size,

            "tot_n_never_backtracked": tot_n_never_backtracked,
            "tot_perc_never_backtracked": tot_n_never_backtracked / self.n_backtrack_streamline.size,

            "tot_n_reached_gm_after": tot_n_reached_gm_after,
            "tot_perc_reached_gm_after": tot_n_reached_gm_after / tot_n_backtracked,

            "tot_n_never_reached_gm_after": nb_streamlines_cant_be_saved,
            "tot_perc_never_reached_gm_after": nb_streamlines_cant_be_saved / tot_n_backtracked,

            "avg_freq_of_backtracking": avg_freq_of_backtracking,
        }
        


class RolloutEnvironment(object):

    def __init__(self,
                 reference: nib.Nifti1Image,
                 oracle: LightningLikeModule,
                 n_rollouts: int = 20,  # Nb of rollouts to try
                 backup_size: int = 3,  # Nb of steps we are backtracking
                 extra_n_steps: int = 2,  # Nb of steps further we need to compare the different rollouts
                 min_streamline_steps: int = 1,  # Min length of a streamline
                 max_streamline_steps: int = 256,  # Max length of a streamline
                 rollout_stats: RolloutStats = RolloutStats(),
                 utility_tracker: RolloutUtilityTracker = None
                 ):

        self.rollout_agent = None
        self.reference = reference
        self.oracle = oracle

        self.n_rollouts = n_rollouts
        self.backup_size = backup_size
        self.extra_n_steps = extra_n_steps
        self.min_streamline_steps = min_streamline_steps
        self.max_streamline_steps = max_streamline_steps
        self.rollout_stats = rollout_stats
        self.rollout_stats.set_nb_rollouts(n_rollouts)
        self.utility_tracker = utility_tracker

    def setup_rollout_agent(self, agent: ActorCritic):
        self.rollout_agent = agent

    def _verify_rollout_agent(self):
        if self.rollout_agent is None:
            raise ValueError("Rollout agent not set. Please call setup_rollout_agent before running rollouts.")

    # TODO: Copied from env
    def _compute_stopping_flags(
        self,
        streamlines: np.ndarray,
        stopping_criteria: dict[StoppingFlags, Callable]
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Checks which streamlines should stop and which ones should
        continue.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        stopping_criteria : dict of int->Callable
            List of functions that take as input streamlines, and output a
            boolean numpy array indicating which streamlines should stop

        Returns
        -------
        should_stop : `numpy.ndarray`
            Boolean array, True is tracking should stop
        flags : `numpy.ndarray`
            `StoppingFlags` that triggered stopping for each stopping
            streamline
        """
        idx = np.arange(len(streamlines))

        should_stop = np.zeros(len(idx), dtype=np.bool_)
        flags = np.zeros(len(idx), dtype=int)

        # For each possible flag, determine which streamline should stop and
        # keep track of the triggered flag
        for flag, stopping_criterion in stopping_criteria.items():
            stopped_by_criterion = stopping_criterion(streamlines)
            flags[stopped_by_criterion] |= flag.value
            should_stop[stopped_by_criterion] = True

        return should_stop, flags

    def rollout(
            self,
            streamlines: np.ndarray,
            in_stopping_idx: np.ndarray,
            in_stopping_flags: np.ndarray,
            current_length: int,
            stopping_criteria: dict[StoppingFlags, Callable],
            format_state_func: Callable[[np.ndarray], np.ndarray],
            format_action_func: Callable[[np.ndarray], np.ndarray],
            affine_vox2rasmm=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._verify_rollout_agent()

        assert self.max_streamline_steps == streamlines.shape[1]

        # Backtrack streamline length
        backup_length = max(1, current_length - self.backup_size)
        start = backup_length

        # For statistics purposes
        n_og_reached_target = count_flags(in_stopping_flags, StoppingFlags.STOPPING_TARGET)

        # If the original streamline isn't long enough to backtrack, just do nothing.
        # We don't backtrack in that case, there's no need to compute the rest
        if backup_length < 1 or backup_length == current_length:  
            print("Streamline too short to backtrack or already at the max length: {}".format(current_length))
            return streamlines, np.array([], dtype=in_stopping_idx.dtype), in_stopping_idx, in_stopping_flags

        n_streamlines = in_stopping_idx.shape[0]
        # If some streamlines are stopping in the gray matter, we don't want
        # to backtrack them, since they are probably valid. We only want to
        # backtrack prematurely stopped streamlines. (e.g. out of mask).
        backtrackable_mask = self._get_backtrackable_mask(in_stopping_flags)
        backtrackable_idx = in_stopping_idx[backtrackable_mask]
        not_backtracked_streamlines = in_stopping_idx[~backtrackable_mask]

        if backtrackable_idx.size <= 0:
            # Can't backtrack, because we're at the start or every streamline ends correctly (in the target).
            return streamlines, np.array([], dtype=in_stopping_idx.dtype), in_stopping_idx, in_stopping_flags

        b_streamlines = streamlines[backtrackable_idx].copy()

        b_streamlines[:, backup_length:current_length, :] = 0  # Backtrack on backup_length
        rollouts = np.repeat(b_streamlines[None, ...], self.n_rollouts,
                             axis=0)  # (n_rollouts, n_streamlines, n_points, 3)

        max_rollout_length = min(current_length + self.extra_n_steps, self.max_streamline_steps)
        
        # List of np.array of size n_rollouts x (n_streamlines,)
        rollouts_continue_idx = [np.arange(b_streamlines.shape[0]) for _ in range(self.n_rollouts)]

        # Every streamline is continuing, thus no stopping flag for each of them
        flags = np.zeros((self.n_rollouts, b_streamlines.shape[0]), dtype=np.int32) # (n_rollouts, n_streamlines)
        true_lengths = np.full((self.n_rollouts, b_streamlines.shape[0]), backup_length - 1, dtype=np.int32) # (n_rollouts, n_streamlines)

        actions_per_rollout = np.zeros((rollouts.shape[0], rollouts.shape[1], 20, 3), dtype=np.float32) # TODO: This might be missing a dimension.
        action_number = 0
        while backup_length < max_rollout_length and not all(np.size(arr) == 0 for arr in rollouts_continue_idx):

            for rollout in range(self.n_rollouts):
                r_continue_idx = rollouts_continue_idx[rollout]

                if r_continue_idx.size <= 0:
                    # No more streamlines to continue for that rollout
                    continue

                # Grow continuing streamlines one step forward
                state = format_state_func(
                    rollouts[rollout, r_continue_idx, :backup_length, :])

                with torch.no_grad():
                    actions = self.rollout_agent.select_action(state, probabilistic=1.1)
                    actions = actions.cpu().numpy()

                    actions_per_rollout[rollout, r_continue_idx, action_number] = actions

                new_directions = format_action_func(actions)

                # Step forward
                rollouts[rollout, r_continue_idx, backup_length, :] = \
                    rollouts[rollout, r_continue_idx, backup_length - 1, :] + new_directions
                true_lengths[rollout, r_continue_idx] = backup_length + 1

                # Get continuing streamlines that should stop and their stopping flag
                should_stop, new_flags = self._compute_stopping_flags(
                    rollouts[rollout, r_continue_idx, :backup_length + 1, :],
                    stopping_criteria
                )

                # See which trajectories is stopping or continuing
                new_continue_idx, stopping_idx = (r_continue_idx[~should_stop],
                                                  r_continue_idx[should_stop])


                rollouts_continue_idx[rollout] = new_continue_idx

                # Keep the reason why tracking stopped (don't delete a streamline that reached the target!)
                flags[rollout, stopping_idx] = new_flags[should_stop]

            backup_length += 1

        # Get the mean distance between each path produced by the rollouts
        # to make sure that there is a good diversity in the rollouts.
        # We don't want all the rollouts to be the same, otherwise the Monte
        # Carlo exploration can't be efficient.
        effective_rollouts = rollouts[:, :, start:, :] # (n_rollouts, n_streamlines, n_points, 3)
        mean_rollout_distance = np.mean(
            np.linalg.norm(
                np.diff(effective_rollouts, axis=2),
                axis=-1
            )
        )

        # Compute the variance of the actions taken by the agent. The variance should be between each action
        # corresponding to the same streamline. We want to make sure that each action for a different rollout is
        # different enough to explore the space efficiently.
        action_variances = np.var(actions_per_rollout, axis=0) # (n_streamlines, 20, 3)
        action_variances = np.mean(action_variances, axis=(1, 2)) # (n_streamlines,)
        mean_action_variance = np.mean(action_variances)

        # Visualize the rollouts
        viz = False
        if viz:
            self.visualize_rollouts(rollouts[:, 0], true_lengths[:, 0], max(1, current_length - self.backup_size))

        save = False
        if save and current_length == 2:
            # Save the rollouts as a tractogram
            self._save_rollouts_as_tractogram(rollouts, true_lengths, max(1, current_length - self.backup_size), affine_vox2rasmm)

        # Get the best rollout for each streamline.
        best_rollouts, new_flags, best_true_lengths = \
            self._filter_best_rollouts(rollouts, flags, backtrackable_idx, true_lengths)
        
        # Get the indices of the rollouts that improved the streamline
        # and convert it to the actual streamline index.
        rollout_improvement_idx = self._get_improvement_idx(current_length, best_true_lengths, new_flags)
        streamline_improvement_idx = backtrackable_idx[rollout_improvement_idx]

        # Squash the retained rollouts to the current_length
        best_rollouts[:, current_length:, :] = 0

        # Replace the original streamlines with the best rollouts.
        streamlines[streamline_improvement_idx, :, :] = best_rollouts[rollout_improvement_idx]

        # We want to get the indices of the streamlines that were improved
        # and that does not end.
        # TODO: does this make sense with the streamline_improvement_idx?
        not_ending_rollouts = (new_flags == 0).nonzero()[0]
        new_continuing_streamlines = backtrackable_idx[not_ending_rollouts]

        # Update the stopping flags of the streamlines that were changed.
        # For example, streamlines that aren't stopping anymore have their
        # flags reset to 0, while if we have a streamline that is now stopping
        # in the gray matter (STOPPING_TARGET), that new flag is kept.
        indices = np.searchsorted(in_stopping_idx, streamline_improvement_idx)
        new_in_stopping_flags = in_stopping_flags.copy()
        new_in_stopping_flags[indices] = \
            new_flags[rollout_improvement_idx]

        # Remove the stopping indices that are not stopping anymore
        new_in_stopping_idx = in_stopping_idx[new_in_stopping_flags > 0]

        assert new_in_stopping_idx.shape[0] == np.sum(new_in_stopping_flags > 0)

        # Update the stats
        if self.rollout_stats:
            n_reached_target = count_flags(new_in_stopping_flags, StoppingFlags.STOPPING_TARGET) - n_og_reached_target
            n_saved = new_continuing_streamlines.shape[0]
            n_not_saved = n_streamlines - n_saved - n_reached_target
            self.rollout_stats.update_step(rdist=mean_rollout_distance,
                                           perc_saved=n_saved/n_streamlines,
                                           perc_reach_gm=n_reached_target/n_streamlines,
                                           perc_not_saved=n_not_saved/n_streamlines,
                                           action_variance=mean_action_variance)


        if self.utility_tracker:
            self.utility_tracker.update(
                backtrackable_idx, # All streamlines indices that were backtracked
                not_backtracked_streamlines,
                new_in_stopping_flags[backtrackable_mask] # All flags of the stopping streamlines
            )

        intersection = np.intersect1d(new_in_stopping_idx, new_continuing_streamlines)
        assert intersection.size == 0, \
            "Conflict of indices between stopping and continuing streamlines."

        return streamlines, new_continuing_streamlines, new_in_stopping_idx, new_in_stopping_flags
    
    @staticmethod
    def _padded_streamlines_to_array_sequence(
            streamlines: np.ndarray,
            true_lengths: np.ndarray) -> ArraySequence:
        """ Convert padded streamlines to an ArraySequence object.

        streamlines: np.ndarray of shape (n_streamlines, max_nb_points, 3)
            containing the streamlines with padding up to max_nb_points.
        true_lengths: np.ndarray of shape (n_streamlines,) containing the
            effective number of points in each streamline.
        """
        assert true_lengths[true_lengths == 0].size == 0, \
            "Streamlines should at least have one point."

        total_points = np.sum(true_lengths)
        _data = np.zeros((total_points, 3), dtype=np.float32)
        _offsets = np.zeros_like(true_lengths)
        _offsets[1:] = np.cumsum(true_lengths)[:-1]
        _lengths = true_lengths

        for i, (streamline, length) in enumerate(zip(streamlines, true_lengths)):
            _data[_offsets[i]:_offsets[i] + length] = streamline[:length]

        array_seq_streamlines = ArraySequence()
        array_seq_streamlines._data = _data
        array_seq_streamlines._offsets = _offsets
        array_seq_streamlines._lengths = _lengths

        return array_seq_streamlines
    
    def _save_rollouts_as_tractogram(self, rollouts, true_lengths, backtrack_start_idx, affine_vox2rasmm):
        # Select only the rollouts of the first streamline
        n_streamlines_to_save = min(rollouts.shape[1], 10)

        streamlines_indices = np.random.choice(
            rollouts.shape[1], n_streamlines_to_save, replace=False)

        streamlines_to_save = rollouts[:, streamlines_indices] # Select all the rollouts for the specified streamlines
        streamlines_to_save = streamlines_to_save.reshape(-1, streamlines_to_save.shape[2], 3)

        true_lengths_to_save = true_lengths[:, streamlines_indices]
        true_lengths_to_save = true_lengths_to_save.reshape(-1)

        arseq = self._padded_streamlines_to_array_sequence(streamlines_to_save, true_lengths_to_save)

        # Create a new tractogram
        tractogram = Tractogram(
            streamlines=arseq) # These streamlines should all have the same number of points.

        tractogram.apply_affine(affine_vox2rasmm)

        sft = StatefulTractogram(
            streamlines=tractogram.streamlines,
            reference=self.reference,
            space=Space.RASMM)
        
        save_tractogram(sft, 'tmp/rollouts.trk', bbox_valid_check=False)


    def visualize_rollouts(self, streamline_rollouts: np.ndarray, streamline_lengths: np.ndarray, backtrack_start_idx: int):
        """
        This function uses Fury to visualize the rollouts of the streamline i.

        streamline_rollouts: np.ndarray of shape (n_rollouts, max_n_points, 3)
        streamline_lengths: np.ndarray of shape (n_rollouts,) containing the true length of each streamline.
        """


        original_streamline = streamline_rollouts[0, :backtrack_start_idx, :] # (nb_points, 3)
        rollouts_only = self._trim_zeros(streamline_rollouts, streamline_lengths, backtrack_start_idx) # (n_rollouts, nb_points, 3) # Different length for each rollout

        # backtracking_streamlines = streamlines[backtrackable_idx, :backtracked_length, :]
        # backtracking_rollouts = rollouts[:, backtrackable_idx, :rollout_length, :]
        # chosen_streamline = self._trim_zeros(backtracking_rollouts[:, 0, ...], true_lengths[:, 0])
        # chosen_streamline_scores = rollout_scores[:, 0]
        # max_score_idx = np.argmax(chosen_streamline_scores)
        # best_streamline = [chosen_streamline[max_score_idx]]
        # chosen_streamline.pop(max_score_idx)

        # fa_actor = actor.slicer(self.reference_data, opacity=0.7, interpolation='nearest')

        # original_reference_streamline = original_streamline[None, 0, ...]
        # reference_streamline = backtracking_streamlines[None, 0, ...]
        original_actor = actor.line(original_streamline, colors=(65/255, 143/255, 205/255), linewidth=3)
        # reference_actor = actor.line(reference_streamline, (65/255, 143/255, 205/255), linewidth=3)
        rollouts_actor = actor.line(rollouts_only, colors=(1, 0, 0), linewidth=3)
        # best_rollout_actor = actor.line(best_streamline, (0, 1, 0), linewidth=3)

        scene = window.Scene()
        # scene.add(fa_actor)
        scene.add(original_actor)
        scene.add(rollouts_actor)
        # scene.add(best_rollout_actor)
        # scene.add(reference_actor)

        # window.show(scene, size=(600, 600), reset_camera=False)
        window.record(scene, out_path='rollouts.png', size=(600, 600))


    def _trim_zeros(self, rollouts, lengths, backtrack_start_idx):
        """ Trim the zeros from the rollouts to the true length of the streamlines.
        """
        trimmed_rollouts = []
        for i, (rollout, length) in enumerate(zip(rollouts, lengths)):
            if length > backtrack_start_idx:
                trimmed_streamline = rollout[backtrack_start_idx:length]
                trimmed_rollouts.append(trimmed_streamline)

        return trimmed_rollouts

    def _filter_best_rollouts(self,
                              rollouts: np.ndarray,
                              flags,
                              backtrackable_idx: np.ndarray,
                              true_lengths: np.ndarray
                              ):
        """ Filter the best rollouts based on the oracle's predictions.
        """

        rollouts_scores = np.zeros((self.n_rollouts, rollouts.shape[1]), dtype=np.float32)
        for rollout in range(rollouts.shape[0]):
            # Here, the streamlines should be trimmed to their true length
            # which might differ from one streamline to another.
            # We might want to use nibabel's ArraySequence to store the
            # streamlines so we can also resample them to a fixed number of
            # points using dipy's set_number_of_points function.
            array_seq_streamlines = \
                self._padded_streamlines_to_array_sequence(
                    rollouts[rollout], true_lengths[rollout])
            
            array_seq_streamlines = set_number_of_points(array_seq_streamlines, self.oracle.nb_points)

            # Get the scores from the oracle
            scores = self.oracle.predict(array_seq_streamlines)

            # When we reach the target, we want to add a bonus of +1 to the score so it outweighs the other scores
            scores += is_flag_set(flags[rollout], StoppingFlags.STOPPING_TARGET)

            # When the streamline is too short, we want to add a penalty of -10 to the score
            # so it doesn't get chosen.
            too_short_mask = true_lengths[rollout] < self.min_streamline_steps
            scores[too_short_mask] -= 10

            rollouts_scores[rollout] = scores

        # Filter based on the calculated scores and keep the best rollouts and their according flags
        best_rollout_indices = np.argmax(rollouts_scores, axis=0)
        rows = np.arange(rollouts.shape[1]) # Req. for advanced indexing
        best_rollouts = rollouts[best_rollout_indices, rows]
        best_flags = flags[best_rollout_indices, rows]
        best_true_lengths = true_lengths[best_rollout_indices, rows]

        return best_rollouts, best_flags, best_true_lengths

    @staticmethod
    def _get_improvement_idx(current_length: int, rollout_lengths: np.ndarray, flags: np.ndarray):
        assert len(rollout_lengths) == len(flags), \
            "Rollout lengths and flags should have the same length. lengths: {}, flags: {}".format(rollout_lengths.shape, flags.shape)
        
        reaches_target_mask = is_flag_set(flags, StoppingFlags.STOPPING_TARGET)
        long_enough_mask = rollout_lengths >= current_length
        total_mask = reaches_target_mask | long_enough_mask

        # To improve, streamlines should be either longer than the current length,
        # or should reach the target.
        improvement_idx = total_mask.nonzero()[0]
        return improvement_idx

    @staticmethod
    def _get_backtrackable_mask(
            stopping_flags: np.ndarray
    ) -> np.ndarray:
        """ Filter out the stopping flags from which we are able to perform backtracking to retry different paths
        for each stopped streamline.

        For example, if a streamline stopped because of STOPPING_TARGET, we might not want to backtrack since the
        streamline ends in the gray matter.
        """
        flag1 = is_flag_set(stopping_flags, StoppingFlags.STOPPING_TARGET, logical_not=True)
        return flag1