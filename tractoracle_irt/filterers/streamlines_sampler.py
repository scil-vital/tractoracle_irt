import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram
from tractoracle_irt.utils.logging import get_logger

LOGGER = get_logger(__name__)

class IdentitySampler(object):
    def __init__(self):
        pass

    def sample_streamlines(self, valid_sft: StatefulTractogram, invalid_sft: StatefulTractogram):
        return valid_sft, invalid_sft

class StreamlinesSampler(object):
    def __init__(self):
        pass

    def sample_streamlines(self,
                           valid_sft: StatefulTractogram,
                           invalid_sft: StatefulTractogram,
                           force_zero: bool = False):
        """
        Equalize the number of streamlines in the tractogram to the reference.

        Parameters
        ----------
        streamlines : ArraySequence
            The streamlines to be equalized.
        reference : ArraySequence
            The reference streamlines.
        """
        nb_valid = len(valid_sft)
        nb_invalid = len(invalid_sft)

        if nb_valid == nb_invalid:
            return valid_sft, invalid_sft
        elif (nb_valid == 0 or nb_invalid == 0) and force_zero:
            return valid_sft, invalid_sft
        elif nb_valid > nb_invalid:
            return self._sample_streamlines(
                valid_sft, nb_invalid, valid=True), invalid_sft
        else:
            return valid_sft, self._sample_streamlines(
                invalid_sft, nb_valid, valid=False)

    def _sample_streamlines(self,
                              streamlines: StatefulTractogram,
                              n: int,
                              valid: bool = True):
        
        nb_streamlines = len(streamlines)
        indices = np.random.choice(nb_streamlines, n, replace=False)
        validity = "valid" if valid else "invalid"
        LOGGER.info(f"Sampling {n} streamlines from {nb_streamlines} "
                    f"{validity} streamlines.")

        return streamlines[indices]
        