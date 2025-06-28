from abc import abstractmethod, ABCMeta
from typing import Tuple
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram
from tractoracle_irt.filterers.streamlines_sampler \
    import StreamlinesSampler

class Filterer(metaclass=ABCMeta):
    
    def __init__(self, sampler: StreamlinesSampler = None):
        self.sampler = sampler

    def __call__(self, in_directory: str, tractograms: str, out_dir: str):
        """Filter a list of tracts."""
        valid, invalid, subject_ids = self._filter(in_directory, tractograms, out_dir)

        if self.sampler is not None:
            valid, invalid = \
                self.sampler.sample_streamlines(valid, invalid)

            # Make sure the number of valid and invalid is almost equal
            assert np.abs(len(valid) - len(invalid)) < 5, \
                f"Number of valid and invalid streamlines differ by more than 5. " \
                f"Valid: {len(valid)}, invalid: {len(invalid)}"
        
        return valid, invalid, subject_ids
        
    @abstractmethod
    def _filter(self, in_directory: str, tractograms: str, out_dir: str) -> Tuple[StatefulTractogram, StatefulTractogram]:
        """Implement the filtering logic for the tractograms."""
        raise NotImplementedError("Filter method not implemented.")

    

