import os
import tempfile
from collections import namedtuple

import numpy as np
from dipy.io.streamline import load_tractogram
from scilpy.segment.tractogram_from_roi import segment_tractogram_from_roi
from scilpy.tractanalysis.scoring import compute_tractometry

from dipy.io.streamline import load_tractogram
from dipy.io.streamline import save_tractogram

from tractoracle_irt.filterers.filterer import Filterer
from tractoracle_irt.filterers.streamlines_sampler import StreamlinesSampler
from tractoracle_irt.experiment.tractometer_validator import load_and_verify_everything
from pathlib import Path

class TractometerFilterer(Filterer):

    def __init__(
        self,
        base_dir,
        reference,
        dilate_endpoints=1,
        invalid_score=0,
        bbox_valid_check=True,
        sampler: StreamlinesSampler = None
    ):
        super().__init__(sampler)
        self.name = 'Tractometer'
        self.gt_config = os.path.join(base_dir, 'scil_scoring_config.json')
        self.gt_dir = base_dir
        self.dilation_factor = dilate_endpoints
        self.invalid_score = invalid_score
        self.bbox_valid_check = bbox_valid_check

        assert os.path.exists(reference), f"Reference {reference} does not exist."
        self.reference = reference

        # Load
        (self.gt_tails, self.gt_heads, self.bundle_names, self.list_rois,
         self.bundle_lengths, self.angles, self.orientation_lengths,
         self.abs_orientation_lengths, self.inv_all_masks, self.gt_masks,
         self.any_masks) = \
            load_and_verify_everything(
                reference,
                self.gt_config,
                self.gt_dir,
                False)

    @property
    def ends_up_in_orig_space(self):
        return True

    def _filter(self, in_directory: str, tractograms: str, out_dir: str):
        valids = []
        invalids = []
        subject_ids = []

        scored_extension = 'trk' # This might fail for ISMRM2015...
    
        for tractogram in tractograms:
            assert os.path.exists(tractogram), f"Tractogram {tractogram} does not exist."
            filtered_path_valid = os.path.join(out_dir, "valid_scored_{}.{}".format(Path(tractogram).stem, scored_extension))
            filtered_path_invalid = os.path.join(out_dir, "invalid_scored_{}.{}".format(Path(tractogram).stem, scored_extension))
            sft = load_tractogram(tractogram, self.reference,
                                bbox_valid_check=self.bbox_valid_check, trk_header_check=True)
            
            if len(sft.streamlines) == 0:
                return (sft, sft)

            args_mocker = namedtuple('args', [
                'compute_ic', 'save_wpc_separately', 'unique', 'reference',
                'bbox_check', 'out_dir', 'dilate_endpoints', 'no_empty'])

            with tempfile.TemporaryDirectory() as temp:
                
                args = args_mocker(
                    False, False, True, self.reference, False, temp,
                    self.dilation_factor, False)

                # Segment VB, WPC, IB
                (vb_sft_list, wpc_sft_list, ib_sft_list, nc_sft,
                ib_names, _) = segment_tractogram_from_roi(
                    sft, self.gt_tails, self.gt_heads, self.bundle_names,
                    self.bundle_lengths, self.angles, self.orientation_lengths,
                    self.abs_orientation_lengths, self.inv_all_masks, self.any_masks,
                    self.list_rois, args)
                
                valid, invalid = self._merge_with_scores(vb_sft_list, nc_sft)
                
                # Replace saving with directly putting that data into a hdf5 file.
                save_tractogram(valid, filtered_path_valid)
                save_tractogram(invalid, filtered_path_invalid)

            # We could also append the filtered_path instead of the tractogram themselves.
            valids.append(valid)
            invalids.append(invalid)
            subject_ids.append(Path(tractogram).parent.name)

        return valids, invalids, subject_ids

    def _merge_with_scores(self, vb_sft_list, inv_tractogram, merge_valid_invalid=False):
        """
        Merge the streamlines with the scores.

        Returns: (main_trackogram, inv_tractogram)
        To concatenate the two, use merge_valid_invalid=True. That will cause
        the function to return a single tractogram with all the streamlines and their scores.
        """
        main_tractogram = None

        # Add valid streamlines
        for bundle in vb_sft_list:
            num_streamlines = len(bundle.streamlines)

            bundle.data_per_streamline['score'] = np.ones(
                num_streamlines, dtype=np.float32)

            if num_streamlines <= 0:
                continue
            elif main_tractogram is None:
                main_tractogram = bundle
            else:
                main_tractogram = main_tractogram + bundle

        # Add invalid streamlines
        num_streamlines = len(inv_tractogram.streamlines)
        if num_streamlines > 0:
            inv_tractogram.data_per_streamline['score'] = np.full(
                num_streamlines, self.invalid_score, dtype=np.float32)

        if merge_valid_invalid:
            output = main_tractogram + inv_tractogram if main_tractogram is not None else inv_tractogram
        else:
            output = (main_tractogram, inv_tractogram)
        return output
        
