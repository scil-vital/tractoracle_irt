from tractoracle_irt.filterers.filterer import Filterer
from dipy.io.stateful_tractogram import StatefulTractogram

import argparse
import tempfile
import numpy as np
import subprocess

from dipy.io.streamline import load_tractogram
from dipy.io.streamline import save_tractogram
import os
from pathlib import Path
from typing import Union

class TractometerFilterer(Filterer):
    
    def __init__(self):
        super(TractometerFilterer, self).__init__()

    # TODO: This is copied from the TractOracle's generate_and_score_tractograms.py that was naively
    # implemented to generate streamline datasets. This should be refactored to avoid using subprocesses
    # and call directly the Tractometer's API.
    def __call__(self, reference, tractograms, gt_config, out_dir, scored_extension="trk", tmp_base_dir: str = None):
        """Filter a list of tracts."""
        reference = Path(reference)
        gt_config = Path(gt_config)
        out_dir = Path(out_dir)

        assert gt_config.exists(), f"Ground truth config {gt_config} does not exist."
        assert reference.exists(), f"Reference {reference} does not exist."

        filtered_tractograms = []

        for tractogram in tractograms:

            tractogram_path = Path(tractogram)
            assert tractogram_path.exists(), f"Tractogram {tractogram} does not exist."

            with tempfile.TemporaryDirectory(dir=tmp_base_dir) as tmp:

                tmp_path = Path(tmp)

                scoring_args = [
                    tractogram_path, # in_tractogram
                    gt_config, # gt_config
                    tmp_path, # out_dir
                    "--reference", reference
                ]
                c_proc = subprocess.run(["pwd"])
                # Segment and score the tractogram
                c_proc = subprocess.run(["scil_tractogram_segment_and_score.py", *scoring_args])
                c_proc.check_returncode() # Throws if the process failed

                out_path = out_dir / "scored_{}.{}".format(tractogram_path.stem, scored_extension)
                self._merge_with_scores(reference, tmp_path / "segmented_VB", tmp_path / "IS.trk", out_path)
                
                filtered_tractograms.append(out_path)

        return filtered_tractograms

    def _merge_with_scores(self, reference, bundles_dir, inv_streamlines, output):
        """Merge the streamlines with the scores."""
        file_list = os.listdir(bundles_dir)
        main_tractogram = None

        # Add valid streamlines
        for file in file_list:
            tractogram = load_tractogram(os.path.join(bundles_dir, file), str(reference))
            num_streamlines = len(tractogram.streamlines)

            tractogram.data_per_streamline['score'] = np.ones(num_streamlines, dtype=np.float32)

            if main_tractogram is None:
                main_tractogram = tractogram
            else:
                main_tractogram = main_tractogram + tractogram

        assert inv_streamlines.exists(), f"Invalid streamlines {inv_streamlines} does not exist."
        assert Path(reference).exists(), f"Reference {reference} does not exist."

        # Add invalid streamlines
        inv_tractogram = load_tractogram(str(inv_streamlines), str(reference))
        num_streamlines = len(inv_tractogram.streamlines)
        inv_tractogram.data_per_streamline['score'] = np.zeros(num_streamlines, dtype=np.float32)

        main_tractogram = main_tractogram + inv_tractogram

        assert main_tractogram is not None, "No valid streamlines found."
        
        print(f"Main tractogram has {len(main_tractogram.streamlines)} streamlines.")
        print(f"Saving tractogram to {output}")
        save_tractogram(main_tractogram, str(output))