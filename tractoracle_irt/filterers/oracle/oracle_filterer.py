import numpy as np
from dipy.io.streamline import load_tractogram
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map

from tractoracle_irt.filterers.filterer import Filterer
from tractoracle_irt.oracles.oracle import OracleSingleton
from tractoracle_irt.utils.logging import get_logger

LOGGER = get_logger(__name__)

class OracleFilterer(Filterer):

    def __init__(self, checkpoint, device):

        self.name = 'Oracle'

        if checkpoint:
            self.checkpoint = checkpoint
            self.model = OracleSingleton(checkpoint, device)
        else:
            self.checkpoint = None

        self.device = device

    def __call__(self, filename, reference):

        # Bbox check=False, TractoInferno volume may be cropped really tight
        sft = load_tractogram(filename, reference,
                              bbox_valid_check=False, trk_header_check=True)
        sft.to_vox()
        sft.to_corner()

        streamlines = sft.streamlines

        if len(streamlines) == 0:
            LOGGER.info("No streamlines found in the tractogram.")
            return 0, 0

        batch_size = 1024
        N = len(streamlines)
        predictions = np.zeros((N))
        for i in range(0, N, batch_size):

            j = i + batch_size
            scores = self.model.predict(streamlines[i:j])
            predictions[i:j] = scores
        
        final_scores = predictions > 0.5

        nb_valid_streamlines = np.count_nonzero(final_scores)
        nb_nb_streamlines_total = len(streamlines)
        LOGGER.info(f"Number of valid streamlines: {nb_valid_streamlines}")
        LOGGER.info(f"Number of streamlines: {nb_nb_streamlines_total}")

        return nb_valid_streamlines, nb_nb_streamlines_total

    def _filter(self):
        pass
