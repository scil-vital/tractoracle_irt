import numpy as np
import torch
from dipy.tracking.streamline import set_number_of_points

from tractoracle_irt.oracles.transformer_oracle import TransformerOracle
from tractoracle_irt.utils.torch_utils import get_device_str, get_device
from nibabel.streamlines.array_sequence import ArraySequence
from tractoracle_irt.environments.utils import resample_streamlines_if_needed
from tractoracle_irt.utils.logging import get_logger
from tqdm import tqdm

LOGGER = get_logger(__name__)

class OracleSingleton:
    _registered_checkpoints = {
        # '<checkpoint name>': <OracleSingleton instance>
    }

    def __new__(cls, *args, **kwargs):
        checkpoint_str = args[0]
        # Only create one instance of the oracle per checkpoint.
        if checkpoint_str not in cls._registered_checkpoints.keys():
            print('Instanciating new Oracle, should only happen once. '
                  '(ckpt: {})'.format(checkpoint_str))
            instance = super().__new__(cls)
            instance._initialized = False
            cls._registered_checkpoints[checkpoint_str] = instance
        return cls._registered_checkpoints[checkpoint_str]

    def __init__(self, checkpoint: str, device: str, batch_size=4096, lr=None):
        if getattr(self, '_initialized', False):
            return
        else:
            LOGGER.debug("Should we be skipping init? ", self._initialized)
        
        self._initialized = True
        self.checkpoint = torch.load(checkpoint, map_location=get_device(), weights_only=False)

        # The model's class is saved in hparams
        is_pl_checkpoint = "pytorch-lightning_version" in self.checkpoint.keys()
        hparams_key = "hyper_parameters" \
            if is_pl_checkpoint else "hyperparameters"

        hyper_parameters = self.checkpoint[hparams_key]
        self.nb_points = hyper_parameters['input_size'] // 3 + 1
        models = {
            'TransformerOracle': TransformerOracle
        }

        # Load it from the checkpoint
        self.model = models[hyper_parameters[
            'name']].load_from_checkpoint(self.checkpoint, lr).to(device)

        self.model.eval()
        self.batch_size = batch_size
        self.device = device

    def predict(self, streamlines, prefetch_streamlines=False):
        N = len(streamlines)
        result = torch.zeros((N), dtype=torch.float, device=self.device)

        #print("oracle batch predict shape: ", streamlines.shape)

        if prefetch_streamlines:
            placeholder = torch.zeros(
                (self.batch_size, self.nb_points - 1, 3), pin_memory=get_device_str() == "cuda")

            # Get the first batch
            batch = streamlines[:self.batch_size]
            N_batch = len(batch)
            # Resample streamlines to fixed number of point to set all
            # sequences to same length
            data = resample_streamlines_if_needed(batch, self.nb_points)

            # Compute streamline features as the directions between points
            dirs = np.diff(data, axis=1)

            # Send the directions to pinned memory
            placeholder[:N_batch] = torch.from_numpy(dirs)
            # Send the pinned memory to GPU asynchronously
            input_data = placeholder[:N_batch].to(
                self.device, non_blocking=True, dtype=torch.float)
            i = 0

            while i <= N // self.batch_size:
                start = (i+1) * self.batch_size
                end = min(start + self.batch_size, N)
                current_batch_size = end - start
                # Prefetch the next batch
                if start < end:
                    # Resample streamlines to fixed number of point to set all
                    # sequences to same length
                    data = resample_streamlines_if_needed(streamlines[start:end], self.nb_points)

                    # Compute streamline features as the directions between points
                    dirs = np.diff(data, axis=1)
                    # Put the directions in pinned memory
                    placeholder[:current_batch_size] = torch.from_numpy(dirs)

                with torch.amp.autocast(device_type=get_device_str()):
                    with torch.no_grad():
                        predictions = self.model(input_data)
                        result[
                            i * self.batch_size:
                            (i * self.batch_size) + self.batch_size] = predictions
                            
                i += 1
                if i >= N // self.batch_size:
                    break
                # Send the pinned memory to GPU asynchronously
                input_data = placeholder[:end-start].to(
                    self.device, non_blocking=True, dtype=torch.float)

        else:
            for batch_idx in tqdm(range(0, N, self.batch_size), disable=True):
                start = batch_idx
                end = min(start + self.batch_size, N)

                batch = streamlines[start:end]
                data = resample_streamlines_if_needed(batch, self.nb_points)
                dirs = np.diff(data, axis=1)
                dirs = torch.tensor(dirs, device=self.device)

                with torch.amp.autocast(device_type=get_device_str()):
                    with torch.no_grad():
                        predictions = self.model(dirs)
                        result[start:end] = predictions

        return result.cpu().numpy()
