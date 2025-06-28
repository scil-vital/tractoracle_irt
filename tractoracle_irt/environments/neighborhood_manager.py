import torch
import numpy as np

from tractoracle_irt.utils.neighborhood_interpolation import \
    interpolate_volume_in_neighborhood
from tractoracle_irt.utils.dwi_ml import \
    prepare_neighborhood_vectors

from tractoracle_irt.utils.interpolation import neighborhood_interpolation, calc_neighborhood_grid
from tractoracle_irt.utils.torch_utils import get_device


def to_torch(data, device=None):
    if isinstance(data, torch.Tensor):
        out = data
    elif isinstance(data, np.ndarray):
        out = torch.from_numpy(data).float()
    else:
        raise ValueError('Data type not supported: ', type(data))
    
    if device is not None:
        out = out.to(device)

    return out

class NeighborhoodManager(object):
    def __init__(self, *args, method='efficient', **kwargs):
        self.method = method
        
        if method=='efficient':
            self._manager = EfficientNeighborhoodManager(*args, **kwargs)
        elif method=='dwi_ml':
            self._manager = DwiMlNeighborhoodManager(*args, **kwargs)
        else:
            raise ValueError('Method not supported: ', method)

    def get(self, coords, torch_convention=False):
        return self._manager.get(coords, torch_convention)
    
    def get_crops(self, coords):
        return self._manager.get_crops(coords)
    
    @property
    def radius(self):
        return self._manager.radius

class EfficientNeighborhoodManager(object):
    def __init__(self, data_volume, radius, add_neighborhood_vox, flatten, device=get_device(), **kwargs):
        self.device = device
        self.add_neighborhood_vox = add_neighborhood_vox

        self.radius = radius
        self.data_volume = to_torch(data_volume, device=self.device)
        self.grid = calc_neighborhood_grid(radius, device=self.device, resolution=self.add_neighborhood_vox)

        # Warn that flattening is not supported
        self.flatten = flatten

    def get(self, coords, torch_convention):
        # Send the coordinates to the device if they aren't already
        if coords.device != self.device:
            coords = coords.to(self.device)

        # Seem to have better results with align_corners=False
        signal = neighborhood_interpolation(self.data_volume, coords, self.grid, align_corners=False)          

        if self.flatten:
            signal = signal.reshape(signal.shape[0], -1)
            if torch_convention:
                import warnings
                warnings.warn('Flattening is enabled, but torch_convention=True. Ignoring torch_convention=True.')
        elif torch_convention:
            # Permute axes to fit PyTorch's convention of (N, C, D, H, W)
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
            signal = signal.permute(0, 4, 1, 2, 3)

        return signal
    
    def get_crops(self, coords):
        crops = []
        rad = self.radius
        for coord in coords:
            out = torch.zeros_like(self.data_volume)

            coord = coord.long()
            x, y, z = coord
            
            # Crop the volume and zero pad it if necessary
            # crop = self.data_volume[
            #     x-rad: x+rad+1,
            #     y-rad: y+rad+1,
            #     z-rad: z+rad+1]
            # Doing such a thing won't work because the crop might be out of bounds.
            # We need to ensure that if it's out of bounds, we zero pad it.
            crop = torch.zeros((2*rad+1, 2*rad+1, 2*rad+1, self.data_volume.shape[-1]), device=self.device)

            min_x = max(0, x-rad)
            max_x = min(self.data_volume.shape[0], x+rad+1)
            min_y = max(0, y-rad)
            max_y = min(self.data_volume.shape[1], y+rad+1)
            min_z = max(0, z-rad)
            max_z = min(self.data_volume.shape[2], z+rad+1)

            crop[
                min_x-x+rad: max_x-x+rad,
                min_y-y+rad: max_y-y+rad,
                min_z-z+rad: max_z-z+rad] = self.data_volume[
                    min_x: max_x,
                    min_y: max_y,
                    min_z: max_z]
            
            # Copy the crop to the output, but only copy
            # values which are within the bounds of the output
            o_min_x = max(0, rad-x)
            o_max_x = min(out.shape[0], rad+1+(self.data_volume.shape[0]-x))
            o_min_y = max(0, rad-y)
            o_max_y = min(out.shape[1], rad+1+(self.data_volume.shape[1]-y))
            o_min_z = max(0, rad-z)
            o_max_z = min(out.shape[2], rad+1+(self.data_volume.shape[2]-z))


            out[
                x-rad: x+rad+1,
                y-rad: y+rad+1,
                z-rad: z+rad+1] = crop[
                    o_min_x: o_max_x,
                    o_min_y: o_max_y,
                    o_min_z: o_max_z]
            crops.append(out)
        
        return crops


class DwiMlNeighborhoodManager(object):
    def __init__(self, data_volume, radius, add_neighborhood_vox, flatten, neighborhood_type, device=get_device(), **kwargs):
        self.device = device
        self.radius = radius
        self.neighborhood_type = neighborhood_type
        self.add_neighborhood_vox = add_neighborhood_vox
        self.flatten = flatten

        self.data_volume = to_torch(data_volume, device=self.device)

        self.neighborhood_directions = prepare_neighborhood_vectors(
            self.neighborhood_type,
            self.radius,
            self.add_neighborhood_vox).to(
                self.device)
        
        # Sort the neighborhood directions to have a consistent order (lexicographically)
        sorted_indices = torch.argsort(self.neighborhood_directions[:, 2])
        sorted_coords = self.neighborhood_directions[sorted_indices]
        sorted_indices = torch.argsort(sorted_coords[:, 1], stable=True)
        sorted_coords = sorted_coords[sorted_indices]
        sorted_indices = torch.argsort(sorted_coords[:, 0], stable=True)
        sorted_coords = sorted_coords[sorted_indices]
        self.neighborhood_directions = sorted_coords

    def get(self, coords, torch_convention=False, crop_last=False):
        with torch.no_grad():
            signal, _ = interpolate_volume_in_neighborhood(
                self.data_volume,
                coords,
                self.neighborhood_directions, clear_cache=False)

            if not self.flatten and self.neighborhood_type == 'grid':
                signal = self._unflatten_neighborhood(signal)

            if torch_convention:
                # Permute axes to fit PyTorch's convention of (N, C, D, H, W)
                # https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
                signal = signal.permute(0, 4, 1, 2, 3)

        return signal
    
    def _unflatten_neighborhood(self, signal):
        assert self.neighborhood_type == 'grid'

        nb_points = signal.shape[0]
        nb_neighb = len(self.neighborhood_directions)
        out_size = int(self.radius * 2 + 1)
        nb_features = int(signal.shape[1] / nb_neighb) # Nb of features per neighbor

        # The way we perform our interpolation, we get:
        # n1 - f1, n1 - f2, ....,  n2 - f1, n2 - f2, ...
        # For this to work, we need the neighbors vectors to be ordred lexicographically.
        unflattened = signal.view(nb_points, out_size, out_size, out_size, nb_features) # (5, 19, 19, 19, 28)

        return unflattened

    def get_crops(self, coords):
        crops = []
        rad = self.radius
        for coord in coords:
            out = torch.zeros_like(self.data_volume)

            coord = coord.long()
            x, y, z = coord
            
            # Crop the volume and zero pad it if necessary
            # crop = self.data_volume[
            #     x-rad: x+rad+1,
            #     y-rad: y+rad+1,
            #     z-rad: z+rad+1]
            # Doing such a thing won't work because the crop might be out of bounds.
            # We need to ensure that if it's out of bounds, we zero pad it.
            crop = torch.zeros((2*rad+1, 2*rad+1, 2*rad+1, self.data_volume.shape[-1]), device=self.device)

            min_x = max(0, x-rad)
            max_x = min(self.data_volume.shape[0], x+rad+1)
            min_y = max(0, y-rad)
            max_y = min(self.data_volume.shape[1], y+rad+1)
            min_z = max(0, z-rad)
            max_z = min(self.data_volume.shape[2], z+rad+1)

            crop[
                min_x-x+rad: max_x-x+rad,
                min_y-y+rad: max_y-y+rad,
                min_z-z+rad: max_z-z+rad] = self.data_volume[
                    min_x: max_x,
                    min_y: max_y,
                    min_z: max_z]
            
            # Copy the crop to the output, but only copy
            # values which are within the bounds of the output
            o_min_x = max(0, rad-x)
            o_max_x = min(out.shape[0], rad+1+(self.data_volume.shape[0]-x))
            o_min_y = max(0, rad-y)
            o_max_y = min(out.shape[1], rad+1+(self.data_volume.shape[1]-y))
            o_min_z = max(0, rad-z)
            o_max_z = min(out.shape[2], rad+1+(self.data_volume.shape[2]-z))


            out[
                x-rad: x+rad+1,
                y-rad: y+rad+1,
                z-rad: z+rad+1] = crop[
                    o_min_x: o_max_x,
                    o_min_y: o_max_y,
                    o_min_z: o_max_z]
            crops.append(out)
        
        return crops
