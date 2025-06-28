import pytest

import torch
import nibabel as nib
from tractoracle_irt.utils.interpolation import (
    prepare_neighborhood_vectors, unflatten_neighborhood,
    interpolate_volume_in_neighborhood)

from tractoracle_irt.utils.interpolation import neighborhood_interpolation, calc_neighborhood_grid

nib_img_path = 'ismrm2015_1mm/fodfs/ismrm2015_fodf.nii.gz' # TODO

@pytest.fixture
def prepare_fodf_volume_and_target():
    nib_img = nib.load(nib_img_path)
    img = nib_img.get_fdata()
    fodf_volume = torch.tensor(img, dtype=torch.float32)

    coord = torch.tensor([img.shape[0]//2, img.shape[1]//2, img.shape[2]//2], dtype=torch.long)
    radius = 10

    # Crop the fodf_volume to the neighborhood
    target = fodf_volume[
        coord[0].long()-radius:coord[0].long()+radius+1,
        coord[1].long()-radius:coord[1].long()+radius+1,
        coord[2].long()-radius:coord[2].long()+radius+1]

    return fodf_volume, target, coord.float(), radius

@pytest.fixture
def prepare_fodf_volume_and_targets():
    nib_img = nib.load(nib_img_path)
    img = nib_img.get_fdata()
    fodf_volume = torch.tensor(img, dtype=torch.float32)

    coords = torch.tensor([
        [img.shape[0]//2, img.shape[1]//2, img.shape[2]//2],
        [img.shape[0]//2+5, img.shape[1]//2+5, img.shape[2]//2+5],
        [img.shape[0]//2-5, img.shape[1]//2-5, img.shape[2]//2-5],
        [img.shape[0]//2+5, img.shape[1]//2-5, img.shape[2]//2-5],
        [img.shape[0]//2+5, img.shape[1]//2+5, img.shape[2]//2-5]
    ], dtype=torch.long)
    radius = 10

    targets = []
    for coord in coords:
        target = fodf_volume[
            coord[0].long()-radius:coord[0].long()+radius+1,
            coord[1].long()-radius:coord[1].long()+radius+1,
            coord[2].long()-radius:coord[2].long()+radius+1]
        targets.append(target)

    return fodf_volume, targets, coords.float(), radius

def test_dwiml_interpolation_crop(prepare_fodf_volume_and_target):
    fodf_volume, target, coord, radius = prepare_fodf_volume_and_target

    n_coef = fodf_volume.shape[-1]
    neighborhood_type = 'grid'
    neighborhood_resolution = 1.0
    neighborhood_vectors = prepare_neighborhood_vectors(
        neighborhood_type, radius, neighborhood_resolution)
    
    grid_side_size = radius*2 + 1

    # Interpolate with dwi_ml
    signal, _ = interpolate_volume_in_neighborhood(fodf_volume, coord.unsqueeze(0), neighborhood_vectors)
    assert signal.shape == (1, grid_side_size*grid_side_size*grid_side_size*n_coef)

    # Unflatten the neighborhood
    signal = unflatten_neighborhood(
                    signal, neighborhood_vectors, 'grid',
                    radius, neighborhood_resolution)
    assert signal.shape == (1, grid_side_size, grid_side_size, grid_side_size, n_coef)

    difference = torch.abs(target - signal[0])
    error_ratio = difference.sum() / (difference>=0).sum()
    assert error_ratio < 0.03


def test_custom_interpolation_crop(prepare_fodf_volume_and_target):
    # Prepare data
    fodf_volume, target, coord, radius = prepare_fodf_volume_and_target

    grid = calc_neighborhood_grid(radius, device=fodf_volume.device, resolution=1.0)

    # Interpolate with custom interpolation technique
    signal = neighborhood_interpolation(fodf_volume, coord.unsqueeze(0), grid)

    difference = torch.abs(target - signal[0])
    error_ratio = difference.sum() / (difference>=0).sum()
    print("error_ratio:", error_ratio)
    assert error_ratio < 0.04  # TODO: We need to reduce the error ratio here.

def test_custom_interpolation_mutiple_coordinates(prepare_fodf_volume_and_targets):
    fodf_volume, targets, coords, radius = prepare_fodf_volume_and_targets

    grid = calc_neighborhood_grid(radius, device=fodf_volume.device, resolution=1.0)

    # Interpolate with custom interpolation technique
    signal = neighborhood_interpolation(fodf_volume, coords, grid)
    
    differences = []
    for i, target in enumerate(targets):
        difference = torch.abs(target - signal[i])
        error_ratio = difference.sum() / (difference>=0).sum()
        differences.append(error_ratio)
    
    for error_ratio in differences:
        assert error_ratio < 0.04 # TODO: We need to reduce the error ratio here.
