import torch

@torch.no_grad()
def calc_neighborhood_grid(neighborhood_radius: int, device=None, resolution: float = 1.):
    # Get the neighborhood grid for the coordinates
    neighborhood_grid = torch.meshgrid(
        torch.arange(-neighborhood_radius, neighborhood_radius+1),
        torch.arange(-neighborhood_radius, neighborhood_radius+1),
        torch.arange(-neighborhood_radius, neighborhood_radius+1),
        indexing='xy')
    
    # Convert the neighborhood grid to a tensor
    neighborhood_grid = torch.stack(neighborhood_grid, dim=-1).float().to(device=device)

    # Scale the neighborhood_grid to the specified resolution
    neighborhood_grid *= resolution

    return neighborhood_grid

@torch.no_grad()
def neighborhood_interpolation(volume: torch.Tensor, coords: torch.Tensor, neighborhood_grid: torch.Tensor, align_corners: bool = False):
    """
    This function interpolates a volume at given coordinates using a neighborhood
    of points around the coordinates.

    The volume is of shape (C, D, H, W) where C is the number of channels, H is the
    height, W is the width and D is the depth of the volume.

    The coordinates are of shape (N, 3) where N is the number of coordinates and
    the 3 columns are the x, y, z coordinates.

    The neighborhood is a grid of points around the coordinates. The radius of the
    neighborhood is given by neighborhood_radius. For example, if neighborhood_radius
    is 1, the neighborhood will have 27 points. If neighborhood_radius is 2, the
    neighborhood will have 125 points. The neighborhood grid for each point is
    of shape (D_neigh, H_neigh, W_neigh) where D_neigh, H_neigh, W_neigh are the
    depth, height and width of the neighborhood grid.

    We need to use the torch.nn.functional.grid_sample function to interpolate the
    volume at the coordinates. We need to prepare the grid for the grid_sample
    function. The grid is of shape (N, D_neigh, H_neigh, W_neigh, 3) where N is
    the number of coordinates and the last dimension is the x, y, z coordinates

    The function returns the interpolated values at the coordinates. The output
    is of shape (N, C, D_neigh, H_neigh, W_neigh).
    """
    N = coords.shape[0] # Coords are in the x, y, z format

    # Add the coordinates to the neighborhood grid
    neighborhood_grid = neighborhood_grid + coords.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)
    neighborhood_grid = neighborhood_grid.permute(0, 3, 1, 2, 4)

    # Can we create a view of the volume that is of shape (N, C, D, H, W) to use grid_sample?
    # N being the number of coordinates.
    volume_mod = volume.permute(3, 0, 1, 2).unsqueeze(0)
    volume_mod = volume_mod.expand(N, -1, -1, -1, -1)

    # We need to normalize the grid coordinates to be between -1 and 1 before giving it to grid_sample
    offset = 0.0
    if align_corners:
        offset = 0.5
    neighborhood_grid = 2 * (neighborhood_grid + offset) / torch.tensor([volume_mod.shape[2],
                                                              volume_mod.shape[3],
                                                              volume_mod.shape[4]],
                                                              dtype=torch.float32,
                                                              device=volume_mod.device) - 1



    # Interpolate the volume at the coordinates using grid_sample
    # 'bilinear' interpolation is specified, however, according to the documentation
    # (https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html)
    # if the input is 5D, the interpolation mode will actually be 'trilinear'.
    #
    # Note: dwi_ml clamps the indices to the volume shape which is similar to
    # padding_mode='border' in grid_sample. Here we use padding_mode='zeros'
    # as I want the agent to know that it's on the edge of the image.
    #
    interpolated_volume = torch.nn.functional.grid_sample(
        volume_mod, neighborhood_grid, mode='bilinear', align_corners=align_corners, padding_mode='border'
    )

    interpolated_volume = interpolated_volume.permute(0, 2, 3, 4, 1)
    
    return interpolated_volume
