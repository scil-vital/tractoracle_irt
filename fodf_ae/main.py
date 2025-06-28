import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import nibabel as nib
import itertools as it
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tractoracle_irt.environments.neighborhood_manager import NeighborhoodManager
from tractoracle_irt.algorithms.shared.fodf_encoder import SmallWorkingFodfEncoder, SmallWorkingFodfDecoder

EXAMPLE_DATA=os.path.join(os.path.dirname(__file__), 'example_data')
VOLUME_PATH = os.path.join(EXAMPLE_DATA, 'ismrm2015_fodf.nii.gz')
WM_MASK_PATH = os.path.join(EXAMPLE_DATA, 'ismrm2015_wm.nii.gz')

dim_2d = False
conv_layer = nn.Conv3d
bn_layer = nn.BatchNorm3d
conv_t_layer = nn.ConvTranspose3d
get_flat_size = lambda dim_size: dim_size**3
get_flat_shape = lambda dim_size: (dim_size, dim_size, dim_size)

class NeighborhoodDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, n_coefs=28, neighborhood_size=3, method='crop', torch_convention=False):
        self.is_train = train
        self.volume_nib = nib.load(VOLUME_PATH)
        self.volume_data = self.volume_nib.get_fdata()
        self.affine = self.volume_nib.affine

        self.neighborhood_size = neighborhood_size
        self.rng = np.random.RandomState(42)
        self.coords = self._get_coordinates()
        self.method = method
        self.n_coefs = n_coefs
        self.torch_convention = torch_convention

        if method == 'interpolate':
            print("Using DWI_ML interpolation method...")
            self.neigh_manager = NeighborhoodManager(
                self.volume_data,
                self.neighborhood_size,
                1,
                False,
                'grid',
                method='dwi_ml',
                device='cpu')
            
            self.coords = torch.from_numpy(np.array(self.coords)).float()


    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if self.method == 'crop':
            return self._crop_at_coordinate(self.coords[idx])
        elif self.method == 'interpolate':
            return self.coords[idx]
        else:
            raise ValueError('Invalid method: {}'.format(self.method))
    
    def _crop_at_coordinate(self, coord):
        x, y, z = coord
        rad = self.neighborhood_size

        # Keep everything within bounds
        min_x = np.clip(x-rad, 0, self.volume_data.shape[0])
        max_x = np.clip(x+rad, 0, self.volume_data.shape[0])
        min_y = np.clip(y-rad, 0, self.volume_data.shape[1])
        max_y = np.clip(y+rad, 0, self.volume_data.shape[1])
        min_z = np.clip(z-rad, 0, self.volume_data.shape[2])
        max_z = np.clip(z+rad, 0, self.volume_data.shape[2])

        crop = self.volume_data[min_x:max_x, min_y:max_y, min_z:max_z]

        # We need to pad the indices where we are out of bounds
        # For example, if x-rad is negative, we need to pad the left side of the crop
        # if x+rad is greater than the volume_data.shape[0], we need to pad the right side of the crop
        # Same thing for y and z
        def get_padding(x, rad, max_value):
            pad_min = 0
            pad_max = 0
            if x-rad < 0:
                pad_min = rad - x

            if x+rad > max_value:
                pad_max = x + rad - max_value
            
            return pad_min, pad_max

        pad_left, pad_right = get_padding(x, rad, self.volume_data.shape[0])
        pad_top, pad_bottom = get_padding(y, rad, self.volume_data.shape[1])
        pad_front, pad_back = get_padding(z, rad, self.volume_data.shape[2])

        placeholder = np.zeros((2*rad, 2*rad, 2*rad, self.n_coefs), dtype=np.float32)
        placeholder[pad_left:2*rad-pad_right, pad_top:2*rad-pad_bottom, pad_front:2*rad-pad_back] = crop

        if self.torch_convention:
            placeholder = np.transpose(placeholder, (3, 0, 1, 2))

        if dim_2d:
            return placeholder[:, 0, :, :] # 2D image
        else:
            return placeholder[:, :, :, :] # 3D image

    def _interpolate_at_coordinate(self, coord):
        interp = self.neigh_manager.get(coord, torch_convention=self.torch_convention)
        
        # Crop it to be evenly sized
        interp = interp[:, :, :-1, :-1, :-1]

        # The result is of an odd size, we need to crop the last dim
        if dim_2d:
            return interp[:, 0, :, :]
        else:
            return interp[:, :, :, :]


    def _get_coordinates(self):
        all_x = np.arange(0, self.volume_data.shape[0])[self.volume_data.shape[0]//2-20:self.volume_data.shape[0]//2+20]
        all_y = np.arange(0, self.volume_data.shape[1])[self.volume_data.shape[1]//2-20:self.volume_data.shape[1]//2+20]
        all_z = np.arange(0, self.volume_data.shape[2])[self.volume_data.shape[2]//2-20:self.volume_data.shape[2]//2+20]

        all_coords = list(it.product(all_x, all_y, all_z))
        self.rng.shuffle(all_coords)
        if self.is_train:
            coords = all_coords[:int(0.8*len(all_coords))]
        else:
            coords = all_coords[int(0.8*len(all_coords)):]

        # For testing purposes, we will only use one coordinate and repeat it multiple times
        # coords = [coords[0]] * 10000
        return coords

def setup_neighborhood_datasets(neighborhood_size=3, method='crop'):
    trainset = NeighborhoodDataset(train=True, neighborhood_size=neighborhood_size, method=method, torch_convention=True)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=10)

    testset = NeighborhoodDataset(train=False, neighborhood_size=neighborhood_size, method=method, torch_convention=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=True, num_workers=10)

    return trainloader, testloader


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = SmallWorkingFodfEncoder()
        self.decoder = SmallWorkingFodfDecoder()
    
    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out


def train(model, interp='crop'):
    trainloader, testloader = setup_neighborhood_datasets(neighborhood_size=4, method=interp)
    model.train()
    model = model.to(device='cuda')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    last_best_loss = np.inf
    nb_epochs = 100
    for epoch in range(nb_epochs):
        for i, inputs in enumerate(tqdm(trainloader)):
            inputs = inputs.to(device='cuda')
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                loss_item = loss.item()
                print('Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, i, loss_item))

                if loss_item < last_best_loss:
                    last_best_loss = loss_item
                    torch.save(model.state_dict(), 'best_model.pth')

    # Vizualize some reconstructions from the testloader
    n = 5 # number of reconstructions to vizualize
    fig, axes = plt.subplots(2, n, figsize=(20, 5))
    for i, inputs in enumerate(testloader):
        inputs = inputs.to(device='cuda')
        outputs = model(inputs)
        for j in range(n):
            if dim_2d:
                target = inputs[j, 0][None, ...].permute(1, 2, 0).cpu().detach().numpy()
                recons = outputs[j, 0][None, ...].permute(1, 2, 0).cpu().detach().numpy()
            else:
                target = inputs[j, 0][None, 0].permute(1, 2, 0).cpu().detach().numpy()
                recons = outputs[j, 0][None, 0].permute(1, 2, 0).cpu().detach().numpy()
            print("loss ({}): {}".format(j, criterion(outputs[j], inputs[j])))
            
            axes[0, j].imshow(target)
            axes[1, j].imshow(recons)

        break

    if dim_2d:
        test(model, 'reconstructions_2d.png')
    else:
        test(model, 'reconstructions_3d.png')

def test(model, out_file=None, interp='crop'):
    _, testloader = setup_neighborhood_datasets(neighborhood_size=4, method=interp)
    neigh_manager = NeighborhoodManager(
        data_volume=nib.load(VOLUME_PATH).get_fdata(),
        radius=16,
        add_neighborhood_vox=1, #0.375,
        neighborhood_type='grid',
        flatten=False,
        device='cuda',
        method='dwi_ml')
    model.eval()
    model = model.to(device='cuda')
    criterion = nn.MSELoss()
    n = 20 # number of reconstructions to vizualize
    fig, axes = plt.subplots(4, n//2, figsize=(20, 5))
    for i, inputs in enumerate(testloader):
        inputs = inputs.to(device='cuda')
        if interp=='interpolate':
            inputs = neigh_manager.get(inputs, torch_convention=True)
            inputs = inputs[:, :, :-1, :-1, :-1]
        outputs = model(inputs)

        print('Loss: {}'.format(criterion(outputs, inputs)))
        for j in range(n):
            if dim_2d:
                target = inputs[j, 0][None, ...].permute(1, 2, 0).cpu().detach().numpy()
                recons = outputs[j, 0][None, ...].permute(1, 2, 0).cpu().detach().numpy()
                target_slice = target
                recons_slice = recons
            else:
                target = inputs[j, 0][None].permute(1, 2, 3, 0).cpu().detach().numpy()
                recons = outputs[j, 0][None].permute(1, 2, 3, 0).cpu().detach().numpy()
                target_slice = target[:, 0]
                recons_slice = recons[:, 0]
            print("loss ({}): {}".format(j, criterion(outputs[j], inputs[j])))
            
            if j < n//2:
                axes[0, j].imshow(target_slice)
                axes[1, j].imshow(recons_slice)
            else:
                axes[2, j-n//2].imshow(target_slice)
                axes[3, j-n//2].imshow(recons_slice)

            # Save the images to disk as nifti images
            ex_dir = Path('examples')
            target_nib = nib.Nifti1Image(target, affine=np.eye(4))
            recons_nib = nib.Nifti1Image(recons, affine=np.eye(4))
            nib.save(target_nib, ex_dir / 'target_{}.nii.gz'.format(j))
            nib.save(recons_nib, ex_dir / 'recons_{}.nii.gz'.format(j))

    for i, inputs in enumerate(tqdm(testloader)):
        inputs = inputs.to(device='cuda')
        outputs = model(inputs)
        for j in range(n):
            print("j: {} n: {}".format(j, n))
            if dim_2d:
                target = inputs[j, 0][None, ...].permute(1, 2, 0).cpu().detach().numpy()
                recons = outputs[j, 0][None, ...].permute(1, 2, 0).cpu().detach().numpy()
            else:
                target = inputs[j, 0][None, 0].permute(1, 2, 0).cpu().detach().numpy()
                recons = outputs[j, 0][None, 0].permute(1, 2, 0).cpu().detach().numpy()
            print("loss ({}): {}".format(j, criterion(outputs[j], inputs[j])))
            
            if j < n//2:
                axes[0, j].imshow(target)
                axes[1, j].imshow(recons)
            else:
                axes[2, j-n//2].imshow(target)
                axes[3, j-n//2].imshow(recons)
        break

    if out_file is not None:
        plt.savefig(out_file)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--interp', choices=['crop', 'interpolate'], default='crop')
    args = parser.parse_args()

    model = Model()
    if args.ckpt is not None:
        print('Loading model from checkpoint...')
        model.load_state_dict(torch.load(args.ckpt))

    if args.test:
        print('Testing...')
        test(model, args.out_file, args.interp)
    else:
        train(model, args.interp)
