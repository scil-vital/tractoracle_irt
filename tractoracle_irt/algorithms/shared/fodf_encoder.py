import torch
import torch.nn as nn
import numpy as np
from tractoracle_irt.algorithms.shared.utils import ResidualBlock, ResNextBlock
from tractoracle_irt.algorithms.shared.batch_renorm import BatchRenorm1d, BatchRenorm3d
from tractoracle_irt.utils.utils import count_parameters

class DummyFodfEncoder(nn.Module):
    """
    This should not be a bottleneck
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        print(f"{self.__class__.__name__} __init__ with {count_parameters(self)} parameters")

    def forward(self, x):
        return x[:, :64, :3, :3, :3]
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        return
    
    def state_dict(self):
        return {}
    
class ExpFodfEncoder(nn.Module):
    """
    This should not be a bottleneck
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

        sc = 64 # Start channels
        self.layers = nn.Sequential(
            # 28x19x19x19
            nn.Conv3d(in_channels=28, out_channels=28, kernel_size=1, stride=1, padding=0),  # 28x19x19x19
            nn.ReLU(),
            nn.Conv3d(in_channels=28, out_channels=sc, kernel_size=3, stride=1, padding=1),  # 64x19x19x19
            nn.ReLU(),

            nn.Conv3d(in_channels=sc, out_channels=sc*2, kernel_size=3, stride=1, padding=1),  # 128x19x19x19
            nn.ReLU(),
            self.make_layer(in_channels=sc*2, cardinality=8, num_blocks=3, stride=1),  # 128x19x19x19

            nn.Conv3d(in_channels=sc*2, out_channels=sc*4, kernel_size=3, stride=1, padding=1),  # 256x19x19x19
            nn.ReLU(),
            self.make_layer(in_channels=sc*4, cardinality=16, num_blocks=3, stride=1),  # 256x19x19x19

            # MAXPOOL 19x19x19 -> 9x9x9
            nn.MaxPool3d(kernel_size=3, stride=2),  # 256x9x9x9
            nn.Conv3d(in_channels=sc*4, out_channels=sc*8, kernel_size=3, stride=1, padding=1),  # 512x9x9x9
            nn.ReLU(),
            self.make_layer(in_channels=sc*8, cardinality=16, num_blocks=2, stride=1),  # 512x9x9x9

            # MAXPOOL 9x9x9 -> 4x4x4
            nn.MaxPool3d(kernel_size=2, stride=2),  # 28x4x4x4
            nn.Conv3d(in_channels=sc*8, out_channels=sc*16, kernel_size=3, stride=1, padding=1),  # 1024x4x4x4
            nn.ReLU(),
            self.make_layer(in_channels=sc*16, cardinality=16, num_blocks=1, stride=1),  # 1024x4x4x4

            nn.Conv3d(in_channels=sc*16, out_channels=sc*32, kernel_size=3, stride=1, padding=1),  # 2048x4x4x4


            # MAXPOOL 4x4x4 -> 2x2x2
            # nn.MaxPool3d(kernel_size=2, stride=2),  # 28x2x2x2
            # nn.Conv3d(in_channels=sc*8, out_channels=sc*16, kernel_size=3, stride=1, padding=1),  # 2048x2x2x2
            # nn.ReLU(),
            # self.make_layer(in_channels=sc*16, cardinality=16, num_blocks=1, stride=1),  # 2048x2x2x2

            # MAXPOOL 2x2x2 -> 1x1x1
            # nn.MaxPool3d(kernel_size=2, stride=2),  # 4096x1x1x1
            # nn.Conv3d(in_channels=sc*16, out_channels=sc*32, kernel_size=3, stride=1, padding=1),  # 4096x1x1x1
            # nn.ReLU(),
            # self.make_layer(in_channels=sc*32, cardinality=16, num_blocks=1, stride=1),  # 4096x1x1x1
        )

        # self.decoding_layers = nn.Sequential(
        #     self.make_layer(in_channels=sc*32, cardinality=16, num_blocks=1, stride=1),  # 4096x1x1x1

        #     # UPSAMPLE 1x1x1 -> 2x2x2
        #     nn.ConvTranspose3d(in_channels=sc*32, out_channels=sc*16, kernel_size=2, stride=2, padding=0),  # 4096x2x2x2
        #     # nn.Conv3d(in_channels=sc*32, out_channels=sc*16, kernel_size=3, stride=1, padding=1),  # 2048x2x2x2
        #     # nn.ReLU(),
        #     self.make_layer(in_channels=sc*16, cardinality=16, num_blocks=1, stride=1),  # 2048x2x2x2

        #     # UPSAMPLE 2x2x2 -> 4x4x4
        #     nn.ConvTranspose3d(in_channels=sc*16, out_channels=sc*8, kernel_size=2, stride=2, padding=0),  # 28x4x4x4
        #     # nn.Conv3d(in_channels=sc*16, out_channels=sc*8, kernel_size=3, stride=1, padding=1),  # 1024x4x4x4
        #     # nn.ReLU(),
        #     self.make_layer(in_channels=sc*8, cardinality=16, num_blocks=1, stride=1),  # 1024x4x4x4

        #     # UPSAMPLE 4x4x4 -> 9x9x9
        #     nn.ConvTranspose3d(in_channels=sc*8, out_channels=sc*4, kernel_size=3, stride=2, padding=0),  # 28x9x9x9
        #     # nn.Conv3d(in_channels=sc*8, out_channels=sc*4, kernel_size=3, stride=1, padding=1),  # 512x9x9x9
        #     # nn.ReLU(),
        #     self.make_layer(in_channels=sc*4, cardinality=16, num_blocks=2, stride=1),  # 512x9x9x9

        #     # UPSAMPLE 9x9x9 -> 19x19x19
        #     nn.ConvTranspose3d(in_channels=sc*4, out_channels=sc*2, kernel_size=3, stride=2, padding=0),  # 28x19x19x19
        #     self.make_layer(in_channels=sc*2, cardinality=8, num_blocks=3, stride=1),  # 128x19x19x19

        #     nn.Conv3d(in_channels=sc*2, out_channels=sc, kernel_size=3, stride=1, padding=1),  # 128x19x19x19
        #     nn.ReLU(),
        #     nn.Conv3d(in_channels=sc, out_channels=28, kernel_size=1, stride=1, padding=0),  # 28x19x19x19
        # )

        print(f"{self.__class__.__name__} __init__ with {count_parameters(self)} parameters")

    def forward(self, x):
        out = self.layers(x)

        # print("encoder.out.shape: ", out.shape)
        # out = self.decoding_layers(out)
        # print("decoder.out.shape: ", out.shape)

        # assert False

        return out
    
    def make_layer(self, in_channels, cardinality, num_blocks, stride=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResNextBlock(in_channels, in_channels//2, cardinality, stride))
        return nn.Sequential(*layers)

    def load_state_dict(self, state_dict, strict = True, assign = False):
        return
    
    def state_dict(self):
        return {}
    
class ExpFodfDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sc = 64 # Start channels
        self.decoding_layers = nn.Sequential(

            # UPSAMPLE 1x1x1 -> 2x2x2
            # nn.ConvTranspose3d(in_channels=sc*32, out_channels=sc*16, kernel_size=2, stride=2, padding=0),  # 4096x2x2x2
            # nn.Conv3d(in_channels=sc*32, out_channels=sc*16, kernel_size=3, stride=1, padding=1),  # 2048x2x2x2
            # nn.ReLU(),
            # self.make_layer(in_channels=sc*16, cardinality=16, num_blocks=1, stride=1),  # 2048x2x2x2

            self.make_layer(in_channels=sc*32, cardinality=16, num_blocks=1, stride=1),  # 2048x4x4x4
            nn.Conv3d(in_channels=sc*32, out_channels=sc*16, kernel_size=3, stride=1, padding=1),  # 1024x4x4x4
            nn.ReLU(),

            # UPSAMPLE 2x2x2 -> 4x4x4
            # nn.ConvTranspose3d(in_channels=sc*32, out_channels=sc*16, kernel_size=2, stride=2, padding=0),  # 28x4x4x4
            # nn.Conv3d(in_channels=sc*16, out_channels=sc*8, kernel_size=3, stride=1, padding=1),  # 1024x4x4x4
            # nn.ReLU(),
            self.make_layer(in_channels=sc*16, cardinality=16, num_blocks=2, stride=1),  # 1024x4x4x4

            # UPSAMPLE 4x4x4 -> 9x9x9
            nn.ConvTranspose3d(in_channels=sc*16, out_channels=sc*8, kernel_size=3, stride=2, padding=0),  # 28x9x9x9
            # nn.Conv3d(in_channels=sc*8, out_channels=sc*4, kernel_size=3, stride=1, padding=1),  # 512x9x9x9
            # nn.ReLU(),
            self.make_layer(in_channels=sc*8, cardinality=16, num_blocks=2, stride=1),  # 512x9x9x9

            # UPSAMPLE 9x9x9 -> 19x19x19
            nn.ConvTranspose3d(in_channels=sc*8, out_channels=sc*4, kernel_size=3, stride=2, padding=0),  # 28x19x19x19
            self.make_layer(in_channels=sc*4, cardinality=8, num_blocks=3, stride=1),  # 128x19x19x19

            nn.Conv3d(in_channels=sc*4, out_channels=sc*2, kernel_size=3, stride=1, padding=1),  # 128x19x19x19
            nn.ReLU(),
            nn.Conv3d(in_channels=sc*2, out_channels=sc, kernel_size=3, stride=1, padding=1),  # 128x19x19x19
            nn.ReLU(),
            nn.Conv3d(in_channels=sc, out_channels=28, kernel_size=1, stride=1, padding=0),  # 28x19x19x19
        )
    
    def forward(self, x):
        out = self.decoding_layers(x)
        return out
    
    def make_layer(self, in_channels, cardinality, num_blocks, stride=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResNextBlock(in_channels, in_channels//2, cardinality, stride))
        return nn.Sequential(*layers)

class SimpleFodfEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sc = 32 # Start channels
        # self.layers = nn.Sequential(
        #     # 28x19x19x19
        #     nn.Conv3d(in_channels=28, out_channels=28, kernel_size=1, stride=1, padding=0),  # 28x19x19x19
        #     nn.Tanh(),
        #     nn.Conv3d(in_channels=28, out_channels=sc, kernel_size=1, stride=1, padding=0),  # 64x19x19x19
        #     nn.Tanh(),
        #     nn.Conv3d(in_channels=sc, out_channels=sc*2, kernel_size=3, stride=1, padding=1),  # 128x19x19x19
        #     nn.Tanh(),

        # )

        # MLP
        self.layers = nn.Sequential(
            nn.Linear(28*19*19*19, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.Tanh(),
        )

    def forward(self, x):
        x_flat = x.reshape(x.size(0), -1)
        out = self.layers(x_flat)
        return out

class SimpleFodfDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.Tanh(),
            nn.Linear(2048, 28*19*19*19),
        )

        # sc = 32 # Start channels
        # self.layers = nn.Sequential(
        #     # 64x19x19x19
        #     nn.Conv3d(in_channels=sc*2, out_channels=sc, kernel_size=1, stride=1, padding=0),  # 128x19x19x19
        #     nn.Tanh(),
        #     # nn.BatchNorm3d(sc),
        #     nn.Conv3d(in_channels=sc, out_channels=28, kernel_size=1, stride=1, padding=0),  # 28x19x19x19
        #     nn.Tanh(),
        # )

    def forward(self, x):
        out = self.layers(x)
        out = out.reshape(x.size(0), 28, 19, 19, 19)
        return out
    

class FodfEncoder(nn.Module):
    def __init__(self, n_coeffs=28, renorm=False, activation=nn.ReLU):
        super().__init__()

        norm_layer = nn.BatchNorm3d if not renorm else BatchRenorm3d

        # self.encoder = nn.Sequential(
        #     nn.Conv3d(in_channels=n_coeffs, out_channels=n_coeffs, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm3d(n_coeffs),  # 28x19x19x19
        #     activation(),
        #     nn.Conv3d(in_channels=n_coeffs, out_channels=n_coeffs, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm3d(n_coeffs),  # 28x19x19x19
        #     activation(),
        #     nn.Conv3d(in_channels=n_coeffs, out_channels=32, kernel_size=3, stride=1, padding=1),  # 90x108x90x64
        #     norm_layer(32),
        #     activation(),

        #     ResidualBlock(32, norm_layer=norm_layer),  # 64x19x19x19
        #     ResidualBlock(32, norm_layer=norm_layer),  # 64x19x19x19
        #     nn.MaxPool3d(kernel_size=2, stride=2),  # 64x10x10x10
        #     nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # 128x10x10x10

        #     ResidualBlock(64, norm_layer=norm_layer),  # 128x10x10x10
        #     ResidualBlock(64, norm_layer=norm_layer),  # 128x10x10x10
        #     nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # 256x5x5x5

        #     ResidualBlock(128, norm_layer=norm_layer),  # 256x5x5x5
        #     ResidualBlock(128, norm_layer=norm_layer),  # 256x5x5x5
        #     ResidualBlock(128, norm_layer=norm_layer),  # 256x5x5x5
        #     nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),  # 512x3x3x3

        #     ResidualBlock(256, norm_layer=norm_layer),  # 512x3x3x3
        #     ResidualBlock(256, norm_layer=norm_layer),  # 512x3x3x3

        #     # Reduce the number of channels, otherwise the latent space is too large to fit in memory.
        #     # nn.Conv3d(256, 128, kernel_size=1, stride=1, padding=0),  # 256x3x3x3
        #     # norm_layer(128),
        #     # activation(),

        #     # nn.Conv3d(128, 64, kernel_size=1, stride=1, padding=0),  # 128x3x3x3
        #     # norm_layer(64),
        #     # activation(),
        # )

        self.activ = activation()

        sc = 32

        # Layers
        self.conv1x1_1 = nn.Conv3d(n_coeffs, n_coeffs, kernel_size=1) # 28x19x19x19
        self.conv1x1_2 = nn.Conv3d(n_coeffs, sc, kernel_size=1) # 64x19x19x19
        
        self.conv3x3_3 = nn.Conv3d(sc, sc*2, kernel_size=3, stride=1, padding=1) # 128x19x19x19
        self.bn_3 = nn.BatchNorm3d(sc*2) # 128x19x19x19

        self.conv_4 = nn.Conv3d(sc*2, sc*4, kernel_size=3, stride=1, padding=1) # 128x19x19x19
        self.bn_4 = nn.BatchNorm3d(sc*4) # 128x19x19x19
        self.layer_1 = self.make_layer(in_channels=sc*4, cardinality=8, num_blocks=3, stride=1) # 128x19x19x19

        self.conv_5 = nn.Conv3d(sc*4, sc*8, kernel_size=3, stride=2, padding=1) # 256x10x10x10
        self.bn_5 = nn.BatchNorm3d(sc*8) # 256x10x10x10
        self.layer_2 = self.make_layer(in_channels=sc*8, cardinality=16, num_blocks=3, stride=1) # 256x10x10x10

        self.conv_6 = nn.Conv3d(sc*8, sc*16, kernel_size=3, stride=2, padding=1) # 512x5x5x5
        self.bn_6 = nn.BatchNorm3d(sc*16) # 512x5x5x5
        self.layer_3 = self.make_layer(in_channels=sc*16, cardinality=32, num_blocks=3, stride=1) # 512x5x5x5

        self.conv_7 = nn.Conv3d(sc*16, sc*32, kernel_size=3, stride=2, padding=1) # 1024x3x3x3
        self.bn_7 = nn.BatchNorm3d(sc*32) # 1024x3x3x3
        self.layer_4 = self.make_layer(in_channels=sc*32, cardinality=64, num_blocks=3, stride=1) # 1024x3x3x3

        self.flattener = nn.Flatten()

        self._flat_output_size = (sc*32) * 3 * 3 * 3

        print(f"{self.__class__.__name__} __init__ with {count_parameters(self)} parameters")

    def make_layer(self, in_channels, cardinality, num_blocks, stride=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResNextBlock(in_channels, in_channels//2, cardinality, stride))
        return nn.Sequential(*layers)

    @property
    def flat_output_size(self):
        return self._flat_output_size
    
    def forward(self, x, flatten=False, swap_channels=False):
        if swap_channels:
            x = x.permute(0, 4, 1, 2, 3)
        
        x = self.conv1x1_1(x)
        x = self.activ(x)
        x = self.conv1x1_2(x)
        x = self.activ(x)

        x = self.conv3x3_3(x)
        x = self.bn_3(x)
        x = self.activ(x)

        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.activ(x)
        x = self.layer_1(x)

        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.activ(x)
        x = self.layer_2(x)

        x = self.conv_6(x)
        x = self.bn_6(x)
        x = self.activ(x)
        x = self.layer_3(x)

        x = self.conv_7(x)
        x = self.bn_7(x)
        x = self.activ(x)
        x = self.layer_4(x)

        if flatten:
            x = self.flattener(x)
            assert x.shape[1] == self.flat_output_size, \
                "The flattened output is not the expected size of " \
                f"{self.flat_output_size}. Make sure that the input size is " \
                "in the correct order (N, C, D, H, W) as specified in the " \
                "PyTorch documentation about Conv3d layers."

        return x
    
class NoDownsampleFodfEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            nn.Conv3d(in_channels=28, out_channels=64, kernel_size=1, stride=1, padding=0),  # 28x19x19x19
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),  # 64x17x17x17
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0), # 128x15x15x15
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0),  # 256x13x13x13
            nn.ReLU(),
            nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0),  # 512x11x11x11
            # nn.ReLU(),
            # nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0),  # 1024x9x9x9
            # nn.ReLU(),
            # nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0),  # 1024x7x7x7
            # nn.ReLU(),
            # nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0),  # 1024x5x5x5
            # nn.ReLU(),
            # nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0),  # 1024x3x3x3
            # nn.ReLU(),
            # nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0),  # 1024x1x1x1
        )

    def forward(self, x):
        out = self.layers(x)
        return out

class NoDownsampleFodfDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            # nn.ConvTranspose3d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0),  # 1024x3x3x3
            # nn.ReLU(),
            # nn.ConvTranspose3d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0),  # 1024x5x5x5
            # nn.ReLU(),
            # nn.ConvTranspose3d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0),  # 1024x7x7x7
            # nn.ReLU(),
            # nn.ConvTranspose3d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0),  # 1024x9x9x9
            # nn.ReLU(),
            # nn.ConvTranspose3d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0),  # 1024x11x11x11
            # nn.ReLU(),
            nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=0),  # 512x13x13x13
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0),  # 256x15x15x15
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0),  # 128x17x17x17
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),  # 64x19x19x19
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=28, kernel_size=1, stride=1, padding=0),  # 28x19x19x19
        )

    def forward(self, x):
        out = self.layers(x)
        return out

class LinLatentEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = (28, 19, 19, 19)
        self.sc = 64
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels=28, out_channels=self.sc, kernel_size=3, stride=1, padding=1),  # 64x19x19x19
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels=self.sc, out_channels=self.sc*2, kernel_size=3, stride=2, padding=0),  # 128x9x9x9
            nn.BatchNorm3d(self.sc*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels=self.sc*2, out_channels=self.sc*4, kernel_size=3, stride=2, padding=0),  # 256x4x4x4
            nn.BatchNorm3d(self.sc*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels=self.sc*4, out_channels=self.sc*8, kernel_size=3, stride=1, padding=0), # 512x2x2x2
            nn.BatchNorm3d(self.sc*8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flat_fts = self.get_flat_fts(self.convs)
        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )

    def get_flat_fts(self, fts):
        f = fts(torch.ones(1, *self.input_size))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.convs(x.view(-1, *self.input_size))
        x = x.reshape(-1, self.flat_fts)
        x = self.linear(x)
        return x

class LinLatentDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc_in_dim = 1024
        self.fc_out_dim = 4096
        self.sc = 64

        self.linear = nn.Sequential(
            nn.Linear(1024, 512*2*2*2),
            nn.ReLU(),
            nn.BatchNorm1d(512*2*2*2),
        )

        self.convs = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.sc*8, out_channels=self.sc*4, kernel_size=3, stride=1, padding=0),  # 256x4x4x4
            nn.BatchNorm3d(self.sc*4),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=self.sc*4, out_channels=self.sc*2, kernel_size=3, stride=2, padding=0),  # 128x9x9x9
            nn.BatchNorm3d(self.sc*2),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=self.sc*2, out_channels=self.sc, kernel_size=3, stride=2, padding=0),  # 64x19x19x19
            nn.BatchNorm3d(self.sc),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=self.sc, out_channels=28, kernel_size=3, stride=1, padding=1),  # 28x19x19x19
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.sc*8, 2, 2, 2)
        x = self.convs(x)
        return x

class ResidualBlockV2(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(in_channels),
            # nn.ReLU()
        )

    def forward(self, x):
        residue = x
        out = self.block(x)
        out += residue
        return out
    
def downsampling_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

class LinLatentEncoderV2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = (1, 31, 31, 31)
        self.sc = 64
        self.convs = nn.Sequential(
            nn.ConstantPad3d(padding=(1, 0, 1, 0, 1, 0), value=0.0), # 28x32x32x32
            nn.Conv3d(in_channels=1, out_channels=28, kernel_size=3, stride=1, padding=0), # 28x32x32x32
            nn.ReLU(),

            ResidualBlockV2(in_channels=28), # 28x32x32x32
            ResidualBlockV2(in_channels=28), # 28x32x32x32
            ResidualBlockV2(in_channels=28), # 28x32x32x32
            ResidualBlockV2(in_channels=28), # 28x32x32x32

            downsampling_block(in_channels=28, out_channels=32), # 32x16x16x16

            ResidualBlockV2(in_channels=32), # 32x16x16x16
            ResidualBlockV2(in_channels=32), # 32x16x16x16
            ResidualBlockV2(in_channels=32), # 32x16x16x16
            ResidualBlockV2(in_channels=32), # 32x16x16x16

            downsampling_block(in_channels=32, out_channels=48), # 48x8x8x8

            ResidualBlockV2(in_channels=48), # 48x8x8x8
            ResidualBlockV2(in_channels=48), # 48x8x8x8
            ResidualBlockV2(in_channels=48), # 48x8x8x8
            ResidualBlockV2(in_channels=48), # 48x8x8x8

            downsampling_block(in_channels=48, out_channels=96), # 128x4x4x4

            ResidualBlockV2(in_channels=96), # 96x4x4x4
            ResidualBlockV2(in_channels=96), # 96x4x4x4
            ResidualBlockV2(in_channels=96), # 96x4x4x4
            ResidualBlockV2(in_channels=96), # 96x4x4x4
        )

        self.flat_fts = self.get_flat_fts(self.convs)
        print("Flat fts: ", self.flat_fts)
        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_fts, 16384)
        )

    def get_flat_fts(self, fts):
        f = fts(torch.ones(1, *self.input_size))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.convs(x)
        x = self.lin(x)
        return x

def upsampling_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

class LinLatentDecoderV2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc_in_dim = 16384
        self.fc_out_dim = 96*4*4*4
        self.sc = 64

        self.linear = nn.Sequential(
            nn.Linear(self.fc_in_dim, self.fc_out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_out_dim),
        )

        self.convs = nn.Sequential(
            nn.Unflatten(1, (96, 4, 4, 4)),
            
            ResidualBlockV2(in_channels=96), # 96x4x4x4
            ResidualBlockV2(in_channels=96), # 96x4x4x4

            upsampling_block(in_channels=96, out_channels=64), # 64x8x8x8

            ResidualBlockV2(in_channels=64), # 64x8x8x8
            ResidualBlockV2(in_channels=64), # 64x8x8x8

            upsampling_block(in_channels=64, out_channels=48), # 48x16x16x16

            ResidualBlockV2(in_channels=48), # 48x16x16x16
            ResidualBlockV2(in_channels=48), # 48x16x16x16
            ResidualBlockV2(in_channels=48), # 48x16x16x16
            ResidualBlockV2(in_channels=48), # 48x16x16x16

            upsampling_block(in_channels=48, out_channels=32), # 32x32x32x32

            ResidualBlockV2(in_channels=32), # 32x32x32x32
            ResidualBlockV2(in_channels=32), # 32x32x32x32
            ResidualBlockV2(in_channels=32), # 32x32x32x32
            ResidualBlockV2(in_channels=32), # 32x32x32x32

            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1), # 28x32x32x32
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.convs(x)
        return x[..., :-1, :-1, :-1]
    
###########################################################################
# Working FODF Encoder/Decoder !
###########################################################################
dim_2d = False
conv_layer = nn.Conv2d if dim_2d else nn.Conv3d
bn_layer = nn.BatchNorm2d if dim_2d else nn.BatchNorm3d
conv_t_layer = nn.ConvTranspose2d if dim_2d else nn.ConvTranspose3d
get_flat_size = lambda dim_size: dim_size**2 if dim_2d else dim_size**3
get_flat_shape = lambda dim_size: (dim_size, dim_size) if dim_2d else (dim_size, dim_size, dim_size)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.block = nn.Sequential(
            conv_layer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            bn_layer(in_channels),
            nn.ReLU(),
            conv_layer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            bn_layer(in_channels),
        )

    def forward(self, x):
        residue = x
        out = self.block(x)
        out += residue
        return out
    
def downsampling_block(in_channels, out_channels):
    return nn.Sequential(
        conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU()
    )

def upsampling_block(in_channels, out_channels):
    return nn.Sequential(
        conv_t_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

def small_downsampling_block(in_channels, out_channels):
    return nn.Sequential(
        conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0), # No padding => reduces by 2.
        nn.ReLU(),
        conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU()
    )

def small_upsampling_block(in_channels, out_channels):
    return nn.Sequential(
        conv_t_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

class WorkingFodfEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            # 32x32x3
            ResidualBlock(in_channels=28), # 32x32x32
            ResidualBlock(in_channels=28), # 32x32x32
            ResidualBlock(in_channels=28), # 32x32x32
            ResidualBlock(in_channels=28), # 32x32x32

            downsampling_block(in_channels=28, out_channels=48), # 16x16x48
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),

            downsampling_block(in_channels=48, out_channels=96), # 8x8x96
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            # downsampling_block(in_channels=96, out_channels=192), # 4x4x192
            # downsampling_block(in_channels=192, out_channels=96), # 2x2x96

            # nn.Flatten(),
            # nn.Linear(get_flat_size(8)*96, self.latent_space_size),
            downsampling_block(in_channels=96, out_channels=32), # 4x4x64
            ResidualBlock(in_channels=32),
            ResidualBlock(in_channels=32),
            ResidualBlock(in_channels=32),
            ResidualBlock(in_channels=32),
        )

        self.output_size = get_flat_size(4)*32 # 4x4x4x32 = 2048

        print(f'{self.__class__.__name__}: {count_parameters(self)} params')

    def forward(self, x):
        return self.layers(x)

class WorkingFodfDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            # nn.Linear(self.latent_space_size, get_flat_size(8)*96),
            # nn.Unflatten(1, (96, *get_flat_shape(8))),
            ResidualBlock(in_channels=32),
            ResidualBlock(in_channels=32),
            ResidualBlock(in_channels=32),
            ResidualBlock(in_channels=32),
            upsampling_block(in_channels=32, out_channels=96), # 8x8x96
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            # self.upsampling_block(in_channels=96, out_channels=192), # 4x4x192
            # self.upsampling_block(in_channels=192, out_channels=96), # 8x8x96
            upsampling_block(in_channels=96, out_channels=48), # 16x16x48
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),

            conv_t_layer(in_channels=48, out_channels=48, kernel_size=2, stride=2), # 32x32x48
            nn.ReLU(),
            conv_layer(in_channels=48, out_channels=28, kernel_size=3, stride=1, padding=1),
        )

        print(f'{self.__class__.__name__}: {count_parameters(self)} params')

    def forward(self, x):
        return self.layers(x)
    
class SmallWorkingFodfEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            # 32x32x3
            ResidualBlock(in_channels=28), # 9x9x9
            ResidualBlock(in_channels=28), # 9x9x9
            ResidualBlock(in_channels=28), # 9x9x9
            ResidualBlock(in_channels=28), # 9x9x9

            small_downsampling_block(in_channels=28, out_channels=48), # 7x7x7
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),

            small_downsampling_block(in_channels=48, out_channels=96), # 5x5x5
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),

            small_downsampling_block(in_channels=96, out_channels=32), # 3x3x3
            ResidualBlock(in_channels=32),
            ResidualBlock(in_channels=32),
            ResidualBlock(in_channels=32),
            ResidualBlock(in_channels=32),
        )

        self.output_size = get_flat_size(3)*32 # 4x4x4x32 = 2048

        print(f'{self.__class__.__name__}: {count_parameters(self)} params')

    def forward(self, x):
        return self.layers(x)

class SmallWorkingFodfDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            # 3x3x3
            ResidualBlock(in_channels=32),
            ResidualBlock(in_channels=32),
            ResidualBlock(in_channels=32),
            ResidualBlock(in_channels=32),
            small_upsampling_block(in_channels=32, out_channels=96), # 5x5x5

            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            ResidualBlock(in_channels=96),
            small_upsampling_block(in_channels=96, out_channels=48), # 7x7x7

            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),
            ResidualBlock(in_channels=48),
            conv_t_layer(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=0), # 9x9x9
            nn.ReLU(),
            conv_layer(in_channels=48, out_channels=28, kernel_size=3, stride=1, padding=1),
        )

        print(f'{self.__class__.__name__}: {count_parameters(self)} params')

    def forward(self, x):
        return self.layers(x)


def encoding_layers(in_channels, norm_func=nn.Identity, **kwargs):
    # Assume that the input is of size 5x5x5
    return nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1), # 5x5x5
        norm_func(32, **kwargs),
        nn.ReLU(),
        nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0), # 3x3x3
        norm_func(64, **kwargs),
        nn.ReLU(),
        nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 3x3x3
        norm_func(64, **kwargs),
        nn.ReLU(),
        nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0), # 1x1x1
        norm_func(128, **kwargs),
        nn.ReLU(),
        nn.Flatten()
    ), 128
