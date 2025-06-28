import torch
from dataclasses import dataclass

class StateShape(object):
    def __init__(self,
                 nb_streamlines: int,
                 neighborhood_common_shape: tuple,
                 prev_dirs_size: int):
        self.nb_streamlines = nb_streamlines
        self.neighborhood_common_shape = neighborhood_common_shape
        self.prev_dirs_size = prev_dirs_size[-1] if isinstance(prev_dirs_size, torch.Size) else prev_dirs_size

        # Validate the input parameters.
        assert self.nb_streamlines >= 0, "Number of streamlines must be positive."

        for dim in self.neighborhood_common_shape:
            assert dim >= 0, "Common shape dimensions must be positive."

        assert self.prev_dirs_size >= 0, "Number of previous directions must be positive."

    @property
    def is_flat(self):
        return True

    def to_dict(self):
        return {
            'nb_streamlines': self.nb_streamlines,
            'neighborhood_common_shape': self.neighborhood_common_shape,
            'prev_dirs_size': self.prev_dirs_size,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            nb_streamlines=d['nb_streamlines'],
            neighborhood_common_shape=d['neighborhood_common_shape'],
            prev_dirs_size=d['prev_dirs_size']
        )
    
    def __eq__(self, other):
        # Make sure each attribute of the two objects are equal
        for attr in self.__dict__:
            if getattr(self, attr) != getattr(other, attr):
                return False
            
        return True
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.nb_streamlines}, {self.neighborhood_common_shape}, {self.prev_dirs_size})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.nb_streamlines}, {self.neighborhood_common_shape}, {self.prev_dirs_size})"    


class State(object):
    def __init__(self, neighborhood=None, previous_directions=None, coords=None, device=None):
        if neighborhood is not None:
            self._neighborhood = neighborhood
        else:
            self._neighborhood = torch.tensor([], dtype=torch.float32)
            raise ValueError("Neighborhood must be provided to create a state.")

        if previous_directions is not None:
            self._previous_directions = previous_directions
        else:
            self._previous_directions = torch.tensor([], dtype=torch.float32)
            raise ValueError("Previous directions must be provided.")

        self._coords = coords

        self._shape = self._init_shape(self.neighborhood.shape, self._previous_directions.shape)

    def _init_shape(self, neighborhood_shape, prev_dirs_shape):
        assert neighborhood_shape[0] == prev_dirs_shape[0], "Number of streamlines must be the same."
        return StateShape(
            nb_streamlines=neighborhood_shape[0],
            neighborhood_common_shape=neighborhood_shape[1:],
            prev_dirs_size=prev_dirs_shape[1:]
        )

    @classmethod
    def zeros(cls, shape, prev_dirs_size, device=None, dtype=torch.float32):
        print("State shape: {}".format(shape))
        state_conv = torch.zeros(shape, device=device, dtype=dtype)
        previous_directions = torch.zeros((shape[0], prev_dirs_size), device=device, dtype=dtype)
        coords = torch.zeros((shape[0], 3), device=device, dtype=dtype)
        return cls(state_conv, previous_directions, coords)
    
    @classmethod
    def ones(cls, shape, prev_dirs_size, device=None, dtype=torch.float32):
        print("State shape: {}".format(shape))
        state_conv = torch.ones(shape, dtype=dtype, device=device)
        previous_directions = torch.ones((shape[0], prev_dirs_size), dtype=dtype, device=device)
        coords = torch.zeros((shape[0], 3), dtype=dtype, device=device)
        return cls(state_conv, previous_directions, coords)

    def to(self, device, copy=False, non_blocking=False):
        self._neighborhood = self._neighborhood.to(device, copy=copy, non_blocking=non_blocking)
        self._previous_directions = self._previous_directions.to(device, copy=copy, non_blocking=non_blocking)

        if self._coords is not None:
            self._coords = self._coords.to(device, copy=copy, non_blocking=non_blocking)

        return self
    
    def pin_memory(self):
        self._neighborhood = self._neighborhood.pin_memory()
        self._previous_directions = self._previous_directions.pin_memory()
        return self
    
    def index_select(self, dim, index):
        state_slice = self._neighborhood.index_select(dim, index)
        dirs_slice = self._previous_directions.index_select(dim, index)

        coords_slice = None
        if coords_slice is not None:
            coords_slice = self._coords.index_select(dim, index)

        return self.__class__(state_slice, dirs_slice, coords_slice)

    @property
    def shape(self):
        return self._shape
    
    def __len__(self):
        return self._neighborhood.shape[0]

    @property
    def neighborhood(self):
        return self._neighborhood
    
    @neighborhood.setter
    def neighborhood(self, value):
        self._neighborhood = value

    @property
    def prev_dirs(self):
        return self._previous_directions
    
    @prev_dirs.setter
    def prev_dirs(self, value):
        self._previous_directions = value

    @property
    def coords(self):
        return self._coords
    
    @coords.setter
    def coords(self, value):
        self._coords = value

    def __getitem__(self, indices):
        state_slice = self._neighborhood[indices]
        dirs_slice = self._previous_directions[indices]

        coords_slice = None
        if self._coords is not None:
            coords_slice = self._coords[indices]

        return self.__class__(state_slice, dirs_slice, coords_slice)
    
    def __setitem__(self, indices, other):
        if isinstance(other, State):
            self._neighborhood[indices] = other._neighborhood
            self._previous_directions[indices] = other._previous_directions
            if self._coords is not None:
                self._coords[indices] = other._coords
            else:
                self._coords = other._coords
        elif isinstance(other, tuple) or isinstance(other, list):
            assert len(other) == 2 or len(other) == 3, "Expected a tuple of tensors holding" \
                " the state and previous directions only."
            
            self._neighborhood[indices] = other[0]
            self._previous_directions[indices] = other[1]
            if len(other) == 3:
                if self._coords is not None:
                    self._coords[indices] = other[2]
                else:
                    self._coords = other[2]
        elif isinstance(other, torch.Tensor):
            assert other.shape[0] == 2 or other.shape[0] == 3, "Expected a tensor of dim 2 holding" \
                " the state and previous directions only."
            self._neighborhood[indices] = other[0]
            self._previous_directions[indices] = other[1]
            if other.shape[0] == 3:
                if self._coords is not None:
                    self._coords[indices] = other[2]
                else:
                    self._coords = other[2]
        else:
            raise ValueError("Expected a State object or a tuple of tensors.")

class ConvStateShape(StateShape):

    def __init__(self, nb_streamlines, *args, prev_dirs_size=None):
        """
        This constructor accepts the following parameters only:
        1. ConvStateShape(nb_streamlines, neighborhood_common_shape, prev_dirs_size)
        2. ConvStateShape(nb_streamlines, nb_sh_coefs, depth, height, width, prev_dirs_size)
        """
        if len(args) == 1:
            neighborhood_common_shape = args[0]
        elif len(args) == 4:
            nb_sh_coefs, depth, height, width = args
            neighborhood_common_shape = (nb_sh_coefs, depth, height, width)
        else:
            raise ValueError("Invalid number of arguments.")

        super().__init__(nb_streamlines, neighborhood_common_shape, prev_dirs_size)

        self.nb_sh_coefs = neighborhood_common_shape[0]
        self.depth = neighborhood_common_shape[1]
        self.height = neighborhood_common_shape[2]
        self.width = neighborhood_common_shape[3]

    @property
    def is_flat(self):
        return False

    def to_dict(self):
        super_dict = super().to_dict()
        super_dict.update({
            'nb_sh_coefs': self.nb_sh_coefs,
            'depth': self.depth,
            'height': self.height,
            'width': self.width
        })
        return super_dict

    @classmethod
    def from_dict(cls, d):
        return cls(
            d['nb_streamlines'],
            d['nb_sh_coefs'],
            d['depth'],
            d['height'],
            d['width'],
            prev_dirs_size=d['prev_dirs_size'],
        )
    
class ConvState(State):
    def _init_shape(self, neighborhood_shape, prev_dirs_shape):
        assert neighborhood_shape[0] == prev_dirs_shape[0], "Number of streamlines must be the same."
        return ConvStateShape(
            neighborhood_shape[0],
            neighborhood_shape[1:],
            prev_dirs_size=prev_dirs_shape[1:]
        )
