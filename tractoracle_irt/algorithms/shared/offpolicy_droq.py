import torch
import torch.nn as nn
from tractoracle_irt.algorithms.shared.offpolicy import SACActorCritic, MaxEntropyActor
from tractoracle_irt.algorithms.shared.utils import (
    format_widths, make_fc_network, make_conv_network)
from tractoracle_irt.environments.state import State, StateShape
from tractoracle_irt.algorithms.shared.batch_renorm import BatchRenorm1d, BatchRenorm3d
from tractoracle_irt.algorithms.shared.fodf_encoder import FodfEncoder

class DroQCritic(nn.Module):
    def __init__(
        self,
        state_dim: StateShape,
        action_dim: int,
        hidden_dims: str,
        critic_size_factor=1,
        dropout_rate=0.01
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: str
                String representing layer widths

        """
        super(DroQCritic, self).__init__()
        self.state_is_flat = state_dim.is_flat

        print("Init DroQCritic with state_flat: ", self.state_is_flat)

        if self.state_is_flat:
            flat_neigh_size = state_dim.neighborhood_common_shape[0]
        else:
            conv_state_shape = (state_dim.nb_sh_coefs, state_dim.depth,
                        state_dim.height, state_dim.width)
            self.q_neighbor_encoder = FodfEncoder(n_coeffs=conv_state_shape[0])
            flat_neigh_size = self.q_neighbor_encoder.flat_output_size
        
        full_fc_state_dim = flat_neigh_size + state_dim.prev_dirs_size
        print("Full FC state dim: ", full_fc_state_dim)
        self.hidden_layers = format_widths(
            hidden_dims) * critic_size_factor

        self.activation = nn.ReLU
        
        # From DroQ implementation, we need to introduce
        # dropout layers followed by layer normalization.
        input_size = full_fc_state_dim + action_dim
        print("input size: ", input_size)
        self.q1 = nn.Sequential(
            nn.Linear(input_size, self.hidden_layers[0]),
            nn.Dropout1d(dropout_rate),
            nn.LayerNorm(self.hidden_layers[0]),
            nn.ReLU(),

            nn.Linear(self.hidden_layers[0], self.hidden_layers[1]),
            nn.Dropout1d(dropout_rate),
            nn.LayerNorm(self.hidden_layers[1]),
            nn.ReLU(),

            nn.Linear(self.hidden_layers[1], self.hidden_layers[2]),
            nn.Dropout1d(dropout_rate),
            nn.LayerNorm(self.hidden_layers[2]),
            nn.ReLU(),

            nn.Linear(self.hidden_layers[2], 1)
        )

    def forward(self, state: State, action) -> torch.Tensor:

        # Predict on single state-action pair
        if self.state_is_flat:
            encoded_neighborhood_1 = state.neighborhood
        else:
            encoded_neighborhood_1 = self.q_neighbor_encoder(state.neighborhood, flatten=True)
        q1_states_input = torch.cat([encoded_neighborhood_1, state.prev_dirs], -1)
        q1_input = torch.cat([q1_states_input, action], -1)
        pred = self.q1(q1_input).squeeze(-1)
        return pred

class DroQDoubleCritic(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.q1 = DroQCritic(*args, **kwargs)
        self.q2 = DroQCritic(*args, **kwargs)
    
    def forward(self, state, action) -> torch.Tensor:
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2

class DroQActorCritic(SACActorCritic):
    def __init__(
            self,
            state_dim: StateShape,
            action_dim,
            hidden_dims,
            device,
            dropout_rate=0.01
    ):
        self.device = device
        self.actor = MaxEntropyActor(
            state_dim, action_dim, hidden_dims,
        ).to(device)

        self.critic = DroQDoubleCritic(
            state_dim, action_dim, hidden_dims, dropout_rate=dropout_rate
        ).to(device)