import os
from dataclasses import dataclass, field, fields

@dataclass
class HParams:
    algorithm: str = field(default=None, init=False, repr=False)

    # General experiment parameters
    experiment: str
    experiment_id: str
    experiment_path: str
    workspace: str
    offline: bool

    # Data parameters
    target_sh_order: int
    dataset_file: str

    # Environment parameters
    step_size: int
    noise: float
    reward_with_gt: bool
    binary_stopping_threshold: float
    neighborhood_radius: int
    neighborhood_type: str
    flatten_state: bool
    conv_state: bool # Redundant with flatten_state
    fodf_encoder_ckpt: str
    interpolation: str
    exclude_direct_neigh: bool

    # Oracle parameters
    oracle_crit_checkpoint: str
    oracle_reward_checkpoint: str
    oracle_bonus: float
    oracle_validator: bool
    oracle_stopping_criterion: bool

    # Tractometer parameters
    tractometer_validator: bool
    tractometer_dilate: int
    tractometer_reference: str
    scoring_data: str

    # Extractor parameters
    extractor_validator: bool
    extractor_target: str

    # RBX Filterer parameters
    rbx_validator: bool
    singularity_image: str
    atlas_directory: str

    # Verifyber Filterer parameters
    verifyber_validator: bool
    verifyber_image_path: str

    # Tracking parameters
    npv: int
    theta: float # Angular thresholds
    min_length: int
    max_length: int
    n_actor: int
    n_dirs: int

    # Learning parameters
    max_ep: int
    lr: float
    gamma: float
    alignment_weighting: float

    # Logging parameters
    use_comet: bool
    log_interval: int
    
    # Other parameters
    hidden_dims: str
    rng_seed: int
    backup_dir: str

    def __post_init__(self):
        pass

    @classmethod
    def from_dict(cls, config: dict, filter_extra_keys=True):
        if filter_extra_keys:
            valid_keys = {field.name for field in fields(cls) if field.init}
            filtered_config = {k: v for k, v in config.items() if k in valid_keys}
            extra_keys = set(config.keys()) - valid_keys
            if extra_keys:
                print(f"Warning: Ignoring unsupported parameters: {extra_keys}")
            return cls(**filtered_config)
        else:
            return cls(**config)
    
    def to_dict(self):
        return self.__dict__
