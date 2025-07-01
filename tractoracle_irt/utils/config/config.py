import yaml
import os
import subprocess
from datetime import datetime
import argparse
from pathlib import Path
import logging

from typing import Union
from copy import deepcopy

from tractoracle_irt.utils.utils import get_project_root_dir
from tractoracle_irt.utils.config.misc import check_file_exists
from tractoracle_irt.utils.config.public_files import PUBLIC_FILES, download_file_data

DEFAULT_SOURCEDIR = os.path.join(get_project_root_dir())

class Config:
    def __init__(self, config_file, is_local=False, default_source_dir=DEFAULT_SOURCEDIR):
        self._is_local = is_local

        # Load the YAML config file
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        
        # Extract global variables
        self.vars = {"PROJECTDIR": get_project_root_dir()}
        other_vars = self._select_local_or_cluster_paths(config.get("vars", {}))
        self.vars.update(other_vars)

        # Extract global settings
        self._global_config = self._select_local_or_cluster_paths(config["global"])

        # Extract extras settings
        self._extras = self._select_local_or_cluster_paths(config.get("extras", {}))

        # Extract data settings
        self._data_config = self._select_local_or_cluster_paths(config["data"])
        self.SOURCEDIR = os.path.expanduser(self._data_config.get("SOURCEDIR", default_source_dir))
        self.EXPDIR = os.path.expanduser(self._data_config["EXPDIR"])
        self.DATADIR = os.path.expanduser(self._data_config["DATADIR"])

        # Extract and build experiment settings
        self._experiments = self._select_local_or_cluster_paths(config["experiments"])

        # Build combined config
        # self._experiments = [self.experiment_config(i) for i in range(len(self._experiments))]
        _experiments = []
        for i in range(len(self._experiments)):
            exp = self.experiment_config(i)
            rng_experiments = self._expand_rng_seeds(exp)
            _experiments.extend(rng_experiments)
        
        self._experiments = _experiments
        self._nb_experiments = len(self._experiments)
        self._add_exp_ids()
        self._add_dest_folder()


        ######################################################
        # Make sure everything is in order with each experiment
        # before generating the SLURM jobs.
        ######################################################
        self._perform_checks()
        self.download_public_files()

    @property
    def global_config(self):
        return self._global_config

    def _expand_rng_seeds(self, exp):
        """
        Expand the rng_seed field in the experiments list.
        If rng_seed is a list, it will be expanded to a list of experiments.
        If rng_seed is an int, it will be expanded to a list of experiments with the same seed.
        """
        expanded_experiments = []

        # print("exp:", exp)
        if isinstance(exp["seeds"], list):
            # Expand the rng_seed field
            rng_seeds = exp["seeds"]
            del exp["seeds"]
            for seed in rng_seeds:
                new_exp = {**exp}
                new_exp["seed"] = seed

                expanded_experiments.append(new_exp)
        else:
            exp["seed"] = exp["seeds"]
            expanded_experiments.append(exp)
        
        return expanded_experiments

    def _select_local_or_cluster_paths(self, data_config: Union[dict, list], do_substitute_vars=True):
        """
        This class has an internal flag that determines if we're running on the cluster or locally.
        In the data_config dict provided, we have:

        { ...
          "<some_key>": {
            "location": {
              "cluster": "<some_cluster_path>",
              "local": "<some_local_path>"
            },
            ...
          },
          "<some_other_key>": {
            "location": "<some_path>",
            ...
          }
        }

        If the flag is set to local, the path will be replaced by the local path.
        Otherwise, it will be replaced by the cluster path.

        Output (on cluster):
        { ...
          "<some_key>": {
            "location": "<some_cluster_path>"
          },
          "<some_other_key>": {
            "location": "<some_cluster_path>"
          }
        }
        """

        def expand_user(path):
            if path is not None:
                return os.path.expanduser(path)
            else:
                return path
            
        def substitute_vars(path):
            """
            Substitute the variables in the path with their values.
            We only replace if we detect a variable in the path.
            A variable is defined as a string that starts with "{{" and ends with "}}".
            """
            if path is None:
                return path
            
            if not isinstance(path, str):
                return path

            if not do_substitute_vars:
                return path
            
            old_path = None
            while old_path != path:
                old_path = path
                for var_name, var_value in self.vars.items():
                    path = path.replace(f"{{{{{var_name}}}}}", str(var_value))

            return path

        def finalize_path(path):
            # 0. If the path refers to a public file, we download it
            # and update the path accordingly.
            if isinstance(path, str):
                if path.startswith("public://"):
                    # Remove the "public://" prefix
                    path = path[9:]
                    public_file_data = PUBLIC_FILES.get(path, None)
                    if public_file_data is not None:
                        download_file_data(public_file_data)
                        path = public_file_data.path # Update the path to the downloaded file

            # 2. Substitute the variables in the path if there are any
            path = substitute_vars(path)

            # 3. Expand the user directory
            path = expand_user(path)

            return path

        # We need to depth-check the dictionary to see if we have key with 
        # "location", "cluster" and "local" keys.
        def select_paths(config, is_local):
            updated_config = {}
            for key, value in config.items():
                if isinstance(value, dict):
                    if "local" in value and "cluster" in value:
                        # Make sure there's no other key in the dictionary (location is tolerated)
                        assert len(value) == 2, f"Invalid data configuration for key '{key}': {value}. Only 'cluster' and 'local' keys are allowed if any of them is specified."
                    
                        updated_config[key] = finalize_path(value["local" if is_local else "cluster"])
                    else:
                        updated_config[key] = select_paths(value, is_local)
                elif isinstance(value, str):
                    updated_config[key] = finalize_path(value)
                else:
                    updated_config[key] = value
            return updated_config
        
        if isinstance(data_config, dict):
            updated_config = select_paths(data_config, self._is_local)
        elif isinstance(data_config, list):
            updated_config = [select_paths(exp, self._is_local) for exp in data_config]
        return updated_config

    def _add_exp_ids(self):
        already_created_exp_ids = []
        for exp in self._experiments:
            exp_id = f"{exp['exp_name']}_{exp['seed']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

            # Make sure we don't create two experiments with the same ID
            if exp_id in already_created_exp_ids:
                raise ValueError(f"Experiment ID {exp_id} already exists.")
        
            already_created_exp_ids.append(exp_id)
            exp["exp_id"] = exp_id

    def _add_dest_folder(self):
        for i in range(self._nb_experiments):
            exp = self._experiments[i]
            exp["dest_folder"] = \
                os.path.join(self.EXPDIR,
                exp['exp_name'],
                exp['exp_id'],
                str(exp['seed'])
            )

    @property
    def experiments_iter(self):
        """
        Generator that yields the experiment configurations.
        """
        for i in range(self._nb_experiments):
            yield self._experiments[i]

    def experiment_config(self, indice):
        merged_config = {**self.global_config} 
        merged_config.update(self._experiments[indice])

        # Handle extras
        extra_config = {}
        extra_type = merged_config.get("extra_type", None)
        if extra_type is not None and isinstance(extra_type, str):
            if extra_type not in self._extras:
                raise ValueError(f"Extra type '{extra_type}' not found in configuration. "
                                f"The only extras available are: {list(self._extras.keys())}")
            elif self._extras[extra_type] is None:
                raise ValueError(f"Extra type '{extra_type}' has an empty configuration. ")
            
            extra_config.update(self._extras[extra_type])
        elif extra_type is not None and isinstance(extra_type, list):
            for et in extra_type:
                if et not in self._extras:
                    raise ValueError(f"Extra type '{et}' not found in configuration. "
                                    f"The only extras available are: {list(self._extras.keys())}")
                elif self._extras[et] is None:
                    raise ValueError(f"Extra type '{et}' has an empty configuration. ")
                
                extra_config.update(self._extras[et])
        elif extra_type is not None:
            raise ValueError(f"Invalid extra type '{extra_type}' in configuration. "
                                f"Expected a string or a list of strings.")

        # Overwrite values of the extra config with the specific config
        for key in merged_config.keys():
            if key in extra_config:
                extra_config[key] = merged_config[key]

        final_config = {**extra_config}
        final_config.update(merged_config) # Overwrite the extra config with the specific config

        return final_config

    @property
    def data_config(self):
        return self._data_config
    
    def _perform_checks(self):
        for i, (config) in enumerate(self.experiments_iter):
            # Also, make sure that the dataset exist for every experiment
            dataset_name = config["dataset"]
            dataset_path = self._data_config[dataset_name].get("location", None)

            if not dataset_path:
                raise ValueError(f"Dataset '{dataset_name}' not found in configuration.")
            
            check_file_exists(dataset_path)
            
            # Make sure the oracle checkpoints exist
            check_file_exists(config["reward_ckpt"])
            check_file_exists(config["crit_ckpt"])
            
            # Make sure the launch script exists
            launch_script_path = os.path.join(self.SOURCEDIR, config["launch_script"])
            check_file_exists(launch_script_path)
            
            # If there's a FODF encoder checkpoint, make sure it exists
            check_file_exists(config.get("agent_checkpoint", None), can_be_none=True)
                
            # Check the mutually exclusive flags
            if sum([
                config.get("flatten_state", False),
                config.get("fodf_encoder_ckpt") is not None,
                config.get("conv_state", False)]) != 1:
                raise ValueError(f"Must specify one of flatten_state, fodf_encoder_ckpt or conv_state for experiment {i}")
            
            if config.get("extra_type", None) is not None:
                # We need to make sure that if the extra_type is "rlhf", we also have one or more
                # of the following flags set: rbx_validator, tractometer_validator, extractor_validator or verifyber_validator
                if config["extra_type"] == "rlhf":
                    if not (config.get("rbx_validator", False) or
                            config.get("tractometer_validator", False) or
                            config.get("extractor_validator", False) or
                            config.get("verifyber_validator", False)):
                        raise ValueError(f"For experiment {i}, when using extra_type 'rlhf', "
                                         f"at least one of rbx_validator, tractometer_validator, "
                                         f"extractor_validator or verifyber_validator must be set.")
                elif "rlhf" in config["extra_type"]:
                    # If the extra_type is a list, we need to check if one of the elements is "rlhf"
                    # and rlhf is contained in the list, we need to make sure that at least one of the
                    # following flags is set: rbx_validator, tractometer_validator, extractor_validator or verifyber_validator
                    # If rlhf is not in the list, we don't need to check

                    if not (config.get("rbx_validator", False) or
                            config.get("tractometer_validator", False) or
                            config.get("extractor_validator", False) or
                            config.get("verifyber_validator", False)):
                        raise ValueError(f"For experiment {i}, when using extra_type containing 'rlhf', "
                                         f"at least one of rbx_validator, tractometer_validator, "
                                         f"extractor_validator or verifyber_validator must be set.")

    def download_public_files(self):
        """
        Download the public files defined in the configuration.
        """
        

        