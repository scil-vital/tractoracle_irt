import yaml
import os
import subprocess
from datetime import datetime
import argparse
from pathlib import Path

from typing import Union
from copy import deepcopy

def parse_args():
    parser = argparse.ArgumentParser(description="Submit experiments to SLURM.")
    parser.add_argument("config", type=str, default="config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--submit", action="store_true", help="Also submit the script to be executed (qc (local) | sbatch (SLURM)).")
    parser.add_argument("--local", action="store_true", help="Run the experiments locally.")
    parser.add_argument("--dry-run", action="store_true", help="Don't write any scripts.")
    parser.add_argument("--account", type=str, default="def-pmjodoin", choices=["def-pmjodoin", "rrg-descotea", "def-descotea"], help="SLURM account to use.")
    args = parser.parse_args()
    return args

class Config:
    def __init__(self, config_file, is_local=False):
        self._is_local = is_local

        # Load the YAML config file
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        
        # Extract global settings
        self._global_config = self._select_local_or_cluster_paths(config["global"])

        # Extract extras settings
        self._extras = self._select_local_or_cluster_paths(config.get("extras", {}))

        # Extract data settings
        self._data_config = self._select_local_or_cluster_paths(config["data"])
        self.SOURCEDIR = os.path.expanduser(self._data_config["SOURCEDIR"])
        self.EXPDIR = os.path.expanduser(self._data_config["EXPDIR"])
        self.DATADIR = os.path.expanduser(self._data_config["DATADIR"])

        # Extract and build experiment settings
        self._experiments = self._select_local_or_cluster_paths(config["experiments"])

        # Build combined config
        # self._experiments = [self.experiment_config(i) for i in range(len(self._experiments))]
        _experiments = []
        _extras_managers = []
        for i in range(len(self._experiments)):
            exp, extras_manager = self.experiment_config(i)
            rng_experiments = self._expand_rng_seeds(exp)
            _experiments.extend(rng_experiments)
            for _ in range(len(rng_experiments)):
                _extras_managers.append(deepcopy(extras_manager))
        
        self._experiments = list(zip(_experiments, _extras_managers))

        self._nb_experiments = len(self._experiments)
        self._add_exp_ids()
        self._add_dest_folder()


        ######################################################
        # Make sure everything is in order with each experiment
        # before generating the SLURM jobs.
        ######################################################
        self._perform_checks()

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

    def _select_local_or_cluster_paths(self, data_config: Union[dict, list]):
        """
        This class has an internal flag that determines if we're running on the cluster or locally.
        In the data_config dict provided, we have:

        { ...
          "<some_key>": {
            "location": "<some_path>",
            "cluster": "<some_cluster_path>",
            "local": "<some_local_path>"
          },
          "<some_other_key>": {
            "location": "<some_path>",
            "cluster": "<some_cluster_path>",
            "local": "<some_local_path>"
          }
        }

        If the flag is set to local, the path will be replaced by the local path.
        Otherwise, it will be replaced by the cluster path.

        Example for cluster path:
        Input:
        { ...
          "<some_key>": {
            "location": "<some_path>",
            "cluster": "<some_cluster_path>",
            "local": "<some_local_path>"
          },
          "<some_other_key>": {
            "location": "<some_path>",
            "cluster": "<some_cluster_path>",
            "local": "<some_local_path>"
          }
        }

        Output:
        { ...
          "<some_key>": "<some_cluster_path>",
          "<some_other_key>": <some_cluster_path>"
        }
        """

        def expand_user(path):
            if path is not None:
                return os.path.expanduser(path)
            else:
                return path

        # We need to depth-check the dictionary to see if we have key with 
        # "location", "cluster" and "local" keys.
        def select_paths(data_config, is_local):
            selected_paths = {}
            for key, value in data_config.items():
                if isinstance(value, dict):
                    if "local" in value and "cluster" in value:
                        # Make sure there's no other key in the dictionary (location is tolerated)
                        assert len(value) == 2, f"Invalid data configuration for key '{key}': {value}. Only 'cluster' and 'local' keys are allowed if any of them is specified."
                    
                        selected_paths[key] = expand_user(value["local" if is_local else "cluster"])
                    else:
                        selected_paths[key] = select_paths(value, is_local)
                elif isinstance(value, str):
                    selected_paths[key] = expand_user(value)
                else:
                    selected_paths[key] = value
            return selected_paths
        
        if isinstance(data_config, dict):
            selected_paths = select_paths(data_config, self._is_local)
        elif isinstance(data_config, list):
            selected_paths = [select_paths(exp, self._is_local) for exp in data_config]
        return selected_paths

    def _add_exp_ids(self):
        already_created_exp_ids = []
        for exp, _ in self._experiments:
            exp_id = f"{exp['exp_name']}_{exp['seed']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

            # Make sure we don't create two experiments with the same ID
            if exp_id in already_created_exp_ids:
                raise ValueError(f"Experiment ID {exp_id} already exists.")
        
            already_created_exp_ids.append(exp_id)
            exp["exp_id"] = exp_id

    def _add_dest_folder(self):
        # for i in range(self._nb_experiments):
        #     exp, _ = self.experiment_config(i, handle_extras=False)
        #     self._experiments[i]["dest_folder"] = \
        #         os.path.join(self.EXPDIR,
        #         exp['exp_name'],
        #         exp['exp_id'],
        #         str(exp['seed'])
        #     )
        for i in range(self._nb_experiments):
            exp, _ = self._experiments[i]
            self._experiments[i][0]["dest_folder"] = \
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

    def experiment_config(self, indice, handle_extras=True):
        merged_config = {**self.global_config}
        specific_config = self._experiments[indice]
        merged_config.update(specific_config)

        extras_manager = None
        if handle_extras:
            # Get extras config
            extra_config = {}
            extra_type = specific_config.get("extra_type", None)
            if extra_type is None:
                pass
            elif isinstance(extra_type, str):
                if extra_type not in self._extras:
                    raise ValueError(f"Extra type '{extra_type}' not found in configuration. "
                                    f"The only extras available are: {list(self._extras.keys())}")
                elif self._extras[extra_type] is None:
                    raise ValueError(f"Extra type '{extra_type}' has an empty configuration. ")
                
                extra_config.update(self._extras[extra_type])
            elif isinstance(extra_type, list):
                for et in extra_type:
                    if et not in self._extras:
                        raise ValueError(f"Extra type '{et}' not found in configuration. "
                                        f"The only extras available are: {list(self._extras.keys())}")
                    elif self._extras[et] is None:
                        raise ValueError(f"Extra type '{et}' has an empty configuration. ")
                    
                    extra_config.update(self._extras[et])
            else:
                raise ValueError(f"Invalid extra type '{extra_type}' in configuration. "
                                 f"Expected a string or a list of strings.")

            # Overwrite values of the extra config with the specific config
            for key in merged_config.keys():
                if key in extra_config:
                    extra_config[key] = merged_config[key]

            # Add the extras to the ExtrasManager
            extras_manager = ExtrasManager()
            extras_manager.add_config(extra_config)

        return merged_config, extras_manager

    @property
    def data_config(self):
        return self._data_config
    
    def _perform_checks(self):
        for i, (config, _) in enumerate(self.experiments_iter):
            # Also, make sure that the dataset exist for every experiment
            dataset_name = config["dataset"]
            dataset_path = os.path.join(self._data_config[dataset_name].get("location", None))
            if not dataset_path:
                raise ValueError(f"Dataset '{dataset_name}' not found in configuration.")
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset '{dataset_name}' not found at location '{dataset_path}'.")
            
            # Make sure the oracle checkpoints exist
            if not os.path.exists(config["reward_ckpt"]):
                raise FileNotFoundError(f"Oracle reward checkpoint '{config['reward_ckpt']}' not found.")
            if not os.path.exists(config["crit_ckpt"]):
                raise FileNotFoundError(f"Oracle critic checkpoint '{config['crit_ckpt']}' not found.")
            
            # Make sure the launch script exists
            launch_script_path = os.path.join(self.SOURCEDIR, config["launch_script"])
            if not os.path.exists(launch_script_path):
                raise FileNotFoundError(f"Launch script '{launch_script_path}' not found.")
            
            # If there's a FODF encoder checkpoint, make sure it exists
            if config.get("fodf_encoder_ckpt", None) is not None:
                if not os.path.exists(config["fodf_encoder_ckpt"]):
                    raise FileNotFoundError(f"FODF encoder checkpoint '{config['fodf_encoder_ckpt']}' not found.")
                
            # Check the mutually exclusive flags
            state_state_specified = False
            if config.get("flatten_state", False):
                state_state_specified = True
            if config.get("fodf_encoder_ckpt", None) is not None:
                assert not state_state_specified, f"Can only specify one of flatten_state, fodf_encoder_ckpt or conv_state for experiment {i}"
                state_state_specified = True
            if config.get("conv_state", False):
                assert not state_state_specified, f"Can only specify one of flatten_state, fodf_encoder_ckpt or conv_state for experiment {i}"
                state_state_specified = True
            if not state_state_specified:
                assert False, f"Must specify one of flatten_state, fodf_encoder_ckpt or conv_state for experiment {i}"

class ExtrasManager:
    def __init__(self, separator=" ", prefix="--", indent_nb_spaces=4):
        self._flags = []
        self.separator = separator
        self.prefix = prefix
        self.indentation = indent_nb_spaces * " "

    def add_flag(self, flag, value=None):
        if flag is None:
            raise ValueError("Flag cannot be None.")
        
        if flag.endswith("_") and value is None:
            return # Skip flags that end with an underscore and have no value

        if isinstance(value, bool) and value is False:
            return # Skip flags that are False
        
        if not flag.startswith(self.prefix):
            flag = self.prefix + flag

        self._flags.append(str(flag))

        if value is not None \
            and not isinstance(value, bool):
            self._flags.append(str(value))

    def add_config(self, extras_config: dict):
        for key, value in extras_config.items():
            self.add_flag(key, value)

    def compile_flags(self, linebreak=False, start_with_linebreak=False, indent=0):
        if linebreak:
            start = ""
            sep = "\\\n" + (self.indentation * indent)
            if start_with_linebreak:
                start += sep
            return start + sep.join(self._flags)
        else:
            return self.separator.join(self._flags)
    
    def __str__(self):
        return self.compile_flags(linebreak=False)

def get_dataset_type(dataset_path):
    """Determine dataset type based on file extension."""
    if dataset_path.endswith(".tar.gz") or dataset_path.endswith(".tar"):
        return "tar"
    elif dataset_path.endswith(".hdf5"):
        return "hdf5"
    else:
        raise ValueError(f"Unknown dataset format: {dataset_path}")
    
def get_prepare_dataset_command(dataset_name, data_config, data_dir, is_local):
    dataset_path = os.path.join(data_config[dataset_name].get("location", None))
    dataset_type = get_dataset_type(dataset_path)
    if is_local:
        dataset_dir = os.path.join(data_dir, "datasets")
    else:
        dataset_dir = data_dir

    if dataset_type == "tar":
        archive_path = dataset_path
        output_path = data_dir
        
        slurm_dataset_dir = os.path.join(data_dir, "ismrm2015_2mm")
        hdf5_path = os.path.join(slurm_dataset_dir, "ismrm2015.hdf5")

        return f"tar xf {archive_path} -C {output_path}", slurm_dataset_dir, hdf5_path
    elif dataset_type == "hdf5":
        hdf5_path = Path(dataset_dir) / f"{dataset_name}.hdf5"
        return f"cp {dataset_path} {dataset_dir}/{dataset_name}.hdf5", dataset_dir, str(hdf5_path)
    else:
        raise ValueError(f"Unknown dataset format: {dataset_path}")

def main():
    args = parse_args()
    config_manager = Config(args.config, args.local)

    ######################################################
    # Generate SLURM Jobs for each experiment.
    ######################################################
    all_jobs = []
    for i, (config, extra_flags_manager) in enumerate(config_manager.experiments_iter):
        exp_name = config["exp_name"]

        # Prepare extra flags
        if config.get("use_comet", False):
            extra_flags_manager.add_flag("--use_comet")

        # Those are mutually exclusive flags
        state_state_specified = False
        if config.get("flatten_state", False):
            extra_flags_manager.add_flag("--flatten_state")
            state_state_specified = True
        if config.get("fodf_encoder_ckpt", None) is not None:
            assert not state_state_specified, f"Can only specify one of flatten_state, fodf_encoder_ckpt or conv_state for experiment {i}"
            extra_flags_manager.add_flag("--fodf_encoder_ckpt", config['fodf_encoder_ckpt'])
            state_state_specified = True
        if config.get("conv_state", False):
            assert not state_state_specified, f"Can only specify one of flatten_state, fodf_encoder_ckpt or conv_state for experiment {i}"
            extra_flags_manager.add_flag("--conv_state")
            state_state_specified = True
        if not state_state_specified:
            raise ValueError(f"Must specify one of flatten_state, fodf_encoder_ckpt or conv_state for experiment {i}")
        
        if config.get("agent_checkpoint", False):
            extra_flags_manager.add_flag("--agent_checkpoint", config["agent_checkpoint"])
        # from tractoracle_irt.utils.utils import prettier_dict
        # print(prettier_dict(config))

        # Add the oracle flags
        if config.get("oracle_validator", False):
            extra_flags_manager.add_flag("--oracle_validator")
        if config.get("oracle_stopping_criterion", False):
            extra_flags_manager.add_flag("--oracle_stopping_criterion")
        if config.get("oracle_bonus", None) is not None:
            extra_flags_manager.add_flag("--oracle_bonus", config["oracle_bonus"])
        if config.get("reward_ckpt", None) is not None and config.get("oracle_bonus", 0) > 0:
            extra_flags_manager.add_flag("--oracle_reward_checkpoint", config["reward_ckpt"])
        if config.get("crit_ckpt", None) is not None and config.get("oracle_stopping_criterion", False):
            extra_flags_manager.add_flag("--oracle_crit_checkpoint", config["crit_ckpt"])


        # Paths to the dataset
        dataset_name = config["dataset"]
        prepare_ds_cmd, ds_dir, hdf5_path = get_prepare_dataset_command(dataset_name, config_manager.data_config, config_manager.DATADIR, args.local)

        # If the dataset has a field "tractometer_reference", add it to the extra flags
        if config_manager.data_config[dataset_name].get("tractometer_reference", None) is not None:
            extra_flags_manager.add_flag("--tractometer_reference", os.path.join(ds_dir, config_manager.data_config[dataset_name]['tractometer_reference']))
            extra_flags_manager.add_flag("--tractometer_validator")
        if config_manager.data_config[dataset_name].get("scoring_data", None) is not None:
            extra_flags_manager.add_flag("--scoring_data", os.path.join(ds_dir, config_manager.data_config[dataset_name]['scoring_data']))
        if config.get("exclude_direct_neigh", False):
            extra_flags_manager.add_flag("--exclude_direct_neigh")
        if config.get("batch_renorm", False):
            extra_flags_manager.add_flag("--batch_renorm")

        if config.get('first_oracle_train_steps', None) is not None:
            extra_flags_manager.add_flag("--first_oracle_train_steps", config['first_oracle_train_steps'])
        if config.get('disable_oracle_training', False):
            extra_flags_manager.add_flag("--disable_oracle_training")
        if config.get("offline", False):
            extra_flags_manager.add_flag("--offline")

        # SLURM script content
        script_path = os.path.join(config_manager.SOURCEDIR, config["launch_script"])
        extra_flags_string = extra_flags_manager.compile_flags(linebreak=True, indent=1, start_with_linebreak=True)
        slurm_script = f"""#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={config["cpus"]}
#SBATCH --mem={config["mem"]}
#SBATCH --time={config["time"]}
#SBATCH --job-name={config["exp_name"]}
#SBATCH --mail-user=jeremi.levesque@usherbrooke.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# THIS SCRIPT IS AUTOMATICALLY GENERATED BY submit_experiments.py
# DO NOT EDIT MANUALLY. INSTEAD, MODIFY THE CONFIGURATION FILE.

set -e

# Check if the script is ran locally or on a cluster node.
if [ -z $SLURM_JOB_ID ]; then
    islocal=1
else
    islocal=0
fi

# Only run that when on a cluster node
if [ $islocal -eq 0 ]; then
    echo "Loading modules and virtual env..."
    module load StdEnv/2020
    module load nextflow/21.10.3
    module load apptainer/1.1.8
    module load python/3.10 cuda cudnn httpproxy
    source ~/tractoracle_irt/venv/bin/activate

    export COMET_API_KEY=$(cat ~/.comet_api_key)
    export COMET_GIT_DIRECTORY="/home/levj1404/tractoracle_irt"

    # Prepare directories
    mkdir -p $SLURM_TMPDIR/data
    mkdir -p $SLURM_TMPDIR/experiments

    # Extract dataset
    echo "Preparing {dataset_name} dataset..."
    {prepare_ds_cmd}

    # Define paths
    ORACLE_REWARD_CHECKPOINT=$SLURM_TMPDIR/data/oracle_reward.ckpt
    ORACLE_CRIT_CHECKPOINT=$SLURM_TMPDIR/data/oracle_crit.ckpt

    # Prepare oracle checkpoints
    echo "Preparing oracle checkpoints..."
    cp {config["reward_ckpt"]} $ORACLE_REWARD_CHECKPOINT
    cp {config["crit_ckpt"]} $ORACLE_CRIT_CHECKPOINT

else
    # Define paths
    ORACLE_REWARD_CHECKPOINT={config["reward_ckpt"]}
    ORACLE_CRIT_CHECKPOINT={config["crit_ckpt"]}
fi

DEST_FOLDER="{config['dest_folder']}"

mkdir -p $DEST_FOLDER

# Run training script
echo "Running experiment..."
python -O {script_path} \\
    $DEST_FOLDER \\
    "{config['project_name']}" \\
    "{config["exp_id"]}" \\
    "{hdf5_path}" \\
    --max_ep {config["max_ep"]} \\
    --hidden_dims "{config['hidden_dims']}" \\
    --workspace {config["workspace"]} \\
    --rng_seed {config["seed"]} \\
    --n_actor {config["n_actors"]} \\
    --npv {config["npv"]} \\
    --min_length {config["min_length"]} \\
    --max_length {config["max_length"]} \\
    --noise {config["noise"]} \\
    --batch_size {config["batch_size"]} \\
    --replay_size {config["replay_size"]} \\
    --lr {config["lr"]} \\
    --gamma {config["gamma"]} \\
    --theta {config["theta"]} \\
    --binary_stopping_threshold {config["binary_stopping_threshold"]} \\
    --n_dirs {config["n_dirs"]} \\
    --neighborhood_type "{config['neighborhood_type']}" \\
    --neighborhood_radius {config["neighborhood_radius"]} \\
    --log_interval {config["log_interval"]} \\
    --utd {config["utd"]} {extra_flags_string}
    
echo "Experiment available on {config["dest_folder"]}"
    """

        slurm_script_path = os.path.join(config_manager.SOURCEDIR, "slurm_scripts", f"{config['exp_id']}.sh")
        os.makedirs(os.path.dirname(slurm_script_path), exist_ok=True)

        # Save and submit the SLURM script
        if not args.dry_run:
            with open(slurm_script_path, "w") as f:
                f.write(slurm_script)

            all_jobs.append((exp_name, config["exp_id"], slurm_script_path))
            print(f"Generated SLURM script for experiment {exp_name}: {slurm_script_path}")
        else:
            print(f"Dry run: SLURM script for experiment {exp_name} would be generated at {slurm_script_path}")

    ######################################################
    # Submit the jobs to SLURM.
    ######################################################
    if args.submit:
        for exp_name, exp_id, slurm_script_path in all_jobs:
            if args.local:
                print(f"Running experiment {exp_name} locally with job script: {slurm_script_path}")
                subprocess.run(["qc", "bash", slurm_script_path], check=True)
            else:
                print(f"Submitted experiment {exp_name} with job script: {slurm_script_path}")
                subprocess.run(["sbatch", "--account", args.account, slurm_script_path], check=True)
    else:
        print("No jobs submitted. Launch the created scripts manually or use the --submit flag to submit jobs.")

if __name__ == "__main__":
    main()