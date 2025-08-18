import os
import subprocess
import argparse
from pathlib import Path
import logging

from tractoracle_irt.utils.config.config import Config
from tractoracle_irt.utils.config.params_manager import ParamsManager
from tractoracle_irt.utils.config.builders import BashScriptBuilder
from tractoracle_irt.utils.config.misc import get_dataset_type

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO, handlers=[console_handler])
LOGGER = logging.getLogger("submit_experiments")

def parse_args():
    parser = argparse.ArgumentParser(description="Submit experiments to SLURM.")
    parser.add_argument("config", type=str, default="config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--submit", action="store_true", help="Also submit the script to be executed (qc (local) | sbatch (SLURM)).")
    parser.add_argument("--cluster", action="store_true", help="Run the experiments locally.")
    parser.add_argument("--dry-run", action="store_true", help="Don't write any scripts.")
    parser.add_argument("--account", type=str, default="def-pmjodoin", choices=["def-pmjodoin", "rrg-descotea", "def-descotea"], help="SLURM account to use.")
    parser.add_argument("--no-download", action="store_true", help="Do not download the URLs in the config file.")
    args = parser.parse_args()
    return args

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
    config_manager = Config(args.config, not args.cluster)

    ######################################################
    # Generate SLURM Jobs for each experiment.
    ######################################################
    all_jobs = []
    for i, config in enumerate(config_manager.experiments_iter):
        exp_name = config["exp_name"]
        pm = ParamsManager()

        script_path = os.path.join(config_manager.SOURCEDIR, config["launch_script"])
        pm.register_script(script_path)

        # Command to prepare the dataset
        dataset_name = config["dataset"]
        hdf5_path = os.path.join(config_manager.data_config[dataset_name].get("location", None))
        prepare_dataset_command=""
        if args.cluster:
            commands = [
                "# Extract dataset",
                f'echo "Preparing {dataset_name} dataset..."',
                "mkdir -p $SLURM_TMPDIR/experiments",
                "mkdir -p $SLURM_TMPDIR/datasets",
            ]
            if get_dataset_type(hdf5_path) == "tar":
                # If the dataset is a tar file, we need to extract it
                # In our case, the tar file is specifically the ISMRM2015 dataset.
                commands += f"tar xf {hdf5_path} -C $SLURM_TMPDIR/datasets"
            else:
                commands += f"cp {hdf5_path} $SLURM_TMPDIR/datasets/{dataset_name}.hdf5"

            prepare_dataset_command = "\n".join(commands)
            hdf5_path = os.path.join("$SLURM_TMPDIR/datasets", f"{dataset_name}.hdf5")

        #########################################
        # ParamsManager setup
        # 
        # The params manager is used to register the command line arguments that
        # will be passed to the python script. As you add new arguments to your
        # script, you should also register them here so that they can be
        # properly parsed and passed to the script.
        #
        # Disclaimer: This process could be highly improved, but for now
        # it is a simple way to manage the parameters.
        #########################################

        # Register the permanent parameters
        pm.add_pos_arg(config['dest_folder'])
        pm.add_pos_arg(config["project_name"])
        pm.add_pos_arg(config["exp_id"])
        pm.add_pos_arg(hdf5_path)
        
        pm.add_param("--max_ep", config.get("max_ep", None), required=True)
        pm.add_param("--workspace", config.get("workspace", None), required=True)
        pm.add_param("--hidden_dims", config.get("hidden_dims", None))
        pm.add_param("--rng_seed", config.get("seed", None))
        pm.add_param("--n_actor", config.get("n_actors", None))
        pm.add_param("--npv", config.get("npv", None))
        pm.add_param("--min_length", config.get("min_length", None))
        pm.add_param("--max_length", config.get("max_length", None))
        pm.add_param("--noise", config.get("noise", None))
        pm.add_param("--batch_size", config.get("batch_size", None))
        pm.add_param("--replay_size", config.get("replay_size", None))
        pm.add_param("--lr", config.get("lr", None))
        pm.add_param("--gamma", config.get("gamma", None))
        pm.add_param("--theta", config.get("theta", None))
        pm.add_param("--binary_stopping_threshold", config.get("binary_stopping_threshold", None))
        pm.add_param("--n_dirs", config.get("n_dirs", None))
        pm.add_param("--neighborhood_type", config.get("neighborhood_type", None))
        pm.add_param("--neighborhood_radius", config.get("neighborhood_radius", None))
        pm.add_param("--log_interval", config.get("log_interval", None))
        pm.add_param("--utd", config.get("utd", None))

        # Prepare extra flags
        pm.add_flag_if_true("use_comet", config.get("use_comet", False))
        pm.add_flag_if_true("--flatten_state", config.get("flatten_state", False))
        pm.add_flag_if_true("--conv_state", config.get("conv_state", False))
        pm.add_param("--fodf_encoder_ckpt", config.get('fodf_encoder_ckpt', None))
        pm.add_param("--agent_checkpoint", config.get("agent_checkpoint", None))
        pm.add_param("--alg", config.get("alg", None))
        pm.add_param("--agent_train_steps", config.get("agent_train_steps", None))
        pm.add_param("--oracle_train_steps", config.get("oracle_train_steps", None))
        pm.add_param("--first_oracle_train_steps", config.get('first_oracle_train_steps', None))
        pm.add_param("--oracle_lr", config.get("oracle_lr", None))
        pm.add_param("--dataset_to_augment", config.get("dataset_to_augment", None))
        pm.add_param("--nb_new_streamlines_per_iter", config.get("nb_new_streamlines_per_iter", None))
        pm.add_param("--rlhf_inter_npv", config.get("rlhf_inter_npv", None))
        pm.add_param("--oracle_batch_size", config.get("oracle_batch_size", None))
        pm.add_param("--num_workers", config.get("num_workers", None))

        # Add the oracle flags
        pm.add_flag_if_true("--oracle_validator", config.get("oracle_validator", False))
        pm.add_flag_if_true("--oracle_stopping_criterion", config.get("oracle_stopping_criterion", False))
        pm.add_param("--oracle_bonus", config.get("oracle_bonus", None))
        pm.add_param("--oracle_reward_checkpoint", config.get("reward_ckpt", None),
                                      required=config.get("oracle_bonus", 0) > 0)
        pm.add_param("--oracle_crit_checkpoint", config.get("crit_ckpt", None),
                                      required=config.get("oracle_stopping_criterion", False))

        pm.add_flag_if_true("--exclude_direct_neigh", config.get("exclude_direct_neigh", False))
        pm.add_flag_if_true("--batch_renorm", config.get("batch_renorm", False))
        pm.add_flag_if_true("--disable_oracle_training", config.get('disable_oracle_training', False))
        pm.add_flag_if_true("--offline", config.get("offline", False))

        # For now, we need to manually register the extra parameters too.
        # If the dataset has a field "tractometer_reference", add it to the extra flags
        pm.add_param("--warmup_agent_steps", config.get("warmup_agent_steps", None))
        pm.add_flag_if_true("--tractometer_validator", config.get("tractometer_validator", False))
        pm.add_param("--tractometer_reference", config.get("tractometer_reference", None))
        pm.add_param("--scoring_data", config_manager.data_config[dataset_name].get("scoring_data", None))

        pm.add_flag_if_true("--rbx_validator", config.get("rbx_validator", False))
        pm.add_param("--atlas_directory", config.get("atlas_directory", None))

        pm.add_flag_if_true("--extractor_validator", config.get("extractor_validator", False))
        pm.add_param("--extractor_target", config.get("extractor_target", None))

        pm.add_flag_if_true("--verifyber_validator", config.get("verifyber_validator", False))
        pm.add_param("--verifyber_sif_img_path", config.get("verifyber_sif_img_path", None))

        # SLURM script content
        builder = BashScriptBuilder(config, not args.cluster, pm, prepare_dataset_command)
        slurm_script = builder.build_script()

        slurm_script_path = os.path.join(config_manager.SOURCEDIR, "slurm_scripts", f"{config['exp_id']}.sh")
        os.makedirs(os.path.dirname(slurm_script_path), exist_ok=True)

        # Save and submit the SLURM script
        if not args.dry_run:
            with open(slurm_script_path, "w") as f:
                f.write(slurm_script)

            all_jobs.append((exp_name, config["exp_id"], slurm_script_path))
            LOGGER.info(f"Generated SLURM script for experiment {exp_name}: {slurm_script_path}")
        else:
            LOGGER.info(f"Dry run: SLURM script for experiment {exp_name} would be generated at {slurm_script_path}")

    ######################################################
    # Submit the jobs to SLURM.
    ######################################################
    if not args.cluster and args.submit:
        LOGGER.warning("Running in local mode. The --submit flag will not submit jobs to SLURM.")
    elif args.submit and args.cluster:
        for exp_name, exp_id, slurm_script_path in all_jobs:
            LOGGER.info(f"Submitted experiment {exp_name} with job script: {slurm_script_path}")
            subprocess.run(["sbatch", "--account", args.account, slurm_script_path], check=True)
    else:
        LOGGER.info("No jobs submitted. Launch the created scripts manually or use the --submit flag to submit jobs.")

if __name__ == "__main__":
    main()