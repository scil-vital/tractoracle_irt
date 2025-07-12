# TractOracle-IRT: Exploring the robustness of TractOracle methods in RL-based tractography

TractOracle-IRT is a reinforcement learning (RL) framework applied to [tractography](https://tractography.io/about/) where the **reward model is iteratively aligned with the policy** during training.

It is a continuation which builds on top of previous works from *Théberge et al.* ([see below](#reference-work)).

## Getting started

### (Required) Install the project
**This repository supports Python 3.10.**

It is recommended to use a python [virtual environment](https://virtualenv.pypa.io/en/latest/user_guide.html) to run the code. As the project depends on scilpy, the Python version supported is strongly linked with the versions supported by scilpy.

``` bash
# 1. Create virtual environment

# Using virtualenv
python -m virtualenv .venv
# Using uv:
uv venv --python 3.10

# 2. Activate the environment
source .venv/bin/activate
```

Then, install the project and its required dependencies with

``` bash
# 0. (Optional) Install PyTorch with specific CUDA binaries.
./install_torch.sh

# 1. Install the project

# Using pip
pip install -e .

# Using uv
uv pip install -e .
```

### Install external dependencies
- [**Nextflow**](https://www.nextflow.io/docs/latest/install.html): We recommend installing version 21.10.3 (or a similar version), as most of the flows we use (i.e. rbx_flow, extractor_flow) do not support Nextflow's DSL2. Nextflow is required to run [RecobundlesX](https://github.com/levje/rbx_flow) and [extractor_flow](https://github.com/scilus/extractor_flow), which are used during the IRT procedure.
- [**Docker**](https://www.docker.com/get-started/): docker is required to run IRT training when running [RecobundlesX](https://github.com/levje/rbx_flow), [extractor_flow](https://github.com/scilus/extractor_flow) and Verifyber.
- [**Apptainer**](https://apptainer.org/docs/admin/main/installation.html): Although we prefer the use of docker to spawn containers, we also support Apptainer/Singularity images which requires the installation of Apptainer. To use Apptainer instead of Docker containers for IRT training, please refer to [this guide](docs/using_apptainer.md).

## Publicly available files
For convenience and to ensure reproducibility of our experiments, we have made several files available that can be downloaded automatically upon request in [configs/defaults/public_files.yml](configs/defaults/public_files.yml).
- Oracles checkpoints.
- IRT trained agents checkpoints.
- RBX Atlas.
- Extractor MNI registration target.
- Potentially others upon request.

You can specify those files as an `--agent_checkpoint` path when tracking or anywhere in your [agent training configuration](#training-an-agent). Simply specify `public://<name_of_file>` instead of the path. The file(s) will be automatically downloaded in the *.data/* folder if not already there and the path will be automatically substituted. Please consult the YAML entry names of the [public files list](configs/defaults/public_files.yml) to appropriately use those files.

## Tracking

You will need a trained agent for tracking. We provide the weights of our IRT-trained agents on TractoInferno and HCP. To use any of those weights, specify the following arguments:
- `--agent_checkpoint public://sac_irt_inferno`
- `--agent_checkpoint public://crossq_irt_inferno`
- `--agent_checkpoint public://sac_irt_hcp`
- `--agent_checkpoint public://crossq_irt_hcp`

By default, the tracking script will use the `public://sac_irt_inferno` checkpoint. Those are also available in the docker image specified [below](#track-with-a-docker-image).

You can also provide your own You will need to provide fODFs, a seeding mask and a WM mask. The seeding mask **must** represent the interface of white matter and gray matter. For optimal results, the interface mask should contain voxels from both the white and gray matter.

Agents used for tracking are constrained by their training regime. For example, the agents provided in `models` were trained on a volume with a resolution of ~1mm iso voxels and a step size of 0.75mm using fODFs of order 6, `descoteaux07` basis. When tracking on arbitrary data, the step-size and fODF order and basis will be adjusted accordingly automatically (i.e resulting in a step size of 0.375mm on 0.5mm iso diffusion data). **However**, if using fODFs in the `tournier07` basis, you will need to set the `--sh_basis` argument accordingly.

### Track with a docker image
In case you want to avoid installing the tools above and you simply want to quickly test how our agents perform on your data, this section is for you. However, using Docker, we only provide the SAC-IRT or CrossQ-IRT checkpoints (RBX-based, trained on TractoInferno) to perform tracking, which are the best performing tracking agents. If you want to perform tracking using one of your checkpoints or any other checkpoint, please track locally as described [above](#tracking).

Prerequisites:
1. Install and make sure the [docker](https://www.docker.com/get-started/) daemon is running. (running `docker ps` should not give an error).
2. (Recommended) Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html): This is required to use the docker container with GPU support (which performs tracking significantly faster). Only CUDA-based GPUs are supported at the moment.

Spin up a container and watch the magic happen:
``` bash
docker run [--gpus all] \
    -v $out_dir:/output \
    -v $in_odf:/input/in_odf.nii.gz \
    -v $in_interface:/input/in_seed.nii.gz \
    -v $in_mask:/input/in_mask.nii.gz \
    -t mrzarfir/tractoracle-irt:latest \
    --out_tractogram /output/$out_tractogram
```

For your convenience, the `track_docker.sh` script is provided which guardrails the inputs and outputs. You can use this script the following way:
``` bash
bash track_docker.sh {cpu,gpu} \ 
    <input_odf>.nii.gz \ 
    <input_interface>.nii.gz \ 
    <input_mask>.nii.gz \ 
    <output_tractogram_name> \ 
    <output_directory>
```

### How to track locally
The previous tracking procedure using Docker containers is simple and effective, but might constrain some users (especially if you develop an extension of this work). To run the tracking yourself without an intermediate technology, you can run the `tractoracle_irt/runners/ttl_track.py` script. There is an extensive list of arguments that you can modify, run `tractoracle_irt/runners/ttl_track.py` to get the full list.

To track, use:
``` bash
python tractoracle_irt/runners/ttl_track.py \  
    <in_odf>.trk \  
    <in_seed>.trk \  
    <in_mask>.trk \  
    [--out_tractogram <path_output_tractogram>.trk]  
    [--agent_checkpoint <path>/last_model_state.ckpt]  # Should have the hparams file beside the ckpt.
    [--in_peaks <peaks_path>.trk]  # Quicker tracking
```

## Training an oracle
Before training any agent, it is required to have a trained oracle. You shouldn't have to train a custom oracle yourself as we provide 6 different checkpoints that you can use:
- `public://inferno_rbx`
- `public://inferno_ext`
- `public://inferno_ver`
- `public://hcp_rbx`
- `public://hcp_ext`
- `public://hcp_ver`

However, you might want to train or fine-tune an oracle on your data and/or with a different reference filtering method. For additional instructions on how to do that, please [refer to this file](docs/train_oracle.md).

## Training an agent

Before training any agent, you need to have a **trained oracle network**. You can use one of the provided oracles as specified [above](#training-an-oracle) or [train your own](docs/train_oracle.md).

> In order to train any model, you have to correctly install the project locally as no docker image is provided to train. Once you have correctly installed the project, please continue to the next subsections to be able to train your first tracking agent.

### Preparing your data
In order to train an agent, you'll need to compile your dataset into a single HDF5 file. TL;DR: use the `tractoracle_irt/datasets/create_dataset_tracking.py` script. You'll require a configuration file that you can create using `tractoracle_irt/datasets/create_config_tracking.py`.

For additional details on how to prepare your data to train your agent, please consult [this guide](./docs/build_dataset_tracking.md).

### Configure your training experiment.
The main scripts to train any agents are available in the `tractoracle_irt/trainers/` directory. In principle, you can manually run any of these scripts and provide the required arguments according to the experiment you want to run. However, those scripts offer a very long list of customizable arguments which can be cumbersome to individually set for each experiment. Additionnally, we ran the experiments on a SLURM-based system and we had to run a lot of experiments, using different seeds, different datasets, training regimes and more.

Instead of running the scripts manually, we proceed with YAML configuration files. An example configuration file for two training experiments is available in [configs/inferno_example.yml](configs/inferno_example.yml). To configure your training, you can copy this file and adjust the paths and arguments to fit your needs.

Once you have your configuration file set up, you'll have to use `submit_experiments.py`. This script reads the configuration file, makes sure some paths exist and performs some additional preprocessing steps which are detailed in the provided [example](configs/inferno_example.yml) (e.g. choosing between cluster and local paths, performing additional checks, etc.). Here is an overview on how to call the script:  
```bash
python submit_experiments.py \
    [--cluster] \
    [--submit] \
    [--dry-run]
    config/file.yaml
```

This creates one to several bash files that you can run either manually, or submit them all to a SLURM system, making it easier to start multiple jobs at once. It also allows to have experiments that are permanent on disks and easily reproducible.

## Contributing

Contributions are welcome. Please refer to the [contribution guidelines](./docs/CONTRIBUTING.md)

## Reference work

This work is a **continuation** of the [TrackToLearn/TractOracle-RL framework](https://github.com/scil-vital/TrackToLearn) from *Théberge et al.*.

> Théberge, A., Descoteaux, M., & Jodoin, P. M. (2024). TractOracle: towards an anatomically-informed reward function for RL-based tractography. Accepted at MICCAI 2024.

To see the reference version or previous versions of this work, please visit [this work](https://github.com/scil-vital/TrackToLearn).
