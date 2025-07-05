# TractOracle-IRT: Exploring the robustness of TractOracle methods in RL-based tractography

TractOracle-IRT is a reinforcement learning (RL) framework applied to [tractography](https://tractography.io/about/) where the **reward model is iteratively aligned with the policy** during training.

It is a continuation which builds on top of previous works from *Théberge et al.* ([see below](#reference-work)).

## Getting started

### (Required) Install the project
**This repository supports Python 3.10 and 3.11.**

It is recommended to use a python [virtual environment](https://virtualenv.pypa.io/en/latest/user_guide.html) to run the code.

``` bash
# 1. Create virtual environment

# Using virtualenv
python -m virtualenv .venv
# Using uv:
uv venv --python 3.11

# 2. Activate the environment
source .venv/bin/activate
```

Then, install the project and its required dependencies with

``` bash
# 0. (Optional) Install PyTorch with specific CUDA binaries.
./install_torch.sh

# 2. Install the project

# Using pip
pip install -e .

# Using uv
uv pip install -e .
```

### Install external dependencies
- [**Nextflow**](https://www.nextflow.io/docs/latest/install.html): We recommend installing version 21.10.3 (or a similar version), as most of the flows we use (i.e. rbx_flow, extractor_flow) do not support Nextflow's DSL2. Nextflow is required to run [RecobundlesX](https://github.com/levje/rbx_flow) and [extractor_flow](https://github.com/scilus/extractor_flow), which are used during the IRT procedure.
- [**Docker**](https://www.docker.com/get-started/): docker is required to run IRT training when running [RecobundlesX](https://github.com/levje/rbx_flow), [extractor_flow](https://github.com/scilus/extractor_flow) and Verifyber.
- [**Apptainer**](https://apptainer.org/docs/admin/main/installation.html): Although we prefer the use of docker to spawn containers, we also support Apptainer/Singularity images which requires the installation of Apptainer.

## Tracking

You will need a trained agent for tracking. One is provided in the `models` folder and is loaded autmatically when tracking. You can then track by running `ttl_track.py`.

You will need to provide fODFs, a seeding mask and a WM mask. The seeding mask **must** represent the interface of white matter and gray matter.

Agents used for tracking are constrained by their training regime. For example, the agents provided in `models` were trained on a volume with a resolution of ~1mm iso voxels and a step size of 0.75mm using fODFs of order 6, `descoteaux07` basis. When tracking on arbitrary data, the step-size and fODF order and basis will be adjusted accordingly automatically (i.e resulting in a step size of 0.375mm on 0.5mm iso diffusion data). **However**, if using fODFs in the `tournier07` basis, you will need to set the `--sh_basis` argument accordingly.

### How to track

### Track with a docker image
In case you want to avoid installing the tools above and you simply want to quickly test how our agents perform on your data, this section is for you. However, using Docker, we only provide the SAC-IRT or CrossQ-IRT checkpoints (RBX-based, trained on TractoInferno) to perform tracking, which are the best performing tracking agents. If you want to perform tracking using one of your checkpoints or any other checkpoint, please track locally as described [above](#tracking).

Prerequisites:
1. Install and make sure the [docker](https://www.docker.com/get-started/) daemon is running. (running `docker ps` should not give an error).
2. (Recommended) Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html): This is required to use the docker container with GPU support (which performs tracking significantly faster). Only CUDA-based GPUs are supported at the moment.

Spin up a container and watch the magic happen:
``` bash
docker run [--gpus all] \
    -v <your_output_directory>:/output\
    -v <your_input_fodf>:/input/in_odf.nii.gz
    -v <your_interface_mask>:/input/in_seed.nii.gz
    -v <your_tracking_mask>:/input/in_mask.nii.gz
    -t mrzarfir/tractoracle-irt:latest \
    <out_tractogram_name.trk>
```

For your convenience, you can run `bash track_docker_cpu.sh` or `bash track_docker_gpu.sh` which take care of running that command for you with additional guidance. 
For your convenience, the `track_docker.sh` script is provided which guardrails the inputs and outputs. You can use this script the following way:
``` bash
bash track_docker {cpu, gpu} <input_odf>.nii.gz <input_interface>.nii.gz <input_mask>.nii.gz <output_tractogram_name> <output_directory>
```


## Training

In order to train any model, you have to correctly install the project locally as no docker image is provided to train. Once you have correctly installed the project, please continue to the next subsections to be able to train your first tracking agent.

### Choosing what you want to train.
### Training configuration

## Contributing

Contributions are welcome. Please refer to the [contribution guidelines](./docs/CONTRIBUTING.md)

## Reference work

This work is a **continuation** of the [TrackToLearn/TractOracle-RL framework](https://github.com/scil-vital/TrackToLearn) from *Théberge et al.*.

> Théberge, A., Descoteaux, M., & Jodoin, P. M. (2024). TractOracle: towards an anatomically-informed reward function for RL-based tractography. Accepted at MICCAI 2024.

To see the reference version or previous versions of this work, please visit [this work](https://github.com/scil-vital/TrackToLearn).
