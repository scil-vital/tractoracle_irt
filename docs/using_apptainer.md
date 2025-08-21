# Using Apptainer (or Singularity) instead of Docker for IRT

The IRT training scheme uses Docker containers by default to run the filtering algorithms. Some systems (especially HPC clusters) do not allow the use of Docker to run containers. In that case, or in the case you simply prefer using Apptainer/Singularity containers, we provide this quick guide on how to generate our containers to run filtering algorithms on your data.

> Note: currently, we support building apptainer images for RecobundlesX, extractor_flow and Verifyber. If you require an apptainer image to perform tracking, please open an issue to suggest this feature.

## Prerequisites
- Make sure you have properly installed [**Apptainer**](https://apptainer.org/docs/admin/main/installation.html) and [**Nextflow**](https://www.nextflow.io/docs/latest/install.html) as specified in [the requirements of the project](../README.md#install-external-dependencies). 
- For the following steps, you'll need to have [nf-core](https://pypi.org/project/nf-core/) correctly installed. If you followed the [installation steps](../README.md), `nf-core` should be installed within your Python virtual environment.

## Download the singularity images

> Especially when training on a system that does not have internet access, **pre-downloading the required images is required**.

RecobundlesX and extractor_flow both use different containers for different modules. The **nf-core** tool will allow to automatically download all required containers. Simply execute the following commands:  

0. Make sure that you have apptainer and nextflow installed (or loaded if running on HPC):
```bash
# For HPC:
module load apptainer nextflow

# Activate your virtual environment:
# e.g. source venv/activate/bin
```
1. Define where nextflow should download/find the apptainer images:
```bash
# Define where your singularity images should be downloaded.
# The images can take a lot of space, choose the path accordingly.
export NXF_SINGULARITY_CACHEDIR=$HOME/.nextflow-images
export APPTAINER_CACHEDIR=$NXF_SINGULARITY_CACHEDIR/.apptainer/cache

# Add them to your path
echo "export NXF_SINGULARITY_CACHEDIR=$NXF_SINGULARITY_CACHEDIR" >> ~/.bashrc
echo "export APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR" >> ~/.bashrc
```
2. Download the images (this can take a while)
```bash
# Download images for RecobundlesX
nf-core pipelines download levje/nf-rbx -r main --container-cache-utilisation amend --container-system singularity --compress none -l docker.io

# Download images for extractor_flow
nf-core pipelines download levje/nf-extractor -r main --container-cache-utilisation amend --container-system singularity --compress none -l docker.io

# Build images for Verifyber
sudo apptainer build $NXF_SINGULARITY_CACHEDIR/verifyber_image.sif docker://mrzarfir/verifyber:latest
```

3. Use the images: In your IRT training config, set the flag `use_apptainer` to True. For Verifyber filtering, specify the path to your apptainer image for the `verifyber_sif_img_path` argument pointing towards the `$NXF_SINGULARITY_CACHEDIR/verifyber_image.sif`.
