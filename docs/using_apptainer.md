# Using Apptainer (or Singularity) instead of Docker for IRT

The IRT training scheme uses Docker containers by default to run the filtering algorithms. Some systems (especially HPC clusters) do not allow the use of Docker to run containers. In that case, or in the case you simply prefer using Apptainer/Singularity containers, we provide this quick guide on how to generate our containers to run filtering algorithms on your data.

> Note: currently, we support building apptainer images for RecobundlesX, extractor_flow and Verifyber. If you require an apptainer image to perform tracking, please open an issue to suggest this feature.

## Prerequisites
Make sure you have properly installed [**Apptainer**](https://apptainer.org/docs/admin/main/installation.html) as specified in [the requirements of the project](../README.md#install-external-dependencies).

## Building the images

**For RecobundlesX**
``` bash
sudo apptainer build rbx_image.sif docker://mrzarfir/scilus:1.6.0
```
In your IRT training config, specify the path to your apptainer image for the `rbx_sif_img_path` argument.

> This image contains some manually made modifications, for more information, please consult the description [here](https://hub.docker.com/repository/docker/mrzarfir/scilus/general).

**For extractor_flow**
``` bash
sudo apptainer build extractor_image.sif docker://mrzarfir/extractorflow-fixed:latest
```
In your IRT training config, specify the path to your apptainer image for the `extractor_sif_img_path` argument.

**For Verifyber**
``` bash
sudo apptainer build verifyber_image.sif docker://mrzarfir/verifyber:latest
```
In your IRT training config, specify the path to your apptainer image for the `verifyber_sif_img_path` argument.
