# Build your streamlines dataset (to train an oracle)

> N.B: This guide assumes that you are familiar with dMRI image processing. At this point,
> you should have access to fODFs, peaks, tissue segmentations, interface mask, etc.
> How to process your data to obtain the required files is not covered by this guide
> and you should refer to the paper behind this repository which should be linked on
> the main page.

## 0. Generate your data
To create your dataset, you should have at your disposal a few streamlines (hopefully a few millions) that you have generated using any tractography algorithm (e.g. iFOD2, sd_stream, PTT, etc.), but we recommend using more than one tracking method for improved generalization. You should then filter (or label) each streamline you want to include in your dataset. You can use any filtering method you want (e.g. RecobundlesX, extractor_flow, Verifyber) in order to filter your tractogram files. After filtering, for each subject, you should have a file containing plausible (recognized), a file containing implausible (unrecognized) streamlines and you should also have a reference image in the diffusion space (.nii/.nii.gz).

How to generate and filter those streamlines is out of scope for this guide, but most algorithms are well-documented to help you achieve what you want to do.

## 1. Prepare your data

To build a streamlines dataset, you'll need (a lot) of streamlines already generated and *scored/filtered*.
This means that you should have the following files for each subject:
- Recognized (i.e. plausible) streamlines file (preferably .trk)
- Unrecognized (i.e. implausible) streamlines file (preferably .trk)
- Reference image (i.e. image in the diffusion space) as a .nii/.nii.gz file.

In the following steps, we assume that your have your data well organized
in a structure **somewhat similar to** the following:
```
example_directory/
├── trainset
│   ├── 100610
│   │   ├── 100610__fa.nii.gz
│   │   ├── 100610__recognized.trk
│   │   └── 100610__unrecognized.trk
│   └── 101006
│       ├── 101006__fa.nii.gz
│       ├── 101006__recognized.trk
│       └── 101006__unrecognized.trk
├── validset
│   ├── 136126
│   │   ├── 136126__fa.nii.gz
│   │   ├── 136126__recognized.trk
│   │   └── 136126__unrecognized.trk
│   └── 136227
│       ├── 136227__fa.nii.gz
│       ├── 136227__recognized.trk
│       └── 136227__unrecognized.trk
└── testset
    ├── 139435
    │   ├── 139435__fa.nii.gz
    │   ├── 139435__recognized.trk
    │   └── 139435__unrecognized.trk
    └── 140117
        ├── 140117__fa.nii.gz
        ├── 140117__recognized.trk
        └── 140117__unrecognized.trk
```
**The bare minimum you should have** is having a root directory (i.e. `example_directory` in this case)
containing the splits of your datasets (train, valid and test directories).

- If the names of the splits is different, you'll be able to indicate in the following
procedure as an additional argument when creating the configuration.
- If streamlines or reference files are nested in the directory structure of the subject's directory
you'll also be able to specify that in the following procedure instead of moving all your files around.

## 1. Create the configuration

To build your own dataset, you'll first need to create a configuration file
as this is required by the subsequent script that compiles all the required
files into the final dataset. To do this, we provide the convenience script
`tractoracle_irt/datasets/create_dataset_streamlines.py`. If this script
doesn't suit your needs, feel free to modify it or to build your own configuration.

You can provide several different arguments to help the script correctly identify
the files you need to add to the dataset. The --help prints out a detailed explanation
on how to organize your data and what arguments to specify as input to successfully
create the configuration file.

``` bash
python tractoracle_irt/datasets/create_config_streamlines.py --help
```
## 2. Create the dataset
The only step left is to compile all those files specified within your configuration
into a single HDF5 file that will be used to train your agents. In most cases, although
there are additional arguments you can provide, you should just have to run the following
line (and replace with the appropriate paths):
``` bash
python tractoracle_irt/datasets/create_dataset_streamlines.py \
    [--nb_points 32] \  
    <config_file>.json \  
    <output_file>.hdf5
```

# Next steps
Now that you have a dataset to train an oracle, the logical next step would be to **train that oracle**! Refer to [this guide](./train_oracle.md).
