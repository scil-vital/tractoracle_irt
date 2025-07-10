# Build your tracking dataset (to train an RL-based tractography agent)
> N.B: This guide assumes that you are familiar with dMRI image processing. At this point,
> you should have access to fODFs, peaks, tissue segmentations, interface mask, etc.
> How to process your data to obtain the required files is not covered by this guide
> and you should refer to the paper behind this repository which should be linked on
> the main page.

## 0. Prepare your data
Much similarily to the streamlines dataset just above, **the bare minimum you should have**
is having a root directory containing the splits of your datasets (train, valid and test directories).
By default, the configuration creation script (described in the next step), will assume that the
file structure of this root and subjects directory will look like the following:

```
example_directory/
├── trainset
│   ├── 100610
│   │   ├── fodfs/
│   │   │   ├── 100610__fodf.nii.gz
│   │   │   └── 100610__peaks.nii.gz
│   │   ├── masks/
│   │   │   ├── 100610__mask_gm.nii.gz
│   │   │   └── 100610__mask_wm.nii.gz
│   │   ├── maps/
│   │   │   └── 100610__interface.nii.gz
│   │   ├── dti/
│   │   │   └── 100610__fa.nii.gz
│   │   └── anat/
│   │       └── 100610_T1.nii.gz
│   └── 100711
│       ├── fodfs/
│       ...
├── validset
│   ├── 136126
│   │   ├── fodfs/
│   │   │   ├── 136126__fodf.nii.gz
│   │   │   └── 136126__peaks.nii.gz
│   │   ├── masks/
│   │   │   ├── 136126__mask_gm.nii.gz
│   │   │   └── 136126__mask_wm.nii.gz
│   │   ├── maps/
│   │   │   └── 136126__interface.nii.gz
│   │   ├── dti/
│   │   │   └── 136126__fa.nii.gz
│   │   └── anat/
│   │       └── 136126_T1.nii.gz
│   └── 136227
│       ├── fodfs/
│       ...
└── testset
    ├── 139435
    │   ├── fodfs/
    │   │   ├── 139435__fodf.nii.gz
    │   │   └── 139435__peaks.nii.gz
    │   ├── masks/
    │   │   ├── 139435__mask_gm.nii.gz
    │   │   └── 139435__mask_wm.nii.gz
    │   ├── maps/
    │   │   └── 139435__interface.nii.gz
    │   ├── dti/
    │   │   └── 139435__fa.nii.gz
    │   └── anat/
    │       └── 139435_T1.nii.gz
    └── 140117
        ├── fodfs/
        ...
```

Of course, this structure is somewhat flexible as you'll be able to specify the differences
when building the configuration file, but you essentially need to have the following required
files within each subject directory:
- FODF image
- Peaks image
- Tracking mask
- Seeding image
- Anatomical image
- Fractional anisotropy (FA) image
- Gray matter mask

If the files or directory names differ from what's expected by the default, you can specify
that in the following step.

## 1. Create the configuration
To build your own dataset, you'll first need to create a configuration file
as this is required by the subsequent script that compiles all the required
files into the final dataset. To do this, we provide the convenience script
`tractoracle_irt/datasets/create_dataset_tracking.py`. If this script
doesn't suit your needs, feel free to modify it or to build your own configuration.

You can provide several different arguments to help the script correctly identify
the files you need to add to the dataset. The --help prints out a detailed explanation
on how to organize your data and what arguments to specify as input to successfully
create the configuration file.

``` bash
python tractoracle_irt/datasets/create_config_tracking.py --help
```
## 2. Create the dataset
The only step left is to compile all those files specified within your configuration
into a single HDF5 file that will be used to train your agents. In most cases, although
there are additional arguments you can provide, you should just have to run the following
line (and replace with the appropriate paths):
``` bash
python tractoracle_irt/datasets/create_dataset_tracking.py \
    <config_file>.json \
    <output_file>.hdf5
```

# Next step
Now that you have a dataset to train an RL tracking agent, the logical next step would be to **train that agent**! Refer to [this section](../README.md#training-an-agent).
