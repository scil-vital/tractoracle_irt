# This script organizes the output of tractoflow
#
# We aim here to reorganize multiple subjects' directories.
# The script should create the new directories and move only the necessary files to the new location.
# Each subject should have a directory tree with the following structure:
# subjectID/
# ├── anat/
# │  ├── <subjectID>_T1.nii.gz 
# ├── bundles/
# ├── dti/
# │   ├── <subjectID>__fa.nii.gz 
# │   └── <subjectID>__rgb.nii.gz
# ├── dwi/
# ├── fodfs/
# │   ├── <subjectID>__fodf.nii.gz 
# │   └── <subjectID>__peaks.nii.gz  
# ├── maps/
# │   ├── <subjectID>__map_csf.nii.gz 
# │   ├── <subjectID>__map_wm.nii.gz 
# │   ├── <subjectID>__map_gm.nii.gz
# │   └── <subjectID>__interface.nii.gz 
# └── masks/
#     ├── <subjectID>__mask_wm.nii.gz
#     ├── <subjectID>__mask_csf.nii.gz
#     └── <subjectID>__mask_gm.nii.gz
# 
MOVE_CMD=cp

manage_anat_directory() {
    # $1: the subject directory from the tractoflow output
    # $2: the subject ID
    subid=$2
    sub_out_dir=$3/$subid

    echo "manage_anat_directory: $1 $2 $3"

    # Create the anat directory
    mkdir -p $sub_out_dir/anat

    # Move the T1 file
    $MOVE_CMD $1/$subid/Resample_T1/${subid}__t1_resampled.nii.gz $sub_out_dir/anat/${subid}_T1.nii.gz
}

manage_bundles_directory() {
    # $1: the subject directory from the tractoflow output
    # $2: the subject ID
    subid=$2
    sub_out_dir=$3/$subid

    # Create the bundles directory
    mkdir -p $sub_out_dir/bundles
}

manage_dti_directory() {
    # $1: the subject directory from the tractoflow output
    # $2: the subject ID
    subid=$2
    sub_out_dir=$3/$subid

    # Create the dti directory
    mkdir -p $sub_out_dir/dti

    $MOVE_CMD $1/$subid/DTI_Metrics/${subid}__fa.nii.gz $sub_out_dir/dti/
    $MOVE_CMD $1/$subid/DTI_Metrics/${subid}__rgb.nii.gz $sub_out_dir/dti/
    $MOVE_CMD $1/$subid/DTI_Metrics/${subid}__ad.nii.gz $sub_out_dir/dti/
    $MOVE_CMD $1/$subid/DTI_Metrics/${subid}__md.nii.gz $sub_out_dir/dti/
    $MOVE_CMD $1/$subid/DTI_Metrics/${subid}__rd.nii.gz $sub_out_dir/dti/
}

manage_dwi_directory() {
    # $1: the subject directory from the tractoflow output
    # $2: the subject ID
    subid=$2
    sub_out_dir=$3/$subid

    # Create the dwi directory
    mkdir -p $sub_out_dir/dwi

    $MOVE_CMD $1/$subid/Resample_DWI/${subid}__dwi_resampled.nii.gz $sub_out_dir/dwi/
}

manage_fodfs_directory() {
    # $1: the subject directory from the tractoflow output
    # $2: the subject ID
    subid=$2
    sub_out_dir=$3/$subid

    # Create the fodfs directory
    mkdir -p $sub_out_dir/fodfs

    $MOVE_CMD $1/$subid/FODF_Metrics/${subid}__fodf.nii.gz $sub_out_dir/fodfs/
    $MOVE_CMD $1/$subid/FODF_Metrics/${subid}__peaks.nii.gz $sub_out_dir/fodfs/
}

manage_maps_directory() {
    # $1: the subject directory from the tractoflow output
    # $2: the subject ID
    subid=$2
    sub_out_dir=$3/$subid

    # Create the maps directory
    mkdir -p $sub_out_dir/maps

    $MOVE_CMD $1/$subid/PFT_Maps/${subid}__interface.nii.gz $sub_out_dir/maps/
    $MOVE_CMD $1/$subid/PFT_Maps/${subid}__map_include.nii.gz $sub_out_dir/maps/
    $MOVE_CMD $1/$subid/PFT_Maps/${subid}__map_exclude.nii.gz $sub_out_dir/maps/

    $MOVE_CMD $1/$subid/Segment_Tissues/${subid}__map_csf.nii.gz $sub_out_dir/maps/
    $MOVE_CMD $1/$subid/Segment_Tissues/${subid}__map_wm.nii.gz $sub_out_dir/maps/
    $MOVE_CMD $1/$subid/Segment_Tissues/${subid}__map_gm.nii.gz $sub_out_dir/maps/
}

manage_masks_directory() {
    # $1: the subject directory from the tractoflow output
    # $2: the subject ID
    subid=$2
    sub_out_dir=$3/$subid

    # Create the masks directory
    mkdir -p $sub_out_dir/masks

    $MOVE_CMD $1/$subid/Segment_Tissues/${subid}__mask_csf.nii.gz $sub_out_dir/masks/
    $MOVE_CMD $1/$subid/Segment_Tissues/${subid}__mask_wm.nii.gz $sub_out_dir/masks/
    $MOVE_CMD $1/$subid/Segment_Tissues/${subid}__mask_gm.nii.gz $sub_out_dir/masks/
}

manage_tracking_directory() {
    # $1: the subject directory from the tractoflow output
    # $2: the subject ID
    subid=$2
    sub_out_dir=$3/$subid

    # Create the tracking directory
    mkdir -p $sub_out_dir/tracking

    $MOVE_CMD $1/$subid/Tracking/${subid}__tracking_filteredLength.trk $sub_out_dir/tracking/
}

organize_tractoflow_output() {
    # $1: the directory containing the tractoflow output
    # $2: the directory where the organized output should be stored

    # Loop through the subjects
    for subject in $(ls $1); do
        echo "Processing subject $subject"
        # Create the subject directory
        subject_dir=$2/$subject
        mkdir -p $subject_dir

        # Manage the anat directory
        manage_anat_directory $1 $subject $2

        # Manage the bundles directory
        manage_bundles_directory $1 $subject $2

        # Manage the dti directory
        manage_dti_directory $1 $subject $2

        # Manage the dwi directory
        manage_dwi_directory $1 $subject $2

        # Manage the fodfs directory
        manage_fodfs_directory $1 $subject $2

        # Manage the maps directory
        manage_maps_directory $1 $subject $2

        # Manage the masks directory
        manage_masks_directory $1 $subject $2

        # Manage the tracking directory
        manage_tracking_directory $1 $subject $2
    done
}

# Example usage
organize_tractoflow_output "tractoflow/" "tractoflow_organized/"
