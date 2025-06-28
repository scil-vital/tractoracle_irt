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

    # Move the T1 file
    $MOVE_CMD $1/$subid/Resample_T1/${subid}__t1_resampled.nii.gz $sub_out_dir/${subid}_t1.nii.gz
}

manage_tracking_directory() {
    # $1: the subject directory from the tractoflow output
    # $2: the subject ID
    subid=$2
    sub_out_dir=$3/$subid

    $MOVE_CMD $1/$subid/Tracking/${subid}__tracking_filteredLength.trk $sub_out_dir/
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

        # Manage the tracking directory
        manage_tracking_directory $1 $subject $2
    done
}

# Example usage
organize_tractoflow_output "tractoflow/" "extractor/input/"
