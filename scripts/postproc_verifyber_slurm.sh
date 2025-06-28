#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=0-05:00:00
#SBATCH --mail-user=jeremi.levesque@usherbrooke.ca
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --job-name=pp_ver_part{{PART}}

echo "Job started."

source ~/tractoracle_irt/venv/bin/activate

module load apptainer

# Path where the results/ directory is located.
INPUT_DIR={{INPUT_DIR}}
RESULTS=results
SUBJECTS_LIST={{SUBJECTS_LIST}}
TRACKING_DIR={{TRACKING_DIR}}
SUBTRACT_DIR={{SUBTRACT_DIR}}

echo "Input directory: $INPUT_DIR"
echo "Results directory: $RESULTS"
echo "Tracking directory: $TRACKING_DIR/<subid>/$SUBTRACT_DIR"
echo "Subjects list: $SUBJECTS_LIST"

cat ${SUBJECTS_LIST} | while read subid; do

    # Run if the started or finished file does not exist.
    if [ -f "$INPUT_DIR/$RESULTS/$subid/started" ] || [ -f "$INPUT_DIR/$RESULTS/$subid/finished" ]; then
        echo "Skipping $subid, already processed."
        continue
    fi

    TRACTOGRAM=$TRACKING_DIR/$subid/${SUBTRACT_DIR}/*.trk

    touch "$INPUT_DIR/$RESULTS/$subid/started"
    python ~/tractoracle_irt/postproc_verifyber.py \
        ${TRACTOGRAM} \
        --invalid_streamlines "$INPUT_DIR/$RESULTS/$subid/idxs_non-plausible.txt" \
        --invalid_output "$INPUT_DIR/$RESULTS/$subid/${subid}_invalid.trk" \
        --valid_streamlines "$INPUT_DIR/$RESULTS/$subid/idxs_plausible.txt" \
        --valid_output "$INPUT_DIR/$RESULTS/$subid/${subid}_valid.trk"
    touch "$INPUT_DIR/$RESULTS/$subid/finished"
    rm "$INPUT_DIR/$RESULTS/$subid/started"
done

echo "Job finished."