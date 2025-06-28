#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=22000M
#SBATCH --time=0-40:00:00
#SBATCH --mail-user=jeremi.levesque@usherbrooke.ca
#SBATCH --mail-type=ALL

EXPNAME=OracleNet-TractoInferno
EXPID=OracleNet-TractoInferno-32-Classif
MAXEPOCHS=75
NB_STREAMLINES_POINTS=32
NUM_WORKERS=20
DENSE=0
PARTIAL=0

additionnal_args=()
if [ $DENSE -eq 1 ]; then
    additionnal_args+=("--dense")
fi

if [ $PARTIAL -eq 1 ]; then
    additionnal_args+=("--partial")
fi

# Check if the script is ran locally or on a cluster node.
if [ -z $SLURM_JOB_ID ]; then
    islocal=1
else
    islocal=0
fi

if [ $islocal -eq 1 ]; then
    echo "Running locally"
    EXPPATH=data/experiments/TractOracleNet/${EXPNAME}
    DATASET_FILE=train_test_classical_tracts_antoine_valid.hdf5
else
    echo "Running on HPC..."
    module load python/3.10 cuda cudnn httpproxy nextflow
    source ~/tractoracle_irt/venv/bin/activate
    export COMET_API_KEY=$(cat ~/.comet_api_key)

    EXPPATH=${SLURM_TMPDIR}/experiment/${EXPNAME}
    mkdir -p ${EXPPATH}

    DATASET_TO_USE=tractoinferno
    # Prepare datasets
    # ISMRM2015
    echo "Copying dataset ${DATASET_TO_USE}..."
    if [ $DATASET_TO_USE == "ismrm2015" ]; then
        cp ~/train_test_classical_tracts_antoine_valid.hdf5 $SLURM_TMPDIR/${DATASET_TO_USE}_st_dataset.hdf5
    elif [ $DATASET_TO_USE == "tractoinferno" ]; then
        cp ~/tractoinferno_tracts.hdf5 $SLURM_TMPDIR/${DATASET_TO_USE}_st_dataset.hdf5
    else
        echo "Error: Unknown dataset specified."
        exit 1
    fi
    DATASET_FILE=$SLURM_TMPDIR/${DATASET_TO_USE}_st_dataset.hdf5

fi


mkdir -p ${EXPPATH}

# BATCH SIZE and GRAD ACCUMULATION
# The original batch size is 2816, but since we want to use a significantly smaller
# batch size, we need to increase the number of gradient accumulation steps to
# compensate for the smaller batch size. The original batch size is 2048 and we want
# to use a batch size of 512.
TOTAL_BATCH_SIZE=2048
MICRO_BATCH_SIZE=1024 #512 # Should reduce or increase this based on the GPU memory available.
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / MICRO_BATCH_SIZE)) # 88

echo "Total batch size: ${TOTAL_BATCH_SIZE}"
echo "Micro batch size: ${MICRO_BATCH_SIZE}"
echo "Gradient accumulation steps: ${GRAD_ACCUM_STEPS}"

python tractoracle_irt/trainers/tractoraclenet_train.py \
    ${EXPPATH} \
    ${EXPNAME} \
    ${EXPID} \
    ${MAXEPOCHS} \
    ${DATASET_FILE} \
    --lr 0.0001 \
    --oracle_batch_size ${MICRO_BATCH_SIZE} \
    --grad_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --n_head 4 \
    --n_layers 4 \
    --out_activation sigmoid \
    --nb_streamlines_points ${NB_STREAMLINES_POINTS} \
    --num_workers ${NUM_WORKERS} \
    --use_comet \
    "${additionnal_args[@]}"

# Archive into .tar.gz everything in $SCRATCH_TMPDIR and copy it to ~/scratch/ with the name TractOracleNet-aaaa-mm-dd-hh-mm-ss.tar.gz
if [ $islocal -ne 1 ]; then
    echo "Archiving experiment..."
    ARCHIVE_NAME=TractOracleNet-$(date +"%Y-%m-%d-%H%M%S").tar.gz
    tar -cvf ${SLURM_TMPDIR}/${ARCHIVE_NAME} $EXPPATH
    cp ${SLURM_TMPDIR}/${ARCHIVE_NAME} ~/scratch/${ARCHIVE_NAME}
fi

