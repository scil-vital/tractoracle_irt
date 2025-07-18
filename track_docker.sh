#!/bin/bash
set -e

use_gpu=$1
gpu_flag=""
if [ "$use_gpu" == "gpu" ]; then
    gpu_flag="--gpus all"
fi

# Remove the first argument from the list of arguments
# in_odf should be the first argument
shift

# Check if the correct number of arguments is provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <gpu|cpu> <input_odf> <input_interface> <input_mask> <output_tractogram> <output_directory>"
    exit 1
fi

in_odf=$1
in_interface=$2
in_mask=$3
out_dir=$4
out_tractogram=$5

# Check if the input files exist
if [ ! -f "$in_odf" ]; then
    echo "Error: Input ODF file '$in_odf' does not exist."
    exit 1
fi
if [ ! -f "$in_interface" ]; then
    echo "Error: Input interface file '$in_interface' does not exist."
    exit 1
fi
if [ ! -f "$in_mask" ]; then
    echo "Error: Input mask file '$in_mask' does not exist."
    exit 1
fi
# Check if the output directory exists, if not create it
if [ ! -d "$out_dir" ]; then
    echo "Output directory '$out_dir' does not exist. Creating it."
    mkdir -p "$out_dir"
fi
# Check if the output tractogram name is provided
if [ -z "$out_tractogram" ]; then
    echo "Error: Output tractogram name is not provided."
    exit 1
fi
# Check if the output tractogram name ends with .trk or .tck
if [[ "$out_tractogram" != *.trk && "$out_tractogram" != *.tck ]]; then
    echo "Error: Output tractogram name must end with .trk or .tck"
    exit 1
fi

# Make sure all the input files are absolute paths
in_odf=$(realpath "$in_odf")
in_interface=$(realpath "$in_interface")
in_mask=$(realpath "$in_mask")
out_dir=$(realpath "$out_dir")

echo "=========================="
echo "Running Tractoracle IRT in Docker"
echo "=========================="
echo "Input ODF: $in_odf"
echo "Input Interface: $in_interface"
echo "Input Mask: $in_mask"
echo "Output Tractogram: $out_dir/$out_tractogram"
echo "=========================="

docker run $gpu_flag \
    -v $out_dir:/output \
    -v $in_odf:/input/in_odf.nii.gz \
    -v $in_interface:/input/in_seed.nii.gz \
    -v $in_mask:/input/in_mask.nii.gz \
    -t mrzarfir/tractoracle-irt:latest \
    --out_tractogram /output/$out_tractogram

echo "=========================="