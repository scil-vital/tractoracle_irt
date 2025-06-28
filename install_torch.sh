#!/bin/bash
set -e 

# Install required packages
echo "Platform:" $(uname )
echo "Python version: $(python --version)"

# If cuda is installed, check the version
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "Found GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "Found CUDA version: $(nvidia-smi | grep "CUDA Version" | awk '{print $9}')"

    FOUND_CUDA=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | sed 's/\.//g')
    if (( $FOUND_CUDA >= 128 )); then
        CUDA_VERSION="cu128"
    elif (( $FOUND_CUDA >= 126 )); then
        CUDA_VERSION="cu126"
    elif (( $FOUND_CUDA >= 118 )); then
        CUDA_VERSION="cu118"
    else
      CUDA_VERSION="cpu"
      echo "CUDA version ${FOUND_CUDA} is not compatible. Installing PyTorch without CUDA support."
    fi
else
    echo "No GPU or CUDA installation found. Installing PyTorch without CUDA support."
    CUDA_VERSION="cpu"
fi

echo "Updating pip ..."
pip install --upgrade pip

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing PyTorch 2.7.1"
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
else
    # Install pytorch
    echo "Installing PyTorch 2.7.1+${CUDA_VERSION}"
    # Install PyTorch with CUDA support
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION} --quiet
fi

