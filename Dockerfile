FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && \
    apt-get install curl git zip unzip build-essential python3-dev -y \
    && apt-get install -y libxrender1 libgl1 libgl1-mesa-glx

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /tractoracle_irt/

# Create the virtual environment
RUN uv venv --python 3.10

# This activates the virtual environment
# as it will be the first available python
# executable in the PATH. No need to source it.
ENV PATH="/app/.venv/bin:$PATH"

# Install requirements first
COPY pyproject.toml .

# We mount the cache to the local cache.
# This makes the image size smaller
# and improves performance across builds.
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r pyproject.toml

# Install the project itself
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv --no-cache pip install .

# Download required checkpoint files
RUN uv --no-cache run scripts/download_public_files.py --remove_zip \
    sac_irt_inferno \
    crossq_irt_inferno \
    sac_irt_hcp \
    crossq_irt_hcp

# Setup the files to be used by the entrypoint
RUN mkdir /input \
    && mkdir /output

ENV TRACTORACLE_IRT_OUTPUT_TRACTOGRAM="/output/tractogram.trk"
ENTRYPOINT [ "uv", "run", "tractoracle_irt/runners/ttl_track.py", "/input/in_odf.nii.gz", "/input/in_seed.nii.gz", "/input/in_mask.nii.gz" ]