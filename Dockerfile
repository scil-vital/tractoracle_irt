FROM ubuntu:22.04

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install curl git neovim zip unzip build-essential python3-dev -y

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin/:$PATH"

# Clone the repository
# RUN git clone https://github.com/scil-vital/tractoracle_irt.git
COPY . /app/tractoracle_irt

# Create the virtual environment
RUN cd tractoracle_irt \
    && uv venv --python 3.10

# This activates the virtual environment
# as it will be the first available python
# executable in the PATH. No need to source it.
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies
RUN cd tractoracle_irt \
    && uv pip install -e .

# Download required checkpoint files
RUN cd tractoracle_irt \
    && uv run scripts/download_public_files.py --remove_zip \
        sac_irt_inferno \
        crossq_irt_inferno \
        sac_irt_hcp \
        crossq_irt_hcp

ENTRYPOINT [ "cd", "/app/tractoracle_irt", "&&", "uv", "run", "tractoracle_irt/runners/ttl_track.py"]

