# Most of this is from: https://gitlab.com/NERSC/nersc-official-images/-/blob/main/nersc/python/3.9-anaconda-2021.11/Dockerfile
FROM docker.io/library/ubuntu:latest
WORKDIR /opt

RUN \
    apt-get update && apt-get install --yes \
    build-essential \
    gfortran \
    git \
    wget \
    libgl1 libegl1 libxext6 libsm6 libxrender1 && \
    apt-get clean all && rm -rf /var/lib/apt/lists/*

# miniforge
# Download and install the latest Miniforge (comes with Mamba and conda-forge)
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/miniconda && \
    rm /tmp/miniforge.sh

# Update PATH environment variable to include Miniforge
ENV PATH="/opt/miniconda/bin:$PATH"


# Install base packages + dependencies for leaf segmentation notebook
# PyTorch stack, scipy/scikit-image/pandas/matplotlib/tqdm/Pillow, psutil
RUN mamba install --yes -c conda-forge -c pytorch -c nvidia \
    python=3.10 \
    jupyterlab scipy scikit-learn scikit-image numpy pandas matplotlib tqdm pillow psutil \
    pytorch torchvision torchaudio pytorch-cuda=12.4 \
    && mamba clean --all -f -y

# Install leaf-segmentation-specific pip packages
RUN pip install \
    segmentation-models-pytorch \
    albumentations \
    wandb

# for NERSC jupyterhub environment
RUN pip install batchspawner
ENV NERSC_JUPYTER_IMAGE=YES

# expected by entrypoint script
RUN mkdir -p /alsuser /alsdata

WORKDIR /alsuser
