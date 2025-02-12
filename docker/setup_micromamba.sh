#!/bin/bash
set -ef -o pipefail

# Install and initialize micromamba
curl -Ls https://anaconda.org/conda-forge/micromamba/1.5.10/download/linux-64/micromamba-1.5.10-0.tar.bz2 | tar -xvj bin/micromamba
mv bin/micromamba ${MAMBA_EXE}

# Set the system wide base conda config
mkdir -p /etc/conda
cat <<HERE | tee /etc/conda/.condarc
channels:
  - conda-forge
  - nvidia
  - nodefaults
show_channel_urls: True
default_threads: 4
pkgs_dirs:
  - /scratch/micromamba/pkgs
envs_dirs:
  - /scratch/micromamba/envs
HERE
