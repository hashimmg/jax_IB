#! /bin/bash

set -ef -o pipefail

eval "$("${MAMBA_EXE}" shell hook --shell=bash)"
micromamba activate jax-cfd

exec bash -o pipefail -c "$@"
