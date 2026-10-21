#!/usr/bin/env bash
# Activate the project virtualenv and expose the pip-installed CUDA 11.8
# libraries required by TensorFlow 2.12 (CUDA 11.8 + cuDNN 8.6).
#
# Usage:
#   source scripts/activate_gpu.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/.venv/bin/activate"

NVIDIA_LIB_DIRS="$(find "$PROJECT_ROOT/.venv/lib"/python*/site-packages/nvidia -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)"

if [ -n "$NVIDIA_LIB_DIRS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

unset PROJECT_ROOT NVIDIA_LIB_DIRS
