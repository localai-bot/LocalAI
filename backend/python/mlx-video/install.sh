#!/bin/bash
# Installation script for MLX Video Backend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
PIP="${PIP:-$PYTHON -m pi}"

echo "Installing MLX Video Backend dependencies..."

# Detect platform and install appropriate requirements
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - using MPS requirements"
    if [[ -f "requirements-mps.txt" ]]; then
        $PIP install -r requirements-mps.txt
    else
        $PIP install -r requirements.txt
    fi
elif command -v nvidia-smi &> /dev/null; then
    echo "Detected CUDA - using CUDA requirements"
    if [[ -f "requirements-cublas12.txt" ]]; then
        $PIP install -r requirements-cublas12.txt
    else
        $PIP install -r requirements.txt
    fi
else
    echo "Using CPU requirements"
    if [[ -f "requirements-cpu.txt" ]]; then
        $PIP install -r requirements-cpu.txt
    else
        $PIP install -r requirements.txt
    fi
fi

# Install the backend itself
$PIP install -e .

echo "MLX Video Backend installed successfully!"
