#!/bin/bash
# Run script for MLX Video Backend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
PORT="${PORT:-50052}"

# Check if model_id is provided
if [ -z "$MODEL_ID" ]; then
    echo "Error: MODEL_ID environment variable is required"
    echo "Usage: MODEL_ID=<model_name> ./run.sh"
    exit 1
fi

echo "Starting MLX Video Backend..."
echo "  Model: $MODEL_ID"
echo "  Port: $PORT"

# Detect platform and set appropriate flags
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  Platform: macOS (MPS enabled)"
    export TORCH_ENABLE_MPS=1
elif command -v nvidia-smi &> /dev/null; then
    echo "  Platform: CUDA enabled"
else
    echo "  Platform: CPU (no GPU acceleration)"
fi

exec $PYTHON backend.py --port "$PORT"
