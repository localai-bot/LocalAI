#!/bin/bash
# Test script for MLX Video Backend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"

echo "Running MLX Video Backend tests..."

# Run the test client
$PYTHON test.py

echo "All tests passed!"
