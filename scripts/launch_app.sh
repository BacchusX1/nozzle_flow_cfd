#!/bin/bash
# Launch the nozzle CFD design tool

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Launching Nozzle CFD Design Tool..."

# Activate the su2_env if not already active
if [[ "$CONDA_DEFAULT_ENV" != "su2_env" ]]; then
    echo "📦 Activating su2_env conda environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate su2_env
fi

cd "$PROJECT_ROOT"
python main.py
