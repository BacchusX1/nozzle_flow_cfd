#!/bin/bash
# Setup script for Nozzle CFD Design Tool
# Creates a conda environment with all dependencies including SU2 solver

set -e  # Exit on error

ENV_NAME="su2_env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  Nozzle CFD Design Tool - Environment Setup"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "   Please install Miniconda or Miniforge first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "📦 Conda found: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists."
    read -p "   Do you want to remove and recreate it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "   Updating existing environment instead..."
        UPDATING=true
    fi
fi

if [ "$UPDATING" = true ]; then
    echo ""
    echo "📥 Updating environment '${ENV_NAME}'..."
    conda activate "${ENV_NAME}" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "${ENV_NAME}"
    
    # Install/update packages
    echo "   Installing SU2 solver from conda-forge (version 8.1.0)..."
    conda install -c conda-forge su2=8.1.0 -y
    
    echo "   Installing Python dependencies..."
    conda install -c conda-forge pyside6 numpy scipy matplotlib gmsh python-gmsh -y
    
    echo "   Installing test dependencies..."
    pip install pytest pytest-qt
else
    echo ""
    echo "🔨 Creating new conda environment '${ENV_NAME}'..."
    echo "   This may take several minutes..."
    echo ""

    # Create environment with Python and all dependencies
    # Pin SU2 to version 8.1.0 which is tested and working
    conda create -n "${ENV_NAME}" -c conda-forge \
        python=3.11 \
        su2=8.1.0 \
        pyside6 \
        numpy \
        scipy \
        matplotlib \
        gmsh \
        python-gmsh \
        pytest \
        -y

    echo ""
    echo "📥 Activating environment to install additional packages..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
    
    # Install pytest-qt via pip (not always in conda-forge)
    pip install pytest-qt
fi

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "=========================================="
echo "  How to use:"
echo "=========================================="
echo ""
echo "  1. Activate the environment:"
echo "     conda activate ${ENV_NAME}"
echo ""
echo "  2. Run the application:"
echo "     cd ${PROJECT_ROOT}"
echo "     python main.py"
echo ""
echo "  3. Run tests:"
echo "     cd ${PROJECT_ROOT}"
echo "     pytest tests/ -v"
echo ""
echo "  4. Verify SU2 installation:"
echo "     SU2_CFD --help"
echo ""
echo "=========================================="

# Verify SU2 is installed
echo ""
echo "🔍 Verifying SU2 installation..."
if command -v SU2_CFD &> /dev/null; then
    echo "✅ SU2_CFD found: $(which SU2_CFD)"
    SU2_CFD --help 2>&1 | head -5 || true
else
    echo "⚠️  SU2_CFD not found in PATH after installation"
    echo "   You may need to activate the environment first:"
    echo "   conda activate ${ENV_NAME}"
fi

echo ""
echo "🎉 Setup complete! Activate with: conda activate ${ENV_NAME}"
