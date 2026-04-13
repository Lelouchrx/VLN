#!/bin/bash
# Install habitat-lab and habitat-sim
set -e

# Activate environment
eval "$(conda shell.bash hook)"
CONDA_ENV="${CONDA_ENV:-vln}"
HABITAT_VERSION="${HABITAT_VERSION:-v0.2.4}"
conda activate "${CONDA_ENV}"

# Clean up old installations
pip uninstall habitat-lab habitat-sim -y || true
rm -rf habitat-lab habitat-sim

# Install habitat-lab
echo "Installing habitat-lab..."
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout "${HABITAT_VERSION}"
pip install -e habitat-lab
cd ..

# Install habitat-sim
echo "Installing habitat-sim..."
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout "${HABITAT_VERSION}"

# Build options
export HABITAT_BUILD_GUI_VIEWERS=OFF
export HABITAT_WITH_CUDA=ON
export HABITAT_WITH_BULLET=ON
export HABITAT_WITH_AUDIO=OFF

pip install . --no-build-isolation
cd ..

echo "Installation complete!"