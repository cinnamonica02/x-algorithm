#!/bin/bash
# GPU Setup Script for RunPod/Cloud Training
# This script ensures JAX with CUDA support is installed correctly

set -e

echo "=================================================================="
echo "JAX GPU Setup for E-commerce Training"
echo "=================================================================="

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1)
    echo "Detected CUDA version: $CUDA_VERSION.x"
else
    echo "WARNING: nvidia-smi not found. Assuming CUDA 12.x"
    CUDA_VERSION=12
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install JAX with correct CUDA support
echo ""
echo "Installing JAX with CUDA $CUDA_VERSION support..."
if [ "$CUDA_VERSION" = "12" ]; then
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
elif [ "$CUDA_VERSION" = "11" ]; then
    pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
    echo "ERROR: Unsupported CUDA version: $CUDA_VERSION"
    exit 1
fi

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install dm-haiku optax tqdm pandas numpy

# Verify GPU is detected
echo ""
echo "=================================================================="
echo "Verifying GPU detection..."
echo "=================================================================="
python -c "
import jax
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
print('Backend:', jax.default_backend())
backend = jax.default_backend()
if backend == 'gpu':
    print('✓ SUCCESS: GPU detected and ready!')
elif backend == 'cuda':
    print('✓ SUCCESS: CUDA detected and ready!')
else:
    print('✗ ERROR: GPU not detected. Backend is:', backend)
    print('Training will be 100x slower on CPU!')
    exit(1)
"

echo ""
echo "=================================================================="
echo "Setup complete! Ready to train."
echo "=================================================================="
echo ""
echo "Next steps:"
echo "  python train_ecommerce_ranking.py --epochs 10"
