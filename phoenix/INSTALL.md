# Installation Guide

## Quick Start (GPU Training - Recommended)

### For RunPod / Cloud GPU

```bash
cd phoenix

# Run the automated setup script
bash setup_gpu.sh

# Verify GPU is working
python -c "import jax; print('Backend:', jax.default_backend()); print('Devices:', jax.devices())"
# Should show: Backend: gpu (or cuda), Devices: [cuda(id=0)]

# Start training
python train_ecommerce_ranking.py --epochs 10
```

### Manual Installation (if script fails)

**Step 1: Check CUDA version**
```bash
nvidia-smi  # Look for "CUDA Version: X.X"
```

**Step 2: Install JAX with GPU support**

For CUDA 12.x (most RunPod/Lambda pods):
```bash
pip install --upgrade pip
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For CUDA 11.x:
```bash
pip install --upgrade pip
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Step 3: Install other dependencies**
```bash
pip install dm-haiku optax tqdm pandas numpy
```

**Step 4: Verify GPU**
```bash
python -c "import jax; print('Backend:', jax.default_backend()); print('Devices:', jax.devices())"
```

Expected output:
```
Backend: gpu
Devices: [cuda(id=0)]
```

## Local Development (CPU or GPU)

### Using uv (for local development only)

**WARNING:** Do NOT use `uv sync` on RunPod - it doesn't install GPU-enabled JAX correctly!

For local CPU-only testing:
```bash
cd phoenix
pip install uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

For local GPU:
```bash
cd phoenix
pip install uv
uv venv
source .venv/bin/activate
# Don't use uv sync - install JAX manually:
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv sync --no-install-project  # Install other deps only
```

## Troubleshooting

### "RuntimeError: GPU not detected"

The training script checks for GPU and will fail if not detected. This is intentional to prevent slow CPU training.

**Solution:**
```bash
# Uninstall any existing JAX
pip uninstall jax jaxlib -y

# Reinstall with CUDA support
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify
python -c "import jax; print(jax.default_backend())"
```

### "WARNING: Falling back to cpu"

JAX was installed without CUDA support. Follow the solution above.

### "CUDA version mismatch"

Your system CUDA version doesn't match the installed JAX CUDA version.

**Solution:**
```bash
# Check your CUDA version
nvidia-smi

# Install matching JAX version
# For CUDA 11.x:
pip uninstall jax jaxlib -y
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 12.x:
pip uninstall jax jaxlib -y
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Training still slow even with GPU

Verify JAX is actually using the GPU:

```bash
python -c "
import jax
import jax.numpy as jnp
import time

# Test GPU
x = jax.random.normal(jax.random.PRNGKey(0), (10000, 10000))
start = time.time()
y = jnp.dot(x, x).block_until_ready()
elapsed = time.time() - start

print(f'Backend: {jax.default_backend()}')
print(f'Devices: {jax.devices()}')
print(f'Matrix multiply time: {elapsed:.3f}s')
print(f'Expected: <0.1s on GPU, >2s on CPU')
"
```

If it's still slow (>2 seconds), JAX is not using the GPU properly.

## Dependencies Reference

**Core dependencies:**
- JAX (with CUDA support): 0.4.20+
- Haiku: 0.0.13+
- Optax: 0.2.0+
- NumPy: 1.26.4+
- Pandas: 2.0.0+
- tqdm: 4.66.0+

**Python version:** 3.11+

## Testing Installation

Quick test:
```bash
python test_ecommerce_model.py
```

Expected output:
```
Testing e-commerce ranking model...
✓ Model forward pass works
✓ Output shape correct
✓ Gradients computed successfully
[OK] All tests passed!
```
