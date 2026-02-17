# RunPod Training Setup - FIXED

This guide provides the **correct workflow** for training on RunPod GPU.

## Common Issue: Slow Training / CPU Fallback

**Problem:** Using `uv sync` installs the wrong JAX version without GPU support.

**Solution:** Use the `setup_gpu.sh` script or follow manual installation below.

## Quick Start (Recommended)

### 1. Spin Up Pod

- GPU: RTX 4090, A6000, or similar
- Template: PyTorch 2.0+ (includes CUDA)
- Storage: 30GB minimum

### 2. SSH Into Pod

```bash
ssh root@<pod-ip> -p <pod-port>
```

### 3. Clone/Upload Code

**Option A: Clone from GitHub**
```bash
cd /workspace
git clone <your-repo-url> x-algorithm
cd x-algorithm/phoenix
```

**Option B: Upload from local (from your local machine)**
```bash
# From local machine (Windows):
# First, compress the folder
tar -czf phoenix.tar.gz phoenix/

# Upload to RunPod
scp -P <pod-port> phoenix.tar.gz root@<pod-ip>:/workspace/

# On RunPod, extract
cd /workspace
tar -xzf phoenix.tar.gz
cd phoenix
```

### 4. Run Automated Setup

```bash
cd /workspace/x-algorithm/phoenix  # or wherever you uploaded

# Make script executable
chmod +x setup_gpu.sh

# Run setup (installs JAX with GPU support)
bash setup_gpu.sh
```

Expected output:
```
==================================================================
JAX GPU Setup for E-commerce Training
==================================================================
Detected CUDA version: 11.x
...
Installing JAX with CUDA 11 support...
...
✓ SUCCESS: GPU detected and ready!
==================================================================
Setup complete! Ready to train.
```

### 5. Verify Data Files

```bash
ls -lh data/processed/
```

Should see:
- `train_sequences.pkl`
- `val_sequences.pkl`
- `test_sequences.pkl`
- `vocabularies.pkl`

If missing, you need to upload the processed data or run preprocessing:
```bash
python data/prepare_retail_rocket.py
```

### 6. Start Training

```bash
# Quick test (5 minutes)
python train_ecommerce_ranking.py --epochs 2 --steps_per_epoch 100

# Default training (25 minutes)
python train_ecommerce_ranking.py --epochs 10

# Full training (50 minutes)
python train_ecommerce_ranking.py --epochs 20 --batch_size 256 --emb_size 256 --num_layers 6
```

## Manual Installation (if script fails)

### Step 1: Check CUDA Version

```bash
nvidia-smi | grep "CUDA Version"
```

### Step 2: Install JAX with GPU Support

**For CUDA 11.x (most RunPod pods):**
```bash
pip install --upgrade pip
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**For CUDA 12.x:**
```bash
pip install --upgrade pip
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Step 3: Install Other Dependencies

```bash
pip install dm-haiku optax tqdm pandas numpy
```

### Step 4: Verify GPU Works

```bash
python -c "import jax; print('Backend:', jax.default_backend()); print('Devices:', jax.devices())"
```

**Expected:** `Backend: gpu` and `Devices: [cuda(id=0)]`

**If you see "cpu":** JAX is not detecting GPU. See Troubleshooting below.

## What NOT to Do

❌ **Don't use `uv sync` on RunPod** - it installs CPU-only JAX
❌ **Don't use `pip install jax[cuda12]`** - this doesn't work, you need `cuda12_pip`
❌ **Don't skip the `-f https://storage.googleapis.com/...` flag** - GPU JAX is not on standard PyPI

## Training Output

Expected progress:
```
==================================================================
E-commerce Ranking Model Training (JIT-compiled)
==================================================================

JAX devices: [cuda(id=0)]
JAX backend: gpu
✓ GPU detected - training will be fast!

...

Compiling training and evaluation functions (JIT)...
Warming up JIT compilation (first iteration is slow)...
✓ JIT compilation complete - training will be fast now!

Training: 10 epochs, 500 steps/epoch, batch_size=128

==================================================================
Epoch 1/10
loss=0.4521, acc=0.823: 100%|████████| 500/500 [01:15<00:00,  6.67it/s]
Train: loss=0.4521, acc=0.823
Val: acc=0.817
```

**Speed check:** Each epoch should take ~1-2 minutes on RTX 4090. If it takes >10 minutes, GPU is not being used.

## Monitoring

### GPU Usage

```bash
# In another terminal, watch GPU usage
watch -n 1 nvidia-smi
```

Should show:
- GPU utilization: 80-100%
- Memory usage: 4-8GB (depending on batch size)

### Training Checkpoints

```bash
ls -lh checkpoints/
```

Checkpoints saved every 2 epochs:
- `checkpoint_step_1000.pkl`
- `checkpoint_step_2000.pkl`
- etc.

## Download Results

After training completes:

```bash
# From your local machine
scp -P <pod-port> -r root@<pod-ip>:/workspace/x-algorithm/phoenix/checkpoints/ ./
```

## Expected Timing (RTX 4090)

| Configuration | Epochs | Time | Cost @ $0.69/hr |
|--------------|--------|------|-----------------|
| Quick test | 2 | 5 min | $0.06 |
| Default | 10 | 25 min | $0.29 |
| Full | 20 | 50 min | $0.58 |
| Large model | 20 | 90 min | $1.04 |

## Troubleshooting

### Training script errors with "GPU not detected"

This is intentional - the script refuses to train on CPU because it's 100x slower.

**Fix:**
```bash
pip uninstall jax jaxlib -y
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -c "import jax; print(jax.default_backend())"  # Should show 'gpu'
```

### Training is very slow (>10 min per epoch)

JAX is running on CPU, not GPU.

**Fix:** See above, or run `bash setup_gpu.sh`

### Out of memory error

Reduce batch size or model size:
```bash
python train_ecommerce_ranking.py --batch_size 64 --emb_size 128 --num_layers 4
```

### Data files not found

Either upload processed data or run preprocessing:
```bash
# Make sure retail_rocket/ folder is present
ls retail_rocket/

# Run preprocessing
python data/prepare_retail_rocket.py
```

### Connection lost during training

Use `tmux` or `screen` to keep training running:
```bash
tmux new -s training
python train_ecommerce_ranking.py --epochs 10
# Press Ctrl+B then D to detach

# Later, reattach:
tmux attach -t training
```

## Cost Optimization

- Use preemptible/spot instances (50% cheaper)
- Use RTX 4090 instead of A100 (4x cheaper, similar speed for this workload)
- Start with quick test (2 epochs) to verify everything works before full training
- Terminate pod immediately after training completes

## Next Steps

After training:
1. Download checkpoints
2. Create inference script for product recommendations
3. Evaluate on test set
4. (Optional) Train retrieval model using similar setup
