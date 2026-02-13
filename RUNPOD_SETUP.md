# RunPod Training Setup

Quick guide for training on RunPod with RTX 4090.

## 1. Spin Up Pod

```bash
# RunPod: Select RTX 4090 pod
# Template: PyTorch 2.0+ with CUDA 11.8+
# Storage: 30GB minimum
```

## 2. SSH Into Pod

```bash
ssh root@<pod-ip> -p <pod-port>
```

## 3. Clone & Setup

```bash
# Clone your repo or rsync the data
cd /workspace

# If syncing from local:
# rsync -avz -e "ssh -p <pod-port>" \
#   C:/Users/Maria\ Guevara/Desktop/xAI_rec_sys/x-algorithm/ \
#   root@<pod-ip>:/workspace/x-algorithm/

# Install dependencies
cd x-algorithm/phoenix
pip install --upgrade pip
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install dm-haiku optax tqdm pandas numpy

# Verify GPU
python -c "import jax; print(jax.devices())"
# Should show: [cuda(id=0)]
```

## 4. Verify Data

```bash
ls -lh data/processed/
# Should see: train_sequences.pkl, val_sequences.pkl, vocabularies.pkl

python data/ecommerce_dataset.py
# Should pass without errors
```

## 5. Test Model

```bash
python test_ecommerce_model.py
# Should show: [OK] All tests passed!
```

## 6. Start Training

```bash
# Default config (recommended for first run)
python train_ecommerce_ranking.py \
    --epochs 10 \
    --batch_size 128 \
    --steps_per_epoch 500

# Fast test run (5 minutes)
python train_ecommerce_ranking.py \
    --epochs 2 \
    --batch_size 64 \
    --steps_per_epoch 100

# Full training (45-60 minutes on 4090)
python train_ecommerce_ranking.py \
    --epochs 20 \
    --batch_size 256 \
    --steps_per_epoch 500 \
    --emb_size 256 \
    --num_layers 6
```

## 7. Monitor Training

Training logs will show:
```
Epoch 1/10
loss=0.4521, acc=0.823: 100%|████| 500/500 [01:15<00:00,  6.67it/s]
Train: loss=0.4521, acc=0.823
Val: acc=0.817
```

## 8. Checkpoints

Saved to `checkpoints/`:
```bash
ls -lh checkpoints/
checkpoint_step_1000.pkl
checkpoint_step_2000.pkl
...
```

Download after training:
```bash
# From local machine:
scp -P <pod-port> -r root@<pod-ip>:/workspace/x-algorithm/phoenix/checkpoints/ ./
```

## Expected Training Time (RTX 4090)

| Config | Time | Cost @ $0.69/hr |
|--------|------|-----------------|
| Test (2 epochs) | 5 min | $0.06 |
| Default (10 epochs) | 25 min | $0.29 |
| Full (20 epochs) | 50 min | $0.58 |

## Troubleshooting

**Out of memory:**
```bash
# Reduce batch size
python train_ecommerce_ranking.py --batch_size 64
```

**JAX not finding GPU:**
```bash
# Reinstall JAX with correct CUDA version
pip uninstall jax jaxlib
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Data loading slow:**
```bash
# Verify data is on fast storage (not network mount)
df -h data/processed/
```

## After Training

Test the trained model:
```bash
# TODO: Create inference script
python test_trained_model.py --checkpoint checkpoints/checkpoint_step_5000.pkl
```
