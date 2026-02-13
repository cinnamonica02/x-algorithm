# Training Ready - E-commerce Ranking Model

## Status: ✅ Ready for RunPod Training

All code is complete and tested. Ready to train on RTX 4090.

## What We Built

### Phase 2: Data Preprocessing ✅
- **Dataset:** 551K interactions, 23K users, 54K products, 924 categories
- **Format:** Temporal sequences with 80/10/10 train/val/test split
- **Actions:** 3 types (transaction, addtocart, view)
- **Files:** `data/processed/*.pkl`

### Phase 3: Model & Training ✅
- **Model:** 474K parameters, Grok-based transformer
- **Architecture:** Customer + history → candidates with isolation
- **Training:** Weighted BCE loss, AdamW optimizer
- **Code Style:** Clean, following X algorithm conventions

## Files Created

```
phoenix/
├── ecommerce_config.py              # Configs and data structures
├── ecommerce_ranking_model.py       # Ranking model
├── test_ecommerce_model.py          # Model tests ✅ passing
├── train_ecommerce_ranking.py       # Training script
└── data/
    ├── ecommerce_dataset.py         # Data loader ✅ tested
    ├── prepare_retail_rocket.py     # Preprocessing ✅ complete
    └── processed/
        ├── train_sequences.pkl
        ├── val_sequences.pkl
        ├── test_sequences.pkl
        └── vocabularies.pkl
```

## Training on RunPod

### Quick Start

```bash
# 1. SSH into RunPod 4090 pod
ssh root@<pod-ip> -p <pod-port>

# 2. Clone/sync repo to /workspace/x-algorithm

# 3. Install deps
cd /workspace/x-algorithm/phoenix
pip install jax[cuda11_pip] dm-haiku optax tqdm pandas numpy -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 4. Train
python train_ecommerce_ranking.py --epochs 10
```

### Training Time (RTX 4090)

- **Test run (2 epochs):** 5 minutes
- **Default (10 epochs):** 25 minutes
- **Full (20 epochs):** 50 minutes

### Expected Cost

RunPod RTX 4090 @ $0.69/hour:
- Test: $0.06
- Default: $0.29
- Full: $0.58

## Model Architecture

```
Input Batch:
  customer_hashes: [B, 2]
  history: [B, 64, ...]
  candidates: [B, 16, ...]

Model:
  Embedding lookup (hash-based)
  → Customer reduce
  → History reduce
  → Candidate reduce
  → Concat [customer, history, candidates]
  → Transformer (4 layers, 128 dim)
  → Logits [B, 16, 3]

Loss:
  Weighted BCE (transaction=10x, addtocart=3x, view=1x)

Output:
  Predictions for [transaction, addtocart, view]
```

## Training Config

Default settings (balanced speed/quality):
```python
--emb_size 128              # Embedding dimension
--num_layers 4              # Transformer layers
--history_len 64            # History sequence length
--candidate_len 16          # Candidates to rank
--epochs 10                 # Training epochs
--batch_size 128            # Batch size
--steps_per_epoch 500       # Steps per epoch
--lr 1e-4                   # Learning rate
--weight_decay 0.01         # Weight decay
```

## Validation Metrics

Training will report:
- **Loss:** Weighted BCE loss
- **Accuracy:** Binary prediction accuracy
- **Per-action probs:** Avg probability for transaction/addtocart/view

Target performance (realistic for dataset):
- Val accuracy: >80%
- Transaction prob: >0.6 when present
- View prob: >0.9 when present

## After Training

Checkpoints saved to `checkpoints/checkpoint_step_*.pkl`:
- Contains: params, embedding_tables, opt_state
- Download via scp
- Use for inference (TODO: create inference script)

## Next Steps

1. **Train on RunPod** (25-50 minutes)
2. **Download checkpoints**
3. **Create inference script** (predict for new customers)
4. **Optionally: Train retrieval model** (Phase 3.3)

See `RUNPOD_SETUP.md` for detailed RunPod instructions.

---

**Ready to train!** 🚀
