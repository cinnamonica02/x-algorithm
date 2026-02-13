# E-Commerce Recommendation System Adaptation Plan

## Context

This project adapts X's open-sourced recommendation algorithm (Grok-based transformers with candidate isolation) to e-commerce. The goal is to:
1. **Learn** how production-scale recommendation systems work
2. **Build** a functional MVP using public e-commerce data
3. **Compare** the approach to Allegro's existing recommendations
4. **Adapt** the trained system to Allegro's catalog (transfer learning)

### Why This Approach?
X's system eliminates hand-engineered features in favor of learning directly from user engagement sequences using transformers. This is cutting-edge for RecSys and offers a unique learning opportunity. The modular architecture (two-tower retrieval + transformer ranking with candidate isolation) is production-proven and adaptable to e-commerce.

### Constraints
- **Solo developer** (you!)
- **Budget**: ~$20-50 for training (using Retail Rocket; $50-100 if using H&M)
- **Time**: 2-3 weeks for MVP with Retail Rocket (3-4 weeks with H&M)
- **No real user data** from Allegro initially
- **Resources**: Local GPU or Colab Pro (~$10/month)

---

## Phase 1: Public Dataset Demo (Option A)

### Dataset Selection

**Recommended: Retail Rocket E-commerce Dataset (Kaggle)** ⭐
- **Size**: 2.7M events, ~235K unique visitors, ~235K products
- **Rich data**: Three explicit engagement types (view, addtocart, transaction) + timestamps
- **Perfect for**: Maps directly to X's multi-action prediction (view → cart → purchase pipeline)
- **Advantages for solo dev**:
  - Smaller = faster iteration and debugging
  - Multiple event types built-in (exactly what we need!)
  - Diverse product categories (not just fashion, closer to Allegro)
  - Already structured for RecSys (less preprocessing)
  - Can train models in hours not days
- **Download**: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

**Alternative: H&M Personalized Fashion Recommendations**
- **Size**: 31M transactions, 1.3M products, 1.4M users
- **Rich data**: Customer transactions, product metadata, images
- **Events**: Purchases over time (implicit feedback)
- **Use this if**: You want larger scale, focus on fashion specifically, or need richer product metadata
- **Trade-off**: Much larger, longer training times (days vs hours), fashion-only
- **Download**: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations

**Alternative: Amazon Product Reviews (if need even more data)**
- **Size**: Varies by category (5M-100M+ reviews)
- **Rich data**: Reviews, ratings, purchase history, product metadata
- **Events**: Rating, review, purchase timestamps
- **Good for**: Multi-task learning (rating prediction + purchase prediction)

**Key Requirements:**
- User-item interaction sequences (temporal order matters!)
- Multiple engagement types (view, cart, purchase, wishlist if available)
- Product metadata (category, price, brand)
- At least 100K users with 5+ interactions each

---

## Phase 2: Data Preprocessing & Feature Engineering

### 2.1 Data Transformation (X → E-commerce Mapping)

**X Format** → **E-commerce Format**

| X Concept | E-commerce Equivalent | Implementation |
|-----------|----------------------|----------------|
| `user_id` | `customer_id` | Direct mapping |
| `tweet_id` | `product_id` | Direct mapping |
| `author_id` | `brand_id` or `category_id` | Use brand as "author" |
| `user_action_sequence` | `interaction_sequence` | List of (product_id, action_type, timestamp) |
| `history_actions` (19 types) | **8 action types**: view, add_to_cart, purchase, add_to_wishlist, remove_from_cart, remove_from_wishlist, click, share | Multi-hot encoding [B, seq_len, 8] |
| `product_surface` | `product_category` | Categorical: electronics, fashion, home, etc. (vocab_size=20-50) |
| Post embeddings | Product embeddings | Hash-based: 2 hashes per product |
| Author embeddings | Brand embeddings | Hash-based: 2 hashes per brand |

### 2.2 Preprocessing Pipeline

**Step 1: Filter Users & Products**
```python
# Keep users with at least 10 interactions (need history)
# Keep products with at least 5 interactions (cold start handled separately)
min_user_interactions = 10
min_product_interactions = 5
```

**Step 2: Create Interaction Sequences**
```python
# Sort by timestamp, create sequences
# Format: user_id → [(product_id, brand_id, action_type, timestamp), ...]
# Split: 80% train, 10% val, 10% test (temporal split!)
```

**Step 3: Action Type Mapping**

For **Retail Rocket** (3 native event types):
```python
action_mapping = {
    'transaction': 0,        # Strongest signal (purchase)
    'addtocart': 1,
    'view': 2,
}
# Can extend with synthetic actions if needed:
# 'remove_from_cart': 3, 'click': 4, etc.
```

For **H&M or other datasets** (infer from data):
```python
action_mapping = {
    'purchase': 0,           # Strongest signal
    'add_to_cart': 1,
    'add_to_wishlist': 2,
    'view': 3,
    'click': 4,
    'share': 5,
    'remove_from_cart': 6,   # Negative signal
    'remove_from_wishlist': 7 # Negative signal
}
```

**Step 4: Create Training Examples**
```python
# For each user interaction sequence:
# History: last N interactions (N=32 or 64)
# Candidates: Next M interactions to predict (M=8-16)
# Labels: Multi-hot vector [M, 8] indicating which actions happened
```

### 2.3 Embedding Tables

**Hash-Based Embeddings** (like X's approach):
```python
# Customer embeddings: 2 hash functions
customer_vocab_size = 10_000  # Modulo hash
customer_emb_dim = 128

# Product embeddings: 2 hash functions
product_vocab_size = 50_000
product_emb_dim = 128

# Brand embeddings: 2 hash functions
brand_vocab_size = 5_000
brand_emb_dim = 128

# Category embeddings: Single vocab
category_vocab_size = 50  # Direct lookup
```

---

## Phase 3: Model Adaptation

### 3.1 Two-Tower Retrieval Model

**File to adapt**: `phoenix/recsys_retrieval_model.py`

**Changes:**
1. **Input batch structure**:
   ```python
   @dataclass
   class EcommerceRetrievalBatch:
       customer_hashes: jnp.ndarray  # [B, 2]
       history_product_hashes: jnp.ndarray  # [B, seq_len, 2]
       history_brand_hashes: jnp.ndarray  # [B, seq_len, 2]
       history_actions: jnp.ndarray  # [B, seq_len, 3] for Retail Rocket (multi-hot)
       history_category: jnp.ndarray  # [B, seq_len] (vocab index)
       # Candidates only needed for corpus pre-computation
       candidate_product_hashes: jnp.ndarray  # [N, 2]
       candidate_brand_hashes: jnp.ndarray  # [N, 2]
       candidate_category: jnp.ndarray  # [N]
   ```

2. **Model config**:
   ```python
   config = PhoenixRetrievalModelConfig(
       emb_size=128,
       history_seq_len=64,  # Typical e-commerce history
       candidate_seq_len=16,
       num_actions=3,  # For Retail Rocket: transaction, addtocart, view
       category_vocab_size=50,
       model=TransformerConfig(
           emb_size=128,
           key_size=64,
           num_q_heads=4,
           num_kv_heads=2,  # GQA
           num_layers=4,  # Scale up from demo
           widening_factor=2.0,
       ),
       hash_config=HashConfig(
           num_user_hashes=2,
           num_item_hashes=2,
           num_author_hashes=2,
       ),
   )
   ```

3. **Training objective**:
   - **Metric learning**: Maximize similarity between customer embedding and purchased products
   - **Loss**: Contrastive loss (InfoNCE) or triplet loss
   - **Negative sampling**: Sample non-purchased products as negatives

**Training estimate**:
- **Model size**: ~3-5M parameters
- **Training time**: 3-6 hours on single GPU with Retail Rocket (6-12 hours with H&M)
- **Cost**: Free on Colab, or ~$5-10 on GCP

### 3.2 Transformer Ranking Model

**File to adapt**: `phoenix/recsys_model.py`

**Changes:**
1. **Input batch structure** (same as retrieval + candidates):
   ```python
   @dataclass
   class EcommerceRankingBatch:
       # Same as retrieval batch, PLUS:
       candidate_product_hashes: jnp.ndarray  # [B, C, 2]
       candidate_brand_hashes: jnp.ndarray  # [B, C, 2]
       candidate_category: jnp.ndarray  # [B, C]
   ```

2. **Model config**:
   ```python
   config = PhoenixModelConfig(
       emb_size=128,
       num_actions=3,  # For Retail Rocket: transaction, addtocart, view
       history_seq_len=64,
       candidate_seq_len=16,
       category_vocab_size=50,
       model=TransformerConfig(
           emb_size=128,
           key_size=64,
           num_q_heads=4,
           num_kv_heads=2,
           num_layers=4,  # More layers than retrieval
           widening_factor=2.0,
       ),
       hash_config=HashConfig(...),
   )
   ```

3. **Multi-task prediction heads**:
   ```python
   # Output: [B, num_candidates, 3] for Retail Rocket
   # Actions: transaction (purchase), addtocart, view
   ```

4. **Training objective**:
   - **Multi-task learning**: Predict probability of each action
   - **Loss**: Weighted binary cross-entropy (weight purchase > view)
   - **Weighting for Retail Rocket**: `transaction=10.0, addtocart=3.0, view=1.0`

**Training estimate**:
- **Model size**: ~5-8M parameters
- **Training time**: 12-24 hours on single GPU with Retail Rocket (24-48 hours with H&M)
- **Cost**: ~$15-30 on GCP (preemptible instances), or Colab Pro

### 3.3 Candidate Isolation Attention Mask

**File**: `phoenix/grok.py` - `make_recsys_attn_mask()`

**No changes needed!** This function is domain-agnostic:
```python
def make_recsys_attn_mask(seq_len: int, candidate_start_offset: int):
    # candidate_start_offset = 1 (user) + history_seq_len
    # Candidates can attend to user+history, but not each other
    # Returns: [1, 1, seq_len, seq_len] mask
```

**Usage**:
```python
candidate_start = 1 + history_seq_len  # After user + history
mask = make_recsys_attn_mask(total_seq_len, candidate_start)
```

---

## Phase 4: Training Strategy

### 4.1 Training Loop

**File**: Create `phoenix/train_ecommerce.py` (new file)

**Approach**:
```python
# Stage 1: Train retrieval model (6-12 hours)
# - Learns customer & product embeddings
# - Outputs: customer_tower_params, product_tower_params

# Stage 2: Freeze embeddings, train ranker (24-48 hours)
# - Uses same embeddings from Stage 1
# - Learns multi-task prediction heads
# - Fine-tunes transformer on ranking objective
```

**Optimization**:
- **Optimizer**: AdamW with learning rate 1e-4
- **Batch size**: 256 (retrieval), 128 (ranking)
- **Gradient clipping**: 1.0
- **Learning rate schedule**: Cosine decay
- **Mixed precision**: Use JAX's `jax.lax.Precision.DEFAULT` for faster training

### 4.2 Data Loading

**File**: Create `phoenix/data_loader.py` (new file)

**Requirements**:
- Stream data from disk (can't fit all in memory)
- Random shuffling per epoch
- Prefetching for GPU utilization
- Handle variable-length sequences (pad to max_len)

**Libraries**: Use `tf.data` or `torch.utils.data.DataLoader` with JAX

### 4.3 Evaluation Metrics

**Retrieval Metrics**:
- **Recall@K** (K=10, 50, 100): % of next purchases in top-K retrieved
- **MRR** (Mean Reciprocal Rank): Position of first relevant item
- **NDCG@K**: Ranking quality

**Ranking Metrics**:
- **AUC-ROC**: Per-action classification quality
- **Precision@K / Recall@K**: Top-K recommendation quality
- **Purchase Rate@K**: % of top-K with actual purchase

**Target Performance** (realistic for public dataset):
- Retrieval: Recall@100 > 0.40
- Ranking: Purchase Rate@10 > 0.15, AUC > 0.75

---

## Phase 5: Inference & Serving (Simplified MVP)

### 5.1 Retrieval Pipeline

**File**: Adapt `phoenix/run_retrieval.py`

**Flow**:
```python
# 1. Pre-compute product embeddings (offline, once)
corpus_embeddings = candidate_tower.encode_all(all_products)
# Save to disk: corpus_embeddings.npy

# 2. At inference time:
user_embedding = user_tower.encode(customer_history)
similarity_scores = user_embedding @ corpus_embeddings.T
top_k_indices = jnp.argsort(-similarity_scores)[:100]
```

**Output**: Top 100 candidate products per user

### 5.2 Ranking Pipeline

**File**: Adapt `phoenix/run_ranker.py`

**Flow**:
```python
# Input: customer_id + top_k_candidates (from retrieval)
# 1. Fetch customer history
# 2. Create batch with candidates
# 3. Forward pass → probabilities [K, 3] for Retail Rocket
# 4. Compute weighted score:
#    score = 10*P(transaction) + 3*P(addtocart) + 1*P(view)
# 5. Sort by score, return top N
```

**Output**: Ranked list of N products (N=20-50)

### 5.3 End-to-End Demo

**File**: Create `phoenix/demo_ecommerce.py` (new file)

```python
# Simple script:
# python demo_ecommerce.py --customer_id 12345 --top_n 20

# 1. Load retrieval model & corpus embeddings
# 2. Retrieve top 100 candidates
# 3. Load ranking model
# 4. Rank candidates
# 5. Print top 20 with scores
```

---

## Phase 6: Comparison with Allegro

### 6.1 Allegro API Exploration

**First Step**: Explore Allegro's public API
- Endpoint: https://api.allegro.pl/
- Documentation: https://developer.allegro.pl/
- Goal: Understand what data is available

**Expected endpoints**:
- `/offers` - Search/browse products
- `/recommendations` - Allegro's existing recommendations (if exposed)
- `/users/me` - Your account data

**Data collection**:
```python
# 1. Query Allegro API for products in same category
# 2. Record Allegro's recommendation order
# 3. Re-rank using your model
# 4. Compare:
#    - Top-K overlap (Jaccard similarity)
#    - Rank correlation (Spearman's rho)
#    - Click-through rate (manual testing)
```

### 6.2 Qualitative Comparison

**Method**: Side-by-side evaluation
1. Pick 5-10 test "users" (simulated personas with different preferences)
2. For each user:
   - Get Allegro's recommendations (via API or scraping)
   - Get your model's recommendations
   - Manually evaluate: relevance, diversity, novelty
3. Document differences

**Example Personas**:
- Budget-conscious shopper (filters by price)
- Brand-loyal (prefers specific brands)
- Impulse buyer (diverse, trending items)
- Niche enthusiast (specific category deep-dive)

---

## Phase 7: Allegro Adaptation (Option C - Transfer Learning)

### 7.1 Data Collection Strategy

**Option A: API-Based** (Preferred if available)
```python
# Use Allegro API to collect:
# 1. Product catalog (IDs, metadata, categories, prices)
# 2. Your own browsing history (if API exposes it)
# 3. Popular/trending products
# 4. Category structure
```

**Option B: Scraping** (If API insufficient)
```python
# Respectful scraping:
# 1. robots.txt compliance
# 2. Rate limiting (1 req/second max)
# 3. Use public search results only
# 4. Focus on product metadata, not user data
```

**Minimal data needed**:
- ~10K products with metadata (name, category, brand, price)
- ~100 "synthetic users" with interaction sequences
- Category taxonomy (for product_surface mapping)

### 7.2 Transfer Learning Approach

**Step 1: Domain Adaptation**
```python
# Keep: Transformer weights (learned from H&M data)
# Retrain: Embedding tables (different product/brand vocabulary)
# Fine-tune: Prediction heads (may need different action weights)
```

**Step 2: Synthetic User Generation**
```python
# Since you don't have real Allegro user data:
# 1. Create 100 synthetic users with diverse preferences
# 2. Simulate interactions based on:
#    - Category preferences (electronics, fashion, home)
#    - Price sensitivity (budget, mid-range, premium)
#    - Brand affinity (local Polish brands vs international)
# 3. Use these to "warm up" the model
```

**Step 3: Few-Shot Adaptation**
```python
# Use your own Allegro browsing history (even 20-50 interactions)
# Fine-tune model with strong regularization
# Goal: Adapt to Allegro's product distribution
```

**Training estimate**:
- **Time**: 2-4 hours (just fine-tuning embeddings)
- **Cost**: ~$5-10

### 7.3 Evaluation on Allegro

**Metrics**:
1. **Diversity**: How many unique categories in top-20?
2. **Novelty**: How many recommended items you haven't seen before?
3. **Relevance**: Manual rating (1-5 stars) for top-10
4. **Comparison**: Overlap with Allegro's recommendations

**Demo**: Build a simple web UI
```python
# Streamlit or Gradio app:
# - Input: Your Allegro user ID (or manual history entry)
# - Output: Side-by-side recommendations
#   - Left: Allegro's recommendations (from API)
#   - Right: Your model's recommendations
# - Allow manual rating/feedback
```

---

## Critical Files to Work With

### Phase 1-2 (Data Prep)
- **New file**: `data/prepare_retail_rocket.py` - Download & preprocess Retail Rocket data
- **New file**: `data/ecommerce_dataset.py` - PyTorch/TF Dataset class for e-commerce
- **Optional**: `data/prepare_hm_dataset.py` - If scaling up to H&M later

### Phase 3 (Model Adaptation)
- **Adapt**: `phoenix/recsys_retrieval_model.py` - Change input format, action count
- **Adapt**: `phoenix/recsys_model.py` - Change input format, num_actions=8
- **Keep**: `phoenix/grok.py` - No changes needed!
- **Adapt**: `phoenix/runners.py` - Update for e-commerce batch format

### Phase 4 (Training)
- **New file**: `phoenix/train_retrieval_ecommerce.py` - Retrieval training loop
- **New file**: `phoenix/train_ranking_ecommerce.py` - Ranking training loop
- **New file**: `phoenix/data_loader.py` - Data loading utilities

### Phase 5 (Inference)
- **Adapt**: `phoenix/run_retrieval.py` - E-commerce inference
- **Adapt**: `phoenix/run_ranker.py` - E-commerce inference
- **New file**: `phoenix/demo_ecommerce.py` - End-to-end demo

### Phase 6-7 (Allegro)
- **New file**: `allegro/collect_data.py` - API/scraping utilities
- **New file**: `allegro/compare_recommendations.py` - Comparison metrics
- **New file**: `allegro/demo_ui.py` - Streamlit side-by-side UI

---

## Verification & Testing

### Unit Tests
```bash
# Test data preprocessing
uv run pytest data/test_ecommerce_dataset.py

# Test model forward pass
uv run pytest phoenix/test_recsys_model.py

# Test attention masking
uv run pytest phoenix/test_recsys_model.py::TestMakeRecsysAttnMask
```

### Integration Tests
```bash
# Test retrieval pipeline
uv run python phoenix/run_retrieval.py --config config/retrieval_ecommerce.yaml

# Test ranking pipeline
uv run python phoenix/run_ranker.py --config config/ranking_ecommerce.yaml

# Test end-to-end
uv run python phoenix/demo_ecommerce.py --customer_id test_user_1
```

### Model Checkpointing
```python
# Save checkpoints every N steps
# Keep best model based on validation Recall@100
# Test on held-out test set at end
```

---

## Timeline Estimate (Solo Developer)

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Week 1** | 4-5 days | Data prep complete, models adapted |
| - Day 1 | | Download Retail Rocket dataset, explore data |
| - Day 2-3 | | Preprocess into training format |
| - Day 4 | | Adapt model code, verify shapes |
| **Week 2** | 4-5 days | Training complete |
| - Day 1 | | Train retrieval model (faster with Retail Rocket!) |
| - Day 2-4 | | Train ranking model (+ debugging) |
| **Week 3** | 5 days | Evaluation + Allegro exploration |
| - Day 1-2 | | Evaluate models, tune hyperparameters |
| - Day 3-4 | | Explore Allegro API, collect data |
| - Day 5 | | Build comparison framework |
| **Week 4** | 3-5 days | Allegro adaptation + demo |
| - Day 1-2 | | Transfer learning to Allegro |
| - Day 2-3 | | Build demo UI, comparison |
| - Day 4 | | Documentation, final evaluation |

**Total**: ~2-3 weeks for complete MVP (using Retail Rocket; add 1-2 weeks if using H&M)

---

## Success Criteria

### Minimum Viable Product (Must Have)
- ✅ Retrieval model trained on Retail Rocket (Recall@100 > 0.30)
- ✅ Ranking model trained on Retail Rocket (AUC > 0.65)
- ✅ End-to-end inference pipeline working
- ✅ Basic comparison with Allegro (manual evaluation)

### Stretch Goals (Nice to Have)
- 🎯 Automated comparison metrics with Allegro
- 🎯 Web UI for side-by-side comparison
- 🎯 Transfer learning to Allegro working well
- 🎯 Blog post documenting findings

---

## Risk Mitigation

### Risk: Training too expensive
**Mitigation**:
- Use smaller model (2 layers instead of 4)
- Use Colab Pro ($10/month with longer sessions)
- Train on subset of data first

### Risk: Dataset too large to handle
**Mitigation**:
- Start with Retail Rocket (smaller, more manageable)
- If using H&M: Sample 10% of data (still ~3M interactions)
- Focus on single category if needed (e.g., women's fashion for H&M)
- Use streaming data loader for larger datasets

### Risk: Can't get Allegro data
**Mitigation**:
- Focus on public product metadata only
- Use your own browsing history
- Compare conceptually rather than quantitatively

### Risk: Model doesn't converge
**Mitigation**:
- Start with smaller model and verify training
- Use pre-trained embeddings (Word2Vec on product titles)
- Simplify to single-task (purchase prediction only)

---

## Next Steps (After Plan Approval)

1. **Download Retail Rocket dataset** from Kaggle (or H&M if you prefer larger scale)
2. **Set up Python environment** (create `requirements.txt` with JAX, Haiku, pandas, etc.)
3. **Explore the data** (run basic statistics, understand event types and schema)
4. **Adapt one model** (start with retrieval - simpler than ranking)
5. **Train small version** (proof of concept with subset of data first)
6. **Iterate** based on results!

Let's build this! 🚀
