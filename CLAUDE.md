# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains the X For You feed recommendation algorithm AND an in-progress e-commerce adaptation project.

### Original X Algorithm
The base code is X's open-sourced recommendation system: a hybrid Rust/Python system that retrieves, ranks, and filters content for users. It combines in-network content (from followed accounts) with out-of-network content (ML-discovered posts) and ranks everything using a Grok-based transformer model.

**Key Philosophy:** The system eliminates hand-engineered features and heuristics, relying on the Grok transformer to learn relevance directly from user engagement sequences.

### E-Commerce Adaptation (Phases 2-3 Complete - Ready to Train)
This repository also includes work to adapt X's architecture to e-commerce product recommendations. The goal is to:
1. Learn how production-scale recommendation systems work
2. Build an MVP using the Retail Rocket e-commerce dataset (included in `retail_rocket/`)
3. Adapt the transformer-based approach to product recommendation

**Current Status (Updated 2026-02-17):**
- ✅ **Phase 1**: Architecture understanding complete
- ✅ **Phase 2**: Data preprocessing complete (551K interactions, 23K users, 54K products)
- ✅ **Phase 3**: Model adaptation complete (474K parameter ranking model)
- 🚀 **Current**: Ready to train on GPU (see `RUNPOD_SETUP.md` for GPU training)

**Key Files:**
- `PHASE2_COMPLETE.md`: Data preprocessing documentation
- `TRAINING_READY.md`: Model architecture and training guide
- `RUNPOD_SETUP.md`: GPU setup instructions (includes JAX GPU fix)
- `phoenix/INSTALL.md`: Installation guide with troubleshooting

## Quick Start: What to Focus On

**If exploring X's architecture**: Read the System Architecture and Critical Design Patterns sections. Run the Phoenix models with `uv run run_ranker.py` or `uv run run_retrieval.py` to see them in action.

**If working on e-commerce adaptation**:
1. Read `TRAINING_READY.md` for current status and training guide
2. Read `RUNPOD_SETUP.md` for GPU setup (includes JAX GPU fix)
3. Review the E-Commerce Implementation Files section below
4. Focus on Python code in `phoenix/` - ignore Rust code for now

### ⚠️ CRITICAL: GPU Setup Issue

**Problem:** `pyproject.toml` cannot install GPU-enabled JAX because it requires a custom repository. The default `uv sync` installs CPU-only JAX, making training 100x slower.

**Why this happens:**
- `jax[cuda12]` from PyPI = **CPU-only** (despite the name!)
- `jax[cuda12_pip]` from Google's repository = **actual GPU support**
- pip/uv cannot specify custom repositories in pyproject.toml

**Solution:**
```bash
cd phoenix
bash setup_gpu.sh  # Installs JAX from Google's GPU repository
```

The training script will check for GPU and refuse to run on CPU. See Installation section below for details.

## Quick Reference

### Documentation Files
- `CLAUDE.md` (this file): Complete repository guide
- `TRAINING_READY.md`: Model architecture, training guide, and current status
- `RUNPOD_SETUP.md`: GPU training setup (CRITICAL: includes JAX GPU fix)
- `PHASE2_COMPLETE.md`: Data preprocessing documentation
- `phoenix/INSTALL.md`: Installation guide with troubleshooting
- `phoenix/data/README.md`: Data pipeline documentation
- `REC_SYS_OUTLINE.md`: Original adaptation plan (historical reference)
- `human_guide.md`: Technical analysis of X's recommendation system

### Dataset Info
**Location:** `phoenix/retail_rocket/` (raw), `phoenix/data/processed/` (preprocessed)
- Raw: 2.7M interactions across 3 action types
- Processed: 551K interactions, 23K users, 54K products, 924 categories
- Actions: transaction (0), addtocart (1), view (2)

### E-Commerce Files (Implemented)
**Data Pipeline:**
- `phoenix/data/prepare_retail_rocket.py`: Preprocessing (complete)
- `phoenix/data/ecommerce_dataset.py`: JAX data loader (complete)

**Models:**
- `phoenix/ecommerce_config.py`: Configurations and data structures
- `phoenix/ecommerce_ranking_model.py`: Ranking model (474K params)
- `phoenix/train_ecommerce_ranking.py`: Training script

**Tests:**
- `phoenix/test_ecommerce_model.py`: Model tests
- `phoenix/data/test_data_loading.py`: Data pipeline tests

## Technology Stack

**Runnable Components (Python/JAX):**
- **JAX**: Numerical computing and automatic differentiation
- **Haiku**: Neural network library (transformer layers)
- **Optax**: Gradient processing and optimization
- **Python**: >=3.11 required

**Reference Components (Rust - not buildable):**
- **Rust**: Service layer and pipeline framework (reference architecture only)
- **gRPC**: Inter-service communication (original X architecture)
- **Kafka**: Real-time ingestion (Thunder service, original X architecture)

**Development:**
- Dependencies managed via `pyproject.toml`
- `uv` can be used for local development but **NOT for GPU training**
- See Installation section below for proper GPU setup

## Installation

### GPU Training (RunPod/Cloud) - CRITICAL

**⚠️ IMPORTANT:** Do NOT use `uv sync` on GPU systems - it installs CPU-only JAX!

**Correct approach:**
```bash
cd phoenix

# Automated setup (installs everything - JAX + other deps)
bash setup_gpu.sh

# DO NOT run "uv sync" after this!
# It will overwrite GPU JAX with CPU-only JAX

# Verify GPU detection
python -c "import jax; print('Backend:', jax.default_backend()); print('Devices:', jax.devices())"
# Should show: Backend: gpu, Devices: [cuda(id=0)]
```

**Manual installation:**
```bash
# For CUDA 11.x (most RunPod pods)
pip install --upgrade pip
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install dm-haiku optax tqdm pandas numpy

# For CUDA 12.x
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install dm-haiku optax tqdm pandas numpy
```

**Why this matters:**
- GPU-enabled JAX is NOT on standard PyPI - it requires Google's special repository
- `pyproject.toml` cannot specify custom repositories, so `uv sync` installs CPU-only JAX
- The training script checks for GPU and will refuse to run on CPU (100x slower)
- The `-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html` flag is REQUIRED

**Design decision:** `pyproject.toml` now has CPU-only JAX as the default dependency. GPU users must use `setup_gpu.sh` or manual installation. This prevents confusion and ensures local development works with `uv sync`.

See `RUNPOD_SETUP.md` for complete GPU setup guide and `phoenix/INSTALL.md` for troubleshooting.

### Local Development (CPU or Local GPU)

**For CPU-only testing:**
```bash
cd phoenix
pip install uv
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync  # Safe - installs CPU-only JAX for local testing
```

**For local GPU:**
```bash
cd phoenix

# Option A: Use the setup script (recommended, installs everything)
bash setup_gpu.sh
# Done! Do NOT run "uv sync" after this.

# Option B: Use uv, then fix JAX
pip install uv
uv venv
source .venv/bin/activate
uv sync  # Installs CPU-only JAX + other deps
pip uninstall jax jaxlib -y  # Remove CPU-only JAX
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## E-Commerce Implementation Files

### Dataset
`phoenix/retail_rocket/`: Retail Rocket e-commerce dataset (raw)
- `events.csv`: 2.7M user-product interactions (view, addtocart, transaction)
- `item_properties_part1.csv` & `item_properties_part2.csv`: Product metadata
- `category_tree.csv`: Product category hierarchy

`phoenix/data/processed/`: Preprocessed data (ready for training)
- After preprocessing: 551K interactions, 23K users, 54K products, 924 categories
- Actions mapped: transaction=0, addtocart=1, view=2

### Implemented Code

**Data Pipeline:**
- `phoenix/data/prepare_retail_rocket.py`: Preprocessing pipeline ✅
- `phoenix/data/ecommerce_dataset.py`: JAX data loader ✅
- `phoenix/data/test_data_loading.py`: Data tests ✅

**Models:**
- `phoenix/ecommerce_config.py`: Configurations and data structures ✅
- `phoenix/ecommerce_ranking_model.py`: Ranking model (474K params) ✅
- `phoenix/grok.py`: Core transformer (unchanged from X, domain-agnostic) ✅

**Training:**
- `phoenix/train_ecommerce_ranking.py`: Training script ✅
- `phoenix/test_ecommerce_model.py`: Model tests ✅
- `phoenix/setup_gpu.sh`: GPU setup automation ✅

**Documentation:**
- `TRAINING_READY.md`: Model architecture and training guide
- `RUNPOD_SETUP.md`: GPU training setup (includes JAX fix)
- `PHASE2_COMPLETE.md`: Data preprocessing documentation
- `phoenix/INSTALL.md`: Installation and troubleshooting

### Key Design Decisions
1. **Category as Brand Proxy**: Using categoryid as brand_id (dataset has no explicit brand field)
2. **3 Action Types**: transaction, addtocart, view (reduced from X's 19 types)
3. **Candidate Isolation**: Maintained from X architecture - prevents cross-candidate attention
4. **Hash-Based Embeddings**: 2 hash functions per entity (scalable, no embedding matrices)
5. **Temporal Splitting**: Per-user temporal split (80/10/10) simulates real-world prediction

### Next Steps (TODO)
- Train ranking model on GPU (25-50 minutes on RTX 4090)
- Create inference script for product recommendations
- Evaluate on test set with ranking metrics (hitrate@K, NDCG@K)
- Optionally: Adapt and train retrieval model

## Development Commands

### E-Commerce Training (Current Work)

**GPU Training (RunPod/Cloud):**
```bash
cd phoenix

# Setup (CRITICAL - do this first!)
bash setup_gpu.sh

# Verify GPU
python -c "import jax; print('Backend:', jax.default_backend())"
# Must show 'gpu' or 'cuda', NOT 'cpu'

# Quick test (5 minutes)
python train_ecommerce_ranking.py --epochs 2 --steps_per_epoch 100

# Default training (25 minutes)
python train_ecommerce_ranking.py --epochs 10

# Full training (50 minutes)
python train_ecommerce_ranking.py --epochs 20 --batch_size 256 --emb_size 256 --num_layers 6
```

**Local Development/Testing:**
```bash
cd phoenix

# Test model architecture
python test_ecommerce_model.py

# Test data loading
python data/test_data_loading.py

# Run preprocessing (if needed)
python data/prepare_retail_rocket.py
```

### Original X Models (Reference/Demonstration)

```bash
# These demonstrate the original X architecture
# They are NOT used for e-commerce training

cd phoenix

# Run the original ranking model demo
uv run run_ranker.py

# Run the original retrieval model demo
uv run run_retrieval.py

# Run original model tests
uv run pytest test_recsys_model.py test_recsys_retrieval_model.py

# Lint code
uv run ruff check phoenix/
```

**Note:** Phoenix requires Python >=3.11. The original X code is representative example code showing the ML architecture, ported from the Grok-1 transformer implementation.

### Rust Services (Reference Only)

This repository contains reference Rust code for three components:
- **home-mixer**: gRPC server that orchestrates the For You feed pipeline
- **thunder**: In-memory post store with real-time Kafka ingestion
- **candidate-pipeline**: Reusable framework library for building recommendation pipelines

The Rust code demonstrates the production architecture but is not set up as a buildable project in this open-source release.

## System Architecture

### High-Level Data Flow

```
Request → HomeMixer → [Query Hydration]
                    ↓
          [Thunder (in-network) + Phoenix Retrieval (out-of-network)]
                    ↓
          [Candidate Hydration]
                    ↓
          [Pre-Scoring Filters]
                    ↓
          [Phoenix Scoring → Weighted Scoring → Diversity]
                    ↓
          [Selection (top K)]
                    ↓
          [Post-Selection Filters]
                    ↓
          Response (Ranked Posts)
```

### Component Responsibilities

#### home-mixer/
The orchestration layer that assembles feeds using the CandidatePipeline framework.

**Key files:**
- `server.rs`: gRPC service implementation (`ScoredPostsService`)
- `candidate_pipeline/phoenix_candidate_pipeline.rs`: Main pipeline configuration
- `sources/`: Thunder (in-network) and Phoenix (out-of-network) candidate sources
- `query_hydrators/`: Fetch user context (engagement history, features)
- `candidate_hydrators/`: Enrich candidates with metadata (author info, media, etc.)
- `filters/`: Pre/post-scoring filters (dedup, age, muted keywords, visibility, etc.)
- `scorers/`: Phoenix ML scorer, weighted scorer, diversity scorer, OON adjustments
- `selectors/`: Sort by score and select top K
- `side_effects/`: Async operations like caching

#### thunder/
In-memory post store with real-time Kafka ingestion for recent posts from all users.

**Key capabilities:**
- Consumes post create/delete events from Kafka
- Maintains per-user stores (original posts, replies/reposts, videos)
- Serves in-network candidates (posts from followed accounts)
- Auto-trims posts older than retention period
- Enables sub-millisecond lookups without external database

**Key files:**
- `thunder_service.rs`: gRPC service for fetching in-network posts
- `posts/post_store.rs`: In-memory storage with automatic trimming
- `kafka_utils.rs`: Kafka consumer setup

#### phoenix/
JAX/Python ML component for retrieval and ranking using Grok-based transformers.

**Two main functions:**

1. **Retrieval (Two-Tower Model)**: Finds relevant out-of-network posts
   - User Tower: Encodes user features/history → user embedding
   - Candidate Tower: Encodes posts → candidate embeddings
   - Similarity Search: Retrieves top-K via dot product

2. **Ranking (Transformer with Candidate Isolation)**: Predicts engagement probabilities
   - Takes user context + candidate posts as input
   - Special attention masking prevents candidates from attending to each other
   - Outputs P(like), P(reply), P(repost), P(click), etc.

**Key files:**
- `grok.py`: Core Grok-1 transformer architecture with custom attention masking
- `recsys_model.py`: Ranking model with candidate isolation
- `recsys_retrieval_model.py`: Two-tower retrieval model
- `run_ranker.py`: Ranking model inference example
- `run_retrieval.py`: Retrieval model inference example

#### candidate-pipeline/
Reusable framework for building recommendation pipelines with trait-based composition.

**Core traits:**
- `Source`: Fetch candidates from data sources
- `QueryHydrator`: Enrich query with user context
- `Hydrator`: Enrich candidates with additional features
- `Filter`: Remove ineligible candidates
- `Scorer`: Compute relevance scores
- `Selector`: Sort and select top candidates
- `SideEffect`: Run async operations (logging, caching)

**Key features:**
- Parallel execution of independent stages
- Configurable error handling
- Separation of business logic from pipeline execution
- Stage-level logging and monitoring

## E-Commerce Development Workflow

### Current Status: Ready to Train

Phases 1-3 are complete. Current workflow:

**Step 1: GPU Setup (CRITICAL)**

Before training, you MUST properly install JAX with GPU support. Do NOT use `uv sync`!

```bash
cd phoenix
bash setup_gpu.sh

# Verify GPU detection
python -c "import jax; print('Backend:', jax.default_backend())"
# Must show 'gpu' or 'cuda'
```

See `RUNPOD_SETUP.md` for complete GPU setup guide.

**Step 2: Train Ranking Model**

```bash
# Quick test (5 min, $0.06 on RunPod RTX 4090)
python train_ecommerce_ranking.py --epochs 2 --steps_per_epoch 100

# Default training (25 min, $0.29)
python train_ecommerce_ranking.py --epochs 10

# Full training (50 min, $0.58)
python train_ecommerce_ranking.py --epochs 20 --batch_size 256 --emb_size 256
```

Expected training output:
- Each epoch: ~1-2 minutes on RTX 4090
- GPU utilization: 80-100%
- If epoch takes >10 minutes: GPU is not being used, check installation

**Step 3: Download Checkpoints**

```bash
# From local machine
scp -P <pod-port> -r root@<pod-ip>:/workspace/x-algorithm/phoenix/checkpoints/ ./
```

**Step 4: Inference & Evaluation (Next Phase - TODO)**

Create inference script to:
- Load trained model from checkpoint
- Generate product recommendations for test users
- Evaluate with ranking metrics (hitrate@K, NDCG@K)
- Build simple demo

**Step 5: Retrieval Model (Optional Future Work)**

Adapt the two-tower retrieval model following the same pattern as the ranking model.

### What's Already Implemented

✅ **Data Pipeline** (Phase 2):
- Preprocessing: filters, sequences, temporal splits
- Data loader: JAX batching with hash lookups
- Output: 551K interactions ready for training

✅ **Ranking Model** (Phase 3):
- Architecture: Grok-based transformer (474K params)
- Candidate isolation: No cross-candidate attention
- Training loop: Weighted BCE loss, AdamW optimizer
- Checkpointing: Every 2 epochs

✅ **GPU Setup** (Phase 3 fix):
- Automated setup script (`setup_gpu.sh`)
- JAX GPU installation fix documented
- Training script checks for GPU and refuses CPU

### Important Notes

- **Budget**: ~$0.29-0.58 for training on RunPod RTX 4090
- **Time**: 25-50 minutes for full training
- **Critical**: Use `setup_gpu.sh` for installation, NOT `uv sync`
- **Rust code**: Reference only - e-commerce MVP is Python-only

## Critical Design Patterns

### 1. CandidatePipeline Framework
All recommendation pipelines implement the `CandidatePipeline<Q, C>` trait with stages executed in order:
```
QueryHydrator → Sources → Hydrator → Filter → Scorer → Selector → PostSelectionFilter
```

**When adding new pipeline components:**
- Sources run in parallel and are merged
- Hydrators run in parallel, but each operates on all candidates
- Filters run sequentially and can short-circuit
- Scorers run sequentially, each can modify scores
- Follow existing patterns in `home-mixer/sources/`, `filters/`, `scorers/`

### 2. Candidate Isolation (Phoenix Ranking)
The ranking transformer uses special attention masking where candidates can attend to user context and history but **not to each other**. This ensures:
- Scores are independent of batch composition
- Scores are cacheable and consistent
- No leakage between candidates during inference

**Implementation:** See `grok.py:make_recsys_attn_mask()` which creates the attention mask with diagonal-only attention for candidate positions.

### 3. Multi-Action Prediction
Instead of a single relevance score, Phoenix predicts probabilities for multiple actions (like, reply, repost, click, block, mute, report). The `WeightedScorer` combines these with configured weights (positive for desired actions, negative for undesired).

**When modifying scoring:**
- Phoenix outputs are in `candidate.phoenix_scores`
- Weights are configured in `scorers/weighted_scorer.rs`
- New scorers can be added to the pipeline in `phoenix_candidate_pipeline.rs`

### 4. Hash-Based Embeddings
Both retrieval and ranking use multiple hash functions for embedding lookups rather than learned embeddings. This is a key architectural choice for scalability.

### 5. Query and Candidate Separation
- `Query` objects (e.g., `ScoredPostsQuery`) contain user context and request metadata
- `Candidate` objects (e.g., `PostCandidate`) represent posts with features and scores
- Pipeline stages operate on `(Query, Vec<Candidate>)` pairs
- Query is hydrated once; candidates are hydrated, filtered, and scored in stages

## Important Conventions

### Filtering Stages
- **Pre-Scoring Filters**: Run before ML inference to reduce compute (dedup, age, blocked authors, muted keywords, previously seen)
- **Post-Selection Filters**: Run after selection for final validation (visibility filtering, conversation dedup)

### Scoring Flow
Scorers run sequentially and each can read/modify candidate scores:
1. `PhoenixScorer`: Calls Phoenix gRPC service for ML predictions
2. `WeightedScorer`: Combines Phoenix predictions into final relevance score
3. `AuthorDiversityScorer`: Attenuates repeated authors for feed diversity
4. `OONScorer`: Adjusts scores for out-of-network content

### Thunder Post Types
Thunder maintains separate stores:
- Original posts
- Replies/reposts
- Video posts (special handling for duration)

The `ThunderSource` fetches from all relevant stores based on user's following list.

## Phoenix-Rust Integration

The home-mixer Rust service calls the Phoenix Python models via gRPC:
- `scorers/phoenix_scorer.rs` makes prediction requests
- Phoenix service is defined in proto (not included in this open source release)
- Request format: user_id + user_action_sequence + list of candidate tweet_infos
- Response: per-candidate predictions for each action type

## Code Organization Notes

- Each pipeline component is its own file/module
- Filters, scorers, hydrators follow consistent trait patterns
- Error handling: Most stages return `Result<Vec<Candidate>, String>`
- Logging: Use `log::info!` for request tracking, `log::warn!` for recoverable errors
- Stats: Many functions use `#[xai_stats_macro::receive_stats]` for metrics
- Proto definitions referenced but not included (e.g., `xai_home_mixer_proto`, `xai_recsys_proto`)

## Working with This Codebase

### For Understanding X's Architecture
Since the X code is reference code from an open-source release:
- Focus on understanding the architecture and patterns rather than building/running Rust code
- The Rust code shows production patterns but isn't a complete buildable project
- Phoenix Python code is runnable for demonstrating the ML architecture
- When making changes, maintain the separation of concerns between pipeline stages
- Follow the existing trait-based patterns when adding new components

### For E-Commerce Adaptation Work

**Current state:** Data preprocessing and model adaptation are complete. Focus on training and inference.

**Key resources:**
- `TRAINING_READY.md`: Model architecture, training guide, current status
- `RUNPOD_SETUP.md`: GPU training setup (includes JAX GPU fix)
- `PHASE2_COMPLETE.md`: Data preprocessing documentation

**What's implemented:**
- ✅ Data pipeline: `phoenix/data/prepare_retail_rocket.py`, `ecommerce_dataset.py`
- ✅ Ranking model: `phoenix/ecommerce_ranking_model.py` (474K params)
- ✅ Training script: `phoenix/train_ecommerce_ranking.py`
- ✅ GPU setup: `phoenix/setup_gpu.sh` (fixes JAX GPU installation)

**Next work:**
1. Train model on GPU (use `setup_gpu.sh` for installation)
2. Create inference script for product recommendations
3. Evaluate on test set (hitrate@K, NDCG@K metrics)
4. Optionally: Adapt retrieval model (`phoenix/recsys_retrieval_model.py`)

**Critical notes:**
- **Never use `uv sync` on GPU systems** - it installs CPU-only JAX
- Always use `setup_gpu.sh` or manual JAX installation from Google's releases
- The training script checks for GPU and refuses to run on CPU (intentional)
- Maintain candidate isolation in attention masking if modifying models
- The Rust code is reference only - e-commerce MVP is Python-only
