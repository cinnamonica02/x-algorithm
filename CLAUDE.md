# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains the X For You feed recommendation algorithm AND an in-progress e-commerce adaptation project.

### Original X Algorithm
The base code is X's open-sourced recommendation system: a hybrid Rust/Python system that retrieves, ranks, and filters content for users. It combines in-network content (from followed accounts) with out-of-network content (ML-discovered posts) and ranks everything using a Grok-based transformer model.

**Key Philosophy:** The system eliminates hand-engineered features and heuristics, relying on the Grok transformer to learn relevance directly from user engagement sequences.

### E-Commerce Adaptation (In Progress)
This repository also includes work to adapt X's architecture to e-commerce product recommendations. The goal is to:
1. Learn how production-scale recommendation systems work
2. Build an MVP using the Retail Rocket e-commerce dataset (included in `retail_rocket/`)
3. Adapt the transformer-based approach to product recommendation

**See `REC_SYS_OUTLINE.md` for the complete adaptation plan**, including data preprocessing, model changes, and implementation timeline.

## Quick Start: What to Focus On

**If exploring X's architecture**: Read the System Architecture and Critical Design Patterns sections. Run the Phoenix models with `uv run run_ranker.py` or `uv run run_retrieval.py` to see them in action.

**If working on e-commerce adaptation**:
1. Read `REC_SYS_OUTLINE.md` first (complete implementation plan)
2. Review the E-Commerce Adaptation Files section below
3. Follow the E-Commerce Development Workflow section
4. Focus on Python code in `phoenix/` - ignore Rust code for now

## Technology Stack

- **Rust**: Service layer, pipeline framework, and orchestration (reference code only, not buildable)
- **Python/JAX**: ML models for retrieval and ranking (Phoenix) - this is runnable
- **gRPC**: Inter-service communication (original X architecture)
- **Kafka**: Real-time post ingestion (Thunder, original X architecture)
- **uv**: Python dependency management

## E-Commerce Adaptation Files

**Dataset**: `retail_rocket/` contains the Retail Rocket e-commerce dataset:
- `events.csv`: 2.7M user-product interactions (view, addtocart, transaction)
- `item_properties_part1.csv` & `item_properties_part2.csv`: Product metadata
- `category_tree.csv`: Product category hierarchy

**Documentation**:
- `REC_SYS_OUTLINE.md`: Complete adaptation plan with timelines and implementation details
- `human_guide.md`: Technical analysis of X's recommendation system architecture

**Key Adaptations Needed** (see REC_SYS_OUTLINE.md for details):
- Map X concepts to e-commerce: tweet_id → product_id, author_id → brand_id
- Reduce action types from 19 to 3-8 (view, addtocart, transaction, etc.)
- Adapt `phoenix/recsys_model.py` and `phoenix/recsys_retrieval_model.py` for e-commerce batch format
- Create new data preprocessing pipeline for Retail Rocket dataset
- Train retrieval model (6-12 hours) then ranking model (12-24 hours)

## Development Commands

### Python (Phoenix ML Models)

```bash
# Run the ranking model
uv run run_ranker.py

# Run the retrieval model
uv run run_retrieval.py

# Run tests
uv run pytest test_recsys_model.py test_recsys_retrieval_model.py

# Lint code (configured in pyproject.toml)
uv run ruff check phoenix/
```

**Note:** Phoenix requires Python >=3.11. The code is representative example code showing the ML architecture, ported from the Grok-1 transformer implementation.

### Rust Services

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

If working on the e-commerce adaptation, follow these steps:

**Phase 1: Data Preprocessing** (2-3 days)
```bash
# Create new file: data/prepare_retail_rocket.py
# - Load retail_rocket/events.csv
# - Filter users with ≥10 interactions, products with ≥5 interactions
# - Create interaction sequences sorted by timestamp
# - Split: 80% train, 10% val, 10% test (temporal split)
# - Map action types: transaction=0, addtocart=1, view=2
# - Save preprocessed data
```

**Phase 2: Model Adaptation** (1-2 days)
```bash
# Adapt phoenix/recsys_retrieval_model.py:
# - Change num_actions from 19 to 3
# - Update batch structure for e-commerce (customer_id, product_id, brand_id)
# - Adjust sequence lengths (history_seq_len=64, candidate_seq_len=16)

# Adapt phoenix/recsys_model.py:
# - Same changes as retrieval model
# - Update prediction heads for 3 action types
# - Keep candidate isolation attention masking unchanged
```

**Phase 3: Training** (3-5 days)
```bash
# Create phoenix/train_retrieval_ecommerce.py
# Train retrieval model: ~6-12 hours on GPU
uv run python phoenix/train_retrieval_ecommerce.py

# Create phoenix/train_ranking_ecommerce.py
# Train ranking model: ~12-24 hours on GPU
uv run python phoenix/train_ranking_ecommerce.py
```

**Phase 4: Inference & Demo** (1-2 days)
```bash
# Adapt phoenix/run_retrieval.py for e-commerce
# Adapt phoenix/run_ranker.py for e-commerce
# Create phoenix/demo_ecommerce.py for end-to-end testing
uv run python phoenix/demo_ecommerce.py --customer_id 12345
```

**Important Notes**:
- Budget: ~$20-50 for GPU training (use Colab Pro or GCP preemptible)
- The Rust code is reference only - build a Python-only MVP first
- Refer to `REC_SYS_OUTLINE.md` for detailed implementation guidance

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
When working on the e-commerce adaptation:
- Start by reading `REC_SYS_OUTLINE.md` for the complete implementation plan
- The Retail Rocket dataset is in `retail_rocket/` - 2.7M events with 3 action types
- Focus on adapting the Python Phoenix models (`phoenix/*.py`) - these are runnable
- The Rust code is for reference/understanding only - the e-commerce MVP will be Python-only
- Create new files for e-commerce-specific code (e.g., `phoenix/train_ecommerce.py`, `data/prepare_retail_rocket.py`)
- Key files to adapt:
  - `phoenix/recsys_retrieval_model.py`: Two-tower retrieval for products
  - `phoenix/recsys_model.py`: Transformer ranking for products
  - `phoenix/grok.py`: Keep as-is (attention masking is domain-agnostic)
- Maintain the candidate isolation pattern in attention masking (this is crucial)
