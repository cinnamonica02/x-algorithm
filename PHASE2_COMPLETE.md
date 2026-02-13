# Phase 2 Complete: Data Preprocessing Pipeline

## Status: ✅ Ready to Run

Phase 2 of the e-commerce adaptation (Data Preprocessing & Feature Engineering) is now complete and ready to execute.

## What Was Built

### 1. Main Preprocessing Script
**File:** `data/prepare_retail_rocket.py`

A complete pipeline that implements all steps from `REC_SYS_OUTLINE.md` Phase 2:
- ✅ Loads raw Retail Rocket data (events, categories, item properties)
- ✅ Extracts product metadata (category as brand proxy)
- ✅ Filters users (≥10 interactions) and products (≥5 interactions)
- ✅ Creates interaction sequences with full metadata
- ✅ Maps action types: transaction=0, addtocart=1, view=2
- ✅ Performs temporal train/val/test splits (80/10/10)
- ✅ Creates vocabulary mappings for all entities
- ✅ Computes and saves dataset statistics

### 2. Supporting Files

**`data/test_data_loading.py`**
- Quick sanity check to verify data files load correctly
- Run before full preprocessing to catch issues early

**`data/requirements.txt`**
- Dependencies: pandas, numpy

**`data/README.md`**
- Complete documentation
- Usage instructions
- Troubleshooting guide
- Next steps

## Data Mapping (X → E-Commerce)

| X Concept | Retail Rocket Mapping | Status |
|-----------|----------------------|--------|
| user_id | visitorid | ✅ Direct |
| tweet_id | itemid (product_id) | ✅ Direct |
| author_id | categoryid (as brand proxy) | ✅ Implemented |
| action types (19) | 3 types: transaction/addtocart/view | ✅ Mapped |
| user_action_sequence | interaction_sequence with metadata | ✅ Complete |
| product_surface | categoryid from item_properties | ✅ Extracted |

## How to Run

### Step 1: Install Dependencies
```bash
pip install -r data/requirements.txt
```

### Step 2: Test Data Loading (Optional)
```bash
python data/test_data_loading.py
```

Expected output:
```
Testing data loading...
======================================================================

1. Loading events.csv...
   ✓ Loaded 2,756,101 events
   Columns: ['timestamp', 'visitorid', 'event', 'itemid', 'transactionid']
   Event types: {'view': 2664312, 'addtocart': 69332, 'transaction': 22457}
   ...
```

### Step 3: Run Full Preprocessing
```bash
python data/prepare_retail_rocket.py
```

Expected runtime: 2-5 minutes

Expected output:
```
======================================================================
Retail Rocket Data Preprocessing Pipeline
======================================================================
Loading raw data...
Loaded 2,756,101 events
...

Preprocessing complete!
======================================================================

Dataset Summary:
  Users:        [after filtering]
  Products:     [after filtering]
  Categories:   1,242
  Interactions: [after filtering]
  Avg sequence length: [calculated]
```

## Output Files

After successful preprocessing, you'll find in `data/processed/`:

```
data/processed/
├── train_sequences.pkl      # Training data
├── val_sequences.pkl         # Validation data
├── test_sequences.pkl        # Test data
├── vocabularies.pkl          # All mappings (user/product/category/brand/action)
└── statistics.json           # Dataset stats (human-readable)
```

## Sequence Format

Each user's sequence is a list of interactions:
```python
{
    'product_id': int,      # Product ID
    'action_type': int,     # 0=transaction, 1=addtocart, 2=view
    'timestamp': int,       # Unix timestamp (milliseconds)
    'category_id': int,     # Category ID (0 if unknown)
    'brand_id': int,        # Brand ID (same as category_id)
    'event_name': str       # Human-readable: 'transaction'/'addtocart'/'view'
}
```

## What's Next: Phase 3 (Model Adaptation)

After preprocessing completes, proceed to Phase 3:

### 3.1 Create E-Commerce Batch Format

**New file to create:** `phoenix/ecommerce_types.py`
```python
@dataclass
class EcommerceRetrievalBatch:
    customer_hashes: jnp.ndarray  # [B, 2]
    history_product_hashes: jnp.ndarray  # [B, seq_len, 2]
    history_brand_hashes: jnp.ndarray  # [B, seq_len, 2]
    history_actions: jnp.ndarray  # [B, seq_len, 3] multi-hot
    history_category: jnp.ndarray  # [B, seq_len]
    candidate_product_hashes: jnp.ndarray  # [N, 2] or [B, C, 2]
    candidate_brand_hashes: jnp.ndarray  # [N, 2] or [B, C, 2]
    candidate_category: jnp.ndarray  # [N] or [B, C]
```

### 3.2 Adapt Models

**Files to modify:**
1. `phoenix/recsys_retrieval_model.py` - Change num_actions from 19 to 3
2. `phoenix/recsys_model.py` - Change num_actions from 19 to 3

**Files to create:**
1. `phoenix/data_loader.py` - Load processed sequences into JAX batches
2. `phoenix/train_retrieval_ecommerce.py` - Training loop for retrieval
3. `phoenix/train_ranking_ecommerce.py` - Training loop for ranking

### 3.3 Timeline

- **Data loader:** 1 day
- **Model adaptation:** 1 day
- **Training retrieval:** 6-12 hours (GPU)
- **Training ranking:** 12-24 hours (GPU)

**Total:** 3-5 days + ~$20-50 GPU cost

## Design Decisions Made

1. **Category as Brand Proxy**
   - Retail Rocket has no explicit brand field
   - Using categoryid as brand_id (1,242 unique values)
   - Alternative: Extract from property '888' or other fields (future enhancement)

2. **Unknown Category Handling**
   - Products without category metadata assigned category_id=0, brand_id=0
   - Allows model to learn "unknown" category embedding
   - ~10-20% of products expected to have no category (estimate)

3. **Temporal Split Strategy**
   - Per-user temporal split (not global)
   - Ensures train/val/test maintain temporal order
   - Simulates real-world scenario: predict future from past

4. **Minimum Interaction Thresholds**
   - Users: ≥10 interactions (need sufficient history)
   - Products: ≥5 interactions (cold start handled separately)
   - Iterative filtering handles cascading effects

5. **Action Type Weighting (for future training)**
   - transaction (purchase) = strongest signal = weight 10.0
   - addtocart = medium signal = weight 3.0
   - view = weakest signal = weight 1.0
   - Based on conversion funnel importance

## Validation Checklist

Before proceeding to Phase 3, verify:

- [ ] Preprocessing runs without errors
- [ ] Output files exist in `data/processed/`
- [ ] `statistics.json` shows reasonable numbers:
  - [ ] Users > 50,000
  - [ ] Products > 20,000
  - [ ] Categories = 1,242 (or 1,243 including unknown=0)
  - [ ] Avg sequence length between 10-50
  - [ ] Train/val/test splits ~80/10/10%
- [ ] Action distribution matches raw data:
  - [ ] view >> addtocart >> transaction
  - [ ] transaction ~0.8%, addtocart ~2.5%, view ~96.7%

## Questions or Issues?

- Review `data/README.md` for troubleshooting
- Check `REC_SYS_OUTLINE.md` for the full adaptation plan
- Verify input data in `retail_rocket/` is complete

---

**Ready to proceed?** Run the preprocessing and verify the output, then move to Phase 3!
