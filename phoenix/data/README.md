# Data Preprocessing for E-Commerce Recommendation

This directory contains scripts to preprocess the Retail Rocket e-commerce dataset for training the recommendation models.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r data/requirements.txt
   ```

2. **Test data loading (optional but recommended):**
   ```bash
   python data/test_data_loading.py
   ```

3. **Run preprocessing:**
   ```bash
   python data/prepare_retail_rocket.py
   ```

## What the Preprocessing Does

The `prepare_retail_rocket.py` script implements **Phase 2** of `REC_SYS_OUTLINE.md`:

### Data Transformation (X → E-Commerce Mapping)

| X Concept | E-Commerce Equivalent | Implementation |
|-----------|----------------------|----------------|
| `user_id` | `customer_id` (visitorid) | Direct mapping |
| `tweet_id` | `product_id` (itemid) | Direct mapping |
| `author_id` | `brand_id` | Uses categoryid as proxy |
| `user_action_sequence` | `interaction_sequence` | List of (product_id, action_type, timestamp, category_id, brand_id) |
| `history_actions` (19 types) | **3 action types**: transaction (0), addtocart (1), view (2) | Integer encoding |
| `product_surface` | `product_category` | Extracted from item_properties where property='categoryid' |
| Post embeddings | Product embeddings | Will use hash-based embeddings |
| Author embeddings | Brand embeddings | Will use hash-based embeddings |

### Processing Steps

1. **Load Raw Data**
   - events.csv: 2.7M user-product interactions
   - category_tree.csv: Category hierarchy
   - item_properties_part1.csv & part2.csv: Product metadata

2. **Extract Product Metadata**
   - Extract categoryid from item_properties (property-value format)
   - Use category as brand proxy (no explicit brand in dataset)
   - Assign category_id=0 for products without metadata

3. **Filter by Interaction Counts**
   - Keep users with ≥10 interactions
   - Keep products with ≥5 interactions
   - Iterative filtering until stable

4. **Create Interaction Sequences**
   - Sort by timestamp for each user
   - Format: [{product_id, action_type, timestamp, category_id, brand_id}, ...]
   - Map action types: transaction=0, addtocart=1, view=2

5. **Temporal Train/Val/Test Split**
   - Train: First 80% of each user's interactions
   - Validation: Next 10%
   - Test: Last 10%
   - This simulates predicting future interactions

6. **Create Vocabularies**
   - user_to_idx, product_to_idx, category_to_idx, brand_to_idx
   - Enables consistent indexing across splits

## Output Files

After preprocessing, the following files are created in `data/processed/`:

- `train_sequences.pkl`: Training interaction sequences
- `val_sequences.pkl`: Validation interaction sequences
- `test_sequences.pkl`: Test interaction sequences
- `vocabularies.pkl`: All vocabulary mappings
- `statistics.json`: Dataset statistics (human-readable)

## Dataset Statistics (Expected)

Based on the Retail Rocket dataset:

- **Raw data:**
  - 1.4M unique visitors
  - 235K unique products
  - 2.7M interactions (view: 2.66M, addtocart: 69K, transaction: 22K)
  - 1,242 unique categories

- **After filtering (estimated):**
  - ~100K-200K users with ≥10 interactions
  - ~50K-100K products with ≥5 interactions
  - ~1.5M-2M interactions remaining

## Next Steps

After preprocessing is complete, proceed to **Phase 3** (Model Adaptation):

1. Adapt `phoenix/recsys_retrieval_model.py` for e-commerce
2. Adapt `phoenix/recsys_model.py` for e-commerce
3. Create data loaders for training
4. Train retrieval model (6-12 hours)
5. Train ranking model (12-24 hours)

See `REC_SYS_OUTLINE.md` for detailed implementation plan.

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'pandas'`
- **Solution:** Install dependencies with `pip install -r data/requirements.txt`

**Issue:** `FileNotFoundError: [Errno 2] No such file or directory: '.../retail_rocket/events.csv'`
- **Solution:** Ensure the Retail Rocket dataset is in the `retail_rocket/` directory

**Issue:** Out of memory during preprocessing
- **Solution:** The script processes all data in memory. If you encounter OOM errors:
  - Close other applications
  - Use a machine with more RAM (8GB recommended)
  - Or modify the script to use chunked processing with `pd.read_csv(..., chunksize=10000)`

**Issue:** Preprocessing takes too long
- **Solution:** The full preprocessing should take 2-5 minutes on a typical laptop. If it's taking much longer:
  - Check disk speed (SSD recommended)
  - Reduce `MIN_USER_INTERACTIONS` or `MIN_PRODUCT_INTERACTIONS` for faster iteration
