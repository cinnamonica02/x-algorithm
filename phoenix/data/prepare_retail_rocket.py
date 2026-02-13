import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pickle
import json


MIN_USER_INTERACTIONS = 10
MIN_PRODUCT_INTERACTIONS = 5
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


ACTION_MAPPING = {
    'transaction': 0,  # Strongest signal (purchase)
    'addtocart': 1,    # Medium signal
    'view': 2,         # Weakest signal
}




BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'retail_rocket'
OUTPUT_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR.mkdir(exist_ok=True)




def load_raw_data():
    print("Loading raw data...")


    events = pd.read_csv(DATA_DIR / 'events.csv')
    print(f"Loaded {len(events):,} events")
    print(f"Event types: {events['event'].value_counts().to_dict()}")


    category_tree = pd.read_csv(DATA_DIR / 'category_tree.csv')
    print(f"Loaded {len(category_tree):,} categories")


    print("Loading item properties (this may take a minute)...")
    item_props_1 = pd.read_csv(DATA_DIR / 'item_properties_part1.csv')
    item_props_2 = pd.read_csv(DATA_DIR / 'item_properties_part2.csv')
    item_properties = pd.concat([item_props_1, item_props_2], ignore_index=True)
    print(f"Loaded {len(item_properties):,} item properties")

    return events, category_tree, item_properties






def extract_product_metadata(item_properties):
    print("\nExtracting product metadata...")

    category_props = item_properties[item_properties['property'] == 'categoryid'].copy()

    # Convert value to integer (categoryid values are numeric strings)
    # Handle potential errors by dropping invalid entries
    category_props['value'] = pd.to_numeric(category_props['value'], errors='coerce')
    category_props = category_props.dropna(subset=['value'])
    category_props['value'] = category_props['value'].astype(int)

    # Create mapping: product_id -> category_id
    # If a product has multiple categoryid entries, take the most recent one
    category_props = category_props.sort_values('timestamp').drop_duplicates(
        subset=['itemid'], keep='last'
    )
    product_to_category = dict(zip(category_props['itemid'], category_props['value']))

    # For brand, we'll use categoryid as a proxy (as per REC_SYS_OUTLINE.md)
    product_to_brand = product_to_category.copy()  # Using category as brand proxy

    print(f"Extracted categories for {len(product_to_category):,} products")
    print(f"Unique categories: {len(set(product_to_category.values())):,}")

    return product_to_category, product_to_brand







def filter_by_interaction_counts(events, min_user_interactions, min_product_interactions):
    print(f"\nFiltering users (≥{min_user_interactions} interactions) and products (≥{min_product_interactions} interactions)...")

    prev_len = len(events)
    iteration = 0

    while True:
        iteration += 1

        # Count interactions per user and product
        user_counts = events['visitorid'].value_counts()
        product_counts = events['itemid'].value_counts()

        # Filter
        valid_users = user_counts[user_counts >= min_user_interactions].index
        valid_products = product_counts[product_counts >= min_product_interactions].index

        events_filtered = events[
            events['visitorid'].isin(valid_users) &
            events['itemid'].isin(valid_products)
        ].copy()

        print(f"  Iteration {iteration}: {len(events_filtered):,} events, "
              f"{len(valid_users):,} users, {len(valid_products):,} products")


        if len(events_filtered) == prev_len:
            break
        prev_len = len(events_filtered)

    print(f"Filtering complete: kept {len(events_filtered):,} events "
          f"({100*len(events_filtered)/len(events):.1f}% of original)")

    return events_filtered








def create_interaction_sequences(events, action_mapping, product_to_category, product_to_brand):
    events = events.sort_values('timestamp').copy()
    events['action_id'] = events['event'].map(action_mapping)
    events['category_id'] = events['itemid'].map(product_to_category).fillna(0).astype(int)
    events['brand_id'] = events['itemid'].map(product_to_brand).fillna(0).astype(int)

    # Log products without metadata
    no_metadata = (events['category_id'] == 0).sum()
    print(f"Products without category metadata: {no_metadata:,} ({100*no_metadata/len(events):.1f}%)")

    # Group by user and create sequences
    user_sequences = defaultdict(list)

    for _, row in events.iterrows():
        user_sequences[row['visitorid']].append({
            'product_id': int(row['itemid']),
            'action_type': int(row['action_id']),
            'timestamp': int(row['timestamp']),
            'category_id': int(row['category_id']),
            'brand_id': int(row['brand_id']),
            'event_name': row['event']
        })

    print(f"Created sequences for {len(user_sequences):,} users")

    seq_lengths = [len(seq) for seq in user_sequences.values()]
    print(f"Sequence length: mean={np.mean(seq_lengths):.1f}, "
          f"median={np.median(seq_lengths):.0f}, "
          f"min={np.min(seq_lengths)}, "
          f"max={np.max(seq_lengths)}")

    return dict(user_sequences)







def create_temporal_splits(user_sequences, train_ratio, val_ratio, test_ratio):

    train_sequences = {}
    val_sequences = {}
    test_sequences = {}

    for user_id, sequence in user_sequences.items():
        seq_len = len(sequence)


        train_end = int(seq_len * train_ratio)
        val_end = int(seq_len * (train_ratio + val_ratio))

        # Ensure each split has at least one interaction
        if train_end < 1:
            train_end = 1
        if val_end <= train_end:
            val_end = train_end + 1
        if val_end >= seq_len:
            val_end = seq_len - 1


        train_sequences[user_id] = sequence[:train_end]
        val_sequences[user_id] = sequence[train_end:val_end]
        test_sequences[user_id] = sequence[val_end:]


    train_total = sum(len(seq) for seq in train_sequences.values())
    val_total = sum(len(seq) for seq in val_sequences.values())
    test_total = sum(len(seq) for seq in test_sequences.values())
    total = train_total + val_total + test_total

    print(f"Train: {train_total:,} interactions ({100*train_total/total:.1f}%)")
    print(f"Val:   {val_total:,} interactions ({100*val_total/total:.1f}%)")
    print(f"Test:  {test_total:,} interactions ({100*test_total/total:.1f}%)")

    return train_sequences, val_sequences, test_sequences







def create_vocabulary_mappings(user_sequences, product_to_category, product_to_brand):
    all_users = set(user_sequences.keys())
    all_products = set()
    all_categories = set()
    all_brands = set()

    for sequence in user_sequences.values():
        for interaction in sequence:
            all_products.add(interaction['product_id'])
            all_categories.add(interaction['category_id'])
            all_brands.add(interaction['brand_id'])

    # Create mappings (ID -> index)
    # NOTE: We keep the original IDs as keys since they're already integers
    user_to_idx = {user_id: idx for idx, user_id in enumerate(sorted(all_users))}
    product_to_idx = {prod_id: idx for idx, prod_id in enumerate(sorted(all_products))}
    category_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted(all_categories))}
    brand_to_idx = {brand_id: idx for idx, brand_id in enumerate(sorted(all_brands))}

    print(f"Vocabularies: {len(user_to_idx):,} users, {len(product_to_idx):,} products, "
          f"{len(category_to_idx):,} categories, {len(brand_to_idx):,} brands")


    if 0 in all_categories:
        unknown_cat_count = sum(
            1 for seq in user_sequences.values()
            for inter in seq if inter['category_id'] == 0
        )
        print(f"Note: {unknown_cat_count:,} interactions have unknown category (ID 0)")

    vocabs = {
        'user_to_idx': user_to_idx,
        'product_to_idx': product_to_idx,
        'category_to_idx': category_to_idx,
        'brand_to_idx': brand_to_idx,
        'product_to_category': product_to_category,
        'product_to_brand': product_to_brand,
        'action_mapping': ACTION_MAPPING,
        'num_actions': len(ACTION_MAPPING),
    }

    return vocabs


def compute_statistics(user_sequences, vocabs):

    stats = {
        'num_users': len(user_sequences),
        'num_products': len(vocabs['product_to_idx']),
        'num_categories': len(vocabs['category_to_idx']),
        'num_brands': len(vocabs['brand_to_idx']),
        'num_actions': vocabs['num_actions'],
        'total_interactions': sum(len(seq) for seq in user_sequences.values()),
    }


    seq_lengths = [len(seq) for seq in user_sequences.values()]
    stats['seq_length_mean'] = float(np.mean(seq_lengths))
    stats['seq_length_median'] = float(np.median(seq_lengths))
    stats['seq_length_min'] = int(np.min(seq_lengths))
    stats['seq_length_max'] = int(np.max(seq_lengths))
    stats['seq_length_std'] = float(np.std(seq_lengths))


    action_counts = defaultdict(int)
    for sequence in user_sequences.values():
        for interaction in sequence:
            action_counts[interaction['event_name']] += 1

    stats['action_distribution'] = dict(action_counts)

    return stats


def save_processed_data(train_seqs, val_seqs, test_seqs, vocabs, stats):


    with open(OUTPUT_DIR / 'train_sequences.pkl', 'wb') as f:
        pickle.dump(train_seqs, f)
    print(f"Saved train_sequences.pkl ({len(train_seqs):,} users)")

    with open(OUTPUT_DIR / 'val_sequences.pkl', 'wb') as f:
        pickle.dump(val_seqs, f)
    print(f"Saved val_sequences.pkl ({len(val_seqs):,} users)")

    with open(OUTPUT_DIR / 'test_sequences.pkl', 'wb') as f:
        pickle.dump(test_seqs, f)
    print(f"Saved test_sequences.pkl ({len(test_seqs):,} users)")


    with open(OUTPUT_DIR / 'vocabularies.pkl', 'wb') as f:
        pickle.dump(vocabs, f)
    print(f"Saved vocabularies.pkl")

    with open(OUTPUT_DIR / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics.json")

    print(f"\nAll processed data saved to: {OUTPUT_DIR}")










def main():
    print("=" * 70)
    print("Retail Rocket Data Preprocessing Pipeline")
    print("=" * 70)

    events, category_tree, item_properties = load_raw_data()

    product_to_category, product_to_brand = extract_product_metadata(item_properties)
    events_filtered = filter_by_interaction_counts(
        events, MIN_USER_INTERACTIONS, MIN_PRODUCT_INTERACTIONS
    )

    user_sequences = create_interaction_sequences(
        events_filtered, ACTION_MAPPING, product_to_category, product_to_brand
    )

    vocabs = create_vocabulary_mappings(user_sequences, product_to_category, product_to_brand)

    train_sequences, val_sequences, test_sequences = create_temporal_splits(
        user_sequences, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    stats = compute_statistics(user_sequences, vocabs)
    save_processed_data(train_sequences, val_sequences, test_sequences, vocabs, stats)



    print("\n" + "=" * 70)
    print("Preprocessing complete!")
    print("=" * 70)
    print("\nDataset Summary:")
    print(f"  Users:        {stats['num_users']:,}")
    print(f"  Products:     {stats['num_products']:,}")
    print(f"  Categories:   {stats['num_categories']:,}")
    print(f"  Interactions: {stats['total_interactions']:,}")
    print(f"  Avg sequence length: {stats['seq_length_mean']:.1f}")
    print(f"\nAction distribution:")
    for action, count in stats['action_distribution'].items():
        print(f"  {action:12s}: {count:,} ({100*count/stats['total_interactions']:.1f}%)")


if __name__ == '__main__':
    main()
