"""
Quick test to verify data loading works correctly.

This is a sanity check before running the full preprocessing pipeline.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'retail_rocket'

print("Testing data loading...")
print("=" * 70)

# Test 1: Load events.csv
print("\n1. Loading events.csv...")
try:
    events = pd.read_csv(DATA_DIR / 'events.csv')
    print(f"   ✓ Loaded {len(events):,} events")
    print(f"   Columns: {list(events.columns)}")
    print(f"   Event types: {dict(events['event'].value_counts())}")
    print(f"   Unique visitors: {events['visitorid'].nunique():,}")
    print(f"   Unique products: {events['itemid'].nunique():,}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Load category_tree.csv
print("\n2. Loading category_tree.csv...")
try:
    cat_tree = pd.read_csv(DATA_DIR / 'category_tree.csv')
    print(f"   ✓ Loaded {len(cat_tree):,} categories")
    print(f"   Root categories (no parent): {cat_tree['parentid'].isna().sum()}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Load item properties
print("\n3. Loading item_properties (sample)...")
try:
    # Just load first 10000 rows to test
    props = pd.read_csv(DATA_DIR / 'item_properties_part1.csv', nrows=10000)
    print(f"   ✓ Loaded sample of {len(props):,} property entries")
    print(f"   Columns: {list(props.columns)}")
    print(f"   Top properties: {dict(props['property'].value_counts().head())}")

    # Test categoryid extraction
    cat_props = props[props['property'] == 'categoryid']
    if len(cat_props) > 0:
        print(f"   ✓ Found {len(cat_props)} categoryid entries in sample")
        print(f"   Sample categoryid values: {list(cat_props['value'].head())}")
    else:
        print("   ! No categoryid entries in this sample")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 70)
print("Data loading test complete!")
print("\nIf all tests passed, you can run the full preprocessing:")
print("  python data/prepare_retail_rocket.py")
