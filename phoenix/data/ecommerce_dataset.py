"""
E-commerce dataset loader for Retail Rocket data.

This module loads preprocessed sequences from .pkl files and creates
batches suitable for JAX/Haiku models (retrieval and ranking).

Key features:
- Hash-based embeddings (2 hashes per ID, as per X's approach)
- Temporal sequence handling with padding/truncation
- Efficient batching for training

Usage:
    dataset = EcommerceDataset('data/processed', split='train')
    batch = dataset.get_batch(batch_size=32, history_len=64, candidate_len=16)
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class EcommerceBatch:
    """
    Batch format for e-commerce recommendation models.

    Follows the structure from REC_SYS_OUTLINE.md Phase 3.1:
    - customer_hashes: [B, 2] - Two hash values per customer
    - history_product_hashes: [B, seq_len, 2] - Two hash values per product
    - history_brand_hashes: [B, seq_len, 2] - Two hash values per brand
    - history_actions: [B, seq_len, 3] - Multi-hot action encoding (transaction, addtocart, view)
    - history_category: [B, seq_len] - Category indices
    - candidate_product_hashes: [B, C, 2] - For ranking model (C candidates)
    - candidate_brand_hashes: [B, C, 2]
    - candidate_category: [B, C]
    - labels: [B, C, 3] - Multi-hot labels for candidates (what actions occurred)
    """
    # User/history features
    customer_hashes: jnp.ndarray  # [B, 2]
    history_product_hashes: jnp.ndarray  # [B, seq_len, 2]
    history_brand_hashes: jnp.ndarray  # [B, seq_len, 2]
    history_actions: jnp.ndarray  # [B, seq_len, 3]
    history_category: jnp.ndarray  # [B, seq_len]

    # Candidate features (for ranking)
    candidate_product_hashes: Optional[jnp.ndarray] = None  # [B, C, 2]
    candidate_brand_hashes: Optional[jnp.ndarray] = None  # [B, C, 2]
    candidate_category: Optional[jnp.ndarray] = None  # [B, C]

    # Labels
    labels: Optional[jnp.ndarray] = None  # [B, C, 3]


def hash_id(item_id: int, num_buckets: int, hash_idx: int = 0) -> int:
    """
    Hash an ID to a bucket using a simple hash function.

    X's approach uses 2 hash functions per ID for better representation.
    We implement this with a simple modulo hash with different primes.

    Args:
        item_id: The ID to hash
        num_buckets: Number of hash buckets (vocab size)
        hash_idx: Which hash function to use (0 or 1)

    Returns:
        Hash bucket index in [0, num_buckets)
    """
    # Use different large primes for different hash functions
    primes = [2654435761, 2654435789]  # Large primes for better distribution
    prime = primes[hash_idx % 2]

    # Simple hash: (id * prime) % num_buckets
    return (item_id * prime) % num_buckets


def hash_ids(ids: np.ndarray, num_buckets: int) -> np.ndarray:
    """
    Hash an array of IDs to produce 2 hash values per ID.

    Args:
        ids: Array of IDs, shape [..., ]
        num_buckets: Number of hash buckets

    Returns:
        Array of shape [..., 2] with two hash values per ID
    """
    # Convert to int64 to prevent overflow
    ids = ids.astype(np.int64)

    # Use different large primes for different hash functions
    # Vectorized hashing
    hash1 = (ids * np.int64(2654435761)) % num_buckets
    hash2 = (ids * np.int64(2654435789)) % num_buckets

    # Stack to create [..., 2] shape
    return np.stack([hash1, hash2], axis=-1).astype(np.int32)


class EcommerceDataset:
    """
    Dataset loader for e-commerce recommendation.

    Loads preprocessed .pkl files and creates batches for training/evaluation.
    """

    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing processed .pkl files
            split: One of 'train', 'val', 'test'
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # Load sequences
        seq_file = self.data_dir / f'{split}_sequences.pkl'
        with open(seq_file, 'rb') as f:
            self.sequences = pickle.load(f)

        # Load vocabularies
        vocab_file = self.data_dir / 'vocabularies.pkl'
        with open(vocab_file, 'rb') as f:
            self.vocabs = pickle.load(f)

        # Extract vocab sizes for hashing
        self.user_vocab_size = len(self.vocabs['user_to_idx'])
        self.product_vocab_size = len(self.vocabs['product_to_idx'])
        self.brand_vocab_size = len(self.vocabs['brand_to_idx'])
        self.category_vocab_size = len(self.vocabs['category_to_idx'])
        self.num_actions = self.vocabs['num_actions']

        # Convert sequences dict to list for easier batching
        self.user_ids = list(self.sequences.keys())

        print(f"Loaded {split} dataset:")
        print(f"  Users: {len(self.user_ids):,}")
        print(f"  Vocab sizes: users={self.user_vocab_size:,}, "
              f"products={self.product_vocab_size:,}, "
              f"categories={self.category_vocab_size:,}")

    def __len__(self):
        return len(self.user_ids)

    def get_user_sequence(self, user_id: int) -> List[Dict]:
        """Get the full sequence for a user."""
        return self.sequences[user_id]

    def create_training_example(
        self,
        user_id: int,
        history_len: int = 64,
        candidate_len: int = 16,
    ) -> Optional[Dict]:
        """
        Create a training example from a user's sequence.

        Format: Use first `history_len` interactions as history,
                next `candidate_len` as candidates to predict.

        Args:
            user_id: User ID
            history_len: Length of history sequence
            candidate_len: Number of candidates to predict

        Returns:
            Dict with 'history' and 'candidates' keys, or None if insufficient data
        """
        sequence = self.sequences[user_id]

        # Need at least history_len + 1 for meaningful training
        if len(sequence) < history_len + 1:
            return None

        # Randomly sample a split point
        # Ensure we have at least history_len before and some candidates after
        max_start = len(sequence) - history_len - 1
        if max_start < 1:
            return None

        start_idx = np.random.randint(0, max_start + 1)
        history_end = start_idx + history_len
        candidate_end = min(history_end + candidate_len, len(sequence))

        history = sequence[start_idx:history_end]
        candidates = sequence[history_end:candidate_end]

        if len(candidates) == 0:
            return None

        return {
            'user_id': user_id,
            'history': history,
            'candidates': candidates,
        }

    def encode_interactions(self, interactions: List[Dict], max_len: int) -> Tuple:
        """
        Encode a list of interactions into arrays.

        Args:
            interactions: List of interaction dicts
            max_len: Maximum sequence length (will pad/truncate)

        Returns:
            Tuple of (product_ids, brand_ids, action_ids, category_ids)
            Each is a numpy array of length max_len
        """
        # Truncate or pad to max_len
        seq_len = min(len(interactions), max_len)

        product_ids = np.zeros(max_len, dtype=np.int32)
        brand_ids = np.zeros(max_len, dtype=np.int32)
        category_ids = np.zeros(max_len, dtype=np.int32)
        action_ids = np.zeros(max_len, dtype=np.int32)

        for i in range(seq_len):
            product_ids[i] = interactions[i]['product_id']
            brand_ids[i] = interactions[i]['brand_id']
            category_ids[i] = interactions[i]['category_id']
            action_ids[i] = interactions[i]['action_type']

        return product_ids, brand_ids, action_ids, category_ids

    def actions_to_multihot(self, action_ids: np.ndarray) -> np.ndarray:
        """
        Convert action IDs to multi-hot encoding.

        Args:
            action_ids: Array of shape [seq_len] with action indices (0, 1, 2)

        Returns:
            Array of shape [seq_len, 3] with one-hot encoding per position
        """
        seq_len = len(action_ids)
        multihot = np.zeros((seq_len, self.num_actions), dtype=np.float32)

        for i in range(seq_len):
            if action_ids[i] < self.num_actions:  # Ignore padding (0 is valid action)
                multihot[i, action_ids[i]] = 1.0

        return multihot

    def get_batch(
        self,
        batch_size: int = 32,
        history_len: int = 64,
        candidate_len: int = 16,
    ) -> EcommerceBatch:
        """
        Create a random batch for training.

        Args:
            batch_size: Number of examples in batch
            history_len: Length of history sequence
            candidate_len: Number of candidate items

        Returns:
            EcommerceBatch with all features
        """
        # Sample users
        batch_user_ids = np.random.choice(self.user_ids, size=batch_size, replace=True)

        # Initialize batch arrays
        customer_ids = np.zeros(batch_size, dtype=np.int32)
        history_products = np.zeros((batch_size, history_len), dtype=np.int32)
        history_brands = np.zeros((batch_size, history_len), dtype=np.int32)
        history_categories = np.zeros((batch_size, history_len), dtype=np.int32)
        history_actions = np.zeros((batch_size, history_len), dtype=np.int32)

        candidate_products = np.zeros((batch_size, candidate_len), dtype=np.int32)
        candidate_brands = np.zeros((batch_size, candidate_len), dtype=np.int32)
        candidate_categories = np.zeros((batch_size, candidate_len), dtype=np.int32)
        candidate_action_labels = np.zeros((batch_size, candidate_len), dtype=np.int32)

        # Fill batch - retry with different users if needed
        valid_examples = 0
        max_retries = batch_size * 3  # Try up to 3x batch size users
        attempts = 0

        while valid_examples < batch_size and attempts < max_retries:
            user_id = np.random.choice(self.user_ids)
            attempts += 1

            example = self.create_training_example(user_id, history_len, candidate_len)

            if example is None:
                continue

            customer_ids[valid_examples] = user_id

            # Encode history
            prod, brand, actions, cats = self.encode_interactions(
                example['history'], history_len
            )
            history_products[valid_examples] = prod
            history_brands[valid_examples] = brand
            history_categories[valid_examples] = cats
            history_actions[valid_examples] = actions

            # Encode candidates
            prod, brand, actions, cats = self.encode_interactions(
                example['candidates'], candidate_len
            )
            candidate_products[valid_examples] = prod
            candidate_brands[valid_examples] = brand
            candidate_categories[valid_examples] = cats
            candidate_action_labels[valid_examples] = actions

            valid_examples += 1

        if valid_examples == 0:
            raise ValueError(
                f"Could not create any valid examples with "
                f"history_len={history_len}, candidate_len={candidate_len}. "
                f"Try reducing these parameters."
            )

        # Trim to valid examples
        customer_ids = customer_ids[:valid_examples]
        history_products = history_products[:valid_examples]
        history_brands = history_brands[:valid_examples]
        history_categories = history_categories[:valid_examples]
        history_actions = history_actions[:valid_examples]
        candidate_products = candidate_products[:valid_examples]
        candidate_brands = candidate_brands[:valid_examples]
        candidate_categories = candidate_categories[:valid_examples]
        candidate_action_labels = candidate_action_labels[:valid_examples]

        # Hash all IDs (creates [..., 2] arrays)
        customer_hashes = hash_ids(customer_ids, self.user_vocab_size)
        history_product_hashes = hash_ids(history_products, self.product_vocab_size)
        history_brand_hashes = hash_ids(history_brands, self.brand_vocab_size)
        candidate_product_hashes = hash_ids(candidate_products, self.product_vocab_size)
        candidate_brand_hashes = hash_ids(candidate_brands, self.brand_vocab_size)

        # Convert actions to multi-hot
        history_actions_multihot = np.stack([
            self.actions_to_multihot(history_actions[i])
            for i in range(valid_examples)
        ])

        candidate_labels_multihot = np.stack([
            self.actions_to_multihot(candidate_action_labels[i])
            for i in range(valid_examples)
        ])

        # Map category IDs to vocab indices
        history_category_indices = np.vectorize(
            lambda x: self.vocabs['category_to_idx'].get(x, 0)
        )(history_categories)

        candidate_category_indices = np.vectorize(
            lambda x: self.vocabs['category_to_idx'].get(x, 0)
        )(candidate_categories)

        # Convert to JAX arrays
        return EcommerceBatch(
            customer_hashes=jnp.array(customer_hashes),
            history_product_hashes=jnp.array(history_product_hashes),
            history_brand_hashes=jnp.array(history_brand_hashes),
            history_actions=jnp.array(history_actions_multihot),
            history_category=jnp.array(history_category_indices),
            candidate_product_hashes=jnp.array(candidate_product_hashes),
            candidate_brand_hashes=jnp.array(candidate_brand_hashes),
            candidate_category=jnp.array(candidate_category_indices),
            labels=jnp.array(candidate_labels_multihot),
        )


if __name__ == '__main__':
    # Test the dataset loader
    print("Testing EcommerceDataset...")
    print("=" * 70)

    dataset = EcommerceDataset('data/processed', split='train')
    print(f"\nDataset size: {len(dataset)} users")

    # Test batch creation
    print("\nCreating test batch...")
    batch = dataset.get_batch(batch_size=4, history_len=32, candidate_len=8)

    print(f"\nBatch shapes:")
    print(f"  customer_hashes: {batch.customer_hashes.shape}")
    print(f"  history_product_hashes: {batch.history_product_hashes.shape}")
    print(f"  history_brand_hashes: {batch.history_brand_hashes.shape}")
    print(f"  history_actions: {batch.history_actions.shape}")
    print(f"  history_category: {batch.history_category.shape}")
    print(f"  candidate_product_hashes: {batch.candidate_product_hashes.shape}")
    print(f"  candidate_brand_hashes: {batch.candidate_brand_hashes.shape}")
    print(f"  candidate_category: {batch.candidate_category.shape}")
    print(f"  labels: {batch.labels.shape}")

    print("\n[OK] Dataset loader test passed!")
