import jax
import jax.numpy as jnp
import numpy as np

from grok import TransformerConfig
from ecommerce_config import (
    EcommerceBatch,
    EcommerceEmbeddings,
    EcommerceModelConfig,
    HashConfig,
)
from ecommerce_ranking_model import create_ecommerce_ranking_model


def create_dummy_batch(
    batch_size: int = 2,
    history_len: int = 32,
    candidate_len: int = 8,
    num_hashes: int = 2,
    num_actions: int = 3,
    category_vocab: int = 100,
) -> EcommerceBatch:
    """Create dummy batch for testing."""

    customer_hashes = np.random.randint(1, 1000, (batch_size, num_hashes), dtype=np.int32)

    history_product_hashes = np.random.randint(
        1, 5000, (batch_size, history_len, num_hashes), dtype=np.int32
    )

    history_brand_hashes = np.random.randint(
        1, 500, (batch_size, history_len, num_hashes), dtype=np.int32
    )

    history_actions = np.random.rand(batch_size, history_len, num_actions).astype(np.float32)
    history_actions = (history_actions > 0.7).astype(np.float32)  # Sparse multi-hot

    history_category = np.random.randint(0, category_vocab, (batch_size, history_len), dtype=np.int32)

    candidate_product_hashes = np.random.randint(
        1, 5000, (batch_size, candidate_len, num_hashes), dtype=np.int32
    )

    candidate_brand_hashes = np.random.randint(
        1, 500, (batch_size, candidate_len, num_hashes), dtype=np.int32
    )

    candidate_category = np.random.randint(
        0, category_vocab, (batch_size, candidate_len), dtype=np.int32
    )

    return EcommerceBatch(
        customer_hashes=jnp.array(customer_hashes),
        history_product_hashes=jnp.array(history_product_hashes),
        history_brand_hashes=jnp.array(history_brand_hashes),
        history_actions=jnp.array(history_actions),
        history_category=jnp.array(history_category),
        candidate_product_hashes=jnp.array(candidate_product_hashes),
        candidate_brand_hashes=jnp.array(candidate_brand_hashes),
        candidate_category=jnp.array(candidate_category),
    )


def create_dummy_embeddings(
    batch_size: int,
    history_len: int,
    candidate_len: int,
    num_hashes: int,
    emb_size: int,
) -> EcommerceEmbeddings:
    """Create dummy embeddings for testing."""

    customer_emb = np.random.randn(batch_size, num_hashes, emb_size).astype(np.float32)

    history_product_emb = np.random.randn(
        batch_size, history_len, num_hashes, emb_size
    ).astype(np.float32)

    history_brand_emb = np.random.randn(
        batch_size, history_len, num_hashes, emb_size
    ).astype(np.float32)

    candidate_product_emb = np.random.randn(
        batch_size, candidate_len, num_hashes, emb_size
    ).astype(np.float32)

    candidate_brand_emb = np.random.randn(
        batch_size, candidate_len, num_hashes, emb_size
    ).astype(np.float32)

    return EcommerceEmbeddings(
        customer_embeddings=jnp.array(customer_emb),
        history_product_embeddings=jnp.array(history_product_emb),
        candidate_product_embeddings=jnp.array(candidate_product_emb),
        history_brand_embeddings=jnp.array(history_brand_emb),
        candidate_brand_embeddings=jnp.array(candidate_brand_emb),
    )


def test_model_init_and_forward():
    """Test model initialization and forward pass."""

    print("Testing E-commerce Ranking Model")
    print("=" * 70)


    config = EcommerceModelConfig(
        emb_size=128,
        num_actions=3,
        history_seq_len=32,
        candidate_seq_len=8,
        category_vocab_size=100,
        model=TransformerConfig(
            emb_size=128,
            key_size=64,
            num_q_heads=2,
            num_kv_heads=1,
            num_layers=2,
            widening_factor=2.0,
        ),
        hash_config=HashConfig(
            num_customer_hashes=2,
            num_product_hashes=2,
            num_brand_hashes=2,
        ),
    )

    print(f"\nModel config:")
    print(f"  emb_size: {config.emb_size}")
    print(f"  num_actions: {config.num_actions}")
    print(f"  history_seq_len: {config.history_seq_len}")
    print(f"  candidate_seq_len: {config.candidate_seq_len}")
    print(f"  transformer layers: {config.model.num_layers}")


    model = create_ecommerce_ranking_model(config)


    batch = create_dummy_batch(
        batch_size=2,
        history_len=config.history_seq_len,
        candidate_len=config.candidate_seq_len,
        num_hashes=2,
        num_actions=config.num_actions,
        category_vocab=config.category_vocab_size,
    )

    embeddings = create_dummy_embeddings(
        batch_size=2,
        history_len=config.history_seq_len,
        candidate_len=config.candidate_seq_len,
        num_hashes=2,
        emb_size=config.emb_size,
    )

    print(f"\nBatch shapes:")
    print(f"  customer_hashes: {batch.customer_hashes.shape}")
    print(f"  history_product_hashes: {batch.history_product_hashes.shape}")
    print(f"  history_actions: {batch.history_actions.shape}")
    print(f"  candidate_product_hashes: {batch.candidate_product_hashes.shape}")


    print(f"\nInitializing model...")
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, batch, embeddings)

    print(f"  Parameters initialized")
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Total parameters: {param_count:,}")

    # Forward pass
    print(f"\nRunning forward pass...")
    output = model.apply(params, None, batch, embeddings)

    print(f"  Output logits shape: {output.logits.shape}")
    print(f"  Expected shape: (2, 8, 3)")

    assert output.logits.shape == (2, 8, 3), "Unexpected output shape"

    print(f"\n[OK] All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_model_init_and_forward()
