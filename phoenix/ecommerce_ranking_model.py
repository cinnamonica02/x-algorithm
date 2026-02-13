"""
E-commerce ranking model adapted from X's Phoenix ranking system.

Key changes from X's model:
- num_actions: 3 (transaction, addtocart, view) instead of 19
- author → brand (using category as brand proxy)
- product_surface → category
- Simplified for e-commerce use case
"""

import logging
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from grok import Transformer, layer_norm
from ecommerce_config import (
    EcommerceBatch,
    EcommerceEmbeddings,
    EcommerceModelConfig,
    EcommerceModelOutput,
)

logger = logging.getLogger(__name__)


def block_customer_reduce(
    customer_hashes: jnp.ndarray,
    customer_embeddings: jnp.ndarray,
    num_hashes: int,
    emb_size: int,
) -> Tuple[jax.Array, jax.Array]:
    """Combine multiple customer hash embeddings into single representation.
    Returns:
        customer_embedding: [B, 1, D]
        customer_mask: [B, 1] where True = valid
    """
    B = customer_embeddings.shape[0]
    D = emb_size

    customer_flat = customer_embeddings.reshape((B, 1, num_hashes * D))

    proj = hk.get_parameter(
        "customer_proj",
        [num_hashes * D, D],
        dtype=jnp.float32,
        init=hk.initializers.VarianceScaling(1.0, mode="fan_out"),
    )

    customer_emb = jnp.dot(customer_flat, proj)

    customer_mask = (customer_hashes[:, 0] != 0).reshape(B, 1).astype(jnp.bool_)

    return customer_emb, customer_mask


def block_history_reduce(
    history_product_hashes: jnp.ndarray,
    history_product_embeddings: jnp.ndarray,
    history_brand_embeddings: jnp.ndarray,
    history_category_embeddings: jnp.ndarray,
    history_action_embeddings: jnp.ndarray,
    num_product_hashes: int,
    num_brand_hashes: int,
) -> Tuple[jax.Array, jax.Array]:
    """Combine history embeddings (product, brand, category, actions).
    Returns:
        history_embeddings: [B, S, D]
        history_mask: [B, S]
    """
    B, S, _, D = history_product_embeddings.shape

    product_flat = history_product_embeddings.reshape((B, S, num_product_hashes * D))
    brand_flat = history_brand_embeddings.reshape((B, S, num_brand_hashes * D))

    history_features = jnp.concatenate(
        [
            product_flat,
            brand_flat,
            history_action_embeddings,
            history_category_embeddings,
        ],
        axis=-1,
    )

    # Project to embedding dimension
    proj = hk.get_parameter(
        "history_proj",
        [history_features.shape[-1], D],
        dtype=jnp.float32,
        init=hk.initializers.VarianceScaling(1.0, mode="fan_out"),
    )

    history_emb = jnp.dot(history_features, proj).reshape(B, S, D)

    # Hash 0 is padding
    history_mask = (history_product_hashes[:, :, 0] != 0).reshape(B, S)

    return history_emb, history_mask


def block_candidate_reduce(
    candidate_product_hashes: jnp.ndarray,
    candidate_product_embeddings: jnp.ndarray,
    candidate_brand_embeddings: jnp.ndarray,
    candidate_category_embeddings: jnp.ndarray,
    num_product_hashes: int,
    num_brand_hashes: int,
) -> Tuple[jax.Array, jax.Array]:
    """Combine candidate embeddings (product, brand, category).
    Returns:
        candidate_embeddings: [B, C, D]
        candidate_mask: [B, C]
    """
    B, C, _, D = candidate_product_embeddings.shape

    product_flat = candidate_product_embeddings.reshape((B, C, num_product_hashes * D))
    brand_flat = candidate_brand_embeddings.reshape((B, C, num_brand_hashes * D))

    candidate_features = jnp.concatenate(
        [
            product_flat,
            brand_flat,
            candidate_category_embeddings,
        ],
        axis=-1,
    )

    # Project to embedding dimension
    proj = hk.get_parameter(
        "candidate_proj",
        [candidate_features.shape[-1], D],
        dtype=jnp.float32,
        init=hk.initializers.VarianceScaling(1.0, mode="fan_out"),
    )

    candidate_emb = jnp.dot(candidate_features, proj)

    # Hash 0 is padding
    candidate_mask = (candidate_product_hashes[:, :, 0] != 0).reshape(B, C).astype(jnp.bool_)

    return candidate_emb, candidate_mask


def embed_actions(
    history_actions: jnp.ndarray,
    num_actions: int,
    emb_size: int,
) -> jax.Array:
    """Embed action multi-hot vectors.
    Returns:
        action_embeddings: [B, S, D]
    """
    B, S, _ = history_actions.shape


    action_emb_matrix = hk.get_parameter(
        "action_embeddings",
        [num_actions, emb_size],
        dtype=jnp.float32,
        init=hk.initializers.VarianceScaling(1.0, mode="fan_in"),
    )

    # Multi-hot to embedding: sum embeddings for active actions
    action_embeddings = jnp.dot(history_actions, action_emb_matrix)

    return action_embeddings


def embed_category(
    categories: jnp.ndarray,
    vocab_size: int,
    emb_size: int,
) -> jax.Array:
    """Embed category indices.
    Returns:
        category_embeddings: [..., D]
    """
    category_emb_matrix = hk.Embed(
        vocab_size=vocab_size,
        embed_dim=emb_size,
        name="category_embeddings",
    )

    return category_emb_matrix(categories)


class EcommerceRankingModel(hk.Module):
    """E-commerce ranking model using Grok transformer.

    Predicts engagement probabilities (transaction, addtocart, view) for
    candidate products given customer history.
    """

    def __init__(self, config: EcommerceModelConfig, name: str = "ecommerce_ranking"):
        super().__init__(name=name)
        self.config = config

    def __call__(
        self,
        batch: EcommerceBatch,
        embeddings: EcommerceEmbeddings,
    ) -> EcommerceModelOutput:
        """Forward pass.

        Args:
            batch: Input batch with hashes, actions, categories
            embeddings: Pre-looked-up hash embeddings

        Returns:
            Model output with logits [B, C, 3]
        """
        cfg = self.config
        B = batch.customer_hashes.shape[0]
        S = cfg.history_seq_len
        C = cfg.candidate_seq_len
        D = cfg.emb_size


        history_action_emb = embed_actions(
            batch.history_actions,
            cfg.num_actions,
            D,
        )

        history_category_emb = embed_category(
            batch.history_category,
            cfg.category_vocab_size,
            D,
        )

        candidate_category_emb = embed_category(
            batch.candidate_category,
            cfg.category_vocab_size,
            D,
        )

        # Reduce hash embeddings to single representations
        customer_emb, customer_mask = block_customer_reduce(
            batch.customer_hashes,
            embeddings.customer_embeddings,
            cfg.hash_config.num_customer_hashes,
            D,
        )



        history_emb, history_mask = block_history_reduce(
            batch.history_product_hashes,
            embeddings.history_product_embeddings,
            embeddings.history_brand_embeddings,
            history_category_emb,
            history_action_emb,
            cfg.hash_config.num_product_hashes,
            cfg.hash_config.num_brand_hashes,
        )

        candidate_emb, candidate_mask = block_candidate_reduce(
            batch.candidate_product_hashes,
            embeddings.candidate_product_embeddings,
            embeddings.candidate_brand_embeddings,
            candidate_category_emb,
            cfg.hash_config.num_product_hashes,
            cfg.hash_config.num_brand_hashes,
        )


        seq_emb = jnp.concatenate([customer_emb, history_emb, candidate_emb], axis=1)
        seq_mask = jnp.concatenate([customer_mask, history_mask, candidate_mask], axis=1)

        # Transformer forward pass with candidate isolation
        candidate_start = 1 + S  # After customer + history

        transformer = Transformer(
            num_q_heads=cfg.model.num_q_heads,
            num_kv_heads=cfg.model.num_kv_heads,
            key_size=cfg.model.key_size,
            widening_factor=cfg.model.widening_factor,
            attn_output_multiplier=cfg.model.attn_output_multiplier,
            num_layers=cfg.model.num_layers,
            name="transformer",
        )

        transformer_out = transformer(
            seq_emb,
            mask=seq_mask,
            candidate_start_offset=candidate_start,
        )


        candidate_outputs = transformer_out.embeddings[:, candidate_start:, :]

        # Layer norm + projection to action logits
        candidate_outputs = layer_norm(candidate_outputs)

        logits_proj = hk.Linear(
            cfg.num_actions,
            name="logits_projection",
            w_init=hk.initializers.VarianceScaling(1.0, mode="fan_in"),
        )

        logits = logits_proj(candidate_outputs)  # [B, C, 3]

        return EcommerceModelOutput(logits=logits)


def create_ecommerce_ranking_model(
    config: EcommerceModelConfig,
) -> hk.Transformed:
    """Create Haiku transformed model.
    Returns:
        Transformed model with init and apply functions
    """

    def forward(
        batch: EcommerceBatch,
        embeddings: EcommerceEmbeddings,
    ) -> EcommerceModelOutput:
        model = EcommerceRankingModel(config)
        return model(batch, embeddings)

    return hk.transform(forward)
