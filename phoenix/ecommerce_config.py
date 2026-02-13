# Copyright 2026 E-commerce Adaptation Project
# Based on X.AI Corp's recommendation system
# Licensed under the Apache License, Version 2.0

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from grok import TransformerConfig


@dataclass
class HashConfig:
    """Configuration for hash-based embeddings."""

    num_customer_hashes: int = 2
    num_product_hashes: int = 2
    num_brand_hashes: int = 2


@dataclass
class EcommerceModelConfig:
    """Configuration for e-commerce ranking model.

    Adapted from X's RecsysModelConfig:
    - num_actions: 3 (transaction, addtocart, view) instead of 19
    - product_surface → category (924 categories in Retail Rocket)
    - author → brand (using category as brand proxy)
    """

    emb_size: int
    num_actions: int = 3
    history_seq_len: int = 64
    candidate_seq_len: int = 16
    category_vocab_size: int = 1000

    model: TransformerConfig = None
    hash_config: HashConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = TransformerConfig(
                emb_size=self.emb_size,
                key_size=64,
                num_q_heads=4,
                num_kv_heads=2,
                num_layers=4,
                widening_factor=2.0,
            )

        if self.hash_config is None:
            self.hash_config = HashConfig()


@dataclass
class EcommerceRetrievalConfig:
    """Configuration for e-commerce retrieval (two-tower) model."""

    emb_size: int
    num_actions: int = 3
    history_seq_len: int = 64
    candidate_seq_len: int = 16
    category_vocab_size: int = 1000

    model: TransformerConfig = None
    hash_config: HashConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = TransformerConfig(
                emb_size=self.emb_size,
                key_size=64,
                num_q_heads=4,
                num_kv_heads=2,
                num_layers=2,
                widening_factor=2.0,
            )

        if self.hash_config is None:
            self.hash_config = HashConfig()


class EcommerceBatch(NamedTuple):
    """Input batch for e-commerce models.

    Contains feature data (hashes, actions, categories).
    Embeddings are passed separately via EcommerceEmbeddings.
    """

    customer_hashes: jax.typing.ArrayLike
    history_product_hashes: jax.typing.ArrayLike
    history_brand_hashes: jax.typing.ArrayLike
    history_actions: jax.typing.ArrayLike
    history_category: jax.typing.ArrayLike
    candidate_product_hashes: jax.typing.ArrayLike
    candidate_brand_hashes: jax.typing.ArrayLike
    candidate_category: jax.typing.ArrayLike
    labels: jax.typing.ArrayLike = None


class EcommerceEmbeddings(NamedTuple):
    """Pre-looked-up embeddings from hash tables.

    Hash embeddings are combined via reduce functions in the model.
    """

    customer_embeddings: jax.typing.ArrayLike
    history_product_embeddings: jax.typing.ArrayLike
    candidate_product_embeddings: jax.typing.ArrayLike
    history_brand_embeddings: jax.typing.ArrayLike
    candidate_brand_embeddings: jax.typing.ArrayLike


class EcommerceModelOutput(NamedTuple):
    """Output of the e-commerce ranking model.

    logits: [B, C, 3] - predictions for (transaction, addtocart, view)
    """

    logits: jax.Array


class EcommerceRetrievalOutput(NamedTuple):
    """Output of the e-commerce retrieval model.

    customer_embedding: [B, D] - customer tower output
    candidate_embeddings: [B, C, D] - candidate tower outputs
    """

    customer_embedding: jax.Array
    candidate_embeddings: jax.Array
