import argparse
import logging
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from grok import TransformerConfig
from ecommerce_config import EcommerceEmbeddings, EcommerceModelConfig, HashConfig
from ecommerce_ranking_model import create_ecommerce_ranking_model
from data.ecommerce_dataset import EcommerceDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_embedding_tables(vocab_sizes, emb_size, rng):
    """Create hash embedding tables for customer, product, and brand."""
    rng_customer, rng_product, rng_brand = jax.random.split(rng, 3)

    return {
        'customer': jax.random.normal(rng_customer, (vocab_sizes['customer'], emb_size)) * 0.02,
        'product': jax.random.normal(rng_product, (vocab_sizes['product'], emb_size)) * 0.02,
        'brand': jax.random.normal(rng_brand, (vocab_sizes['brand'], emb_size)) * 0.02,
    }


def lookup_embeddings(batch, tables):
    """Look up embeddings from hash tables."""
    return EcommerceEmbeddings(
        customer_embeddings=tables['customer'][batch.customer_hashes],
        history_product_embeddings=tables['product'][batch.history_product_hashes],
        candidate_product_embeddings=tables['product'][batch.candidate_product_hashes],
        history_brand_embeddings=tables['brand'][batch.history_brand_hashes],
        candidate_brand_embeddings=tables['brand'][batch.candidate_brand_hashes],
    )


def weighted_bce_loss(logits, labels, weights):
    """Weighted binary cross-entropy loss."""
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    bce = -(labels * log_p + (1 - labels) * log_not_p)
    weighted = bce * weights.reshape(1, 1, 3)
    return jnp.mean(weighted)


def compute_metrics(logits, labels):
    """Compute accuracy and per-action probabilities."""
    probs = jax.nn.sigmoid(logits)
    preds = (probs > 0.5).astype(jnp.float32)
    accuracy = jnp.mean((preds == labels).astype(jnp.float32))

    metrics = {'accuracy': float(accuracy)}

    for i, name in enumerate(['transaction', 'addtocart', 'view']):
        mask = labels[:, :, i] > 0
        if jnp.sum(mask) > 0:
            avg_prob = jnp.sum(probs[:, :, i] * mask) / jnp.sum(mask)
            metrics[f'{name}_prob'] = float(avg_prob)

    return metrics


def train_step(params, tables, opt_state, batch, model, optimizer, weights):
    """Single training step with gradient update."""
    def loss_fn(params):
        embeddings = lookup_embeddings(batch, tables)
        output = model.apply(params, None, batch, embeddings)
        loss = weighted_bce_loss(output.logits, batch.labels, weights)
        return loss, output.logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    metrics = compute_metrics(logits, batch.labels)
    metrics['loss'] = float(loss)

    return params, opt_state, metrics


def evaluate(params, tables, dataset, model, num_batches, batch_size, hist_len, cand_len):
    """Evaluate model on validation set."""
    all_metrics = []

    for _ in range(num_batches):
        batch = dataset.get_batch(batch_size, hist_len, cand_len)
        embeddings = lookup_embeddings(batch, tables)
        output = model.apply(params, None, batch, embeddings)
        all_metrics.append(compute_metrics(output.logits, batch.labels))

    return {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}



def save_checkpoint(step, params, tables, opt_state, checkpoint_dir):
    """Save training checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_path = checkpoint_dir / f'checkpoint_step_{step}.pkl'
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'step': step, 'params': params, 'tables': tables, 'opt_state': opt_state}, f)

    logger.info(f"Checkpoint saved: {checkpoint_path}")


def main(args):
    logger.info("="*70)
    logger.info("E-commerce Ranking Model Training")
    logger.info("="*70)


    train_data = EcommerceDataset('data/processed', split='train')
    val_data = EcommerceDataset('data/processed', split='val')


    config = EcommerceModelConfig(
        emb_size=args.emb_size,
        num_actions=3,
        history_seq_len=args.history_len,
        candidate_seq_len=args.candidate_len,
        category_vocab_size=train_data.category_vocab_size,
        model=TransformerConfig(
            emb_size=args.emb_size,
            key_size=args.emb_size // 2,
            num_q_heads=4,
            num_kv_heads=2,
            num_layers=args.num_layers,
            widening_factor=2.0,
        ),
        hash_config=HashConfig(num_customer_hashes=2, num_product_hashes=2, num_brand_hashes=2),
    )

    logger.info(f"\nConfig: emb={args.emb_size}, layers={args.num_layers}, "
                f"history={args.history_len}, candidates={args.candidate_len}")


    model = create_ecommerce_ranking_model(config)
    rng = jax.random.PRNGKey(args.seed)
    rng_model, rng_emb = jax.random.split(rng)

    vocab_sizes = {
        'customer': train_data.user_vocab_size,
        'product': train_data.product_vocab_size,
        'brand': train_data.brand_vocab_size,
    }
    tables = create_embedding_tables(vocab_sizes, args.emb_size, rng_emb)

    dummy_batch = train_data.get_batch(2, args.history_len, args.candidate_len)
    dummy_emb = lookup_embeddings(dummy_batch, tables)
    params = model.init(rng_model, dummy_batch, dummy_emb)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"Parameters: {param_count:,}")


    optimizer = optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)


    action_weights = jnp.array([10.0, 3.0, 1.0])
    logger.info("Action weights: transaction=10.0, addtocart=3.0, view=1.0")


    logger.info(f"\nTraining: {args.epochs} epochs, {args.steps_per_epoch} steps/epoch, "
                f"batch_size={args.batch_size}")

    global_step = 0

    for epoch in range(args.epochs):
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")

        epoch_metrics = []
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}")

        for _ in pbar:
            batch = train_data.get_batch(args.batch_size, args.history_len, args.candidate_len)
            params, opt_state, metrics = train_step(
                params, tables, opt_state, batch, model, optimizer, action_weights
            )

            epoch_metrics.append(metrics)
            global_step += 1

            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}", 'acc': f"{metrics['accuracy']:.3f}"})


        avg_loss = sum(m['loss'] for m in epoch_metrics) / len(epoch_metrics)
        avg_acc = sum(m['accuracy'] for m in epoch_metrics) / len(epoch_metrics)
        logger.info(f"Train: loss={avg_loss:.4f}, acc={avg_acc:.3f}")


        if (epoch + 1) % args.eval_every == 0:
            val_metrics = evaluate(
                params, tables, val_data, model, args.eval_batches,
                args.batch_size, args.history_len, args.candidate_len
            )
            logger.info(f"Val: acc={val_metrics['accuracy']:.3f}")

        # Checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(global_step, params, tables, opt_state, args.checkpoint_dir)

    # Final save
    logger.info("\n"+"="*70)
    logger.info("Training complete")
    save_checkpoint(global_step, params, tables, opt_state, args.checkpoint_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--history_len', type=int, default=64)
    parser.add_argument('--candidate_len', type=int, default=16)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--steps_per_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--eval_batches', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=2)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    args = parser.parse_args()
    main(args)
