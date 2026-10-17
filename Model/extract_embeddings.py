"""Export frozen account embeddings with verified checkpoint and split alignment."""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import numpy as np
import tensorflow.compat.v1 as tf
import modeling
from research_protocol import read_provenance, sha256_file, validate_split_accounts, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="zipzap_config.json")
    parser.add_argument("--vocab", default="data/vocab")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--split-manifest", default="data/finetune_splits.json")
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--pooling", choices=["self", "mean"], default="mean")
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-supervised-or-unknown", action="store_true")
    parser.add_argument("--allow-diagnostic-checkpoint", action="store_true",
                        help="Permit a checkpoint pretrained with a holdout that excluded some accounts")
    parser.add_argument("--cross-share", action="store_true", help="Architecture for unverified checkpoints")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    provenance = read_provenance(args.checkpoint)
    eligible = provenance.get("verified", False) and provenance.get("encoder_label_use") is False
    if not eligible and not args.allow_supervised_or_unknown:
        parser.error("Primary embeddings require verified label-independent encoder provenance. "
                     "Use --allow-supervised-or-unknown only for a separately identified ablation.")
    diagnostic = bool(provenance.get("diagnostic_holdout", False))
    if diagnostic and not args.allow_diagnostic_checkpoint:
        parser.error("This checkpoint was pretrained with an account holdout (diagnostic run) and never "
                     "trained on those accounts. Pretrain with gen_pretrain_data.py defaults for downstream "
                     "embeddings, or pass --allow-diagnostic-checkpoint for a separately identified ablation.")
    if provenance.get("verified"):
        if (provenance["config_sha256"] != sha256_file(args.config)
                or provenance["vocab_sha256"] != sha256_file(args.vocab)):
            parser.error("Checkpoint config/vocabulary hashes do not match")
    cross_share = provenance.get("cross_share", args.cross_share)
    with open(args.split_manifest, encoding="utf-8") as source:
        manifest = json.load(source)
    validate_split_accounts(manifest)
    if manifest["vocab_sha256"] != sha256_file(args.vocab):
        parser.error("Split manifest uses a different vocabulary")
    split = manifest["splits"][args.split]
    if split["sha256"] != sha256_file(args.input_file):
        parser.error("Input TFRecord does not match the selected split")
    config = modeling.BertConfig.from_json_file(args.config)
    with open(args.vocab, "rb") as source:
        vocab = pickle.load(source)
    if max(vocab.id_to_tokens) >= config.vocab_size:
        parser.error("Vocabulary exceeds configured embedding coverage")
    length = manifest["max_seq_length"]
    schema = {name: tf.FixedLenFeature([length], tf.int64) for name in (
        "input_ids", "input_positions", "input_counts", "input_mask", "input_io_flags", "input_values")}
    schema.update(address=tf.FixedLenFeature([1], tf.int64), label=tf.FixedLenFeature([1], tf.float32))
    dataset = tf.data.TFRecordDataset(args.input_file).map(lambda r: tf.parse_single_example(r, schema))
    features = dataset.batch(args.batch_size).make_one_shot_iterator().get_next()
    integers = {k: tf.cast(v, tf.int32) for k, v in features.items() if k != "label"}
    model = modeling.BertModel(
        config=config, is_training=False, input_ids=integers["input_ids"],
        input_positions=integers["input_positions"], input_io_flags=integers["input_io_flags"],
        input_amounts=integers["input_values"], input_counts=integers["input_counts"],
        input_mask=integers["input_mask"], cross_share=cross_share)
    sequence = model.get_sequence_output()
    if args.pooling == "self":
        embeddings = sequence[:, 0, :]
    else:
        # Exclude the account token and padding from transaction mean pooling.
        mask = tf.cast(integers["input_mask"][:, 1:], tf.float32)
        denominator = tf.reduce_sum(mask, axis=1, keepdims=True)
        with tf.control_dependencies([tf.assert_positive(denominator)]):
            embeddings = tf.reduce_sum(sequence[:, 1:, :] * mask[:, :, None], axis=1) / denominator
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    vectors, labels, counts = {}, {}, {}
    records = 0
    start = time.perf_counter()
    with tf.Session() as sess:
        saver.restore(sess, args.checkpoint)
        while True:
            try:
                ids, targets, batch, masks = sess.run(
                    [features["address"], features["label"], embeddings, features["input_mask"]])
            except tf.errors.OutOfRangeError:
                break
            for account, label, vector, mask in zip(
                    ids.reshape(-1), targets.reshape(-1), batch, masks):
                account = int(account)
                if label not in (0, 1) or not np.isfinite(vector).all():
                    raise ValueError("Invalid label or embedding")
                if account in labels and labels[account] != label:
                    raise ValueError("Inconsistent account labels")
                # Give longer windows proportionally more influence. Position zero is
                # the account's self token, so it is not counted as a transaction.
                weight = max(float(np.sum(mask) - 1), 1.0)
                vectors[account] = vectors.get(account, np.zeros_like(vector)) + vector * weight
                counts[account] = counts.get(account, 0.0) + weight
                labels[account] = int(label)
                records += 1
    if records != split["records"] or set(vectors) != set(split["account_ids"]):
        raise ValueError("Incomplete split coverage")
    ids = sorted(vectors)
    output = Path(args.output)
    if output.suffix != ".npz":
        parser.error("Output must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    account_vectors = np.stack([vectors[a] / counts[a] for a in ids])
    if args.l2_normalize:
        norms = np.linalg.norm(account_vectors, axis=1, keepdims=True)
        account_vectors = account_vectors / np.maximum(norms, 1e-12)
    np.savez_compressed(output, account_ids=ids, embeddings=account_vectors,
                        labels=[labels[a] for a in ids])
    write_json(str(output) + ".json", {
        "checkpoint": args.checkpoint, "checkpoint_provenance": provenance,
        "encoder_label_independent": eligible, "diagnostic_holdout_checkpoint": diagnostic,
        "pooling": args.pooling,
        "window_aggregation": "transaction_count_weighted_mean", "split": args.split,
        "l2_normalized": args.l2_normalize,
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "output_sha256": sha256_file(output), "records": records, "accounts": len(ids),
        "extraction_seconds": time.perf_counter() - start,
        "evaluation_setting": manifest["evaluation_setting"],
        "split_label_use": manifest["split_label_use"],
        "labels_used_by_encoder_during_extraction": False,
        "total_drop": manifest["total_drop"], "max_seq_length": length,
    })


if __name__ == "__main__":
    main()
