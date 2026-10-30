"""Export account embeddings from a verified label-independent encoder."""
import argparse
import json
import os
import pickle
import time
import glob
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import numpy as np
import tensorflow.compat.v1 as tf

import modeling
from research_protocol import read_provenance, sha256_file, sha256_files, validate_split_accounts, write_json


def resolve_input_files(specification):
    files = []
    for item in specification.split(","):
        matches = sorted(glob.glob(item))
        files.extend(matches if matches else [item])
    if not files or not all(os.path.isfile(path) for path in files):
        raise ValueError("Embedding input does not resolve to existing TFRecord files: " + specification)
    return files


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="zipzap_config.json")
    parser.add_argument("--vocab", default="data/vocab")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--split-manifest", default="data/finetune_splits.json",
                        help="Labelled fine-tuning manifest; omit it together with --split for label-free records")
    parser.add_argument("--split", choices=["train", "validation", "test"],
                        help="Required with --split-manifest")
    parser.add_argument("--max-seq-length", type=int, default=100,
                        help="Required sequence length when no split manifest is supplied")
    parser.add_argument("--pooling", choices=["self", "mean"], default="mean")
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-supervised-or-unknown", action="store_true")
    parser.add_argument("--allow-diagnostic-checkpoint", action="store_true",
                        help="Permit a checkpoint pretrained with an account holdout")
    parser.add_argument("--cross-share", action="store_true", help="Architecture for unverified checkpoints")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_seq_length < 2:
        parser.error("batch-size must be positive and max-seq-length must be at least two")
    if args.split and not args.split_manifest:
        parser.error("--split requires --split-manifest")
    if not args.split:
        args.split_manifest = None
    return args


def labelled_contract(args):
    if not args.split_manifest:
        return None, args.max_seq_length
    with open(args.split_manifest, encoding="utf-8") as source:
        manifest = json.load(source)
    validate_split_accounts(manifest)
    if manifest["vocab_sha256"] != sha256_file(args.vocab):
        raise ValueError("Split manifest uses a different vocabulary")
    split = manifest["splits"][args.split]
    if split["sha256"] != sha256_files(resolve_input_files(args.input_file)):
        raise ValueError("Input TFRecord does not match the selected split")
    return (manifest, split), manifest["max_seq_length"]


def main():
    args = parse_args()
    provenance = read_provenance(args.checkpoint)
    eligible = provenance.get("verified", False) and provenance.get("encoder_label_use") is False
    if not eligible and not args.allow_supervised_or_unknown:
        raise ValueError("Primary embeddings require verified label-independent encoder provenance. "
                         "Use --allow-supervised-or-unknown only for an identified ablation.")
    diagnostic = bool(provenance.get("diagnostic_holdout", False))
    if diagnostic and not args.allow_diagnostic_checkpoint:
        raise ValueError("This checkpoint was pretrained with an account holdout. Pass "
                         "--allow-diagnostic-checkpoint only for an identified ablation.")
    if provenance.get("verified") and (provenance["config_sha256"] != sha256_file(args.config)
                                        or provenance["vocab_sha256"] != sha256_file(args.vocab)):
        raise ValueError("Checkpoint config or vocabulary hash does not match")

    contract, length = labelled_contract(args)
    config = modeling.BertConfig.from_json_file(args.config)
    with open(args.vocab, "rb") as source:
        vocab = pickle.load(source)
    if max(vocab.id_to_tokens) >= config.vocab_size:
        raise ValueError("Vocabulary IDs exceed configured embedding coverage")

    schema = {name: tf.FixedLenFeature([length], tf.int64) for name in (
        "input_ids", "input_positions", "input_counts", "input_mask", "input_io_flags", "input_values")}
    schema["address"] = tf.FixedLenFeature([1], tf.int64)
    labelled = contract is not None
    if labelled:
        schema["label"] = tf.FixedLenFeature([1], tf.float32)
    dataset = tf.data.TFRecordDataset(resolve_input_files(args.input_file),
                                      compression_type=None if labelled else "GZIP").map(
        lambda r: tf.parse_single_example(r, schema))
    features = dataset.batch(args.batch_size).make_one_shot_iterator().get_next()
    integers = {name: tf.cast(value, tf.int32) for name, value in features.items() if name != "label"}
    model = modeling.BertModel(
        config=config, is_training=False, input_ids=integers["input_ids"],
        input_positions=integers["input_positions"], input_io_flags=integers["input_io_flags"],
        input_amounts=integers["input_values"], input_counts=integers["input_counts"],
        input_mask=integers["input_mask"], cross_share=provenance.get("cross_share", args.cross_share))
    sequence = model.get_sequence_output()
    if args.pooling == "self":
        embeddings = sequence[:, 0, :]
    else:
        mask = tf.cast(integers["input_mask"][:, 1:], tf.float32)
        denominator = tf.reduce_sum(mask, axis=1, keepdims=True)
        with tf.control_dependencies([tf.assert_positive(denominator)]):
            embeddings = tf.reduce_sum(sequence[:, 1:, :] * mask[:, :, None], axis=1) / denominator

    vectors, weights, labels = {}, {}, {}
    records = 0
    start = time.perf_counter()
    with tf.Session() as session:
        tf.train.Saver(var_list=tf.trainable_variables()).restore(session, args.checkpoint)
        while True:
            try:
                fetches = [features["address"], embeddings, features["input_mask"]]
                if labelled:
                    fetches.append(features["label"])
                result = session.run(fetches)
            except tf.errors.OutOfRangeError:
                break
            ids, batch, masks = result[:3]
            batch_labels = result[3] if labelled else [None] * len(ids)
            for account, vector, mask, label in zip(ids.reshape(-1), batch, masks, batch_labels):
                account = int(account)
                if not np.isfinite(vector).all():
                    raise ValueError("Non-finite embedding")
                if labelled:
                    if label not in (0, 1):
                        raise ValueError("Invalid label")
                    if account in labels and labels[account] != label:
                        raise ValueError("Inconsistent account labels")
                    labels[account] = int(label)
                weight = max(float(np.sum(mask) - 1), 1.0)
                vectors[account] = vectors.get(account, np.zeros_like(vector)) + vector * weight
                weights[account] = weights.get(account, 0.0) + weight
                records += 1

    if contract:
        manifest, split = contract
        if records != split["records"] or set(vectors) != set(split["account_ids"]):
            raise ValueError("Incomplete split coverage")
    ids = sorted(vectors)
    account_vectors = np.stack([vectors[account] / weights[account] for account in ids])
    if args.l2_normalize:
        account_vectors /= np.maximum(np.linalg.norm(account_vectors, axis=1, keepdims=True), 1e-12)
    output = Path(args.output)
    if output.suffix != ".npz":
        raise ValueError("Output must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {"account_ids": ids, "embeddings": account_vectors}
    if labelled:
        output_data["labels"] = [labels[account] for account in ids]
    np.savez_compressed(output, **output_data)
    report = {
        "checkpoint": args.checkpoint, "checkpoint_provenance": provenance,
        "encoder_label_independent": eligible, "diagnostic_holdout_checkpoint": diagnostic,
        "input_labelled": labelled, "labels_read": labelled, "pooling": args.pooling,
        "window_aggregation": "transaction_count_weighted_mean", "l2_normalized": args.l2_normalize,
        "input_sha256": sha256_files(resolve_input_files(args.input_file)),
        "output_sha256": sha256_file(output),
        "records": records, "accounts": len(ids), "max_seq_length": length,
        "extraction_seconds": time.perf_counter() - start,
    }
    if contract:
        manifest, _ = contract
        report.update(split=args.split, split_manifest_sha256=sha256_file(args.split_manifest),
                      evaluation_setting=manifest["evaluation_setting"],
                      split_label_use=manifest["split_label_use"])
    write_json(str(output) + ".json", report)


if __name__ == "__main__":
    main()
