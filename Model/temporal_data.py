"""Build time-sliced, label-free pretraining and embedding data for the inductive experiment.

Period 0 (history up to the first cutoff) builds the vocabulary and the initial pretraining set.
Each later period k yields
  * train_k : windows with activity newer than cutoff k-1, for continued pretraining,
  * ref_k   : every eligible account seen by cutoff k (history truncated at cutoff k),
  * eval_k  : accounts first seen inside period k, which no earlier model or vocabulary has seen.

Addresses missing from the period-0 vocabulary (or seen fewer than --min-token-count times) map to the
unknown token, and the account's own token is always the unknown token, so every account is scored from
its behaviour and counterparties alone. Labels are written to separate files for evaluation only.
"""
import argparse
import datetime
import math
import os
import pickle
import random
from pathlib import Path

import numpy as np

from gen_pretrain_data import (TrainingInstance, create_masked_lm_predictions, remove_stale_shards,
                               shard_paths, write_instance_to_example_files)
from research_protocol import sha256_files, write_json
from vocab import FreqVocab

UNK = "[NO_USE]"
COUNT_LOWER_BOUND = {8: 8, 9: 11, 10: 21}


def parse_date(text):
    return int(datetime.datetime.strptime(text, "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc).timestamp())


def truncate(seq, cutoff):
    """Drop transactions after `cutoff`; sequences are stored newest first."""
    start = 0
    while start < len(seq) and seq[start][2] > cutoff:
        start += 1
    return seq[start:]


def eligible(seq):
    """Same rule as gen_seq.py (at least five transactions), using the bucketed counts' lower bounds."""
    total = 0
    for transaction in seq:
        total += COUNT_LOWER_BOUND.get(transaction[5], transaction[5])
        if total >= 5:
            return True
    return False


def build_vocab(eoa2seq, min_count):
    vocab = FreqVocab()
    vocab.update(eoa2seq)
    vocab.prune(min_count)
    vocab.generate_vocab()
    return vocab


def anonymise(seq, known):
    """Copy a history with unknown counterparties replaced; the account's own token is always unknown."""
    return [[UNK, 0, 0, 0, 0, 0]] + [[t[0] if t[0] in known else UNK] + list(t[1:]) for t in seq]


def windows(seq, max_num_tokens, sliding_step, np_rng, newer_than=None, cap=500):
    if len(seq) <= max_num_tokens:
        starts = [0]
    else:
        starts = list(range(len(seq) - max_num_tokens, 0, -sliding_step)) + [0]
        if len(starts) > cap:
            starts = [int(i) for i in np_rng.permutation(starts)[:cap]]
    for start in starts:
        window = seq[start:start + max_num_tokens]
        if newer_than is None or window[0][2] > newer_than:
            yield window


def write_embedding_set(directory, accounts, histories, keys, known, vocab, args, labels):
    directory.mkdir(parents=True, exist_ok=True)
    np_rng = np.random.RandomState(args.seed)
    max_num_tokens = args.max_seq_length - 1
    sliding_step = round(args.max_seq_length * 0.6)
    instances, instance_keys = [], []
    for account in accounts:
        for window in windows(histories[account], max_num_tokens, sliding_step, np_rng):
            tokens = anonymise(window, known)
            instances.append(TrainingInstance(address=tokens[0][0], tokens=tokens,
                                              masked_lm_positions=[], masked_lm_labels=[]))
            instance_keys.append(keys[account])
    output = str(directory / "embed.tfrecord")
    files = shard_paths(output, len(instances), args.records_per_shard)
    remove_stale_shards(output, files)
    if instances:
        write_instance_to_example_files(instances, args.max_seq_length, args.max_predictions, vocab,
                                        files, address_ids=instance_keys)
    ordered = sorted(accounts, key=lambda a: keys[a])
    np.savez_compressed(directory / "accounts.npz", keys=np.array([keys[a] for a in ordered], dtype=np.int64),
                        accounts=np.array(ordered))
    if labels is not None:
        np.savez_compressed(directory / "labels.npz",
                            keys=np.array([keys[a] for a in ordered], dtype=np.int64),
                            labels=np.array([int(a.lower() in labels) for a in ordered], dtype=np.int64))
    return len(instances), len(ordered)


def write_training_set(directory, accounts, histories, known, vocab, args, newer_than, period):
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed + period)
    np_rng = np.random.RandomState(args.seed + period)
    max_num_tokens = args.max_seq_length - 1
    sliding_step = round(args.max_seq_length * 0.6)
    vocab_words = vocab.vocab_words[3:]
    instances = []
    for account in accounts:
        for window in windows(histories[account], max_num_tokens, sliding_step, np_rng, newer_than):
            tokens = anonymise(window, known)
            if all(t[0] == UNK for t in tokens[1:]):
                continue  # nothing predictable in an all-unknown window
            address, masked, positions, labels = create_masked_lm_predictions(
                tokens, args.masked_lm_prob, args.max_predictions, rng, 0.8, 0.1, False,
                vocab_words=vocab_words, exclude_tokens={UNK})
            instances.append(TrainingInstance(address, masked, positions, labels))
    if not instances:
        raise ValueError("Period %d produced no training windows" % period)
    rng.shuffle(instances)
    output = str(directory / "train.tfrecord")
    files = shard_paths(output, len(instances), args.records_per_shard)
    remove_stale_shards(output, files)
    write_instance_to_example_files(instances, args.max_seq_length, args.max_predictions, vocab, files)
    return files, len(instances)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data_benchmark",
                        help="Directory holding eoa2seq.pkl from gen_seq.py (benchmark corpus keeps phishers for evaluation)")
    parser.add_argument("--output-dir", default="data_temporal")
    parser.add_argument("--cutoffs", default="2019-01-01,2020-01-01,2021-01-01,2022-01-01",
                        help="Comma-separated UTC dates; the first ends the initial pretraining period")
    parser.add_argument("--phisher-file", default="../Data/phisher_account.txt",
                        help="Used only to write evaluation label files; empty disables labels")
    parser.add_argument("--min-token-count", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=100)
    parser.add_argument("--masked-lm-prob", type=float, default=0.15)
    parser.add_argument("--records-per-shard", type=int, default=25000)
    parser.add_argument("--limit-accounts", type=int, default=0, help="Random subset for smoke tests")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    args.max_predictions = math.ceil(args.max_seq_length * args.masked_lm_prob)
    cutoffs = [parse_date(c) for c in args.cutoffs.split(",")]
    if len(cutoffs) < 2 or cutoffs != sorted(set(cutoffs)):
        parser.error("Provide at least two strictly increasing cutoff dates")

    with open(os.path.join(args.source_dir, "eoa2seq.pkl"), "rb") as source:
        eoa2seq = pickle.load(source)
    if args.limit_accounts:
        rng = random.Random(args.seed)
        chosen = rng.sample(sorted(eoa2seq), min(args.limit_accounts, len(eoa2seq)))
        eoa2seq = {a: eoa2seq[a] for a in chosen}
    phishers = None
    if args.phisher_file:
        with open(args.phisher_file, encoding="utf-8") as source:
            phishers = {line.strip().lower() for line in source if line.strip()}

    keys = {a: i + 1 for i, a in enumerate(sorted(eoa2seq))}
    first_seen = {a: seq[-1][2] for a, seq in eoa2seq.items()}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Period 0 fixes the vocabulary for every later model, so deployment never needs new embedding rows.
    base = {a: h for a, seq in eoa2seq.items() for h in [truncate(seq, cutoffs[0])] if eligible(h)}
    vocab = build_vocab(base, args.min_token_count)
    known = set(vocab.token_to_ids)
    vocab_path = output / "vocab"
    with open(vocab_path, "wb") as sink:
        pickle.dump(vocab, sink, protocol=2)
    print("period 0 accounts: %d, vocabulary: %d tokens" % (len(base), len(vocab.vocab_words)))

    summary = {"cutoffs": args.cutoffs.split(","), "min_token_count": args.min_token_count,
               "vocab_tokens": len(vocab.vocab_words), "seed": args.seed, "periods": []}

    def manifest(files, period, newer_than):
        write_json(str(output / ("phase%d" % period) / "pretrain_manifest.json"), {
            "seed": args.seed, "max_seq_length": args.max_seq_length, "total_drop": False,
            "masked_lm_probability": args.masked_lm_prob, "mask_probability": 0.8,
            "random_token_probability": 0.1, "mask_self_token": False, "dupe_factor": 1,
            "tfrecord_compression": "GZIP", "train_sha256": sha256_files(files), "train_files": files,
            "vocab_sha256": sha256_files([str(vocab_path)]), "validation_sha256": None,
            "account_disjoint_validation": False, "diagnostic_holdout": False,
            "train_covers_all_accounts": True, "evaluation_setting": "temporal_inductive",
            "period": period, "cutoff": args.cutoffs.split(",")[period],
            "newer_than": newer_than, "unknown_token_policy": "frequency_floor_and_unknown_self"})

    for period, cutoff in enumerate(cutoffs):
        newer_than = None if period == 0 else cutoffs[period - 1]
        histories = {}
        for account, seq in eoa2seq.items():
            if first_seen[account] > cutoff:
                continue
            history = truncate(seq, cutoff)
            if eligible(history):
                histories[account] = history
        entry = {"period": period, "cutoff": args.cutoffs.split(",")[period],
                 "accounts_by_cutoff": len(histories)}
        if period < len(cutoffs) - 1:  # the final period is only scored, never trained on
            train_accounts = [a for a, h in histories.items() if newer_than is None or h[0][2] > newer_than]
            files, entry["train_records"] = write_training_set(
                output / ("phase%d" % period), train_accounts, histories, known, vocab, args, newer_than, period)
            manifest(files, period, newer_than)

        _, ref_accounts = write_embedding_set(output / ("ref%d" % period), sorted(histories), histories, keys,
                                              known, vocab, args, phishers)
        entry["reference_accounts"] = ref_accounts
        if period > 0:
            new_accounts = sorted(a for a in histories if first_seen[a] > cutoffs[period - 1])
            _, count = write_embedding_set(output / ("eval%d" % period), new_accounts, histories, keys,
                                           known, vocab, args, phishers)
            entry["eval_accounts"] = count
            if phishers is not None:
                entry["eval_known_phishing"] = sum(a.lower() in phishers for a in new_accounts)
        summary["periods"].append(entry)
        print(entry)
        del histories

    write_json(str(output / "temporal_summary.json"), summary)


if __name__ == "__main__":
    main()
