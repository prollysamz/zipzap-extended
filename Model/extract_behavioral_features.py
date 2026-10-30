"""Export account-level behavioural controls from pre-bucket transaction data."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from research_protocol import sha256_file, validate_split_accounts, write_json


FEATURE_NAMES = [
    "aggregated_transactions", "underlying_transaction_count", "unique_counterparties",
    "incoming_groups", "outgoing_groups", "incoming_fraction", "total_amount",
    "mean_amount", "std_amount", "max_amount", "active_duration_seconds",
    "groups_per_day", "underlying_transactions_per_day",
]


def vector(sequence):
    amounts = np.asarray([row[3] for row in sequence], dtype=float)
    counts = np.asarray([row[5] for row in sequence], dtype=float)
    timestamps = np.asarray([row[2] for row in sequence], dtype=float)
    incoming = sum(row[4] == "IN" for row in sequence)
    duration = float(max(0, timestamps.max() - timestamps.min())) if len(timestamps) else 0.0
    days = max(duration / 86400.0, 1.0)
    return [len(sequence), counts.sum(), len({row[0] for row in sequence}), incoming,
            len(sequence) - incoming, incoming / max(len(sequence), 1), amounts.sum(),
            amounts.mean(), amounts.std(), amounts.max(), duration,
            len(sequence) / days, counts.sum() / days]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", default="data/eoa2seq_behavioral.pkl")
    parser.add_argument("--vocab", default="data/vocab")
    parser.add_argument("--split-manifest", default="data/finetune_splits.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with open(args.sequences, "rb") as source:
        sequences = pickle.load(source)
    with open(args.vocab, "rb") as source:
        vocab = pickle.load(source)
    with open(args.split_manifest, encoding="utf-8") as source:
        manifest = json.load(source)
    validate_split_accounts(manifest)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    known_phishing = set()
    # Labels are recovered from the generated TFRecord split metadata only by
    # account membership; read the canonical label file when available.
    label_file = Path("../Data/phisher_account.txt")
    if not label_file.is_file():
        raise ValueError("Known-phishing label file is required for evaluation labels")
    known_phishing = {line.strip() for line in label_file.read_text().splitlines() if line.strip()}
    for split, details in manifest["splits"].items():
        ids = np.asarray(sorted(details["account_ids"]), dtype=np.int64)
        addresses = [vocab.id_to_tokens[int(account)] for account in ids]
        missing = [address for address in addresses if address not in sequences]
        if missing:
            raise ValueError("Behavioural source lacks split accounts: " + str(missing[:5]))
        values = np.asarray([vector(sequences[address]) for address in addresses], dtype=np.float64)
        labels = np.asarray([int(address in known_phishing) for address in addresses], dtype=np.int8)
        np.savez_compressed(output / (split + ".npz"), account_ids=ids,
                            embeddings=values, labels=labels, feature_names=FEATURE_NAMES)
    write_json(output / "manifest.json", {
        "source": "pre-bucket transaction sequences after three-day counterparty aggregation",
        "source_sha256": sha256_file(args.sequences), "vocab_sha256": sha256_file(args.vocab),
        "split_manifest_sha256": sha256_file(args.split_manifest), "feature_names": FEATURE_NAMES,
        "warning": "This is a behavioural-statistics control, not untouched raw transaction data.",
    })


if __name__ == "__main__":
    main()
