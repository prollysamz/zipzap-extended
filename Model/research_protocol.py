"""Shared data contracts and checkpoint provenance for reproducible experiments."""
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def validate_buckets(bucket_list, vocab_size, factors=None):
    """Buckets use contiguous half-open intervals, including padding ID zero."""
    if not bucket_list or (factors is not None and len(factors) != len(bucket_list)):
        raise ValueError("One factor is required for each embedding bucket")
    if factors is not None and any(factor <= 0 for factor in factors):
        raise ValueError("Embedding factor dimensions must be positive")
    previous = 0
    for lower, upper in bucket_list:
        if lower != previous or upper <= lower:
            raise ValueError("Embedding buckets must be contiguous half-open intervals")
        previous = upper
    if previous != vocab_size:
        raise ValueError("Embedding buckets must cover every ID below vocab_size")


def split_accounts(accounts, labels, seed=12345):
    """Stratified 70/15/15 account split, before generating overlapping windows."""
    from sklearn.model_selection import train_test_split
    accounts = sorted(accounts)
    if len(accounts) != len(set(accounts)):
        raise ValueError("Duplicate account IDs")
    train, held_out = train_test_split(
        accounts, test_size=0.30, random_state=seed,
        stratify=[labels[account] for account in accounts])
    validation, test = train_test_split(
        held_out, test_size=0.5, random_state=seed,
        stratify=[labels[account] for account in held_out])
    return {"train": set(train), "validation": set(validation), "test": set(test)}


def validate_split_accounts(manifest):
    seen = set()
    for name in ("train", "validation", "test"):
        split = manifest["splits"][name]
        ids = split["account_ids"]
        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("Empty, duplicate or overlapping accounts in split manifest")
        if split["records"] < len(ids):
            raise ValueError("Split has fewer records than accounts")
        seen.update(ids)


def checkpoint_files(prefix):
    prefix = Path(prefix)
    files = [Path(str(prefix) + ".index")] + sorted(prefix.parent.glob(prefix.name + ".data-*"))
    if len(files) < 2 or not all(path.is_file() for path in files):
        raise ValueError("Missing checkpoint index or data shards: " + str(prefix))
    return files


def read_provenance(prefix):
    """Unknown ancestry remains unknown; tensor names alone cannot establish it."""
    path = Path(str(prefix) + ".provenance.json")
    if not path.is_file():
        return {"encoder_label_use": "unknown", "verified": False}
    report = json.loads(path.read_text(encoding="utf-8"))
    actual = {p.name: sha256_file(p) for p in checkpoint_files(prefix)}
    if report.get("checkpoint_sha256") != actual:
        raise ValueError("Checkpoint does not match its provenance manifest")
    report["verified"] = True
    return report


def save_provenance(prefix, objective, config_file, vocab_file, parent=None,
                    frozen_encoder=False, **details):
    ancestry = read_provenance(parent) if parent else {"encoder_label_use": False}
    label_use = ancestry["encoder_label_use"]
    if objective == "supervised" and not frozen_encoder:
        label_use = True
    if ancestry.get("diagnostic_holdout"):
        # Descendants of a holdout-pretrained encoder remain marked as diagnostic.
        details["diagnostic_holdout"] = True
        details["trained_on_all_accounts"] = False
    report = dict(details, objective=objective, encoder_label_use=label_use,
                  frozen_encoder=frozen_encoder, parent_checkpoint=parent,
                  checkpoint_sha256={p.name: sha256_file(p) for p in checkpoint_files(prefix)},
                  config_sha256=sha256_file(config_file), vocab_sha256=sha256_file(vocab_file))
    write_json(str(prefix) + ".provenance.json", report)
    return report
