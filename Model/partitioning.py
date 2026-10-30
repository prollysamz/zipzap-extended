"""Generate contiguous half-open frequency buckets for the current vocabulary."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np


def frequency_buckets(frequencies, count):
    if not 1 <= count <= len(frequencies):
        raise ValueError("Bucket count must be between one and the vocabulary size")
    cumulative = np.cumsum(frequencies)
    boundaries = [0]
    for index in range(1, count):
        proposed = int(np.searchsorted(cumulative, cumulative[-1] * index / count)) + 1
        boundaries.append(min(max(boundaries[-1] + 1, proposed), len(frequencies) - (count - index)))
    boundaries.append(len(frequencies))
    return [[lower, upper] for lower, upper in zip(boundaries, boundaries[1:])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", default="./data/vocab")
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--config", help="Base model configuration to update")
    parser.add_argument("--output-config", help="Path for the configuration generated for this vocabulary")
    args = parser.parse_args()
    if bool(args.config) != bool(args.output_config):
        parser.error("--config and --output-config must be supplied together")
    with open(args.vocab, "rb") as source:
        vocab = pickle.load(source)
    frequencies = [vocab.counter.get(vocab.id_to_tokens.get(i), 0)
                   for i in range(max(vocab.id_to_tokens) + 1)]
    buckets = frequency_buckets(frequencies, args.buckets)
    if not args.config:
        print(buckets)
    else:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if len(config["factor_list"]) != args.buckets:
            parser.error("The base config must provide one factor dimension per bucket")
        config["bucket_list"] = buckets
        config["vocab_size"] = len(frequencies)
        Path(args.output_config).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(args.output_config)
