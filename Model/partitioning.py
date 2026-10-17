"""Generate contiguous half-open frequency buckets for the current vocabulary."""
import argparse
import pickle
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
    args = parser.parse_args()
    with open(args.vocab, "rb") as source:
        vocab = pickle.load(source)
    frequencies = [vocab.counter.get(vocab.id_to_tokens.get(i), 0)
                   for i in range(max(vocab.id_to_tokens) + 1)]
    print(frequency_buckets(frequencies, args.buckets))
