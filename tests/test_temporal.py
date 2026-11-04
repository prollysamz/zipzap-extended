import random

import numpy as np

from gen_pretrain_data import create_masked_lm_predictions
from temporal_data import UNK, anonymise, build_vocab, eligible, truncate, windows


def history(timestamps, counts=None):
    counts = counts or [1] * len(timestamps)
    return [["peer_%d" % i, i, ts, 2, "OUT", c] for i, (ts, c) in enumerate(zip(timestamps, counts))]


def test_truncate_keeps_only_transactions_up_to_cutoff():
    seq = history([50, 40, 30, 20, 10])
    assert [t[2] for t in truncate(seq, 30)] == [30, 20, 10]
    assert truncate(seq, 5) == []


def test_eligibility_uses_bucket_lower_bounds():
    assert not eligible(history([4, 3, 2, 1]))
    assert eligible(history([5, 4, 3, 2, 1]))
    assert eligible(history([1], counts=[8]))


def test_unknown_counterparties_and_self_token_map_to_unknown():
    tokens = anonymise(history([3, 2, 1]), {"peer_0"})
    assert [t[0] for t in tokens] == [UNK, "peer_0", UNK, UNK]


def test_vocabulary_drops_rare_tokens_and_keeps_unknown_slot():
    seqs = {"acct": history([6, 5, 4, 3, 2, 1])}
    seqs["other"] = history([6, 5, 4, 3, 2, 1])
    vocab = build_vocab(seqs, min_count=2)
    assert "peer_0" in vocab.token_to_ids and "[NO_USE]" in vocab.token_to_ids
    lonely = build_vocab({"acct": history([6, 5, 4, 3, 2, 1])}, min_count=2)
    assert "peer_0" not in lonely.token_to_ids


def test_windows_filter_on_new_activity_and_masking_skips_unknown():
    seq = history(list(range(250, 0, -1)))
    rng = np.random.RandomState(0)
    old = list(windows(seq, 99, 60, rng))
    new = list(windows(seq, 99, 60, rng, newer_than=200))
    assert 0 < len(new) < len(old)
    assert all(w[0][2] > 200 for w in new)
    tokens = anonymise(history([5, 4, 3, 2, 1]), {"peer_2"})
    for seed in range(10):
        _, _, positions, labels = create_masked_lm_predictions(
            tokens, 0.5, 5, random.Random(seed), vocab_words=["peer_2"], exclude_tokens={UNK})
        assert all(tokens[p][0] == "peer_2" for p in positions)
