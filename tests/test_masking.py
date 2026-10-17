import random

from gen_pretrain_data import create_masked_lm_predictions


def test_masking_excludes_self_and_uses_mask_random_and_keep_paths():
    tokens = [["self", 0, 0, 0, 0, 0]] + [
        ["token_%d" % i, i, i, i, "OUT", 1] for i in range(20)]
    vocabulary = ["replacement_%d" % i for i in range(10)]
    seen = set()
    for seed in range(30):
        _, output, positions, labels = create_masked_lm_predictions(
            tokens, 0.5, 10, random.Random(seed), mask_prob=0.8,
            random_token_prob=0.1, mask_self_token=False, vocab_words=vocabulary)
        assert 0 not in positions
        assert output[0][0] == "self"
        for position, label in zip(positions, labels):
            replacement = output[position][0]
            if replacement == "[MASK]":
                seen.add("mask")
            elif replacement == label:
                seen.add("keep")
            else:
                assert replacement in vocabulary
                seen.add("random")
    assert seen == {"mask", "random", "keep"}
