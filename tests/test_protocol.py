import json
import numpy as np
import pytest
from evaluation import aggregate_accounts, evaluate, select_f1_threshold
from partitioning import frequency_buckets
from anomaly_detection import fit_detector, load_dataset, preprocessing
from research_protocol import (read_provenance, save_provenance, split_accounts,
                               validate_buckets, validate_split_accounts)


def test_stratified_accounts_are_disjoint_complete_and_repeatable():
    labels = {str(i): int(i < 20) for i in range(100)}
    splits = split_accounts(labels, labels, seed=7)
    assert splits == split_accounts(reversed(list(labels)), labels, seed=7)
    assert [len(splits[k]) for k in ("train", "validation", "test")] == [70, 15, 15]
    assert set.union(*splits.values()) == set(labels)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not splits[a] & splits[b]
    assert all({labels[a] for a in accounts} == {0, 1} for accounts in splits.values())


def test_buckets_cover_boundaries_and_generated_vocabulary():
    from pathlib import Path
    config = json.loads((Path(__file__).parents[1] / "Model/zipzap_config.json").read_text())
    validate_buckets(config["bucket_list"], config["vocab_size"], config["factor_list"])
    for token in (0, 63, 64, 751, 752, 4242, 16800, 56218, 175121, 514653, 2300000):
        assert sum(lo <= token < hi for lo, hi in config["bucket_list"]) == 1
    for frequencies in ([0, 0, 0, 20, 5, 1], [0] * 8, [1] * 10):
        buckets = frequency_buckets(frequencies, 4)
        validate_buckets(buckets, len(frequencies))
    with pytest.raises(ValueError):
        validate_buckets([[0, 63], [64, 100]], 100)


def test_split_manifest_rejects_overlapping_or_missing_accounts():
    manifest = {"splits": {name: {"account_ids": [i], "records": 1}
                           for i, name in enumerate(("train", "validation", "test"))}}
    validate_split_accounts(manifest)
    manifest["splits"]["test"]["account_ids"] = [0]
    with pytest.raises(ValueError, match="overlapping"):
        validate_split_accounts(manifest)


def test_account_aggregation_keeps_singletons_and_detects_label_conflicts():
    ids, labels, scores = aggregate_accounts([[7]], [[1]], [0.75])
    assert (ids.tolist(), labels.tolist(), scores.tolist()) == ([7], [1], [0.75])
    ids, labels, scores = aggregate_accounts([2, 1, 2], [0, 1, 0], [0.1, 0.8, 0.3])
    assert ids.tolist() == [1, 2]
    np.testing.assert_allclose(scores, [0.8, 0.2])
    with pytest.raises(ValueError, match="Inconsistent"):
        aggregate_accounts([2, 2], [0, 1], [0.1, 0.8])
    with pytest.raises(ValueError, match="finite"):
        aggregate_accounts([2], [1], [float("nan")])


def test_threshold_and_ranking_metrics_have_known_answers():
    threshold = select_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert threshold == 0.8
    result = evaluate([0, 1, 0, 1], [0.1, 0.9, 0.8, 0.7], threshold, 0.5)
    assert result["k"] == 2
    assert result["precision_at_k"] == result["recall_at_k"] == 0.5
    assert result["false_positives_per_1000_accounts"] == 250
    assert result["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert evaluate([0], [0.2], 0.5)["roc_auc"] is None
    with pytest.raises(ValueError, match="both classes"):
        select_f1_threshold([0, 0], [0.1, 0.2])


def test_provenance_preserves_unknown_ancestry_and_detects_changed_weights(tmp_path):
    config = tmp_path / "config.json"
    vocab = tmp_path / "vocab"
    config.write_text("{}")
    vocab.write_bytes(b"vocab")

    def checkpoint(name):
        prefix = str(tmp_path / name)
        (tmp_path / (name + ".index")).write_bytes(b"index")
        (tmp_path / (name + ".data-00000-of-00001")).write_bytes(b"weights")
        return prefix

    pretrained = checkpoint("pretrained")
    save_provenance(pretrained, "masked_address_prediction", config, vocab)
    assert read_provenance(pretrained)["encoder_label_use"] is False
    frozen = checkpoint("frozen")
    save_provenance(frozen, "supervised", config, vocab, parent=pretrained, frozen_encoder=True)
    assert read_provenance(frozen)["encoder_label_use"] is False
    tuned = checkpoint("tuned")
    save_provenance(tuned, "supervised", config, vocab, parent=pretrained)
    resumed = checkpoint("resumed")
    save_provenance(resumed, "masked_address_prediction", config, vocab, parent=tuned)
    assert read_provenance(resumed)["encoder_label_use"] is True
    unknown = checkpoint("unknown")
    save_provenance(resumed, "masked_address_prediction", config, vocab, parent=unknown)
    assert read_provenance(resumed)["encoder_label_use"] == "unknown"
    (tmp_path / "pretrained.index").write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        read_provenance(pretrained)


@pytest.mark.parametrize("method", ["isolation_forest", "kmeans", "autoencoder"])
def test_anomaly_detectors_produce_finite_account_scores(method):
    rng = np.random.RandomState(4)
    values = rng.normal(size=(60, 6))
    transform = preprocessing(4, True).fit(values)
    transformed = transform.transform(values)
    model, score = fit_detector(method, transformed, seed=4, clusters=3, hidden_size=3)
    result = score(transformed)
    assert result.shape == (60,)
    assert np.isfinite(result).all()


def test_unlabelled_representations_are_loaded_without_reading_labels(tmp_path):
    path = tmp_path / "representations.npz"
    np.savez_compressed(path, account_ids=[2, 1], embeddings=[[2.0], [1.0]])
    ids, values, labels = load_dataset(path)
    assert ids.tolist() == [1, 2]
    assert values.tolist() == [[1.0], [2.0]]
    assert labels is None
    with pytest.raises(ValueError, match="requires labels"):
        load_dataset(path, require_labels=True)
