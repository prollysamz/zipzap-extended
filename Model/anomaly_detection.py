"""Fit label-free anomaly detectors on account-level embedding exports."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

from evaluation import evaluate, select_f1_threshold
from research_protocol import sha256_file, write_json


def load_dataset(path, require_labels=False):
    """Load representations without reading labels unless an explicit mode needs them."""
    with np.load(path) as data:
        required = {"account_ids", "embeddings"}
        if not required.issubset(data.files):
            raise ValueError("NPZ must contain account_ids and embeddings")
        ids = np.asarray(data["account_ids"]).reshape(-1)
        values = np.asarray(data["embeddings"], dtype=np.float64)
        labels = None
        if require_labels:
            if "labels" not in data.files:
                raise ValueError("This mode requires labels in the representation file")
            labels = np.asarray(data["labels"]).reshape(-1).astype(int)
    if values.ndim != 2 or len(ids) != len(values) or (labels is not None and len(labels) != len(values)):
        raise ValueError("Misaligned account representation file")
    if len(ids) != len(np.unique(ids)) or not np.isfinite(values).all():
        raise ValueError("Account IDs must be unique and representations finite")
    if labels is not None and not np.isin(labels, [0, 1]).all():
        raise ValueError("Evaluation labels must be binary")
    order = np.argsort(ids, kind="stable")
    return ids[order], values[order], None if labels is None else labels[order]


def preprocessing(pca_components, normalize):
    steps = [("scale", StandardScaler())]
    if pca_components:
        steps.append(("pca", PCA(n_components=pca_components, random_state=0)))
    if normalize:
        steps.append(("normalize", Normalizer()))
    return Pipeline(steps)


def fit_detector(method, values, seed, clusters, hidden_size):
    if method == "isolation_forest":
        model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(values)
        return model, lambda x: -model.score_samples(x)
    if method == "kmeans":
        model = KMeans(n_clusters=clusters, n_init=20, random_state=seed).fit(values)
        return model, lambda x: np.min(model.transform(x), axis=1)
    if method == "autoencoder":
        model = MLPRegressor(hidden_layer_sizes=(hidden_size,), activation="relu",
                             early_stopping=True, validation_fraction=0.1,
                             max_iter=500, random_state=seed).fit(values, values)
        return model, lambda x: np.mean(np.square(x - model.predict(x)), axis=1)
    raise ValueError("Unknown detector: " + method)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Label-free account embeddings used for fitting")
    parser.add_argument("--score", help="Embeddings to score; defaults to --train")
    parser.add_argument("--validation", help="Only required by --threshold=validation_f1")
    parser.add_argument("--test", help="Optional labelled test embeddings for a separately reported evaluation")
    parser.add_argument("--method", choices=["isolation_forest", "kmeans", "autoencoder"], required=True)
    parser.add_argument("--setting", choices=["mixed_unlabelled", "one_class"], default="mixed_unlabelled")
    parser.add_argument("--threshold", choices=["validation_f1", "train_percentile"], default="train_percentile")
    parser.add_argument("--alert-fraction", type=float, default=0.01)
    parser.add_argument("--pca-components", type=int)
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not 0 < args.alert_fraction <= 1:
        parser.error("alert-fraction must be in (0, 1]")
    if args.setting == "one_class" and args.threshold == "train_percentile":
        parser.error("one_class fitting consumes labels; use mixed_unlabelled for a label-free run")
    if args.threshold == "validation_f1" and not args.validation:
        parser.error("validation_f1 threshold selection requires --validation")

    train_ids, train_x, train_y = load_dataset(args.train, require_labels=args.setting == "one_class")
    fit_mask = train_y == 0 if args.setting == "one_class" else np.ones(len(train_x), dtype=bool)
    fit_x = train_x[fit_mask]
    if not len(fit_x):
        raise ValueError("No accounts available for detector fitting")
    if args.pca_components and args.pca_components > min(fit_x.shape):
        parser.error("pca-components exceeds the fitted matrix rank")
    if args.method == "kmeans" and args.clusters > len(fit_x):
        parser.error("clusters cannot exceed fitted account count")

    transform = preprocessing(args.pca_components, args.l2_normalize).fit(fit_x)
    fit_x = transform.transform(fit_x)
    start = time.perf_counter()
    detector, score = fit_detector(args.method, fit_x, args.seed, args.clusters, args.hidden_size)
    training_seconds = time.perf_counter() - start
    train_scores = score(fit_x)

    validation_ids = None
    if args.threshold == "validation_f1":
        validation_ids, validation_x, validation_y = load_dataset(args.validation, require_labels=True)
        if set(train_ids) & set(validation_ids):
            raise ValueError("Training and validation accounts must be disjoint")
        threshold = select_f1_threshold(validation_y, score(transform.transform(validation_x)))
        threshold_label_use = True
    else:
        threshold = float(np.quantile(train_scores, 1 - args.alert_fraction))
        threshold_label_use = False

    score_path = args.score or args.test or args.train
    score_ids, score_x, _ = load_dataset(score_path)
    start = time.perf_counter()
    scores = score(transform.transform(score_x))
    inference_seconds = time.perf_counter() - start
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessing": transform, "detector": detector}, output / "model.joblib")
    np.savez_compressed(output / "scores.npz", account_ids=score_ids, scores=scores,
                        threshold=threshold)

    report = {
        "method": args.method, "setting": args.setting, "seed": args.seed,
        "fitting_label_use": args.setting == "one_class",
        "threshold_protocol": args.threshold, "threshold_label_use": threshold_label_use,
        "score_label_use": False, "threshold": threshold,
        "alert_fraction": args.alert_fraction, "fitted_accounts": int(len(fit_x)),
        "scored_accounts": int(len(score_ids)), "training_seconds": training_seconds,
        "inference_seconds": inference_seconds, "pca_components": args.pca_components,
        "l2_normalize": args.l2_normalize, "train_sha256": sha256_file(args.train),
        "score_sha256": sha256_file(score_path),
    }
    if args.test:
        test_ids, test_x, test_y = load_dataset(args.test, require_labels=True)
        if set(train_ids) & set(test_ids):
            raise ValueError("Training and test accounts must be disjoint")
        if validation_ids is not None and set(validation_ids) & set(test_ids):
            raise ValueError("Validation and test accounts must be disjoint")
        test_scores = score(transform.transform(test_x))
        report.update(evaluate(test_y, test_scores, threshold, args.alert_fraction))
        report["test_sha256"] = sha256_file(args.test)
        np.savez_compressed(output / "test_scores.npz", account_ids=test_ids,
                            labels=test_y, scores=test_scores)
    write_json(output / "metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
