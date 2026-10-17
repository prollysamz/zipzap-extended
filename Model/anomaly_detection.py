"""Fit and evaluate anomaly detectors on account-level NPZ representations."""
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


def load_dataset(path):
    with np.load(path) as data:
        required = {"account_ids", "embeddings", "labels"}
        if not required.issubset(data.files):
            raise ValueError("NPZ must contain account_ids, embeddings and labels")
        ids = np.asarray(data["account_ids"]).reshape(-1)
        values = np.asarray(data["embeddings"], dtype=np.float64)
        labels = np.asarray(data["labels"]).reshape(-1).astype(int)
    if values.ndim != 2 or len(ids) != len(values) or len(labels) != len(values):
        raise ValueError("Misaligned account representation file")
    if len(ids) != len(np.unique(ids)) or not np.isfinite(values).all():
        raise ValueError("Account IDs must be unique and representations finite")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Evaluation labels must be binary")
    order = np.argsort(ids, kind="stable")
    return ids[order], values[order], labels[order]


def preprocessing(pca_components, normalize):
    steps = [("scale", StandardScaler())]
    if pca_components:
        steps.append(("pca", PCA(n_components=pca_components, random_state=0)))
    if normalize:
        steps.append(("normalize", Normalizer()))
    return Pipeline(steps)


def fit_detector(method, values, seed, contamination, clusters, hidden_size):
    if method == "isolation_forest":
        model = IsolationForest(n_estimators=300, contamination=contamination,
                                random_state=seed, n_jobs=-1)
        model.fit(values)
        return model, lambda x: -model.score_samples(x)
    if method == "kmeans":
        model = KMeans(n_clusters=clusters, n_init=20, random_state=seed).fit(values)
        return model, lambda x: np.min(model.transform(x), axis=1)
    if method == "autoencoder":
        model = MLPRegressor(hidden_layer_sizes=(hidden_size,), activation="relu",
                             early_stopping=True, validation_fraction=0.1,
                             max_iter=300, random_state=seed).fit(values, values)
        return model, lambda x: np.mean(np.square(x - model.predict(x)), axis=1)
    raise ValueError("Unknown detector: " + method)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--method", choices=["isolation_forest", "kmeans", "autoencoder"], required=True)
    parser.add_argument("--setting", choices=["mixed_unlabelled", "one_class"], default="mixed_unlabelled")
    parser.add_argument("--threshold", choices=["validation_f1", "train_percentile"], default="train_percentile")
    parser.add_argument("--alert-fraction", type=float, default=0.01)
    parser.add_argument("--contamination", type=float, default=0.01)
    parser.add_argument("--pca-components", type=int)
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not 0 < args.contamination < 0.5 or not 0 < args.alert_fraction <= 1:
        parser.error("contamination must be in (0, .5), alert-fraction in (0, 1]")
    train_ids, train_x, train_y = load_dataset(args.train)
    val_ids, val_x, val_y = load_dataset(args.validation)
    test_ids, test_x, test_y = load_dataset(args.test)
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise ValueError("Train, validation and test accounts must be disjoint")
    fit_mask = train_y == 0 if args.setting == "one_class" else np.ones(len(train_y), dtype=bool)
    if not fit_mask.any():
        raise ValueError("No accounts available for detector fitting")
    fit_x = train_x[fit_mask]
    if args.pca_components and args.pca_components > min(fit_x.shape):
        parser.error("pca-components exceeds the fitted matrix rank")
    transform = preprocessing(args.pca_components, args.l2_normalize).fit(fit_x)
    fit_x = transform.transform(fit_x)
    start = time.perf_counter()
    detector, score = fit_detector(args.method, fit_x, args.seed, args.contamination,
                                   args.clusters, args.hidden_size)
    training_seconds = time.perf_counter() - start
    train_scores = score(fit_x)
    validation_scores = score(transform.transform(val_x))
    if args.threshold == "validation_f1":
        threshold = select_f1_threshold(val_y, validation_scores)
        threshold_label_use = True
    else:
        threshold = float(np.quantile(train_scores, 1 - args.alert_fraction))
        threshold_label_use = False
    start = time.perf_counter()
    test_scores = score(transform.transform(test_x))
    inference_seconds = time.perf_counter() - start
    report = evaluate(test_y, test_scores, threshold, args.alert_fraction)
    report.update(method=args.method, setting=args.setting, seed=args.seed,
                  fitting_label_use=args.setting == "one_class",
                  fitting_label_use_description=("Known-phishing labels select accounts without known labels"
                                                 if args.setting == "one_class" else "none"),
                  threshold_protocol=args.threshold, threshold_label_use=threshold_label_use,
                  negative_class="accounts_without_known_phishing_labels",
                  false_positive_interpretation="alerts among accounts without known phishing labels",
                  training_seconds=training_seconds, inference_seconds=inference_seconds,
                  pca_components=args.pca_components, l2_normalize=args.l2_normalize,
                  train_sha256=sha256_file(args.train), validation_sha256=sha256_file(args.validation),
                  test_sha256=sha256_file(args.test))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessing": transform, "detector": detector}, output / "model.joblib")
    np.savez_compressed(output / "test_scores.npz", account_ids=test_ids,
                        labels=test_y, scores=test_scores)
    write_json(output / "metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
