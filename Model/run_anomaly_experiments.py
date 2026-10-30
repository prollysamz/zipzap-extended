"""Compare anomaly detectors over exported ZipZap account vectors."""

import argparse
import csv
import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

METHODS = {"isolation_forest", "kmeans", "autoencoder"}
SETTINGS = {"mixed", "benign_only"}


def choices(value, allowed):
    selected = [item.strip() for item in value.split(",") if item.strip()]
    invalid = set(selected) - allowed
    if invalid:
        raise argparse.ArgumentTypeError("Invalid choices: %s" % sorted(invalid))
    return selected


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="anomaly_data/train.npz")
    parser.add_argument("--test_file", default="anomaly_data/test.npz")
    parser.add_argument("--output_dir", default="anomaly_results")
    parser.add_argument("--methods", type=lambda x: choices(x, METHODS),
                        default=sorted(METHODS))
    parser.add_argument("--settings", type=lambda x: choices(x, SETTINGS),
                        default=sorted(SETTINGS))
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--threshold_mode", choices=("validation_f1", "quantile"),
                        default="validation_f1")
    parser.add_argument("--expected_contamination", type=float, default=0.0086)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--ae_epochs", type=int, default=100)
    parser.add_argument("--ae_batch_size", type=int, default=256)
    return parser.parse_args()


def load(path):
    data = np.load(path)
    missing = {"addresses", "labels", "vectors"} - set(data.files)
    if missing:
        raise ValueError("%s is missing %s" % (path, sorted(missing)))
    address = np.asarray(data["addresses"]).reshape(-1)
    label = np.asarray(data["labels"], dtype=np.int8).reshape(-1)
    vector = np.asarray(data["vectors"], dtype=np.float32)
    if vector.ndim != 2 or len(vector) != len(label):
        raise ValueError("Invalid vectors in %s" % path)
    return address, label, vector


def split(label, vector, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    order = np.random.RandomState(seed).permutation(len(label))
    count = max(1, int(round(len(label) * fraction)))
    validation, fitting = order[:count], order[count:]
    if not len(fitting):
        raise ValueError("Training split is empty")
    return label[fitting], vector[fitting], label[validation], vector[validation]


def isolation_forest(x, args):
    model = IsolationForest(n_estimators=args.n_estimators, contamination="auto",
                            n_jobs=-1, random_state=args.seed).fit(x)
    return lambda values: -model.score_samples(values)


def kmeans(x, args):
    model = KMeans(n_clusters=min(args.n_clusters, len(x)), n_init=20,
                   random_state=args.seed).fit(x)
    return lambda values: np.min(model.transform(values), axis=1)


def autoencoder(x, validation, args):
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    inputs = tf.keras.Input(shape=(x.shape[1],))
    encoded = tf.keras.layers.Dense(32, activation="relu")(inputs)
    encoded = tf.keras.layers.Dense(8, activation="relu")(encoded)
    decoded = tf.keras.layers.Dense(32, activation="relu")(encoded)
    outputs = tf.keras.layers.Dense(x.shape[1])(decoded)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(x, x, validation_data=(validation, validation),
              epochs=args.ae_epochs, batch_size=min(args.ae_batch_size, len(x)),
              callbacks=[stop], verbose=0, shuffle=True)

    def score(values):
        predicted = model.predict(values, batch_size=args.ae_batch_size, verbose=0)
        return np.mean(np.square(values - predicted), axis=1)
    return score


def threshold(scores, labels, args):
    if args.threshold_mode == "validation_f1" and len(np.unique(labels)) == 2:
        precision, recall, candidates = precision_recall_curve(labels, scores)
        denominator = precision[:-1] + recall[:-1]
        values = np.divide(
            2 * precision[:-1] * recall[:-1], denominator,
            out=np.zeros_like(denominator), where=denominator != 0)
        return float(candidates[int(np.argmax(values))]), "validation_f1"
    return float(np.quantile(scores, 1 - args.expected_contamination)), "quantile"


def evaluate(labels, scores, selected_threshold):
    prediction = (scores >= selected_threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, prediction, average="binary", zero_division=0)
    result = {
        "accounts": int(len(labels)), "phishing_accounts": int(labels.sum()),
        "threshold": float(selected_threshold), "precision": float(precision),
        "recall": float(recall), "f1": float(f1),
        "flagged_accounts": int(prediction.sum()),
        "flagged_rate": float(prediction.mean()), "roc_auc": None, "pr_auc": None,
    }
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["pr_auc"] = float(average_precision_score(labels, scores))
    return result, prediction


def write_predictions(path, address, label, score, prediction):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("address_id", "label", "anomaly_score", "prediction"))
        writer.writerows(zip(address, label, score, prediction))


def main():
    args = arguments()
    train_address, train_label, train_vector = load(args.train_file)
    test_address, test_label, test_vector = load(args.test_file)
    fit_label, fit_vector, validation_label, validation_vector = split(
        train_label, train_vector, args.validation_fraction, args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for setting in args.settings:
        mask = np.ones(len(fit_label), dtype=bool)
        if setting == "benign_only":
            mask = fit_label == 0
        scaler = StandardScaler().fit(fit_vector[mask])
        x_fit = scaler.transform(fit_vector[mask])
        x_validation = scaler.transform(validation_vector)
        x_test = scaler.transform(test_vector)
        early_stop = x_validation
        if setting == "benign_only" and np.any(validation_label == 0):
            early_stop = x_validation[validation_label == 0]

        for method in args.methods:
            if method == "isolation_forest":
                score = isolation_forest(x_fit, args)
            elif method == "kmeans":
                score = kmeans(x_fit, args)
            else:
                score = autoencoder(x_fit, early_stop, args)
            validation_score = np.asarray(score(x_validation)).reshape(-1)
            test_score = np.asarray(score(x_test)).reshape(-1)
            cutoff, source = threshold(validation_score, validation_label, args)
            result, prediction = evaluate(test_label, test_score, cutoff)
            result.update(method=method, setting=setting, fit_accounts=len(x_fit),
                          threshold_source=source, seed=args.seed)
            results.append(result)
            name = "%s__%s__predictions.csv" % (method, setting)
            write_predictions(os.path.join(args.output_dir, name), test_address,
                              test_label, test_score, prediction)
            print("%s / %s: PR-AUC=%s ROC-AUC=%s F1=%.4f" % (
                method, setting, result["pr_auc"], result["roc_auc"], result["f1"]))

    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    fields = ["method", "setting", "fit_accounts", "accounts", "phishing_accounts",
              "roc_auc", "pr_auc", "precision", "recall", "f1", "flagged_accounts",
              "flagged_rate", "threshold", "threshold_source", "seed"]
    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
