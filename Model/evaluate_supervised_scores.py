"""Evaluate exported supervised scores with validation-only threshold selection."""

import argparse
import json

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load(path):
    data = np.load(path)
    labels = np.asarray(data["labels"], dtype=np.int8).reshape(-1)
    scores = np.asarray(data["classifier_scores"], dtype=np.float64).reshape(-1)
    if len(labels) != len(scores) or not len(scores):
        raise ValueError("Classifier scores are missing or misaligned in %s" % path)
    return labels, scores


def select_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator != 0)
    return float(thresholds[int(np.argmax(f1))])


def main():
    args = arguments()
    train_labels, train_scores = load(args.train_file)
    test_labels, test_scores = load(args.test_file)
    order = np.random.RandomState(args.seed).permutation(len(train_labels))
    count = max(1, int(round(len(train_labels) * args.validation_fraction)))
    validation = order[:count]
    threshold = select_threshold(train_labels[validation], train_scores[validation])
    predictions = (test_scores >= threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average="binary", zero_division=0)
    result = {
        "method": "supervised_mlp",
        "accounts": int(len(test_labels)),
        "phishing_accounts": int(test_labels.sum()),
        "roc_auc": float(roc_auc_score(test_labels, test_scores)),
        "pr_auc": float(average_precision_score(test_labels, test_scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": threshold,
        "flagged_accounts": int(predictions.sum()),
        "flagged_rate": float(predictions.mean()),
        "threshold_source": "validation_f1",
        "seed": args.seed,
    }
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
