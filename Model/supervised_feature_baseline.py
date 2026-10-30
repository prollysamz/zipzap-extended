"""Gradient-boosted supervised control on account-level representations."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from anomaly_detection import load_dataset
from evaluation import evaluate, select_f1_threshold
from research_protocol import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--alert-fraction", type=float, default=0.01)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    train_ids, train_x, train_y = load_dataset(args.train)
    val_ids, val_x, val_y = load_dataset(args.validation)
    test_ids, test_x, test_y = load_dataset(args.test)
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise ValueError("Train, validation and test accounts must be disjoint")
    start = time.perf_counter()
    model = HistGradientBoostingClassifier(class_weight="balanced", max_iter=200,
                                           early_stopping=True, random_state=args.seed).fit(train_x, train_y)
    training_seconds = time.perf_counter() - start
    threshold = select_f1_threshold(val_y, model.predict_proba(val_x)[:, 1])
    start = time.perf_counter()
    scores = model.predict_proba(test_x)[:, 1]
    inference_seconds = time.perf_counter() - start
    report = evaluate(test_y, scores, threshold, args.alert_fraction)
    report.update(method="hist_gradient_boosting", seed=args.seed,
                  fitting_label_use=True, threshold_label_use=True,
                  negative_class="accounts_without_known_phishing_labels",
                  training_seconds=training_seconds, inference_seconds=inference_seconds,
                  train_sha256=sha256_file(args.train), validation_sha256=sha256_file(args.validation),
                  test_sha256=sha256_file(args.test))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    np.savez_compressed(output / "test_scores.npz", account_ids=test_ids, labels=test_y, scores=scores)
    write_json(output / "metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
