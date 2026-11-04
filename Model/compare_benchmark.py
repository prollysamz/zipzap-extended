"""Create a compact comparison report from completed benchmark metrics."""
import csv
import json
from pathlib import Path


ROOT = Path("project_results")
BENCHMARK = ROOT / "benchmark"


def load(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def row(name, family, path, fit_labels, threshold_labels, comparable=True):
    metrics = load(path)
    return {
        "model": name,
        "family": family,
        "same_benchmark": comparable,
        "fit_labels": fit_labels,
        "threshold_labels": threshold_labels,
        "accounts": metrics.get("accounts"),
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision", metrics.get("pr_auc")),
        "precision_at_k": metrics.get("precision_at_k"),
        "recall_at_k": metrics.get("recall_at_k"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "training_seconds": metrics.get("training_seconds"),
        "inference_seconds": metrics.get("inference_seconds"),
        "metrics_file": str(path),
    }


def number(value):
    return "—" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))


def main():
    required = [
        ("Corrected fine-tuned MLP", "supervised", BENCHMARK / "supervised_finetuned" / "metrics.json", "yes", "validation"),
        ("Corrected frozen MLP", "supervised", BENCHMARK / "supervised_frozen" / "metrics.json", "yes", "validation"),
        ("Random frozen MLP", "supervised control", BENCHMARK / "supervised_random_frozen" / "metrics.json", "yes", "validation"),
        ("Isolation Forest", "unsupervised", BENCHMARK / "isolation_forest" / "metrics.json", "no", "no"),
        ("K-means", "unsupervised", BENCHMARK / "kmeans" / "metrics.json", "no", "no"),
        ("Autoencoder", "unsupervised", BENCHMARK / "autoencoder" / "metrics.json", "no", "no"),
    ]
    missing = [str(path) for _, _, path, _, _ in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing benchmark metrics: " + ", ".join(missing))
    rows = [row(name, family, path, fit, threshold) for name, family, path, fit, threshold in required]
    legacy = ROOT / "supervised_summary.json"
    if legacy.is_file():
        rows.append(row("Original supervised MLP (legacy reference)", "legacy supervised", legacy,
                        "yes", "validation", comparable=False))

    BENCHMARK.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (BENCHMARK / "comparison.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (BENCHMARK / "comparison.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    headers = ["Model", "Same benchmark", "Fit labels", "Threshold labels", "ROC-AUC", "AP", "P@K", "R@K", "F1"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for item in rows:
        values = [item["model"], "yes" if item["same_benchmark"] else "no", item["fit_labels"],
                  item["threshold_labels"], number(item["roc_auc"]), number(item["average_precision"]),
                  number(item["precision_at_k"]), number(item["recall_at_k"]), number(item["f1"])]
        lines.append("| " + " | ".join(values) + " |")
    (BENCHMARK / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
