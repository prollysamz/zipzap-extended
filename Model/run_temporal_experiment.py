"""Inductive, continually updated label-free experiment.

Stages (run in order, or --stage all):
  data      temporal_data.py: time-sliced TFRecords, fixed vocabulary, unknown-address policy
  pretrain  period-0 pretraining, then one continued-pretraining update per later period
  embed     export account embeddings for every (model state, reference/evaluation set)
  score     fit label-free detectors on each state's reference accounts, score the next period's new accounts
  report    collect metrics into a table

"static" is the period-0 model; "updated" is the model that has continued pretraining on every period
before the one being scored. Both score accounts first seen in the scored period, which no vocabulary or
model has seen. Labels are used only to evaluate the scores.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

PY = sys.executable


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(c) for c in command], check=True)


def latest_checkpoint(directory):
    steps = [(int(m.group(1)), p) for p in Path(directory).glob("model_*.index")
             for m in [re.fullmatch(r"model_(\d+)\.index", p.name)] if m]
    if not steps:
        raise ValueError("No checkpoint in " + str(directory))
    return str(max(steps)[1])[:-len(".index")].replace("\\", "/")


def pretrain(args, period):
    data = args.data_dir
    command = [PY, "run_pretrain.py", "--bert_config_file=%s/zipzap_config.json" % data,
               "--data_dir=%s/" % data, "--train_input_file=%s/phase%d/train*.tfrecord" % (data, period),
               "--pretrain_manifest=%s/phase%d/pretrain_manifest.json" % (data, period),
               "--checkpointDir=%s/phase%d" % (args.ckpt_dir, period), "--seed=%d" % args.seed,
               "--batch_size=%d" % args.batch_size]
    if period == 0:
        command += ["--epoch=%d" % args.epochs]
    else:
        command += ["--epoch=%d" % args.update_epochs, "--learning_rate=%g" % args.update_lr,
                    "--init_checkpoint=" + latest_checkpoint("%s/phase%d" % (args.ckpt_dir, period - 1))]
    run(command)


def periods(args):
    summary = json.loads((Path(args.data_dir) / "temporal_summary.json").read_text(encoding="utf-8"))
    return len(summary["periods"])


def embed(args, state, name):
    out = Path(args.output_dir) / "embeddings" / ("state%d_%s.npz" % (state, name))
    if out.exists():
        return out
    run([PY, "extract_embeddings.py", "--checkpoint=" + latest_checkpoint("%s/phase%d" % (args.ckpt_dir, state)),
         "--config=%s/zipzap_config.json" % args.data_dir, "--vocab=%s/vocab" % args.data_dir,
         "--input-file=%s/%s/embed*.tfrecord" % (args.data_dir, name), "--batch-size=%d" % args.batch_size,
         "--output=" + str(out)])
    return out


def pairs(count):
    """(model state, scored period): static state 0 for every period, updated state k-1 for period k."""
    result = [("static", 0, k) for k in range(1, count)]
    result += [("updated", k - 1, k) for k in range(2, count)]
    return result


def attach_labels(args, embeddings, period):
    out = Path(args.output_dir) / "embeddings" / (embeddings.stem + "_labelled.npz")
    if out.exists():
        return out
    with np.load(embeddings) as data, np.load(Path(args.data_dir) / ("eval%d" % period) / "labels.npz") as lab:
        ids, vectors = data["account_ids"], data["embeddings"]
        lookup = dict(zip(lab["keys"].tolist(), lab["labels"].tolist()))
        np.savez_compressed(out, account_ids=ids, embeddings=vectors,
                            labels=np.array([lookup[int(i)] for i in ids]))
    return out


def stage_embed(args):
    count = periods(args)
    for mode, state, period in pairs(count):
        embed(args, state, "ref%d" % state)
        embed(args, state, "eval%d" % period)


def stage_score(args):
    count = periods(args)
    for mode, state, period in pairs(count):
        train = embed(args, state, "ref%d" % state)
        test = attach_labels(args, embed(args, state, "eval%d" % period), period)
        for method in args.methods.split(","):
            out = Path(args.output_dir) / "scores" / method / ("%s_state%d_period%d" % (mode, state, period))
            if (out / "metrics.json").exists():
                continue
            run([PY, "anomaly_detection.py", "--train=" + str(train), "--test=" + str(test),
                 "--method=" + method, "--seed=%d" % args.seed, "--output-dir=" + str(out)])


def stage_report(args):
    rows = []
    for metrics in sorted((Path(args.output_dir) / "scores").glob("*/*/metrics.json")):
        report = json.loads(metrics.read_text(encoding="utf-8"))
        mode, state, period = re.fullmatch(r"(\w+)_state(\d+)_period(\d+)", metrics.parent.name).groups()
        rows.append((int(period), mode, int(state), metrics.parent.parent.name, report))
    rows.sort(key=lambda r: (r[0], r[3], r[1]))
    fmt = lambda v: "-" if v is None else "%.4f" % v
    lines = ["| Period | Detector | Encoder | Model state | New accounts | Known phishing | ROC-AUC | AP | P@K | R@K |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for period, mode, state, method, r in rows:
        lines.append("| %d | %s | %s | %d | %d | %d | %s | %s | %s | %s |" % (
            period, method, mode, state, r["accounts"], r["known_phishing"], fmt(r["roc_auc"]),
            fmt(r["average_precision"]), fmt(r["precision_at_k"]), fmt(r["recall_at_k"])))
    text = "\n".join(lines) + "\n"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "temporal_comparison.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["data", "pretrain", "embed", "score", "report", "all"], default="all")
    parser.add_argument("--data-dir", default="data_temporal")
    parser.add_argument("--ckpt-dir", default="ckpt_temporal")
    parser.add_argument("--output-dir", default="project_results/temporal")
    parser.add_argument("--source-dir", default="data_benchmark")
    parser.add_argument("--cutoffs", default="2019-01-01,2020-01-01,2021-01-01,2022-01-01")
    parser.add_argument("--limit-accounts", type=int, default=0)
    parser.add_argument("--methods", default="isolation_forest,kmeans,autoencoder")
    parser.add_argument("--epochs", type=int, default=5, help="Period-0 pretraining passes")
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--update-lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    stages = ["data", "pretrain", "embed", "score", "report"] if args.stage == "all" else [args.stage]
    if "data" in stages:
        command = [PY, "temporal_data.py", "--source-dir=" + args.source_dir, "--output-dir=" + args.data_dir,
                   "--cutoffs=" + args.cutoffs, "--seed=%d" % args.seed]
        if args.limit_accounts:
            command.append("--limit-accounts=%d" % args.limit_accounts)
        run(command)
        run([PY, "partitioning.py", "--vocab=%s/vocab" % args.data_dir, "--config=zipzap_config.json",
             "--output-config=%s/zipzap_config.json" % args.data_dir])
    if "pretrain" in stages:
        for period in range(periods(args) - 1):  # the final period is only ever scored, never trained on
            pretrain(args, period)
    if "embed" in stages:
        stage_embed(args)
    if "score" in stages:
        stage_score(args)
    if "report" in stages:
        stage_report(args)


if __name__ == "__main__":
    main()
