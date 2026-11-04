#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/jakku/.venvs/llmfa/bin/python
DATA=data_benchmark
CONFIG="$DATA/zipzap_config.json"
SPLITS="$DATA/finetune_splits.json"
PRETRAIN=ckpt_benchmark_pretrain/model_1360
RESULTS=project_results/benchmark

while pgrep -f "run_finetune.py.*ckpt_benchmark_finetuned" >/dev/null; do sleep 30; done
test -f "$RESULTS/supervised_finetuned/metrics.json"

COMMON=(--bert_config_file="$CONFIG" --data_dir="$DATA/" --split_manifest="$SPLITS"
  --train_input_file="$DATA/finetune_train.tfrecord"
  --validation_input_file="$DATA/finetune_validation.tfrecord"
  --test_input_file="$DATA/finetune_test.tfrecord" --batch_size=256 --seed=12345)

"$PYTHON" -u run_finetune.py "${COMMON[@]}" --init_checkpoint="$PRETRAIN" --load_cross=True \
  --freeze_encoder=True --checkpointDir=ckpt_benchmark_frozen \
  --output_dir="$RESULTS/supervised_frozen" > benchmark_supervised_frozen.log 2>&1

"$PYTHON" -u run_finetune.py "${COMMON[@]}" --freeze_encoder=True \
  --checkpointDir=ckpt_benchmark_random_frozen \
  --output_dir="$RESULTS/supervised_random_frozen" > benchmark_supervised_random_frozen.log 2>&1

mkdir -p "$RESULTS/embeddings"
for split in train validation test; do
  "$PYTHON" extract_embeddings.py --checkpoint="$PRETRAIN" --config="$CONFIG" \
    --vocab="$DATA/vocab" --input-file="$DATA/finetune_${split}.tfrecord" \
    --split-manifest="$SPLITS" --split="$split" --batch-size=256 \
    --output="$RESULTS/embeddings/${split}.npz" > "benchmark_export_${split}.log" 2>&1
done

for method in isolation_forest kmeans autoencoder; do
  "$PYTHON" anomaly_detection.py --train="$RESULTS/embeddings/train.npz" \
    --test="$RESULTS/embeddings/test.npz" --method="$method" --seed=12345 \
    --output-dir="$RESULTS/$method" > "benchmark_${method}.log" 2>&1
done

"$PYTHON" compare_benchmark.py > benchmark_comparison.log 2>&1
