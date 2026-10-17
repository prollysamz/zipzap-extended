# LLM Applications in Finance: Ethereum Fraud Detection

ZipZap replication and a controlled foundation for studying label-scarce Ethereum phishing detection. The original TensorFlow v1 graph architecture is retained through `tensorflow.compat.v1`.

## Current status

The code fixes corrupted amount features, count-bucket boundaries, address-embedding bucket gaps, singleton batches, the sampled pretraining objective, and errors previously treated as successful completion. It adds account-level validation, validation-only threshold selection, ranking metrics, pretrained and random frozen-encoder controls, full fine-tuning, a behavioural-feature control, three anomaly detectors, checkpoint provenance, and account embedding export.

**No new benchmark results are claimed.** The dataset and checkpoints are distributed separately and are not in this repository. Historical results must be rerun after regenerating data and retraining with the corrected configuration. AUC is threshold-independent; a classification threshold does not determine AUC.

## Environment

Python 3.11 is the verified runtime. Create a virtual environment, then install:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

The TensorFlow flow tests use a tiny synthetic dataset, run actual pretraining and both supervised modes, and check checkpoint restoration, frozen weights, amount features, split coverage, embedding extraction, and failure paths. They do not establish real-world detection performance.

## Data and configurations

Place the raw dataset in `Data/` and run commands from `Model/`. The raw filenames expected by `gen_seq.py` remain unchanged. Place the known phishing address list at `Data/phisher_account.txt`.

```bash
cd Model
python gen_seq.py --dataset=10M
python gen_pretrain_data.py --seed=12345
python gen_finetune_data.py --seed=12345
```

The fine-tuning generator creates a stratified 70/15/15 account split **before assigning windows** and writes:

- `data/finetune_train.tfrecord`
- `data/finetune_validation.tfrecord`
- `data/finetune_test.tfrecord`
- `data/finetune_splits.json`, containing account IDs, record counts, file/vocabulary hashes, seed, and preprocessing metadata.

Stratification uses known phishing labels. It must be disclosed as label-informed benchmark construction, even when downstream fitting uses no labels. The negative class means accounts without known phishing labels, not verified legitimate accounts. Small datasets without sufficient examples in both classes fail stratified splitting explicitly.

`zipzap_config.json` uses corrected contiguous half-open embedding buckets. **Its embedding table shapes differ from historical checkpoints.** Use it for new training. `zipzap_config.legacy.json` preserves the original ranges and explicitly permits their gaps for historical evaluation only. Do not load old weights with the corrected configuration or compare corrected runs as though they reproduce the old preprocessing.

Vocabulary IDs depend on the actual dataset. Check their maximum against the configuration. For a different vocabulary, run `python partitioning.py --vocab=data/vocab`, update `bucket_list` and `vocab_size` together, and retrain. The final upper bound is exclusive and `vocab_size` must exceed the maximum ID. Keep one factor dimension per bucket and the first factor dimension equal to `hidden_size` for this architecture.

Pretraining and downstream generation now both default to 100-token windows with repeat dropping disabled. This trains every learned position used downstream and keeps the sequence policy aligned. Both manifests record those settings, and checkpoint loading rejects a mismatch. `gen_seq.py` still aggregates repeated counterparties into three-day groups before bucketization. It saves a pre-bucket copy for the behavioural control, but this is already aggregated data and must not be described as untouched raw transactions.

## Pretraining and provenance

```bash
python run_pretrain.py --checkpointDir=ckpt_pretrain --seed=12345
python inspect_checkpoint.py ckpt_pretrain/model_STEP
```

The sampled loss uses each masked token's real vocabulary ID, removes accidental true-token negatives, subtracts log sampling probabilities, and has no positive-column bias. The default is 512 negatives rather than 5,000. Masking excludes the account self token and uses 80% `[MASK]`, 10% random-token, and 10% unchanged replacement. By default `gen_pretrain_data.py` pretrains on **all accounts** and writes no holdout, which is the setting for every downstream experiment. A masked-address loss on unseen accounts is available only as a separate diagnostic run: `python gen_pretrain_data.py --do_eval=True --data_dir=data_diag/` holds out 20% of accounts (split by account before windowing, so overlapping windows cannot cross the boundary) and writes `test.tfrecord` and `pretrain_split.json`; pass the same `--data_dir`, `--train_input_file`, `--test_input_file` and `--pretrain_manifest` to `run_pretrain.py`, then evaluate with `--do_train=False --do_eval=True`. A diagnostic encoder never trains on the held-out accounts, so their own-ID embeddings stay essentially untrained; its checkpoint provenance records `diagnostic_holdout: true` (inherited by fine-tuned descendants), and `extract_embeddings.py` refuses it unless `--allow-diagnostic-checkpoint` marks the export as a separate ablation. Do not mix diagnostic checkpoints with all-account checkpoints in one comparison. Evaluation reports the corrected sampled loss; it does not report the old candidate-column accuracy.

Replace `model_STEP` with an actual checkpoint prefix. Each newly saved checkpoint has a `.provenance.json` sidecar recording its objective and preprocessing, encoder label use, parent checkpoint, seed, architecture sharing, config/vocabulary hashes, and checkpoint shard hashes. Resuming pretraining from a supervised encoder retains supervised ancestry. The sidecar is an auditable record, not tamper-proof enforcement; retain training logs and dataset hashes with it.

The existing preprocessing builds the vocabulary and self-supervised training pool before the downstream account split, including accounts used later for evaluation. All current experiments are therefore **transductive**. The raw collection also explicitly adds known phishing accounts; this is not an untouched naturally sampled unlabelled population. No temporal or unseen-account generalization claim is supported. The vocabulary has no learned unknown-address policy; an inductive experiment needs separate preprocessing, time cutoffs, and an explicit address policy.

## Supervised baselines

Train and evaluate the full fine-tuned reference:

```bash
python run_finetune.py --init_checkpoint=ckpt_pretrain --checkpointDir=ckpt_finetuned --output_dir=project_results/finetuned --seed=12345
```

Train the MLP on frozen pretrained representations using the same data and encoder architecture:

```bash
python run_finetune.py --init_checkpoint=ckpt_pretrain --freeze_encoder=True --checkpointDir=ckpt_frozen --output_dir=project_results/frozen --seed=12345
```

Run the matching random frozen-encoder control by omitting the checkpoint:

```bash
python run_finetune.py --freeze_encoder=True --checkpointDir=ckpt_random_frozen --output_dir=project_results/random_frozen --seed=12345
```

A directory passed as `--init_checkpoint` resolves to its latest checkpoint. Keep pretrained and supervised output directories separate. Pretraining defaults to shared layers; downstream training defaults to independent layers initialized from those shared weights (`--load_cross=True`). For an unshared source checkpoint use `--load_cross=False`. Frozen mode disables encoder dropout, stops gradients, and restricts optimizer variables to the MLP.

Evaluation selects the threshold maximizing account-level validation F1 and reads the test file only after that threshold is fixed. Scores are averaged across an account's windows. The frozen MLP reference operates on window representations before averaging probabilities; the exporter below instead averages window embeddings. For a strict account-vector detector comparison, train the supervised head on those same exported vectors or apply every detector per window with the same score aggregation. Do not attribute an aggregation difference to detector choice. The split manifest checks file hashes, record counts, exact account coverage, and disjoint account sets. Unexpected errors propagate; partial evaluations do not produce successful metrics.

`metrics.json` includes Average Precision (AP), ROC-AUC, Precision@K, Recall@K, precision, recall, F1, confusion counts, false positives per 1,000 evaluated accounts, timing, provenance, and threshold label use. AP uses scikit-learn's average-precision definition, not trapezoidal PR area. K is `ceil(account_count * alert_fraction)` with `--alert_fraction=0.01` by default. Tied rankings use account ID order. Inference timing includes graph construction, checkpoint loading, scoring, and aggregation, so report that convention. Scores are exported to `test_scores.npz` with stable account IDs.

Repeat runs with several `--seed` values and distinct output directories; report variability. Keep the same split manifest across model seeds to measure training variability independently of split variability. Regenerating splits with different seeds is a separate experiment.

## Existing supplied checkpoints

Historical files can be placed in `Model/ckpt_dir/` and `Model/data/`. If only the old test TFRecord is available, use a threshold declared in advance; this cannot reproduce an unbiased validation-selected F1 without validation data:

```bash
python run_finetune.py --do_train=False --init_checkpoint=ckpt_dir/bert_finetune --load_cross=False --bert_config_file=zipzap_config.legacy.json --split_manifest= --threshold_protocol=fixed --fixed_threshold=0.5 --output_dir=project_results/legacy
```

This explicit legacy mode reports unverified split coverage and unknown checkpoint provenance if no manifest exists. Evaluation-only opens no training input. Do not use a historical test threshold sweep to select a new threshold. Historical amount corruption remains embedded in already generated records and learned weights; changing the writer does not repair them.

## Frozen embedding export

Export mean-pooled transaction representations from a verified pretrained encoder:

```bash
python extract_embeddings.py --checkpoint=ckpt_pretrain/model_STEP --input-file=data/finetune_train.tfrecord --split=train --output=project_results/train_embeddings.npz
```

Repeat with the validation and test files and matching `--split` values. Mean pooling excludes the account self token and padding, avoiding the self-ID lookup shortcut. Windows are combined in proportion to their transaction counts. `--l2-normalize` optionally normalizes final account vectors; the anomaly runner always fits its scaler on training data only. The NPZ stores account IDs, embeddings, and evaluation labels; extraction does not use labels to compute representations. Downstream unsupervised fitting must not consume the exported labels. Labels remain available for separately identified supervised baselines and evaluation.

The export sidecar records checkpoint ancestry, pooling, window aggregation, split hash, output hash, sequence preprocessing, and extraction time. Supervised or unknown encoders are rejected by default. `--allow-supervised-or-unknown` explicitly permits a separately identified ablation and does not mark its representations label-independent. All split exports for a comparison must use the same checkpoint, pooling, preprocessing, and split manifest.

## Behavioural and unsupervised baselines

Build account-level behavioural features from the pre-bucket aggregate and train the supervised tree control:

```bash
python extract_behavioral_features.py --output-dir=project_results/behavioral_features
python supervised_feature_baseline.py --train=project_results/behavioral_features/train.npz --validation=project_results/behavioral_features/validation.npz --test=project_results/behavioral_features/test.npz --output-dir=project_results/behavioral
```

Run an actual anomaly detector on the three embedding exports:

```bash
python anomaly_detection.py --train=project_results/train_embeddings.npz --validation=project_results/validation_embeddings.npz --test=project_results/test_embeddings.npz --method=isolation_forest --output-dir=project_results/isolation_forest
```

`--method` also accepts `kmeans` and `autoencoder`. Scaling, optional PCA, and optional L2 normalization are fitted on training vectors only. Mixed-unlabelled fitting ignores training labels. `--setting=one_class` selects accounts without known phishing labels using labels and records that use explicitly. The default train-percentile threshold does not use validation labels; `--threshold=validation_f1` is a separately recorded calibrated experiment.

The benchmark is positive-unlabelled: class 0 means “no known phishing label,” not verified legitimate. Accordingly, reported false positives are flags among accounts without known labels, and must not be presented as proven false accusations. Future claims still require repeated seeds, contamination sensitivity, and a temporal or unseen-account protocol with an unknown-address policy.
