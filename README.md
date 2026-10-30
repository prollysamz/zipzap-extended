# LLM Applications in Finance: Ethereum Fraud Detection

This repository contains the replication code for detecting Ethereum fraud using a language model approach, based on the ZipZap paper architecture. 

## Requirements
* Python >= 3.6
* TensorFlow
* tf-keras (Required for legacy Keras layer compatibility)
* NumPy
* Pandas
* Scikit-Learn

```bash
pip install tensorflow tf-keras numpy pandas scikit-learn
```

## How to run the Pre-trained Evaluation Locally

If you have been sent the pre-trained weights (`ckpt_dir`) and test dataset (`data`) separately (they are not tracked in GitHub due to file size limits), follow these steps to evaluate the model on your machine:

### 1. Place the files in the correct directories
Extract the provided files directly into the `Model/` directory so the folder structure looks exactly like this:

```
ZipZap-main/
└── Model/
    ├── ckpt_dir/
    │   ├── bert_finetune.data-00000-of-00001
    │   ├── bert_finetune.index
    │   ├── bert_finetune.meta
    │   └── checkpoint
    ├── data/
    │   ├── finetune_test.tfrecord
    │   └── vocab
    ├── run_finetune.py
    └── ...
```

### 2. Run the evaluation script
Navigate into the `Model` directory and execute the evaluation command. 

> **Important note:** You must include `--load_cross=False` when running the evaluation, otherwise the script will attempt to search for a shared layer that was deactivated during our standard pre-training.

```bash
cd Model
python run_finetune.py --do_train=False --do_eval=True --init_checkpoint=ckpt_dir/bert_finetune --load_cross=False
```

### 3. Understanding the output
The script will loop through the `finetune_test.tfrecord` dataset. Once it reaches the end of the sequence, it will print out the final Area Under the Curve (AUC) metric and classification reports for various thresholds (0.05 to 0.95). 

A standard threshold of `0.5` should yield an AUC of **~0.9766**.

## Reproducing from the raw dataset on Windows

The supplied `Dataset/Dataset.gz` archive contains raw transaction CSV files, not the pre-generated TFRecords or trained checkpoints described above. Extract it at the repository root so that the CSV files are under `Data/`.

Create the pinned Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Generate sequences, vocabulary, and TFRecords:

```powershell
.\run_local.ps1 -Stage prepare
```

Then pre-train and fine-tune:

```powershell
.\run_local.ps1 -Stage pretrain
.\run_local.ps1 -Stage finetune -Checkpoint ckpt_dir/model_64000
```

To evaluate a separately supplied fine-tuned checkpoint:

```powershell
.\run_local.ps1 -Stage evaluate -Checkpoint ckpt_dir/bert_finetune
```

The paper used an NVIDIA RTX 3090 with 24 GB of GPU memory. Full preprocessing and training are compute-intensive; CPU-only execution is supported by TensorFlow but is not expected to reproduce the paper's training time. The raw archive does not include `Model/ckpt_dir`, so pre-trained evaluation cannot run until those checkpoint files are supplied or a model is trained locally.

### WSL2 GPU execution

TensorFlow 2.11 and newer do not support CUDA on native Windows. The included `requirements.txt` therefore installs the CPU package on Windows and the official `tensorflow[and-cuda]` extra on Linux/WSL.

This machine is configured with Python 3.11 and its environment at `/home/jakku/.venvs/llmfa` inside the `Ubuntu-24.04` WSL2 distribution. Verify GPU access from PowerShell:

```powershell
.\run_wsl.ps1 -Stage gpu
```

Run preparation and training from PowerShell:

```powershell
.\run_wsl.ps1 -Stage prepare
.\run_wsl.ps1 -Stage pretrain -BatchSize 32
.\run_wsl.ps1 -Stage finetune -BatchSize 32 -Checkpoint ckpt_dir/model_64000
```

The WSL preparation stage uses a disk-backed DuckDB transformation and streams
TFRecords without retaining the complete transaction graph or duplicated
training instances in RAM. Its database and generated data are stored under
`/home/jakku/llmfa-prep` and `/home/jakku/llmfa-data`, where substantially more
disk space is available than on the Windows drive. Add `-RebuildData` only when
the source CSVs or preparation logic have changed.

The default WSL batch size is 32 to fit an 8 GB RTX 4070 Laptop GPU. Reduce it further if TensorFlow reports an out-of-memory error.

### Stage 2 anomaly detection

Export the same 64-dimensional first-token representation consumed by the
supervised MLP, then compare Isolation Forest, K-Means distance and a compact
autoencoder:

```powershell
.\run_wsl.ps1 -Stage extract -Checkpoint ckpt_dir/bert_finetune -BatchSize 64
.\run_wsl.ps1 -Stage anomaly
```

Add `-CrossShare` when extracting from a pretraining checkpoint containing the
shared-layer encoder. Reports are written under
`/home/jakku/llmfa-anomaly-results`, and per-account scores are retained for
error analysis. Thresholds are selected on a validation subset of the training
accounts, not on the held-out test set.
