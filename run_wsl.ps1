param(
    [ValidateSet("gpu", "prepare", "pretrain", "finetune", "evaluate", "extract", "anomaly", "label-free-data", "label-free-partition", "label-free-pretrain", "label-free-extract", "label-free-anomaly", "benchmark-data", "benchmark-pretrain", "benchmark-supervised", "benchmark-export", "benchmark-anomaly")]
    [string]$Stage = "gpu",
    [string]$Checkpoint = "",
    [ValidateRange(1, 256)]
    [int]$BatchSize = 32,
    [string]$Distro = "Ubuntu-24.04",
    [switch]$CrossShare,
    [switch]$RebuildData,
    [string]$Methods = "isolation_forest,kmeans,autoencoder",
    [string]$Settings = "mixed,benign_only",
    [ValidateSet("1M", "10M")]
    [string]$Dataset = "1M",
    [ValidateSet("isolation_forest", "kmeans", "autoencoder")]
    [string]$Method = "isolation_forest",
    [int]$Seed = 12345
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$wslRoot = (& wsl.exe -d $Distro --cd $root -- pwd).Trim()
$python = "/home/jakku/.venvs/llmfa/bin/python"
$modelDir = "$wslRoot/Model"
$dataDir = "/home/jakku/llmfa-data"
$prepWorkDir = "/home/jakku/llmfa-prep"
$anomalyDataDir = "/home/jakku/llmfa-anomaly-data"
$anomalyResultDir = "/home/jakku/llmfa-anomaly-results"
$correctedDataDir = "data"
$correctedConfig = "$correctedDataDir/zipzap_config.json"
$correctedManifest = "$correctedDataDir/pretrain_manifest.json"
$correctedCheckpointDir = "ckpt_pretrain"
$correctedEmbeddingFile = "project_results/unlabelled_embeddings.npz"
$correctedAnomalyDir = "project_results/isolation_forest"
$benchmarkDataDir = "data_benchmark"
$benchmarkConfig = "$benchmarkDataDir/zipzap_config.json"
$benchmarkManifest = "$benchmarkDataDir/pretrain_manifest.json"
$benchmarkSplitManifest = "$benchmarkDataDir/finetune_splits.json"
$benchmarkCheckpointDir = "ckpt_benchmark_pretrain"
$benchmarkResultDir = "project_results/benchmark"

if (-not $wslRoot) {
    throw "Could not translate the project path for WSL."
}

& wsl.exe -d $Distro -- test -x $python
if ($LASTEXITCODE -ne 0) {
    throw "WSL environment not found at $python. See the WSL GPU setup section in README.md."
}

switch ($Stage) {
    "gpu" {
        & wsl.exe -d $Distro -- $python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
    }
    "prepare" {
        $rebuildArgument = @()
        if ($RebuildData) { $rebuildArgument = @("--rebuild_db") }
        & wsl.exe -d $Distro --cd $modelDir -- $python prepare_streaming.py --data_dir=$wslRoot/Data --output_dir=$dataDir --work_dir=$prepWorkDir @rebuildArgument
    }
    # Corrected, label-free workflow.  These stages deliberately operate in Model/data
    # because the generated vocabulary, manifest, and vocabulary-specific config form one contract.
    "label-free-data" {
        & wsl.exe -d $Distro --cd $modelDir -- $python gen_seq.py --dataset=$Dataset --phisher=False
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & wsl.exe -d $Distro --cd $modelDir -- $python gen_pretrain_data.py --do_embed=True --seed=$Seed --data_dir="$correctedDataDir/"
    }
    "label-free-partition" {
        & wsl.exe -d $Distro --cd $modelDir -- $python partitioning.py --vocab=$correctedDataDir/vocab --config=zipzap_config.json --output-config=$correctedConfig
    }
    "label-free-pretrain" {
        & wsl.exe -d $Distro --cd $modelDir -- $python -u run_pretrain.py --bert_config_file=$correctedConfig --checkpointDir=$correctedCheckpointDir --data_dir="$correctedDataDir/" --train_input_file="$correctedDataDir/train*.tfrecord" --pretrain_manifest=$correctedManifest --batch_size=$BatchSize --seed=$Seed
    }
    "label-free-extract" {
        if (-not $Checkpoint) { throw "-Checkpoint is required for label-free embedding export (for example, ckpt_pretrain/model_STEP)." }
        & wsl.exe -d $Distro --cd $modelDir -- $python extract_embeddings.py --checkpoint=$Checkpoint --config=$correctedConfig --vocab=$correctedDataDir/vocab --input-file="$correctedDataDir/embed-*.tfrecord" --batch-size=$BatchSize --output=$correctedEmbeddingFile
    }
    "label-free-anomaly" {
        $outputDir = if ($Method -eq "isolation_forest") { $correctedAnomalyDir } else { "project_results/$Method" }
        & wsl.exe -d $Distro --cd $modelDir -- $python anomaly_detection.py --train=$correctedEmbeddingFile --method=$Method --seed=$Seed --output-dir=$outputDir
    }
    # Labelled benchmark corpus. Labels are used only to construct account-disjoint
    # splits, train supervised controls, and evaluate the held-out test set.
    "benchmark-data" {
        & wsl.exe -d $Distro --cd $modelDir -- $python gen_seq.py --dataset=$Dataset --phisher=True --output_dir=$benchmarkDataDir
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & wsl.exe -d $Distro --cd $modelDir -- $python gen_pretrain_data.py --do_embed=True --seed=$Seed --data_dir="$benchmarkDataDir/"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & wsl.exe -d $Distro --cd $modelDir -- $python partitioning.py --vocab=$benchmarkDataDir/vocab --config=zipzap_config.json --output-config=$benchmarkConfig
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & wsl.exe -d $Distro --cd $modelDir -- $python gen_finetune_data.py --seed=$Seed --data_dir="$benchmarkDataDir/"
    }
    "benchmark-pretrain" {
        & wsl.exe -d $Distro --cd $modelDir -- $python -u run_pretrain.py --bert_config_file=$benchmarkConfig --checkpointDir=$benchmarkCheckpointDir --data_dir="$benchmarkDataDir/" --train_input_file="$benchmarkDataDir/train*.tfrecord" --pretrain_manifest=$benchmarkManifest --batch_size=$BatchSize --seed=$Seed
    }
    "benchmark-supervised" {
        if (-not $Checkpoint) { throw "-Checkpoint is required for the benchmark supervised models (for example, ckpt_benchmark_pretrain/model_STEP)." }
        & wsl.exe -d $Distro --cd $modelDir -- $python -u run_finetune.py --bert_config_file=$benchmarkConfig --data_dir="$benchmarkDataDir/" --split_manifest=$benchmarkSplitManifest --train_input_file=$benchmarkDataDir/finetune_train.tfrecord --validation_input_file=$benchmarkDataDir/finetune_validation.tfrecord --test_input_file=$benchmarkDataDir/finetune_test.tfrecord --init_checkpoint=$Checkpoint --load_cross=True --checkpointDir=ckpt_benchmark_finetuned --output_dir=$benchmarkResultDir/supervised_finetuned --batch_size=$BatchSize --seed=$Seed
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & wsl.exe -d $Distro --cd $modelDir -- $python -u run_finetune.py --bert_config_file=$benchmarkConfig --data_dir="$benchmarkDataDir/" --split_manifest=$benchmarkSplitManifest --train_input_file=$benchmarkDataDir/finetune_train.tfrecord --validation_input_file=$benchmarkDataDir/finetune_validation.tfrecord --test_input_file=$benchmarkDataDir/finetune_test.tfrecord --init_checkpoint=$Checkpoint --load_cross=True --freeze_encoder=True --checkpointDir=ckpt_benchmark_frozen --output_dir=$benchmarkResultDir/supervised_frozen --batch_size=$BatchSize --seed=$Seed
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & wsl.exe -d $Distro --cd $modelDir -- $python -u run_finetune.py --bert_config_file=$benchmarkConfig --data_dir="$benchmarkDataDir/" --split_manifest=$benchmarkSplitManifest --train_input_file=$benchmarkDataDir/finetune_train.tfrecord --validation_input_file=$benchmarkDataDir/finetune_validation.tfrecord --test_input_file=$benchmarkDataDir/finetune_test.tfrecord --freeze_encoder=True --checkpointDir=ckpt_benchmark_random_frozen --output_dir=$benchmarkResultDir/supervised_random_frozen --batch_size=$BatchSize --seed=$Seed
    }
    "benchmark-export" {
        if (-not $Checkpoint) { throw "-Checkpoint is required for benchmark embedding export." }
        foreach ($split in @("train", "validation", "test")) {
            & wsl.exe -d $Distro --cd $modelDir -- $python extract_embeddings.py --checkpoint=$Checkpoint --config=$benchmarkConfig --vocab=$benchmarkDataDir/vocab --input-file=$benchmarkDataDir/finetune_$split.tfrecord --split-manifest=$benchmarkSplitManifest --split=$split --batch-size=$BatchSize --output=$benchmarkResultDir/embeddings/$split.npz
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
    }
    "benchmark-anomaly" {
        & wsl.exe -d $Distro --cd $modelDir -- $python anomaly_detection.py --train=$benchmarkResultDir/embeddings/train.npz --test=$benchmarkResultDir/embeddings/test.npz --method=$Method --seed=$Seed --output-dir=$benchmarkResultDir/$Method
    }
    "pretrain" {
        & wsl.exe -d $Distro --cd $modelDir -- $python -u run_pretrain_legacy.py --batch_size=$BatchSize --bert_config_file=zipzap_config.legacy.json --data_dir=$dataDir/ --train_input_file=$dataDir/train.tfrecord
    }
    "finetune" {
        if (-not $Checkpoint) { throw "-Checkpoint is required for finetuning (for example, ckpt_dir/model_64000)." }
        & wsl.exe -d $Distro --cd $modelDir -- $python -u run_finetune_legacy.py --batch_size=$BatchSize --bert_config_file=zipzap_config.legacy.json --init_checkpoint=$Checkpoint --load_cross=True --data_dir=$dataDir/ --train_input_file=$dataDir/finetune_train.tfrecord --test_input_file=$dataDir/finetune_test.tfrecord
    }
    "evaluate" {
        if (-not $Checkpoint) { throw "-Checkpoint is required for evaluation (for example, ckpt_dir/bert_finetune)." }
        & wsl.exe -d $Distro --cd $modelDir -- $python -u run_finetune_legacy.py --batch_size=$BatchSize --bert_config_file=zipzap_config.legacy.json --do_train=False --do_eval=True --init_checkpoint=$Checkpoint --load_cross=False --data_dir=$dataDir/ --train_input_file=$dataDir/finetune_train.tfrecord --test_input_file=$dataDir/finetune_test.tfrecord
    }
    "extract" {
        if (-not $Checkpoint) { throw "-Checkpoint is required for representation extraction." }
        $extra = @()
        if ($CrossShare) {
            $extra = @("--cross_share")
        } else {
            $extra = @("--include_classifier")
        }
        & wsl.exe -d $Distro --cd $modelDir -- $python extract_representations.py --input_file=$dataDir/finetune_train.tfrecord --checkpoint=$Checkpoint --bert_config_file=zipzap_config.legacy.json --output_file=$anomalyDataDir/train.npz --batch_size=$BatchSize @extra
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & wsl.exe -d $Distro --cd $modelDir -- $python extract_representations.py --input_file=$dataDir/finetune_test.tfrecord --checkpoint=$Checkpoint --bert_config_file=zipzap_config.legacy.json --output_file=$anomalyDataDir/test.npz --batch_size=$BatchSize @extra
    }
    "anomaly" {
        & wsl.exe -d $Distro --cd $modelDir -- $python -u run_anomaly_experiments.py --train_file=$anomalyDataDir/train.npz --test_file=$anomalyDataDir/test.npz --output_dir=$anomalyResultDir --methods=$Methods --settings=$Settings
    }
}

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
