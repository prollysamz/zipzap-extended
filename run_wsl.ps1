param(
    [ValidateSet("gpu", "prepare", "pretrain", "finetune", "evaluate", "extract", "anomaly")]
    [string]$Stage = "gpu",
    [string]$Checkpoint = "",
    [ValidateRange(1, 256)]
    [int]$BatchSize = 32,
    [string]$Distro = "Ubuntu-24.04",
    [switch]$CrossShare,
    [switch]$RebuildData,
    [string]$Methods = "isolation_forest,kmeans,autoencoder",
    [string]$Settings = "mixed,benign_only"
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
    "pretrain" {
        & wsl.exe -d $Distro --cd $modelDir -- $python -u run_pretrain.py --batch_size=$BatchSize --data_dir=$dataDir/ --train_input_file=$dataDir/train.tfrecord
    }
    "finetune" {
        if (-not $Checkpoint) { throw "-Checkpoint is required for finetuning (for example, ckpt_dir/model_64000)." }
        & wsl.exe -d $Distro --cd $modelDir -- $python -u run_finetune.py --batch_size=$BatchSize --init_checkpoint=$Checkpoint --load_cross=True --data_dir=$dataDir/ --train_input_file=$dataDir/finetune_train.tfrecord --test_input_file=$dataDir/finetune_test.tfrecord
    }
    "evaluate" {
        if (-not $Checkpoint) { throw "-Checkpoint is required for evaluation (for example, ckpt_dir/bert_finetune)." }
        & wsl.exe -d $Distro --cd $modelDir -- $python -u run_finetune.py --batch_size=$BatchSize --do_train=False --do_eval=True --init_checkpoint=$Checkpoint --load_cross=False --data_dir=$dataDir/ --train_input_file=$dataDir/finetune_train.tfrecord --test_input_file=$dataDir/finetune_test.tfrecord
    }
    "extract" {
        if (-not $Checkpoint) { throw "-Checkpoint is required for representation extraction." }
        $extra = @()
        if ($CrossShare) {
            $extra = @("--cross_share")
        } else {
            $extra = @("--include_classifier")
        }
        & wsl.exe -d $Distro --cd $modelDir -- $python extract_representations.py --input_file=$dataDir/finetune_train.tfrecord --checkpoint=$Checkpoint --output_file=$anomalyDataDir/train.npz --batch_size=$BatchSize @extra
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & wsl.exe -d $Distro --cd $modelDir -- $python extract_representations.py --input_file=$dataDir/finetune_test.tfrecord --checkpoint=$Checkpoint --output_file=$anomalyDataDir/test.npz --batch_size=$BatchSize @extra
    }
    "anomaly" {
        & wsl.exe -d $Distro --cd $modelDir -- $python -u run_anomaly_experiments.py --train_file=$anomalyDataDir/train.npz --test_file=$anomalyDataDir/test.npz --output_dir=$anomalyResultDir --methods=$Methods --settings=$Settings
    }
}

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
