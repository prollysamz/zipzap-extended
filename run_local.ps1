param(
    [ValidateSet("prepare", "pretrain", "finetune", "evaluate")]
    [string]$Stage = "prepare",
    [string]$Checkpoint = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$model = Join-Path $root "Model"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

Push-Location $model
try {
    switch ($Stage) {
        "prepare" {
            & $python gen_seq.py --dataset=1M --data_dir=../Data
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            & $python gen_pretrain_data.py
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            & $python gen_finetune_data.py
        }
        "pretrain" {
            & $python run_pretrain.py
        }
        "finetune" {
            if (-not $Checkpoint) { throw "-Checkpoint is required for finetuning (for example, ckpt_dir/model_64000)." }
            & $python run_finetune.py --init_checkpoint=$Checkpoint --load_cross=False
        }
        "evaluate" {
            if (-not $Checkpoint) { throw "-Checkpoint is required for evaluation (for example, ckpt_dir/bert_finetune)." }
            & $python run_finetune.py --do_train=False --do_eval=True --init_checkpoint=$Checkpoint --load_cross=False
        }
    }

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
