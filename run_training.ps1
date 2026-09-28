param(
    [string]$PythonExe = "python",
    [string]$DataRoot = (Join-Path $PSScriptRoot "2026-低空图像语义分割赛道-训练集/train/train"),
    [string]$OutputDir = (Join-Path $PSScriptRoot "runs/b-tta-v1"),
    [int]$Epochs = 50,
    [int]$KeepBest = 3,
    [string]$Resume = "",
    [switch]$SkipFinalTta
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath (Join-Path $DataRoot "images"))) {
    throw "Dataset images directory not found: $DataRoot"
}
$trainingArgs = @(
    (Join-Path $PSScriptRoot "train.py"),
    "--data-root", $DataRoot, "--output", $OutputDir,
    "--epochs", "$Epochs", "--batch-size", "4", "--crop-size", "768",
    "--backbone", "resnet101", "--use-class-weights", "--patience", "10",
    "--keep-best", "$KeepBest", "--tta-batch-size", "2"
)
if ($Resume) { $trainingArgs += @("--resume", $Resume) }
if ($SkipFinalTta) { $trainingArgs += "--skip-final-tta" }

Write-Host "Ordinary val: no TTA; keep top $KeepBest candidates."
Write-Host "Final selection/test: scales 0.75,1.0,1.25 + horizontal flip; AMP."
& $PythonExe @trainingArgs
if ($LASTEXITCODE -ne 0) {
    throw "Training or final TTA selection failed. See the preceding error."
}
if ($SkipFinalTta) {
    Write-Host "Final selection deferred. Run select_tta.py using this output directory."
} else {
    Write-Host "Final submission checkpoint: $(Join-Path $OutputDir 'best_tta.pt')"
    Write-Host "predict.py automatically uses the inference settings saved in this checkpoint."
}

