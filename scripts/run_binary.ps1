<#
.SYNOPSIS
    One-shot pipeline for everything related to the BINARY task
    (NORMAL vs PNEUMONIA) for the final delivery.

.DESCRIPTION
    Runs, in order:
      1. Train custom_cnn  (baseline) for seeds 0,1,2
      2. Train resnet18    (transfer) for seeds 0,1,2
      3. Train densenet121 (transfer) for seeds 0,1,2
      4. Evaluate every checkpoint on the test split
      5. Binary ensemble (resnet18 + densenet121)
      6. Aggregate result tables (final_results.csv / .md)
      7. Generate plots (confusion matrices, training curves, comparison)
      8. Generate Grad-CAM overlays (densenet121 + resnet18, binary)
      9. Build a clean per-epoch Markdown training log (results/run_binary.md)

    Output is produced in two places at once:
      - the FULL raw terminal output is saved to a timestamped .log under
        results/logs/ (via Tee-Object, so it also shows live in the terminal);
      - a clean per-epoch summary table is written to results/run_binary.md.

    By default a model is SKIPPED if its checkpoint already exists; pass
    -Force to retrain everything.

.PARAMETER Force
    Retrain all models from scratch even if checkpoints already exist.

.PARAMETER Seeds
    Seeds to train. Default: 0,1,2.

.EXAMPLE
    .\scripts\run_binary.ps1
    .\scripts\run_binary.ps1 -Force
#>
param(
    [switch]$Force,
    [int[]]$Seeds = @(0, 1, 2)
)

$ErrorActionPreference = "Continue"

# --- Locate project root (parent of this script's folder) and move there ---
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

# --- Pick the Python interpreter (prefer the project venv) ---
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Config        = "configs/default.yaml"
$CheckpointDir = "results/checkpoints"
$LogDir        = "results/logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp   = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$LogFile = Join-Path $LogDir "binary_run_$Stamp.log"

# --- Helpers ------------------------------------------------------------
function Section($title) {
    $bar = "=" * 70
    Write-Output ""
    Write-Output $bar
    Write-Output "  $title"
    Write-Output $bar
}

# Tracks pass/fail per step for the final summary.
$Script:Summary = New-Object System.Collections.Generic.List[string]

function Invoke-Step($label, [scriptblock]$action) {
    Write-Output ""
    Write-Output ">>> $label"
    $start = Get-Date
    # 2>&1 merges the child process's stderr into stdout as plain text, so
    # PowerShell does not render it as red "errors" (Python's tqdm bars and
    # library warnings go to stderr). $LASTEXITCODE below still reflects the
    # real exit code, so genuine failures are still detected as FAIL.
    & $action 2>&1
    $code = $LASTEXITCODE
    $secs = [int]((Get-Date) - $start).TotalSeconds
    if ($code -eq 0 -or $null -eq $code) {
        Write-Output "    OK  ($secs s)  -- $label"
        $Script:Summary.Add("OK    $label  ($secs s)")
    } else {
        Write-Output "    FAIL (exit $code, $secs s)  -- $label"
        $Script:Summary.Add("FAIL  $label  (exit $code)")
    }
}

function Start-TrainIfNeeded($model, $seed, [scriptblock]$trainCmd) {
    $ckpt = Join-Path $CheckpointDir "${model}_seed${seed}.pt"
    if ((Test-Path $ckpt) -and (-not $Force)) {
        Write-Output ">>> SKIP train $model seed $seed (checkpoint exists: $ckpt). Use -Force to retrain."
        $Script:Summary.Add("SKIP  train $model seed $seed (exists)")
        return
    }
    Invoke-Step "train $model seed $seed" $trainCmd
}

# --- The actual pipeline (wrapped so we can Tee everything to a file) ---
function Invoke-Pipeline {
    Section "BINARY PIPELINE  (NORMAL vs PNEUMONIA)"
    Write-Output "Project root : $ProjectRoot"
    Write-Output "Python       : $Python"
    Write-Output "Config       : $Config"
    Write-Output "Seeds        : $($Seeds -join ', ')"
    Write-Output "Force        : $Force"
    Write-Output "Log file     : $LogFile"
    Write-Output "Started      : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    # 1-3. Training
    Section "1. TRAINING"
    foreach ($seed in $Seeds) {
        Start-TrainIfNeeded "custom_cnn" $seed { & $Python scripts/train_baseline.py --config $Config --seed $seed }
    }
    foreach ($model in @("resnet18", "densenet121")) {
        foreach ($seed in $Seeds) {
            Start-TrainIfNeeded $model $seed { & $Python scripts/train_transfer.py --model $model --config $Config --seed $seed }
        }
    }

    # 4. Evaluation
    Section "2. EVALUATION (test split)"
    foreach ($model in @("custom_cnn", "resnet18", "densenet121")) {
        foreach ($seed in $Seeds) {
            $ckpt = Join-Path $CheckpointDir "${model}_seed${seed}.pt"
            if (Test-Path $ckpt) {
                Invoke-Step "evaluate $model seed $seed" { & $Python scripts/evaluate_model.py --model $model --checkpoint $ckpt --seed $seed }
            } else {
                Write-Output ">>> SKIP evaluate $model seed $seed (no checkpoint)"
                $Script:Summary.Add("SKIP  evaluate $model seed $seed (no checkpoint)")
            }
        }
    }

    # 5. Ensemble
    Section "3. ENSEMBLE (resnet18 + densenet121)"
    Invoke-Step "binary ensemble" { & $Python scripts/ensemble.py --config $Config }

    # 6. Aggregate tables
    Section "4. RESULT TABLES"
    Invoke-Step "make_result_tables (binary)" { & $Python scripts/make_result_tables.py --task binary }

    # 7. Plots
    Section "5. PLOTS"
    Invoke-Step "generate_plots_binary" { & $Python scripts/generate_plots_binary.py }

    # 8. Grad-CAM (both models)
    Section "6. GRAD-CAM (densenet121 + resnet18)"
    Invoke-Step "grad-cam densenet121" { & $Python scripts/generate_gradcam.py --model densenet121 --checkpoint (Join-Path $CheckpointDir "densenet121_seed0.pt") }
    Invoke-Step "grad-cam resnet18"    { & $Python scripts/generate_gradcam.py --model resnet18    --checkpoint (Join-Path $CheckpointDir "resnet18_seed0.pt") }

    # 9. Clean per-epoch Markdown training log (from this run's training logs)
    Section "7. MARKDOWN TRAINING LOG"
    Invoke-Step "log_to_markdown" { & $Python scripts/log_to_markdown.py $LogFile --output results/run_binary.md --title "Binary Training Log" }

    # --- Summary ---
    Section "SUMMARY"
    foreach ($line in $Script:Summary) { Write-Output "  $line" }
    Write-Output ""
    Write-Output "Finished     : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Output "Full log     : $LogFile"
    Write-Output "Epoch table  : results/run_binary.md"
}

# Run the pipeline, mirroring all output to the terminal AND the log file.
# PowerShell 5.1's Tee-Object lacks -Encoding; use Out-File pipeline instead.
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Invoke-Pipeline *>&1 | Tee-Object -FilePath $LogFile
