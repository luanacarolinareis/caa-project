<#
.SYNOPSIS
    One-shot pipeline for everything related to the INNOVATIONS for the
    final delivery: three-class classification (NORMAL / BACTERIA / VIRUS),
    ensemble, calibration analysis and Grad-CAM.

.DESCRIPTION
    Runs, in order:
      1. Train resnet18    (3-class) for seeds 0,1,2
      2. Train densenet121 (3-class) for seeds 0,1,2
      3. Evaluate every 3-class checkpoint on the test split
      3b. Hierarchical two-stage (densenet121): train Stage B (bacteria vs
          virus) and evaluate it combined with the existing binary Stage A,
          for a direct comparison against the flat 3-class softmax.
      4. Three-class ensemble (resnet18 + densenet121)
      5. Aggregate result tables (final_3class_results.csv / .md)
      6. Calibration analysis (ECE, reliability diagrams, thresholds)
      7. Generate plots (confusion matrices, training curves, comparison)
      8. Generate Grad-CAM overlays (densenet121 + resnet18, 3-class)
      9. Build a clean per-epoch Markdown training log (results/run_innovation.md)

    NOTE: the hierarchical Stage A reuses the BINARY NORMAL-vs-PNEUMONIA
    checkpoints (densenet121_seed*.pt). If those do not exist yet, run
    run_binary.ps1 first; otherwise the hierarchical eval step is skipped.

    The improved 3-class pipeline uses a patient-aware stratified validation
    split, inverse-frequency class weights and F1-macro checkpoint selection
    (configured in configs/three_class.yaml).

    Output is produced in two places at once:
      - the FULL raw terminal output is saved to a timestamped .log under
        results/logs/ (via Tee-Object, so it also shows live in the terminal);
      - a clean per-epoch summary table is written to results/run_innovation.md.

    By default a model is SKIPPED if its checkpoint already exists; pass
    -Force to retrain everything.

.PARAMETER Force
    Retrain all 3-class models from scratch even if checkpoints exist.

.PARAMETER Seeds
    Seeds to train. Default: 0,1,2.

.EXAMPLE
    .\scripts\run_innovation.ps1
    .\scripts\run_innovation.ps1 -Force
#>
param(
    [switch]$Force,
    [int[]]$Seeds = @(0, 1, 2)
)

$ErrorActionPreference = "Continue"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Config        = "configs/three_class.yaml"
$CheckpointDir = "results/checkpoints"
$LogDir        = "results/logs"
$Models        = @("resnet18", "densenet121")
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp   = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$LogFile = Join-Path $LogDir "innovation_run_$Stamp.log"

# --- Helpers ------------------------------------------------------------
function Section($title) {
    $bar = "=" * 70
    Write-Output ""
    Write-Output $bar
    Write-Output "  $title"
    Write-Output $bar
}

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
    $ckpt = Join-Path $CheckpointDir "${model}_3class_seed${seed}.pt"
    if ((Test-Path $ckpt) -and (-not $Force)) {
        Write-Output ">>> SKIP train $model 3class seed $seed (checkpoint exists: $ckpt). Use -Force to retrain."
        $Script:Summary.Add("SKIP  train $model 3class seed $seed (exists)")
        return
    }
    Invoke-Step "train $model 3class seed $seed" $trainCmd
}

# --- The actual pipeline ------------------------------------------------
function Invoke-Pipeline {
    Section "INNOVATION PIPELINE  (3-class + ensemble + calibration + grad-cam)"
    Write-Output "Project root : $ProjectRoot"
    Write-Output "Python       : $Python"
    Write-Output "Config       : $Config"
    Write-Output "Seeds        : $($Seeds -join ', ')"
    Write-Output "Force        : $Force"
    Write-Output "Log file     : $LogFile"
    Write-Output "Started      : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    # 1-2. Training
    Section "1. TRAINING (3-class)"
    foreach ($model in $Models) {
        foreach ($seed in $Seeds) {
            Start-TrainIfNeeded $model $seed { & $Python scripts/train_three_class.py --model $model --config $Config --seed $seed }
        }
    }

    # 3. Evaluation
    Section "2. EVALUATION (test split)"
    foreach ($model in $Models) {
        foreach ($seed in $Seeds) {
            $ckpt = Join-Path $CheckpointDir "${model}_3class_seed${seed}.pt"
            if (Test-Path $ckpt) {
                Invoke-Step "evaluate $model 3class seed $seed" { & $Python scripts/evaluate_three_class.py --model $model --checkpoint $ckpt --seed $seed }
            } else {
                Write-Output ">>> SKIP evaluate $model 3class seed $seed (no checkpoint)"
                $Script:Summary.Add("SKIP  evaluate $model 3class seed $seed (no checkpoint)")
            }
        }
    }

    # 3b. Hierarchical (two-stage) approach: Stage A reuses the existing binary
    # NORMAL-vs-PNEUMONIA checkpoints; Stage B (bacteria vs virus) is trained
    # here. Only DenseNet121 (the primary model). This mirrors Kermany et al.
    Section "2b. HIERARCHICAL TWO-STAGE (densenet121)"
    $hierModel = "densenet121"
    foreach ($seed in $Seeds) {
        $ckptB = Join-Path $CheckpointDir "${hierModel}_bactvirus_seed${seed}.pt"
        if ((Test-Path $ckptB) -and (-not $Force)) {
            Write-Output ">>> SKIP train $hierModel Stage B seed $seed (checkpoint exists: $ckptB). Use -Force to retrain."
            $Script:Summary.Add("SKIP  train $hierModel Stage B seed $seed (exists)")
        } else {
            Invoke-Step "train $hierModel Stage B (bacteria/virus) seed $seed" { & $Python scripts/train_stage_b.py --model $hierModel --seed $seed }
        }
    }
    foreach ($seed in $Seeds) {
        $stageA = Join-Path $CheckpointDir "${hierModel}_seed${seed}.pt"
        $stageB = Join-Path $CheckpointDir "${hierModel}_bactvirus_seed${seed}.pt"
        if ((Test-Path $stageA) -and (Test-Path $stageB)) {
            Invoke-Step "evaluate hierarchical $hierModel seed $seed" { & $Python scripts/evaluate_hierarchical.py --model $hierModel --seed $seed }
        } else {
            $missing = if (-not (Test-Path $stageA)) { "Stage A binary checkpoint $stageA" } else { "Stage B checkpoint $stageB" }
            Write-Output ">>> SKIP hierarchical eval $hierModel seed $seed (missing $missing)"
            Write-Output "    NOTE: Stage A reuses the BINARY checkpoints. Run run_binary.ps1 first if they are missing."
            $Script:Summary.Add("SKIP  hierarchical eval $hierModel seed $seed (missing checkpoint)")
        }
    }

    # 4. Ensemble
    Section "3. ENSEMBLE (3-class)"
    Invoke-Step "3-class ensemble" { & $Python scripts/ensemble.py --config $Config }

    # 5. Aggregate tables
    Section "4. RESULT TABLES"
    Invoke-Step "make_result_tables (three_class)" { & $Python scripts/make_result_tables.py --task three_class }

    # 6. Calibration
    # NOTE: calibration_analysis.py writes a single combined calibration_results.json
    # and reliability_diagrams.png covering ALL models (binary + 3-class) by reading
    # the saved *_probs.json files. We run it with its default full model list so the
    # 3-class entries are refreshed WITHOUT clobbering the binary ones.
    Section "5. CALIBRATION ANALYSIS (all models, combined)"
    Invoke-Step "calibration_analysis" { & $Python scripts/calibration_analysis.py --seeds $Seeds }

    # 7. Plots
    Section "6. PLOTS"
    Invoke-Step "generate_plots_three_class" { & $Python scripts/generate_plots_three_class.py }

    # 8. Grad-CAM (both models)
    Section "7. GRAD-CAM (densenet121 + resnet18, 3-class)"
    Invoke-Step "grad-cam densenet121 3class" { & $Python scripts/generate_gradcam.py --config $Config --model densenet121 --checkpoint (Join-Path $CheckpointDir "densenet121_3class_seed0.pt") }
    Invoke-Step "grad-cam resnet18 3class"    { & $Python scripts/generate_gradcam.py --config $Config --model resnet18    --checkpoint (Join-Path $CheckpointDir "resnet18_3class_seed0.pt") }

    # 9. Clean per-epoch Markdown training log (from this run's training logs)
    Section "8. MARKDOWN TRAINING LOG"
    Invoke-Step "log_to_markdown" { & $Python scripts/log_to_markdown.py $LogFile --output results/run_innovation.md --title "Innovation (3-class) Training Log" }

    # --- Summary ---
    Section "SUMMARY"
    foreach ($line in $Script:Summary) { Write-Output "  $line" }
    Write-Output ""
    Write-Output "Finished     : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Output "Full log     : $LogFile"
    Write-Output "Epoch table  : results/run_innovation.md"
}

# PowerShell 5.1's Tee-Object lacks -Encoding; use Out-File pipeline instead.
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Invoke-Pipeline *>&1 | Tee-Object -FilePath $LogFile
