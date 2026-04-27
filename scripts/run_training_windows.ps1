param(
    [ValidateSet("compute", "final", "single")]
    [string]$Mode = "compute",
    [string]$Config = "",
    [ValidateSet("cpu", "cuda", "mps")]
    [string]$Device = "cpu",
    [string]$PythonBin = "",
    [switch]$ResumeIfAvailable,
    [switch]$SilenceGymWarning
)

$ErrorActionPreference = "Stop"

function Resolve-PythonBin {
    param([string]$Candidate)

    if ($Candidate) {
        return $Candidate
    }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    throw "Could not find python. Activate your venv or pass -PythonBin."
}

function Invoke-Train {
    param(
        [string]$Py,
        [string]$Cfg,
        [string]$Dev,
        [string]$CheckpointPath,
        [bool]$Resume
    )

    $args = @("-m", "madreamer.cli.train", "--config", $Cfg, "--device", $Dev)
    if ($Resume -and $CheckpointPath -and (Test-Path $CheckpointPath)) {
        Write-Host "Resuming $Cfg from $CheckpointPath"
        $args += @("--resume", $CheckpointPath)
    }
    else {
        Write-Host "Starting $Cfg"
    }

    & $Py @args
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$py = Resolve-PythonBin -Candidate $PythonBin

if ($SilenceGymWarning) {
    Write-Host "Pinning setuptools<81 to reduce gym pkg_resources warning..."
    & $py -m pip install "setuptools<81"
}

if ($Mode -eq "single") {
    if (-not $Config) {
        throw "Mode 'single' requires -Config (for example, configs/compute/shared_h1_single_32k.yaml)."
    }
    Invoke-Train -Py $py -Cfg $Config -Dev $Device -CheckpointPath "" -Resume:$false
    Write-Host "Single training run complete."
    exit 0
}

if ($Mode -eq "compute") {
    $jobs = @(
        @{ Config = "configs/compute/shared_h1_single_32k.yaml"; Checkpoint = "artifacts/compute/shared-h1-single-32k/checkpoints/compute-shared-h1-single-32k_shared_latest.pt" },
        @{ Config = "configs/compute/shared_h3_single_64k.yaml"; Checkpoint = "artifacts/compute/shared-h3-single-64k/checkpoints/compute-shared-h3-single-64k_shared_latest.pt" },
        @{ Config = "configs/compute/shared_h3_multi_64k.yaml"; Checkpoint = "artifacts/compute/shared-h3-multi-64k/checkpoints/compute-shared-h3-multi-64k_shared_latest.pt" },
        @{ Config = "configs/compute/opponent_aware_h3_multi_64k.yaml"; Checkpoint = "artifacts/compute/opponent-aware-h3-multi-64k/checkpoints/compute-opponent-aware-h3-multi-64k_opponent_aware_latest.pt" }
    )

    foreach ($job in $jobs) {
        Invoke-Train -Py $py -Cfg $job.Config -Dev $Device -CheckpointPath $job.Checkpoint -Resume:$ResumeIfAvailable.IsPresent
    }

    Write-Host "Compute matrix complete."
    exit 0
}

$finalJobs = @(
    @{ Config = "configs/final/ppo_ffa.yaml"; Checkpoint = "artifacts/final/ppo-ffa/checkpoints/final-ppo-ffa_ppo_latest.pt" },
    @{ Config = "configs/final/independent_h3_ffa.yaml"; Checkpoint = "artifacts/final/independent-h3-ffa/checkpoints/final-independent-h3-ffa_independent_latest.pt" },
    @{ Config = "configs/final/shared_h3_ffa.yaml"; Checkpoint = "artifacts/final/shared-h3-ffa/checkpoints/final-shared-h3-ffa_shared_latest.pt" },
    @{ Config = "configs/final/opponent_aware_h3_ffa.yaml"; Checkpoint = "artifacts/final/opponent-aware-h3-ffa/checkpoints/final-opponent-aware-h3-ffa_opponent_aware_latest.pt" },
    @{ Config = "configs/final/shared_h1_ffa.yaml"; Checkpoint = "artifacts/final/shared-h1-ffa/checkpoints/final-shared-h1-ffa_shared_latest.pt" },
    @{ Config = "configs/final/shared_h5_ffa.yaml"; Checkpoint = "artifacts/final/shared-h5-ffa/checkpoints/final-shared-h5-ffa_shared_latest.pt" },
    @{ Config = "configs/final/team_shared_h3.yaml"; Checkpoint = "artifacts/final/team-shared-h3/checkpoints/final-team-shared-h3_shared_latest.pt" }
)

foreach ($job in $finalJobs) {
    Invoke-Train -Py $py -Cfg $job.Config -Dev $Device -CheckpointPath $job.Checkpoint -Resume:$ResumeIfAvailable.IsPresent
}

Write-Host "Final matrix complete."
