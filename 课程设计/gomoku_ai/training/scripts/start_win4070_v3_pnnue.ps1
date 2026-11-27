param(
    [int]$Rounds = 8,
    [int]$GamesPerRound = 480,
    [int]$Workers = 16,
    [int]$Depth = 6,
    [int]$VcfDepth = 14,
    [double]$TimeLimit = 0.8,
    [int]$Epochs = 4,
    [int]$BatchSize = 2048,
    [string]$Lr = "1.5e-4",
    [string]$PretrainLr = "6e-4",
    [string]$WeightDecay = "1e-3",
    [int]$Dim = 64,
    [int]$VHidden = 32,
    [int]$MaxBuffer = 220000,
    [int]$EvalGames = 64,
    [int]$EvalDepth = 5,
    [double]$EvalTime = 0.7,
    [double]$GateThresh = 0.505,
    [string]$PretrainBuffer = "training/runs_v3/rapfi_distill.npz",
    [string]$DistillBuffer = "training/runs_v3/rapfi_distill_3000.npz",
    [int]$PretrainEpochs = 24,
    [switch]$ForcePretrain,
    [string]$TeacherBin = "",
    [int]$TeacherGames = 200,
    [int]$TeacherTimeMs = 60,
    [switch]$Resume = $true,
    [string]$MonitorUrl = "http://192.168.137.4:8766",
    [string]$RunName = "gomoku-v3-pnnue"
)

$ErrorActionPreference = "Stop"
$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectDir

$PythonExe = "D:\TinyStories\.venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Cannot find Python at $PythonExe"
}

if ([string]::IsNullOrWhiteSpace($TeacherBin)) {
    $Candidates = @(
        (Join-Path $ProjectDir "teacher\rapfi\pbrain-rapfi-windows-avxvnni.exe"),
        (Join-Path $ProjectDir "teacher\rapfi\pbrain-rapfi-windows-avx2.exe"),
        (Join-Path $ProjectDir "teacher\rapfi\pbrain-rapfi-windows-sse.exe")
    )
    foreach ($c in $Candidates) {
        if (Test-Path $c) {
            $TeacherBin = $c
            break
        }
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "training\runs_v3") | Out-Null

$TrainLog = Join-Path $ProjectDir "training\runs_v3\train_v3_win4070.log"
$TrainErr = Join-Path $ProjectDir "training\runs_v3\train_v3_win4070.err.log"
$RepLog   = Join-Path $ProjectDir "training\runs_v3\reporter_v3_win4070.log"
$RepErr   = Join-Path $ProjectDir "training\runs_v3\reporter_v3_win4070.err.log"

$TrainArgs = @(
    "-u", "training\train_pnnue.py",
    "--rounds", "$Rounds",
    "--games-per-round", "$GamesPerRound",
    "--workers", "$Workers",
    "--depth", "$Depth",
    "--vcf-depth", "$VcfDepth",
    "--time-limit", "$TimeLimit",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--lr", "$Lr",
    "--pretrain-lr", "$PretrainLr",
    "--weight-decay", "$WeightDecay",
    "--dim", "$Dim",
    "--v-hidden", "$VHidden",
    "--max-buffer", "$MaxBuffer",
    "--eval-games", "$EvalGames",
    "--eval-depth", "$EvalDepth",
    "--eval-time", "$EvalTime",
    "--gate-thresh", "$GateThresh"
)
$PretrainList = @()
foreach ($pb in @($PretrainBuffer, $DistillBuffer)) {
    if ($pb -and (Test-Path (Join-Path $ProjectDir $pb))) {
        $PretrainList += $pb
    } elseif ($pb) {
        Write-Output "[V3] Pretrain buffer not found, skipped: $pb"
    }
}
foreach ($pb in $PretrainList) {
    $TrainArgs += @("--pretrain-buffer", "$pb")
}
if ($PretrainList.Count -gt 0) {
    $TrainArgs += @("--pretrain-epochs", "$PretrainEpochs")
}
if ($ForcePretrain) {
    $TrainArgs += "--force-pretrain"
}
if ($TeacherBin -and (Test-Path $TeacherBin) -and $TeacherGames -gt 0) {
    $TrainArgs += @(
        "--teacher-bin", "$TeacherBin",
        "--teacher-games", "$TeacherGames",
        "--teacher-time-ms", "$TeacherTimeMs"
    )
}
if ($Resume) {
    $TrainArgs += "--resume"
}

$RepArgs = @(
    "-u", "training\gomoku_reporter.py",
    "--monitor", "$MonitorUrl",
    "--machine", "windows-4070",
    "--run", "$RunName",
    "--phase", "v3_pnnue",
    "--project-dir", "$ProjectDir",
    "--log-file", "$TrainLog",
    "--interval", "3"
)

function Start-DetachedJob {
    param(
        [string]$BatPath,
        [string[]]$ArgList,
        [string]$StdOut,
        [string]$StdErr
    )
    $EscapedArgs = ($ArgList | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + $_ + '"' } else { $_ }
    }) -join " "
    $BatLines = @(
        "@echo off",
        "cd /d `"$ProjectDir`"",
        "`"$PythonExe`" $EscapedArgs > `"$StdOut`" 2> `"$StdErr`""
    )
    [System.IO.File]::WriteAllLines($BatPath, $BatLines, [System.Text.UTF8Encoding]::new($false))
    $Res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = "cmd.exe /c `"$BatPath`""
        CurrentDirectory = $ProjectDir
    }
    if ($Res.ReturnValue -ne 0) {
        throw "Win32_Process.Create failed with ReturnValue=$($Res.ReturnValue)"
    }
    return $Res.ProcessId
}

$RepBat = Join-Path $ProjectDir "training\runs_v3\launch_reporter_v3.bat"
$TrainBat = Join-Path $ProjectDir "training\runs_v3\launch_train_v3.bat"

$RepPid = Start-DetachedJob -BatPath $RepBat -ArgList $RepArgs -StdOut $RepLog -StdErr $RepErr
$TrainPid = Start-DetachedJob -BatPath $TrainBat -ArgList $TrainArgs -StdOut $TrainLog -StdErr $TrainErr

Write-Output "Started Gomoku V3 Pattern-NNUE training on Windows 4070:"
Write-Output "  Train Shell PID    : $TrainPid"
Write-Output "  Reporter Shell PID : $RepPid"
Write-Output "  Pretrain Buffers   : $($PretrainList -join ', ')"
Write-Output "  Train Log          : $TrainLog"
Write-Output "  Monitor URL        : $MonitorUrl"
