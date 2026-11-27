param(
    [int]$Rounds = 10,
    [int]$GamesPerRound = 480,
    [int]$Workers = 16,
    [int]$Depth = 5,
    [int]$VcfDepth = 14,
    [double]$TimeLimit = 0.9,
    [int]$EvalGames = 64,
    [int]$EvalDepth = 4,
    [double]$EvalTime = 0.8,
    [int]$NnueEpochs = 24,
    [int]$CnnEpochs = 10,
    [string]$NnueLr = "5e-4",
    [string]$CnnLr = "6e-4",
    [int]$BatchSize = 512,
    [int]$MaxBuffer = 120000,
    [double]$GateThresh = 0.515,
    [string]$TeacherBin = "",
    [int]$TeacherGames = 120,
    [int]$TeacherTimeMs = 100,
    [int]$TeacherEvalGames = 20,
    [int]$TeacherEvalMs = 80,
    [int]$TeacherWorkers = 8,
    [switch]$Resume,
    [string]$MonitorUrl = "http://192.168.137.4:8766",
    [string]$RunName = "gomoku-v2"
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

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "training\runs") | Out-Null

$TrainLog = Join-Path $ProjectDir "training\runs\train_win4070.log"
$TrainErr = Join-Path $ProjectDir "training\runs\train_win4070.err.log"
$RepLog   = Join-Path $ProjectDir "training\runs\reporter_win4070.log"
$RepErr   = Join-Path $ProjectDir "training\runs\reporter_win4070.err.log"

$TrainArgs = @(
    "-u", "training\train_loop.py",
    "--rounds", "$Rounds",
    "--games-per-round", "$GamesPerRound",
    "--workers", "$Workers",
    "--depth", "$Depth",
    "--vcf-depth", "$VcfDepth",
    "--time-limit", "$TimeLimit",
    "--eval-games", "$EvalGames",
    "--eval-depth", "$EvalDepth",
    "--eval-time", "$EvalTime",
    "--nnue-epochs", "$NnueEpochs",
    "--cnn-epochs", "$CnnEpochs",
    "--batch-size", "$BatchSize",
    "--nnue-lr", "$NnueLr",
    "--cnn-lr", "$CnnLr",
    "--max-buffer", "$MaxBuffer",
    "--nnue-h1", "128",
    "--nnue-h2", "64",
    "--ch", "48",
    "--blocks", "4",
    "--gate-thresh", "$GateThresh"
)
if ($TeacherBin -and (Test-Path $TeacherBin)) {
    $TrainArgs += @(
        "--teacher-bin", "$TeacherBin",
        "--teacher-games", "$TeacherGames",
        "--teacher-time-ms", "$TeacherTimeMs",
        "--teacher-eval-games", "$TeacherEvalGames",
        "--teacher-eval-ms", "$TeacherEvalMs",
        "--teacher-workers", "$TeacherWorkers"
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
    "--phase", "v2",
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

$RepBat = Join-Path $ProjectDir "training\runs\launch_reporter.bat"
$TrainBat = Join-Path $ProjectDir "training\runs\launch_train.bat"

$RepPid = Start-DetachedJob -BatPath $RepBat -ArgList $RepArgs -StdOut $RepLog -StdErr $RepErr
$TrainPid = Start-DetachedJob -BatPath $TrainBat -ArgList $TrainArgs -StdOut $TrainLog -StdErr $TrainErr

Write-Output "Started Gomoku V2 training on Windows 4070:"
Write-Output "  Train Shell PID    : $TrainPid"
Write-Output "  Reporter Shell PID : $RepPid"
Write-Output "  Teacher Bin        : $TeacherBin"
Write-Output "  Train Log          : $TrainLog"
Write-Output "  Monitor URL        : $MonitorUrl"
