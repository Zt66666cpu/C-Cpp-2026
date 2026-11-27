$ErrorActionPreference = "SilentlyContinue"
$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

$Killed = 0
$Procs = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and (
        $_.CommandLine -match 'train_pnnue\.py|gomoku_reporter\.py' -or
        ($_.CommandLine -match 'multiprocessing\.spawn' -and $_.CommandLine -match 'D:\\GomokuAI')
    )
}

foreach ($p in $Procs) {
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    $Killed++
}

# Also terminate orphaned spawn workers whose parent was train_pnnue.py
$SpawnWorkers = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'from multiprocessing\.spawn import spawn_main'
}
foreach ($w in $SpawnWorkers) {
    $Parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($w.ParentProcessId)" -ErrorAction SilentlyContinue
    if (-not $Parent -or $Parent.CommandLine -match 'train_pnnue\.py') {
        Stop-Process -Id $w.ProcessId -Force -ErrorAction SilentlyContinue
        $Killed++
    }
}

Write-Output "[Stop V3] Stopped $Killed background training/worker process(es). Preserved models and checkpoints under $ProjectDir."
