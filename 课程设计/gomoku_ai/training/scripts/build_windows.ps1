$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectDir

$ToolPaths = @(
    "D:\CLion 2026.2.2\bin\cmake\win\x64\bin",
    "D:\CLion 2026.2.2\bin\ninja\win\x64",
    "D:\CLion 2026.2.2\bin\mingw\bin"
)
foreach ($p in $ToolPaths) {
    if (Test-Path $p) {
        $env:PATH = "$p;$env:PATH"
    }
}

$PythonExe = "D:\TinyStories\.venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Cannot find Python at $PythonExe"
}

Write-Output "[Build] Configuring CMake (Ninja + MinGW Release)..."
& cmake -S . -B build-win -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DBUILD_TESTING=OFF
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }

Write-Output "[Build] Building gomoku_engine..."
& cmake --build build-win --target gomoku_engine
if ($LASTEXITCODE -ne 0) { throw "CMake build failed ($LASTEXITCODE)" }

Write-Output "[Verify] Running PyTorch CUDA vs C11 DLL parity check (V2 NNUE+CNN)..."
& $PythonExe training\train_loop.py --verify-only
if ($LASTEXITCODE -ne 0) { throw "Verify V2 parity failed ($LASTEXITCODE)" }

Write-Output "[Verify] Running PyTorch vs C11 DLL parity check (V3 Pattern-NNUE GPN1)..."
& $PythonExe training\train_pnnue.py --verify-only
if ($LASTEXITCODE -ne 0) { throw "Verify V3 parity failed ($LASTEXITCODE)" }

Write-Output "[Verify] Running V3 self-play/arena smoke test..."
& $PythonExe training\train_pnnue.py --smoke-test
if ($LASTEXITCODE -ne 0) { throw "V3 smoke test failed ($LASTEXITCODE)" }

Write-Output "[Done] Windows engine DLL, V2/V3 parity and V3 smoke test verified."
