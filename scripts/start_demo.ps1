param([switch]$NoTelegram)
$ErrorActionPreference = 'Stop'
$cameraDemoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $cameraDemoRoot
try {
    $cameraDemoHealth = Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 2
    if ($cameraDemoHealth.model_loaded -eq $true) {
        Write-Host 'Backend ready: http://127.0.0.1:8000'
        return
    }
    throw 'An existing backend has no loaded model. Stop its terminal and check MODEL_PATH before restarting.'
} catch {
    if ($_.Exception.Message -like '*existing backend*') { throw }
}
if ($NoTelegram) { $env:TELEGRAM_ENABLED = 'false' } else { Remove-Item Env:TELEGRAM_ENABLED -ErrorAction SilentlyContinue }
$env:LATENCY_DB_PATH = Join-Path $cameraDemoRoot 'results\demo\latency.db'
& '.\.venv\Scripts\python.exe' -m uvicorn server.main:app --host 0.0.0.0 --port 8000
exit $LASTEXITCODE
