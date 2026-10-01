# MaskLayer backend launcher
# Run from repo root or from backend/ — works either way.

$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# Load .env from the repo root (one level up from backend/)
$repoRoot = Split-Path $PSScriptRoot -Parent
$envFile  = Join-Path $repoRoot ".env"

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            $key   = $matches[1].Trim()
            $value = $matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
    Write-Host "Loaded env from $envFile"
} else {
    Write-Warning ".env not found at $envFile — using system environment only."
}

Set-Location $PSScriptRoot
python -m uvicorn app.main:app --port 8000 --reload
