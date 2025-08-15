Param()

Set-Location -Path (Join-Path $PSScriptRoot "..")

# Load .env if present
$envPath = ".env"
if (Test-Path $envPath) {
  Get-Content $envPath | ForEach-Object {
    if ($_ -match "^\s*#") { return }
    if ($_ -match "^\s*$") { return }
    $parts = $_ -split "=", 2
    if ($parts.Length -eq 2) { [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1]) }
  }
}

if (-not $env:STORAGE_MODE) { $env:STORAGE_MODE = "local" }
if (-not $env:ASSETS_DIR) { $env:ASSETS_DIR = "../assets" }
if (-not $env:API_KEY) { $env:API_KEY = "demo-key-123" }
if (-not $env:API_HOST) { $env:API_HOST = "0.0.0.0" }
if (-not $env:API_PORT) { $env:API_PORT = "8080" }

Set-Location -Path "api"
python run_api.py
