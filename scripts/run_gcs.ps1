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

if (-not $env:BUCKET_NAME) {
  Write-Error "BUCKET_NAME is required (set it in .env or export it)"
  exit 1
}

$env:STORAGE_MODE = "gcs"
if (-not $env:API_HOST) { $env:API_HOST = "0.0.0.0" }
if (-not $env:API_PORT) { $env:API_PORT = "8080" }

Set-Location -Path "api"
python run_api.py
