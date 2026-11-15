# Swaps corrected frames into out\frames (originals are kept in out\frames_orig).
# Usage: pwsh -NoProfile -File tools\apply_fix.ps1 <start> <end>     (only if EVERY frame of the range exists in out\frames_fix)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$s = [int]$args[0]; $e = [int]$args[1]
$missing = @(); for ($f = $s; $f -le $e; $f++) { $p = "out\frames_fix\frame_{0:d4}.png" -f $f; if (-not (Test-Path $p) -or (Get-Item $p).Length -lt 3MB) { $missing += $f } }
if ($missing.Count -gt 0) { "NOT APPLIED: $($missing.Count) frames missing in out\frames_fix (first: $($missing[0]))"; exit 1 }
New-Item -ItemType Directory -Force out\frames_orig | Out-Null
for ($f = $s; $f -le $e; $f++) {
    $n = "frame_{0:d4}.png" -f $f
    if (-not (Test-Path "out\frames_orig\$n")) { Move-Item "out\frames\$n" "out\frames_orig\$n" }
    Copy-Item "out\frames_fix\$n" "out\frames\$n" -Force
}
"applied $s..$e ; frames in out\frames: " + (Get-ChildItem out\frames -Filter frame_*.png).Count
