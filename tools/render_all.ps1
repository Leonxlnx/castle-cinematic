# Renders a frame range in small chunks, each in a fresh Blender process (robust against leaks/crashes).
# Usage: pwsh -NoProfile -File tools\render_all.ps1 <start> <end> [chunk=48] [samples=32] [res=1920x1080] [outdir=out/frames] [extra args...]
# Safe to re-run: existing frames are skipped.  Create out\STOP to stop after the current frame.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
# keep the machine from idle-sleeping while this render driver is alive (released automatically when it exits;
# no system setting is changed)
try {
    Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
    [void][Win32.Power]::SetThreadExecutionState([uint32]"0x80000001")
} catch { }
$start = [int]$args[0]; $end = [int]$args[1]
$chunk = if ($args.Count -gt 2) { [int]$args[2] } else { 48 }
$samples = if ($args.Count -gt 3) { "$($args[3])" } else { "32" }
$res = if ($args.Count -gt 4) { "$($args[4])" } else { "1920x1080" }
$outdir = if ($args.Count -gt 5) { "$($args[5])" } else { "out/frames" }
$extra = @(); if ($args.Count -gt 6) { $extra = @($args | Select-Object -Skip 6) }
$log = Join-Path $root "out\logs\render_all.log"
New-Item -ItemType Directory -Force (Join-Path $root "out\logs") | Out-Null
"$(Get-Date -Format s) START $start..$end chunk=$chunk samples=$samples res=$res out=$outdir" | Add-Content $log
for ($s = $start; $s -le $end; $s += $chunk) {
    if (Test-Path (Join-Path $root "out\STOP")) { "$(Get-Date -Format s) STOP file - abort" | Add-Content $log; break }
    $e = [math]::Min($s + $chunk - 1, $end)
    # skip chunks that are already complete
    $missing = $false
    for ($f = $s; $f -le $e; $f++) { $p = Join-Path $root ("{0}\frame_{1:d4}.png" -f $outdir, $f); if (-not (Test-Path $p)) { $missing = $true; break } }
    if (-not $missing) { continue }
    $tries = 0
    do {
        $tries++
        $os = Get-CimInstance Win32_OperatingSystem
        $free = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
        "$(Get-Date -Format s) chunk $s..$e try $tries (free RAM $free GB)" | Add-Content $log
        $o = & pwsh -NoProfile -File tools\bl.ps1 "@blend=out\scene.blend" render_frames.py --start $s --end $e --out $outdir --res $res --samples $samples @extra 2>&1
        $code = $LASTEXITCODE
        ($o | Select-String "\[render\]|ERR|BL_" | ForEach-Object { $_.Line }) | Add-Content $log
        if ($code -ne 0) { Start-Sleep -Seconds 20 }
    } while ($code -ne 0 -and $tries -lt 3)
}
"$(Get-Date -Format s) END $start..$end" | Add-Content $log
Write-Output "RENDER_ALL_DONE $start..$end"
