# Waits until the main render (out\frames, 1080 frames) is complete, then re-renders the given frame ranges with
# another .blend into a separate folder.  Resumable (existing frames are skipped).
# Usage: pwsh -NoProfile -File tools\after_main.ps1 <blend> <outdir> <start> <end> [<start2> <end2> ...]
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
try {
    Add-Type -Namespace Win32 -Name Power2 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
    [void][Win32.Power2]::SetThreadExecutionState([uint32]"0x80000001")
} catch { }
$blend = "$($args[0])"; $outdir = "$($args[1])"
$log = Join-Path $root "out\logs\after_main.log"
"$(Get-Date -Format s) waiting for main render" | Add-Content $log
while ((Get-ChildItem (Join-Path $root "out\frames") -Filter "frame_*.png").Count -lt 1080) {
    if (Test-Path (Join-Path $root "out\STOP")) { "$(Get-Date -Format s) STOP" | Add-Content $log; exit 0 }
    [Threading.Thread]::Sleep(15000)
}
for ($i = 2; $i + 1 -lt $args.Count; $i += 2) {
    $s = [int]$args[$i]; $e = [int]$args[$i + 1]
    $tries = 0
    do {
        $tries++
        "$(Get-Date -Format s) range $s..$e try $tries" | Add-Content $log
        $o = & pwsh -NoProfile -File tools\bl.ps1 "@blend=$blend" render_frames.py --start $s --end $e --out $outdir --res 3840x2160 --samples 12 2>&1
        $code = $LASTEXITCODE
        ($o | Select-String "\[render\] done|FAILED|BL_" | ForEach-Object { $_.Line }) | Add-Content $log
        $missing = 0
        for ($f = $s; $f -le $e; $f++) { if (-not (Test-Path (Join-Path $root ("{0}\frame_{1:d4}.png" -f $outdir, $f)))) { $missing++ } }
    } while ($missing -gt 0 -and $tries -lt 4)
    "$(Get-Date -Format s) range $s..$e finished, missing=$missing" | Add-Content $log
}
"$(Get-Date -Format s) AFTER_MAIN_DONE" | Add-Content $log
Write-Output "AFTER_MAIN_DONE"
