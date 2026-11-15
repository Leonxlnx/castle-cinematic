# Throttled, low-priority Blender launcher. At most TWO Blender jobs run at a time (global semaphore),
# so parallel agents can never overload CPU/GPU/RAM.
# Usage:  pwsh -NoProfile -File tools\bl.ps1 <script.py> [args passed to the script after --]
#         pwsh -NoProfile -File tools\bl.ps1 @blend=out\scene.blend <script.py> [args]
# (no param() block on purpose: every argument is passed through untouched, including --out etc.)
$ErrorActionPreference = "Continue"
$bl = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
$root = Split-Path -Parent $PSScriptRoot
$a = @($args)
$blend = ""
if ($a.Count -gt 0 -and "$($a[0])".StartsWith("@blend=")) { $blend = "$($a[0])".Substring(7); $a = @($a | Select-Object -Skip 1) }
if ($a.Count -lt 1) { Write-Output "usage: bl.ps1 [@blend=file.blend] script.py [args]"; exit 2 }
$script = "$($a[0])"
$rest = @($a | Select-Object -Skip 1)

# RAM guard: refuse to start when less than 6 GB physical RAM is free
$os = Get-CimInstance Win32_OperatingSystem
$freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
if ($freeGB -lt 7) { Write-Output "BL_ABORT low RAM: $freeGB GB free"; exit 3 }

$mutex = New-Object System.Threading.Semaphore(2, 2, "Global\CastleBlenderSem2")
$got = $false
try {
    try { $got = $mutex.WaitOne([TimeSpan]::FromMinutes(120)) } catch [System.Threading.AbandonedMutexException] { $got = $true }
    if (-not $got) { Write-Output "BL_ABORT lock timeout"; exit 4 }
    $argList = @("-b")
    if ($blend -ne "") { $argList += $blend }
    $argList += @("--python-exit-code", "1", "-P", $script, "--")
    $argList += $rest
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $bl
    $psi.Arguments = ($argList | ForEach-Object { if ("$_" -match '\s') { '"' + $_ + '"' } else { "$_" } }) -join ' '
    $psi.WorkingDirectory = $root
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi
    $null = $p.Start()
    try { $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::Normal } catch {}
    $errTask = $p.StandardError.ReadToEndAsync()
    while (($line = $p.StandardOutput.ReadLine()) -ne $null) {
        # drop the noisy per-sample progress lines, keep everything else
        if ($line -match '^\s*$') { continue }
        if ($line -match '^Fra:\d+ ' -or $line -match 'Shadow buffer full' -or $line -match '^Saved: ' -or $line -match '^\s*Time: ') { continue }
        Write-Output $line
    }
    $p.WaitForExit()
    $err = $errTask.Result
    if ($err) { ($err -split "`r?`n") | Where-Object { $_ -match '\S' -and $_ -notmatch 'CUEW|Shadow buffer full|DeprecationWarning|use_nodes = True' } | Select-Object -First 60 | ForEach-Object { Write-Output "ERR: $_" } }
    Write-Output "BL_EXIT $($p.ExitCode)"
    exit $p.ExitCode
}
finally {
    if ($got) { $null = $mutex.Release() }
    $mutex.Dispose()
}



