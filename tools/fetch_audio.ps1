# Downloads the free music + sound effects used by the soundtrack into assets\audio\ (they are NOT in the repo:
# the Mixkit licence does not allow redistributing the raw files).  Sources and licences: assets\audio\CREDITS.md
# Usage: pwsh -NoProfile -File tools\fetch_audio.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$audio = Join-Path $root "assets\audio"
$lines = Get-Content (Join-Path $audio "CREDITS.md")
$items = @()
# "- File: `path` ..." followed by "- Direct download...: url"
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^- File: `([^`]+)`') {
        $path = $Matches[1]
        for ($j = $i + 1; $j -lt [math]::Min($i + 8, $lines.Count); $j++) {
            if ($lines[$j] -match '^- Direct download[^:]*: (\S+)') { $items += , @($path, $Matches[1]); break }
        }
    }
    # table rows: | `path` | title | page | url |
    elseif ($lines[$i] -match '^\| `([^`]+)` \|.*\| (https://\S+) \|\s*$') { $items += , @($Matches[1], $Matches[2]) }
}
foreach ($it in $items) {
    $dst = Join-Path $audio ($it[0] -replace '/', '\')
    if (Test-Path $dst) { "have  $($it[0])"; continue }
    New-Item -ItemType Directory -Force (Split-Path -Parent $dst) | Out-Null
    "get   $($it[0])"
    Invoke-WebRequest -Uri $it[1] -OutFile $dst -UserAgent "Mozilla/5.0"
}
"done: $($items.Count) files listed"
