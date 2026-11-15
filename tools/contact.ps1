# Contact sheet from rendered frames: pwsh -NoProfile -File tools\contact.ps1 <framesDir> <start> <end> <count> <out.jpg> [cols=4] [tileWidth=640]
# Picks <count> frames evenly between start..end that exist, tiles them into one JPG for quick visual QC.
$ErrorActionPreference = "Stop"
$dir = "$($args[0])"; $s = [int]$args[1]; $e = [int]$args[2]; $cnt = [int]$args[3]; $out = "$($args[4])"
$cols = if ($args.Count -gt 5) { [int]$args[5] } else { 4 }
$tw = if ($args.Count -gt 6) { [int]$args[6] } else { 640 }
$files = @()
for ($i = 0; $i -lt $cnt; $i++) {
    $f = if ($cnt -gt 1) { [int][math]::Round($s + ($e - $s) * $i / ($cnt - 1)) } else { $s }
    # nearest existing frame at or after f
    for ($k = 0; $k -lt 12; $k++) { $p = Join-Path $dir ("frame_{0:d4}.png" -f ($f + $k)); if (Test-Path $p) { $files += $p; break } }
}
if ($files.Count -eq 0) { "no frames found"; exit 1 }
$rows = [math]::Ceiling($files.Count / $cols)
$inputs = @(); foreach ($f in $files) { $inputs += @("-i", $f) }
$n = $files.Count
$scale = (0..($n - 1) | ForEach-Object { "[$($_):v]scale=${tw}:-1[v$($_)]" }) -join ";"
$lay = (0..($n - 1) | ForEach-Object { "[v$($_)]" }) -join ""
$fc = "$scale;${lay}xstack=inputs=${n}:layout=" + ((0..($n - 1) | ForEach-Object { $c = $_ % $cols; $r = [math]::Floor($_ / $cols); "$($c * $tw)_$($r * [int]($tw * 9 / 16))" }) -join "|") + ":fill=black[o]"
if ($n -eq 1) { $fc = "[0:v]scale=${tw}:-1[o]" }
& ffmpeg -y -hide_banner -loglevel error @inputs -filter_complex $fc -map "[o]" -q:v 3 $out
"contact sheet: $n frames -> $out"
