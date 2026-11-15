# Muxes the rendered frames + soundtrack into the final film (4K master + 1080p version).
# Usage: pwsh -NoProfile -File tools\make_video.ps1 [framesDir=out\frames] [outName=castle_cinematic] [crf=15]
$ErrorActionPreference = "Stop"
[System.Threading.Thread]::CurrentThread.CurrentCulture = [cultureinfo]::InvariantCulture
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$frames = if ($args.Count -gt 0) { "$($args[0])" } else { "out\frames" }
$name = if ($args.Count -gt 1) { "$($args[1])" } else { "castle_cinematic" }
$crf = if ($args.Count -gt 2) { "$($args[2])" } else { "15" }
New-Item -ItemType Directory -Force "out\final" | Out-Null
$n = (Get-ChildItem $frames -Filter "frame_*.png").Count
$dur = $n / 24.0
$fo = [math]::Max(0.0, [math]::Round($dur - 1.7, 3))
$first = Get-ChildItem $frames -Filter "frame_*.png" | Sort-Object Name | Select-Object -First 1
$w = [int](& ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $first.FullName)
# picture: fade in from black 0.6 s, fade out to black over the last 1.7 s
$fade = "fade=t=in:st=0:d=0.6,fade=t=out:st=${fo}:d=1.7"
$audio = @(); $amap = @()
if (Test-Path "out\audio\mix.wav") { $audio = @("-i", "out\audio\mix.wav"); $amap = @("-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-shortest") }
$common = @("-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", "-movflags", "+faststart")
# native-resolution master
$outM = "out\final\${name}_${w}w.mp4"
& ffmpeg -y -hide_banner -loglevel error -framerate 24 -start_number 1 -i "$frames\frame_%04d.png" @audio `
    -vf "$fade,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p" -c:v libx264 -preset slow -crf $crf -profile:v high -bf 2 -g 48 -threads 8 @common @amap $outM
"master: frames=$n duration=$dur s -> $outM"
& ffprobe -v error -show_entries "format=duration,size,bit_rate:stream=codec_name,width,height,r_frame_rate" -of default=nw=1 $outM
if ($w -gt 1920) {
    # 1080p version: high-quality downscale (supersampled) + a touch of sharpening
    $out2 = "out\final\${name}_1080p.mp4"
    & ffmpeg -y -hide_banner -loglevel error -framerate 24 -start_number 1 -i "$frames\frame_%04d.png" @audio `
        -vf "scale=1920:1080:flags=lanczos:out_color_matrix=bt709:out_range=tv,unsharp=3:3:0.35:3:3:0.0,$fade,format=yuv420p" -c:v libx264 -preset slow -crf 14 -profile:v high -level 4.2 -bf 2 -g 48 -threads 8 @common @amap $out2
    "1080p: -> $out2"
    & ffprobe -v error -show_entries "format=duration,size,bit_rate:stream=width,height" -of default=nw=1 $out2
}


