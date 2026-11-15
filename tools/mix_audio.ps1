# Builds the 45 s soundtrack from the downloaded free music + SFX (no synthesis): out\audio\mix.wav
# plus separate stems (music_stem.wav, sfx_stem.wav) for level checks.
# Shot timing (s): S1 0-8.5 establish | S2 8.5-14.25 waterfall | S3 14.25-20.5 pool | S4 20.5-27 fire |
#                  S5 27-33.75 spires | S6 33.75-36.75 stairs | S7 36.75-45 hero (music climax at 36.75)
$ErrorActionPreference = "Stop"
[System.Threading.Thread]::CurrentThread.CurrentCulture = [cultureinfo]::InvariantCulture
$root = Split-Path -Parent $PSScriptRoot
$A = Join-Path $root "assets\audio"
$out = Join-Path $root "out\audio"
New-Item -ItemType Directory -Force $out | Out-Null
$DUR = 45.0

function Lin([object[]]$kf) {
    # piecewise-linear ffmpeg expression in t from keyframes @(@(t,v),...)
    $expr = "$($kf[-1][1])"
    for ($i = $kf.Count - 2; $i -ge 0; $i--) {
        $t0 = $kf[$i][0]; $v0 = $kf[$i][1]; $t1 = $kf[$i + 1][0]; $v1 = $kf[$i + 1][1]
        $seg = "($v0+($v1-$v0)*(t-$t0)/($t1-$t0))"
        $expr = "if(lt(t,$t1),$seg,$expr)"
    }
    return "if(lt(t,$($kf[0][0])),$($kf[0][1]),$expr)"
}

$in = @(
    @("-ss", "72.0", "-t", "46", "-i", "$A\music\aphelion_scott_buckley\Aphelion.mp3"),                       # 0 music
    @("-ss", "8", "-t", "46", "-i", "$A\sfx\evening_wind_forest_birds_mixkit_1238\mixkit_1238.wav"),         # 1 ambience
    @("-ss", "4", "-t", "46", "-i", "$A\sfx\waterfall_large_loop_mixkit_3030\mixkit_3030.wav"),             # 2 waterfall
    @("-ss", "2", "-t", "8", "-i", "$A\sfx\campfire_night_wind_mixkit_1736\mixkit_1736.wav"),               # 3 fire
    @("-ss", "5", "-t", "8", "-i", "$A\sfx\pool_water_lapping_freesound_573163\573163_97550-hq.mp3"),       # 4 lapping
    @("-ss", "3", "-t", "8", "-i", "$A\sfx\wind_mountain_top_mixkit_1267\mixkit_1267.wav"),                 # 5 high wind
    @("-ss", "4", "-t", "8", "-i", "$A\sfx\flag_flapping_freesound_154794\154794_1661766-hq.mp3"),          # 6 flag
    @("-i", "$A\sfx\whoosh_wind_passing_mixkit_1167\mixkit_1167.wav"),                                       # 7 whoosh A
    @("-i", "$A\sfx\whoosh_transition_windy_mixkit_1474\mixkit_1474.wav"),                                   # 8 whoosh B
    @("-i", "$A\sfx\whoosh_cinematic_tunnel_reverb_mixkit_1486\mixkit_1486.wav"),                            # 9 whoosh C
    @("-i", "$A\sfx\riser_cinematic_suspense_swell_mixkit_786\mixkit_786.wav"),                              # 10 riser
    @("-i", "$A\sfx\boom_movie_logo_intro_impact_mixkit_2900\mixkit_2900.wav"),                              # 11 boom
    @("-i", "$A\sfx\boom_drum_deep_impact_mixkit_563\mixkit_563.wav")                                        # 12 deep hit
)
$fmt = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
$wf = Lin @(@(0, 0.06), @(7.0, 0.07), @(8.5, 0.16), @(9.6, 0.50), @(12.3, 0.60), @(14.25, 0.24), @(16.0, 0.12), @(20.5, 0.08), @(21.5, 0.045),
            @(27.0, 0.035), @(33.75, 0.04), @(36.75, 0.07), @(38.5, 0.13), @(42.0, 0.13), @(45.0, 0.0))
$amb = Lin @(@(0, 0.0), @(2.0, 0.9), @(8.5, 0.9), @(9.5, 0.45), @(14.25, 0.6), @(20.5, 1.0), @(27.0, 0.5), @(33.75, 0.9), @(42.0, 0.9), @(45.0, 0.0))

function D([double]$sec) { $ms = [int]([math]::Round($sec * 1000)); return "adelay=$ms|$ms" }

$sfx = @(
    "[1:a]$fmt,volume='$amb':eval=frame,volume=0.75[a1]",
    "[2:a]$fmt,volume='$wf':eval=frame,volume=0.8[a2]",
    "[3:a]$fmt,atrim=0:7.3,afade=t=in:d=0.5,afade=t=out:st=6.4:d=0.9,volume=1.1,alimiter=limit=0.35:level=false,$(D 20.2)[a3]",
    "[4:a]$fmt,atrim=0:6.9,afade=t=in:d=0.7,afade=t=out:st=5.9:d=1.0,volume=3.0,alimiter=limit=0.22:level=false,$(D 14.0)[a4]",
    "[5:a]$fmt,atrim=0:7.5,afade=t=in:d=0.8,afade=t=out:st=6.4:d=1.1,volume=0.34,$(D 26.7)[a5]",
    "[6:a]$fmt,atrim=0:7.2,afade=t=in:d=0.8,afade=t=out:st=6.2:d=1.0,volume=0.6,$(D 26.9)[a6]",
    "[7:a]$fmt,afade=t=in:d=0.6,volume=0.30,$(D 6.3)[a7]",
    "[8:a]$fmt,volume=0.20,$(D 13.3)[a8]",
    "[9:a]$fmt,volume=0.24,$(D 26.3)[a9]",
    "[10:a]$fmt,afade=t=in:d=2.5,volume=0.17,$(D 25.48)[a10]",
    "[11:a]$fmt,atrim=0:5.0,afade=t=out:st=1.6:d=3.4,volume=0.36,$(D 36.70)[a11]",
    "[12:a]$fmt,volume=0.32,$(D 0.0)[a12]"
)
$music = "[0:a]$fmt,afade=t=in:d=1.2,afade=t=out:st=42.0:d=3.0,volume=0.95[m]"
$sfxmix = "[a1][a2][a3][a4][a5][a6][a7][a8][a9][a10][a11][a12]amix=inputs=12:normalize=0:duration=longest,atrim=0:$DUR,afade=t=out:st=42.5:d=2.5[s]"

$args0 = @("-y", "-hide_banner", "-loglevel", "error")
foreach ($i in $in) { $args0 += $i }

# stems
& ffmpeg @args0 -filter_complex "$music" -map "[m]" -t $DUR -c:a pcm_s24le "$out\music_stem.wav"
& ffmpeg @args0 -filter_complex (($sfx -join ";") + ";" + $sfxmix) -map "[s]" -t $DUR -c:a pcm_s24le "$out\sfx_stem.wav"
# mix + limiter
& ffmpeg -y -hide_banner -loglevel error -i "$out\music_stem.wav" -i "$out\sfx_stem.wav" -filter_complex "[0:a][1:a]amix=inputs=2:normalize=0,alimiter=limit=0.84:attack=5:release=80:level=false[x]" -map "[x]" -t $DUR -c:a pcm_s24le "$out\mix_raw.wav"
# static gain to -14 LUFS integrated, then a safety limiter
$m = & ffmpeg -hide_banner -nostats -i "$out\mix_raw.wav" -af ebur128=peak=true -f null - 2>&1 | Out-String
$I = [double]([regex]::Matches($m, "I:\s+(-?[\d\.]+) LUFS") | Select-Object -Last 1).Groups[1].Value
$gain = [math]::Round(-14.0 - $I, 2)
& ffmpeg -y -hide_banner -loglevel error -i "$out\mix_raw.wav" -af "volume=${gain}dB,alimiter=limit=0.80:attack=4:release=90:level=false" -c:a pcm_s24le "$out\mix.wav"
"mix_raw integrated $I LUFS -> gain $gain dB"
foreach ($f in "music_stem.wav", "sfx_stem.wav", "mix.wav") {
    $r = & ffmpeg -hide_banner -nostats -i "$out\$f" -af ebur128=peak=true -f null - 2>&1 | Out-String
    $i2 = ([regex]::Matches($r, "I:\s+(-?[\d\.]+) LUFS") | Select-Object -Last 1).Groups[1].Value
    $pk = ([regex]::Matches($r, "Peak:\s+(-?[\d\.]+) dBFS") | Select-Object -Last 1).Groups[1].Value
    "{0,-16} I={1} LUFS  peak={2} dBFS" -f $f, $i2, $pk
}



