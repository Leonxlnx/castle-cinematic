# Working log (final state)

Result: `out/final/castle_cinematic_3840w.mp4` (4K master) and `castle_cinematic_1080p.mp4`, 45 s, 24 fps,
with music and sound effects. 1080 frames in `out/frames`.

## Pipeline
1. Build + save the scene: `pwsh -NoProfile -File tools\bl.ps1 build.py --mods all --save out/scene.blend --quality final`
2. Preview a shot: `... build.py --mods all --cam s7_hero --at 0,0.5,1 --res 1280x720 --samples 16 --out out/previews/x`
3. Final frames (resumable): `pwsh -NoProfile -File tools\render_all.ps1 1 1080 120 12 3840x2160 out/frames`
4. Soundtrack: `tools\fetch_audio.ps1`, then `tools\mix_audio.ps1` -> `out/audio/mix.wav`
5. Encode: `pwsh -NoProfile -File tools\make_video.ps1 out\frames castle_cinematic 15`

## Decisions that shaped it
* EEVEE on the integrated Radeon 780M; Cycles was far too slow.
* Final format 3840x2160 at 12 samples, about 16-20 s per frame. On this GPU resolution is cheap and
  geometry/shadow work per sample is expensive, so 4K at 12 samples costs about what 1080p at 24 samples does.
* `scene/mods/perf.py`: the sun shadow was about 70% of the frame time. Coarser sun shadow maps, shadow scale 0.5
  and no shadow casting from the far landscape and forest cut a frame from about 33 s to about 15 s.
* Only the sun and 5 facade floodlights cast shadows. All other lights are shadowless with a cutoff distance.
* `scene/mods/cameras.py` `SHOT_HIDE_OBJ` hides geometry a shot cannot see.
* Shots are cut on the accents of the music (climax at 36.75 s = start of the hero shot).
* `tools\bl.ps1`: at most 2 Blender jobs, RAM guard. `tools\render_all.ps1` renders in chunks, skips finished
  frames and keeps the machine awake while it runs.

## What went wrong along the way
* The laptop went to sleep when the session idled overnight, which stopped the sub-agents and cost about 6 hours.
* First test with shadow-casting point lights: 200 s per frame and 10 GB RAM. Fixed by making practical lights
  shadowless.
* First full render was stopped after 121 frames because the lights were too bright (bloom, sun disc, sun glint
  on the lake, lanterns). Everything was dimmed and the render restarted.
* The sun's reflection on the lake still burned out in shots 1 and 5. Those two shots were rendered again with the
  sun's specular set to 0 and the sky reflection clamped (`perf.py`, `sky.py` `PROBE_MAX`). The code in this repo
  has those settings for every shot, so a fresh render differs slightly from the film in the other five shots
  (less sun sparkle on the wet marble).

## Shots (`scene/mods/cameras.py`)
S1 0-8.5 s establish | S2 8.5-14.25 waterfall rise | S3 14.25-20.5 pool glide | S4 20.5-27 fire lounge |
S5 27-33.75 spires against the sun | S6 33.75-36.75 stairs and great window | S7 36.75-45 hero wide
