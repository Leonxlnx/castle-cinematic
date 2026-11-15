# Castle cinematic – shared brief for all module authors

## Goal
A ~45 s, 1920x1080, 24 fps **photoreal cinematic video** of a gothic cliff-top castle at sunset, rendered with
**Blender 5.2.3 EEVEE** from a fully procedural scene (Python scripts, no downloaded assets, no image files).
Look target: `docs/ref/ref1.jpg` (main reference) and `docs/ref/ref3.jpg` – open them with the Read tool first.
Key qualities: fiery orange/pink/purple sunset with dramatic clouds, glowing warm windows and lanterns,
wet reflective marble terrace, turquoise infinity pool spilling a waterfall down a huge cliff, misty valley with a
lake and forest far below, snow-capped mountains on the horizon, red maple leaves and white flowers framing the
foreground.  The owner wants it **ultra detailed, realistic, zero visible glitches**.  Exterior only.

## How the project is organised
```
build.py                 scene builder + preview renderer (CLI, see its docstring)
tools/bl.ps1             THE ONLY allowed way to run Blender (max 2 jobs at once + low priority + RAM guard)
scene/layout.py          world layout: coordinates, anchors, standard cameras  (READ THIS FIRST, do not edit)
scene/common.py          MB mesh builder, NT node DSL, geometry-nodes helpers   (do not edit)
scene/mats.py            shared material registry                               (do not edit; register your own
                         materials from inside your module with @mats.material("name"))
scene/atmo.py            aerial-perspective fog for far objects                 (do not edit)
scene/setup.py           render settings, fallback world/sun, preview cameras   (do not edit)
scene/mods/<name>.py     one module per subsystem, each exposes  build(ctx)
docs/blender52_api_notes.txt   verified Blender 5.2 API facts (socket names etc.) – read before writing nodes
out/previews/            put your test renders here, prefix them with your module name
```
`ctx` has: `scene`, `quality` ("preview" | "final"), `mods` (list of modules in this build), `fps` (24),
`frame_end` (1080), `cams` (dict of preview cameras), `col(name)` (get/create collection), `layout`.
Put your objects in a collection named after your module.

## Running Blender (mandatory)
```
pwsh -NoProfile -File tools\bl.ps1 build.py --mods <yourmod>[,othermods] --cam hero,valley --res 960x540 --samples 16 --out out/previews/<yourmod>_v01
```
* Working directory must be `D:\fabletests5.2\castle`.  Images land in `out/previews/<prefix>_<cam>.png`; look at
  them with the Read tool and iterate.  Custom view: `--campos x,y,z --camtarget x,y,z --lens 35 --cam custom`.
* Animation check: `--frame N` (1..1080).  Anything animated must be driven by the frame number
  (drivers / keyframes / `#frame`), loop-free and deterministic – the film is rendered frame by frame.
* Never start blender.exe directly, never run two builds at once yourself, never use `run_in_background`
  for Blender.  The launcher queues jobs from all authors; a queued job may wait a few minutes.
  Use a generous timeout (e.g. 900000 ms) on the PowerShell call.
* Iterate at <= 1280x720 and <= 16 samples.  A preview build+render should stay under ~2 minutes.
* Do not install anything and do not download anything (except the audio task, which has its own rules).

## Hard technical rules (laptop safety + render budget)
* Machine: Ryzen 7 (16 threads), integrated Radeon 780M, 62 GB RAM (about 20 GB free).  It must never crash.
* Only the sun casts shadows.  Every other light: `light.use_shadow = False`, and give point/spot lights
  `use_custom_distance = True` with a sensible `cutoff_distance`.  (Shadowed point lights blew up render time.)
* Respect your triangle budget (given in your task) – measured as the `approx tris` line build.py prints.
* Whole-scene target is <= 15 s per 1080p frame at 32 samples; your module alone should add <= 2–3 s.
  build.py prints the render time; report it.
* Shaders: procedural only; keep noise `Detail` <= 6 unless clearly needed; avoid huge node chains per pixel.
* Transparency: prefer `surface_render_method = 'DITHERED'` (alpha hashed, sorts correctly, receives shadows);
  use 'BLENDED' only for thin glass/water sheets where needed.
* Far objects (> ~150 m from the castle) must use `atmo.fog(nt, shader)` so all distant layers share one haze.
* Multiply practical light energies by `layout.PRACTICAL_GAIN` and emission strengths by `layout.EMISSIVE_GAIN`.
* Units are meters, Z up, +X east, +Y north; the sun sets WNW (`layout.sun_dir()`).
* Exposure reference: AgX view transform, exposure 0.  The sky near the sun is roughly 3–20 (scene linear),
  the upper sky 0.05–0.4, sunlit stone ends up around 0.2–0.6.  The scene is a dusk scene: practical lights
  must read as clearly glowing but not blow out.

## Working style
* Own only your files.  If you need something changed in a shared file, say so in your final report instead.
* Look at every render you make; fix what looks fake.  Realism beats cleverness.  Compare with the references.
* Keep a module-level docstring that lists what the module builds and its tunable constants.
* When finished, reply with: what was built, the triangle count and render-time numbers, the preview images that
  best show the result (paths), known weaknesses, and anything the integrator must know (anchors, API functions).

