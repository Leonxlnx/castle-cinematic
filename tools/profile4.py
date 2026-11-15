"""Cumulative shadow/perf fixes, timed at final resolution.
  pwsh -NoProfile -File tools/bl.ps1 @blend=out/scene_wip.blend tools/profile4.py --shot s7_hero --frac 0.6 --samples 16 --res 1920x1080
"""
import bpy, sys, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scene.mods import cameras as C

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
a = {"shot": "s7_hero", "frac": "0.5", "samples": "16", "res": "1920x1080"}
i = 0
while i < len(argv):
    if argv[i].startswith("--"):
        a[argv[i][2:]] = argv[i + 1]; i += 2
    else:
        i += 1
scene = bpy.context.scene
scene.timeline_markers.clear()
s = [x for x in C.SHOTS if x['name'] == a["shot"]][0]
f0, f1 = C.frame_range(s)
scene.frame_set(int(f0 + (f1 - f0) * float(a["frac"])))
scene.camera = bpy.data.objects[a["shot"]]
r = scene.render
r.resolution_x, r.resolution_y = (int(x) for x in a["res"].split("x"))
ee = scene.eevee
ee.taa_render_samples = int(a["samples"])


def run(tag, save=None):
    r.filepath = os.path.join(ROOT, "out", "previews", save or "profile_tmp.png")
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print("PROFILE %-46s %.1fs" % (tag, time.time() - t)); sys.stdout.flush()


print("PROFILE vegetation objects:", [(o.name, o.type) for o in bpy.data.collections['vegetation'].objects])
print("PROFILE landscape objects:", sorted(set(o.name.rstrip('0123456789_.') for o in bpy.data.collections['landscape'].objects)))
sun = [l for l in bpy.data.lights if l.type == 'SUN'][0]
print("PROFILE sun: max_res=%s filter=%s jitter=%s angle=%.4f soft=%s" % (sun.shadow_maximum_resolution, sun.shadow_filter_radius,
      sun.use_shadow_jitter, sun.angle, getattr(sun, 'shadow_soft_size', None)))
print("PROFILE eevee: shadow_pool=%s ray=%s step=%s res_scale=%s" % (ee.shadow_pool_size, ee.shadow_ray_count, ee.shadow_step_count, ee.shadow_resolution_scale))
run("warmup")
run("V0 as is", "p4_v0.png")
sun.shadow_maximum_resolution = 0.05
run("V1 sun max shadow res 0.05 m")
for o in bpy.data.collections['landscape'].objects:
    o.visible_shadow = False
run("V2 + landscape casts no shadow")
for o in bpy.data.collections['vegetation'].objects:
    if 'forest' in o.name.lower():
        o.visible_shadow = False
run("V3 + forest casts no shadow")
ee.use_overscan = False
run("V4 + no overscan", "p4_v4.png")
for o in bpy.data.collections['vegetation'].objects:
    o.visible_shadow = False
run("V5 + no vegetation shadows at all")
for o in bpy.data.collections['vegetation'].objects:
    if 'forest' not in o.name.lower():
        o.visible_shadow = True
ee.shadow_resolution_scale = 0.5
run("V6 (veg garden shadows back) + shadow res scale 0.5")
ee.shadow_resolution_scale = 1.0
for l in bpy.data.lights:
    if l.use_shadow and l.type != 'SUN':
        l.use_shadow_jitter = False
run("V7 + flood shadow jitter off")
ee.shadow_ray_count = 1; ee.shadow_step_count = 3
run("V8 + shadow steps 3")
ee.taa_render_samples = 24
run("V9 same at 24 samples", "p4_v9.png")
print("PROFILE done")
