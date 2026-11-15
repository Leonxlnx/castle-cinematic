"""A/B render-time profiler robust against thermal drift: every test is bracketed by baseline renders.
  pwsh -NoProfile -File tools\bl.ps1 @blend=out\scene_wip.blend tools\profile3.py --shot s7_hero --frac 0.6 --samples 8 --res 960x540
"""
import bpy, sys, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scene.mods import cameras as C

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
a = {"shot": "s7_hero", "frac": "0.5", "samples": "8", "res": "960x540"}
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
r.filepath = os.path.join(ROOT, "out", "previews", "profile_tmp.png")


def t_render():
    t = time.time()
    bpy.ops.render.render(write_still=True)
    return time.time() - t


def P(*x):
    print(*x); sys.stdout.flush()


# ---- inventory
vols = [m.name for m in bpy.data.materials if m.node_tree and any(
    (n.type == 'OUTPUT_MATERIAL' and n.inputs['Volume'].is_linked) for n in m.node_tree.nodes)]
P("PROFILE volume materials:", vols)
P("PROFILE lights:", {t: len([l for l in bpy.data.lights if l.type == t]) for t in ('SUN', 'POINT', 'SPOT', 'AREA')},
  "shadowed:", [l.name for l in bpy.data.lights if l.use_shadow])
for c in bpy.data.collections:
    if c.name.startswith("veg") or c.name in ("landscape", "props", "water"):
        tris = sum(len(o.data.polygons) for o in c.objects if o.type == 'MESH')
        P("PROFILE col %-22s objs=%d polys=%dk children=%s" % (c.name, len(c.objects), tris // 1000, [ch.name for ch in c.children]))
blended = [m.name for m in bpy.data.materials if getattr(m, 'surface_render_method', '') == 'BLENDED']
P("PROFILE blended materials:", blended)

t_render()                      # warmup
base = [t_render(), t_render()]
P("PROFILE baseline %.1fs %.1fs" % tuple(base))


def test(tag, on, off):
    on()
    t = t_render()
    off()
    b = t_render()
    P("PROFILE %-34s %.1fs   (baseline after %.1fs)  saves %+.0f%%" % (tag, t, b, 100.0 * (b - t) / b))


def hide_cols(names):
    cols = [bpy.data.collections.get(n) for n in names]
    cols = [c for c in cols if c]
    return (lambda: [setattr(c, 'hide_render', True) for c in cols], lambda: [setattr(c, 'hide_render', False) for c in cols])


def hide_objs(prefixes):
    obs = [o for o in bpy.data.objects if any(o.name.startswith(p) for p in prefixes)]
    return (lambda: [setattr(o, 'hide_render', True) for o in obs], lambda: [setattr(o, 'hide_render', False) for o in obs])


test("no practical lights", *hide_cols(["castle_lights", "props_lights", "terrace_lights"]))
spots = [l for l in bpy.data.lights if l.use_shadow and l.type != 'SUN']
test("flood shadows off", lambda: [setattr(l, 'use_shadow', False) for l in spots], lambda: [setattr(l, 'use_shadow', True) for l in spots])
jit = [l for l in bpy.data.lights if l.use_shadow and getattr(l, 'use_shadow_jitter', False)]
test("shadow jitter off (%d)" % len(jit), lambda: [setattr(l, 'use_shadow_jitter', False) for l in jit], lambda: [setattr(l, 'use_shadow_jitter', True) for l in jit])
suns = [l for l in bpy.data.lights if l.type == 'SUN']
test("sun shadow off", lambda: [setattr(l, 'use_shadow', False) for l in suns], lambda: [setattr(l, 'use_shadow', True) for l in suns])
test("no vegetation", *hide_cols(["vegetation"]))
test("no landscape terrain+cliff", *hide_objs(["landscape_terrain", "landscape_cliff"]))
test("no mist", *hide_objs(["landscape_mist"]))
test("no far trees", *hide_objs(["landscape_trees"]))
test("no props", *hide_cols(["props"]))
test("no water", *hide_cols(["water"]))
test("no castle", *hide_cols(["castle"]))
test("no terrace", *hide_cols(["terrace"]))


def setp(obj, k, v):
    def f(): setattr(obj, k, v)
    return f

test("no raytracing", setp(ee, 'use_raytracing', False), setp(ee, 'use_raytracing', True))
test("no fast GI", setp(ee, 'use_fast_gi', False), setp(ee, 'use_fast_gi', True))
test("no compositor", setp(r, 'use_compositing', False), setp(r, 'use_compositing', True))
w = scene.world
nt = w.node_tree
out = [n for n in nt.nodes if n.type == 'OUTPUT_WORLD'][0]
old_link = out.inputs['Surface'].links[0].from_socket
bg = nt.nodes.new('ShaderNodeBackground'); bg.inputs[0].default_value = (0.3, 0.2, 0.25, 1)
test("flat world (no sky shader)", lambda: nt.links.new(bg.outputs[0], out.inputs['Surface']), lambda: nt.links.new(old_link, out.inputs['Surface']))
ov = ee.use_overscan
test("no overscan", setp(ee, 'use_overscan', False), setp(ee, 'use_overscan', ov))
old_s = ee.taa_render_samples
test("half samples", setp(ee, 'taa_render_samples', max(1, old_s // 2)), setp(ee, 'taa_render_samples', old_s))
P("PROFILE done")
