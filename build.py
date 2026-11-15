"""Scene builder / preview renderer.  Run through the serialized launcher:

  pwsh -File tools\bl.ps1 build.py --mods castle,terrace --cam hero,pool --res 960x540 --samples 16 --out out/previews/test

Arguments (after the script name):
  --mods a,b,c        modules from scene/mods to build, in order ("all" = full scene in canonical order)
  --cam n1,n2         preview cameras from scene/layout.py CAMERAS to render (default: none = build only)
  --campos x,y,z --camtarget x,y,z [--lens 35]   one extra custom camera named "custom" (add it to --cam)
  --res WxH           render resolution (default 960x540)
  --samples N         EEVEE samples (default 16)
  --frame N           frame to render (default 1)
  --out PATH          output prefix; images are written as <prefix>_<cam>.png  (default out/previews/prev)
  --save PATH         save the built scene as a .blend
  --quality preview|final   passed to modules as ctx.quality (final = full densities)
  --exposure E        view exposure override
  --at 0,0.5,1        for shot cameras (names from scene/mods/cameras.py SHOTS): render at these fractions of the shot
Each module is  scene/mods/<name>.py  exposing  build(ctx).
"""
import bpy, sys, os, time, importlib, types, traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ALL = ["sky", "landscape", "terrace", "castle", "water", "vegetation", "props", "birds", "cameras", "comp", "perf"]


def parse(argv):
    a = {"mods": "", "cam": "", "res": "960x540", "samples": "16", "frame": "1", "out": "out/previews/prev",
         "save": "", "quality": "preview", "campos": "", "camtarget": "", "lens": "35", "exposure": "", "at": ""}
    i = 0
    while i < len(argv):
        k = argv[i]
        if k.startswith("--") and k[2:] in a:
            a[k[2:]] = argv[i + 1]; i += 2
        else:
            print("unknown arg", k); i += 1
    return a


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = parse(argv)
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    from scene import setup, layout, common, mats
    res = tuple(int(x) for x in a["res"].split("x"))
    setup.setup_render(scene, res=res, samples=int(a["samples"]), quality=a["quality"])

    mods = [m for m in (ALL if a["mods"] == "all" else a["mods"].split(",")) if m]
    ctx = types.SimpleNamespace(scene=scene, quality=a["quality"], mods=mods, fps=setup.FPS,
                                frame_end=setup.FRAME_END, col=common.get_col, layout=layout)
    cams = setup.make_cameras(scene)
    ctx.cams = cams
    if "sky" not in mods:
        setup.fallback_world(scene)
    for name in mods:
        path = os.path.join(ROOT, "scene", "mods", name + ".py")
        if not os.path.exists(path):
            print("[build] module %s missing - skipped" % name)
            continue
        t = time.time()
        mod = importlib.import_module("scene.mods." + name)
        mod.build(ctx)
        print("[build] %-12s %.1fs" % (name, time.time() - t))

    if a["campos"] and a["camtarget"]:
        p = tuple(float(x) for x in a["campos"].split(","))
        tg = tuple(float(x) for x in a["camtarget"].split(","))
        cams["custom"] = setup.custom_camera(scene, p, tg, float(a["lens"]))
    if a["exposure"]:
        scene.view_settings.exposure = float(a["exposure"])

    # stats
    dg = bpy.context.evaluated_depsgraph_get()
    tris = 0; nobj = 0
    for inst in dg.object_instances:
        o = inst.object
        if o.type == 'MESH':
            nobj += 1
            tris += len(o.data.loop_triangles) if o.data.loop_triangles else len(o.data.polygons) * 2
    print("[build] objects(instances)=%d  approx tris=%d  build time %.1fs" % (nobj, tris, time.time() - t0))

    if a["save"]:
        p = os.path.join(ROOT, a["save"]) if not os.path.isabs(a["save"]) else a["save"]
        os.makedirs(os.path.dirname(p), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=p, compress=True)
        print("[build] saved", p)

    # preview renders: timeline markers would override the chosen camera, so drop them (the .blend is already saved)
    scene.timeline_markers.clear()
    shots = {}
    try:
        from scene.mods import cameras as _C
        shots = {s["name"]: _C.frame_range(s) for s in _C.SHOTS}
    except Exception:
        pass
    fracs = [float(x) for x in a["at"].split(",") if x] if a["at"] else None
    for cn in [c for c in a["cam"].split(",") if c]:
        cam = cams.get(cn) or bpy.data.objects.get(cn) or bpy.data.objects.get("cam_" + cn)
        if cam is None:
            print("[build] camera %s not found" % cn); continue
        scene.camera = cam
        if cn in shots:      # shot camera: render at fractions of the shot (default: middle)
            f0, f1 = shots[cn]
            frames = [int(round(f0 + (f1 - f0) * q)) for q in (fracs or [0.5])]
        else:
            frames = [int(a["frame"])]
        for fr in frames:
            scene.frame_set(fr)
            tag = cn if (cn not in shots) else "%s_f%04d" % (cn, fr)
            outp = os.path.join(ROOT, a["out"] + "_" + tag + ".png") if not os.path.isabs(a["out"]) else a["out"] + "_" + tag + ".png"
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            scene.render.filepath = outp
            t = time.time()
            bpy.ops.render.render(write_still=True)
            print("[build] rendered %s in %.1fs -> %s" % (tag, time.time() - t, outp))
    print("[build] total %.1fs" % (time.time() - t0))


try:
    main()
except Exception:
    traceback.print_exc()
    sys.exit(1)



