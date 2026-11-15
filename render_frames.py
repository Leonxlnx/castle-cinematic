r"""Final frame renderer (resumable).  Run on the saved scene through the launcher:

  pwsh -NoProfile -File tools\bl.ps1 @blend=out\scene.blend render_frames.py --start 1 --end 60 --out out/frames --res 1920x1080 --samples 32

* Frames that already exist (non-empty file) are skipped, so the job can be restarted at any time.
* Each frame is written to a temp name first and renamed when complete (no half-written frames).
* Creating the file  out/STOP  makes the renderer exit cleanly after the current frame.
* Timings go to out/logs/render_times.csv
Optional: --step N (render every Nth frame, for animatics), --shots s1_establish,s3_pool (only these shots),
          --scale P (resolution percentage), --mblur 0|1, --exposure E
"""
import bpy, sys, os, time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse(argv):
    a = {"start": "1", "end": "1080", "out": "out/frames", "res": "1920x1080", "samples": "32", "step": "1",
         "shots": "", "scale": "100", "mblur": "", "exposure": ""}
    i = 0
    while i < len(argv):
        k = argv[i]
        if k.startswith("--") and k[2:] in a:
            a[k[2:]] = argv[i + 1]; i += 2
        else:
            i += 1
    return a


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = parse(argv)
    from scene.mods import cameras as C
    scene = bpy.context.scene
    r = scene.render
    r.resolution_x, r.resolution_y = (int(x) for x in a["res"].split("x"))
    r.resolution_percentage = int(a["scale"])
    scene.eevee.taa_render_samples = int(a["samples"])
    if a["mblur"]:
        r.use_motion_blur = (a["mblur"] == "1")
    if a["exposure"]:
        scene.view_settings.exposure = float(a["exposure"])
    grp = getattr(scene, 'compositing_node_group', None)
    if grp is not None:
        for nd in grp.nodes:
            if nd.bl_idname == 'CompositorNodeBlur':      # vignette softness scales with the frame height
                try:
                    nd.inputs['Size'].default_value = (0.39 * r.resolution_y, 0.39 * r.resolution_y)
                except Exception:
                    nd.inputs['Size'].default_value = 0.39 * r.resolution_y
    r.image_settings.file_format = 'PNG'
    r.image_settings.color_mode = 'RGB'
    r.image_settings.color_depth = '8'
    r.image_settings.compression = 15
    out = a["out"] if os.path.isabs(a["out"]) else os.path.join(ROOT, a["out"])
    os.makedirs(out, exist_ok=True)
    logdir = os.path.join(ROOT, "out", "logs"); os.makedirs(logdir, exist_ok=True)
    stop = os.path.join(ROOT, "out", "STOP")
    only = [s for s in a["shots"].split(",") if s]
    n_done = 0; t_all = time.time()
    for f in range(int(a["start"]), int(a["end"]) + 1, int(a["step"])):
        if os.path.exists(stop):
            print("[render] STOP file found - exiting"); break
        shot = C.shot_at(f)
        if only and shot['name'] not in only:
            continue
        dst = os.path.join(out, "frame_%04d.png" % f)
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            continue
        cam = bpy.data.objects.get(shot['name'])
        if cam is None:
            print("[render] camera missing for", shot['name']); continue
        scene.camera = cam
        C.apply_shot_visibility(shot['name'])
        scene.frame_set(f)
        tmp = os.path.join(out, "_tmp_%04d.png" % f)
        r.filepath = tmp
        t = time.time()
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t
        if os.path.exists(tmp) and os.path.getsize(tmp) > 1000:
            os.replace(tmp, dst)
            n_done += 1
            with open(os.path.join(logdir, "render_times.csv"), "a") as fh:
                fh.write("%d,%s,%.2f,%s\n" % (f, shot['name'], dt, time.strftime("%H:%M:%S")))
            print("[render] frame %d (%s) %.1fs" % (f, shot['name'], dt)); sys.stdout.flush()
        else:
            print("[render] frame %d FAILED (no output)" % f)
    print("[render] done: %d frames in %.1fs" % (n_done, time.time() - t_all))


main()


