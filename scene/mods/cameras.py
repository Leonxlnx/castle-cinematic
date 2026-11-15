"""Cinematic camera work: 7 shots over 1080 frames (45 s @ 24 fps), cut on the musical accents of the
soundtrack (see assets/audio/ANALYSIS.md).  Each shot has its own camera with baked per-frame keys;
timeline markers bind the cameras, and SHOTS is used by the final renderer.

A shot = dict(name, t0, t1 (seconds), lens (mm) or (lens0, lens1), pos [control points], tgt [control points],
              ease (0 = linear .. 1 = full smoothstep), fstop/None, focus = 'target' | distance, roll deg)
Positions/targets are Catmull-Rom splines through the control points.
"""
import bpy, math
from mathutils import Vector, Quaternion
from .. import layout as L
from ..common import get_col

FPS = 24

SHOTS = [
    # 1 - establishing: from out over the valley SE of the cliff, sun on the left, castle on its rock
    dict(name="s1_establish", t0=0.0, t1=8.5, lens=(30.0, 34.0), ease=0.35,
         pos=[(250.0, -372.0, 20.0), (205.0, -305.0, 31.0), (160.0, -245.0, 40.0)],
         tgt=[(-40.0, 10.0, -8.0), (-34.0, 14.0, 0.0), (-28.0, 18.0, 6.0)]),
    # 2 - rising up the waterfall and cresting the infinity edge
    dict(name="s2_waterfall", t0=8.5, t1=14.25, lens=24.0, ease=0.55,
         pos=[(-58.0, -92.0, -60.0), (-56.0, -84.0, -30.0), (-52.0, -74.0, -6.0), (-49.0, -66.0, 7.5)],
         tgt=[(-40.0, -51.0, -52.0), (-39.0, -50.0, -22.0), (-30.0, -30.0, 0.0), (-8.0, 14.0, 13.0)]),
    # 3 - low glide over the pool toward the glowing great window
    dict(name="s3_pool", t0=14.25, t1=20.5, lens=26.0, ease=0.3, fstop=9.0, focus=40.0,
         pos=[(-35.5, -46.0, 0.75), (-34.5, -30.0, 0.95), (-33.0, -12.0, 1.5)],
         tgt=[(0.0, 22.0, 9.0), (1.0, 22.0, 12.0), (2.0, 22.0, 15.0)]),
    # 4 - the lounge: across the fire pit toward the castle
    dict(name="s4_fire", t0=20.5, t1=27.0, lens=35.0, ease=0.4, fstop=5.6, focus='target',
         pos=[(35.6, -64.6, 0.70), (33.2, -64.9, 0.92), (30.4, -65.0, 1.25)],
         tgt=[(31.0, -55.0, -0.3), (29.5, -46.0, 1.8), (21.0, -20.0, 7.5)]),
    # 5 - aerial: spires and flags against the sun
    dict(name="s5_spires", t0=27.0, t1=33.75, lens=50.0, ease=0.3,
         pos=[(118.0, -8.0, 66.0), (112.0, 6.0, 69.0), (104.0, 20.0, 72.0)],
         tgt=[(10.0, 50.0, 58.0), (10.0, 51.0, 60.0), (10.0, 52.0, 62.0)]),
    # 6 - up the grand stair to the great window (short pre-climax beat)
    dict(name="s6_stairs", t0=33.75, t1=36.75, lens=22.0, ease=0.25, fstop=8.0, focus=18.0,
         pos=[(2.0, 2.5, 1.25), (2.0, 5.0, 1.45), (2.0, 7.5, 1.7)],
         tgt=[(2.0, 22.0, 9.0), (2.0, 22.0, 12.5), (2.0, 22.0, 16.0)]),
    # 7 - the hero wide: pull back past the maple leaves to the full panorama (reference image 1)
    dict(name="s7_hero", t0=36.75, t1=45.0, lens=(30.0, 26.0), ease=0.45, fstop=None,
         pos=[(37.0, -70.5, 6.2), (38.6, -72.4, 8.0), (40.6, -75.2, 10.2)],
         tgt=[(-12.0, 20.0, 12.0), (-13.0, 21.0, 12.5), (-14.0, 22.0, 13.0)]),
]


# collections hidden per shot (things that cannot be seen in that shot) - purely a render-time optimisation.
# Filled in after profiling; names are bpy.data.collections names.
SHOT_HIDE = {}
# object-name prefixes hidden per shot (verified invisible from that shot's camera path)
SHOT_HIDE_OBJ = {
    "s3_pool":   ["landscape_cliff"],
    "s4_fire":   ["landscape_cliff"],
    "s5_spires": ["landscape_cliff"],
    "s6_stairs": ["landscape_", "veg_forest"],
}


def apply_shot_visibility(shot_name):
    names = set()
    for v in SHOT_HIDE.values():
        names.update(v)
    hide = set(SHOT_HIDE.get(shot_name, []))
    for n in names:
        col = bpy.data.collections.get(n)
        if col is not None:
            col.hide_render = (n in hide)
    prefixes = set()
    for v in SHOT_HIDE_OBJ.values():
        prefixes.update(v)
    hp = tuple(SHOT_HIDE_OBJ.get(shot_name, []))
    ap = tuple(prefixes)
    if ap:
        for o in bpy.data.objects:
            if o.name.startswith(ap):
                o.hide_render = bool(hp) and o.name.startswith(hp)

def frame_range(s):
    """Inclusive frame range of a shot (frame 1 = t 0)."""
    f0 = int(round(s['t0'] * FPS)) + 1
    f1 = int(round(s['t1'] * FPS))
    return f0, f1


def _catmull(pts, u):
    """u in 0..1 along the whole control polygon (uniform Catmull-Rom, endpoints duplicated)."""
    P = [Vector(p) for p in pts]
    n = len(P)
    if n == 1:
        return P[0].copy()
    if n == 2:
        return P[0].lerp(P[1], u)
    x = u * (n - 1)
    i = min(int(x), n - 2)
    t = x - i
    p0 = P[max(i - 1, 0)]; p1 = P[i]; p2 = P[i + 1]; p3 = P[min(i + 2, n - 1)]
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def _ease(u, k):
    s = u * u * (3 - 2 * u)
    return u * (1 - k) + s * k


def build(ctx):
    scene = ctx.scene
    col = get_col("cameras")
    scene.timeline_markers.clear()
    ctx.shot_cams = {}
    for s in SHOTS:
        f0, f1 = frame_range(s)
        cd = bpy.data.cameras.new(s['name'])
        cd.sensor_width = 36.0
        cd.clip_start = 0.25; cd.clip_end = 90000.0
        lens = s['lens']
        cam = bpy.data.objects.new(s['name'], cd)
        col.objects.link(cam)
        cam.rotation_mode = 'XYZ'
        fstop = s.get('fstop')
        if fstop:
            cd.dof.use_dof = True
            cd.dof.aperture_fstop = fstop
            cd.dof.aperture_blades = 9
        prev = None
        n = max(1, f1 - f0)
        for f in range(f0, f1 + 1):
            u = _ease((f - f0) / n, s.get('ease', 0.4))
            p = _catmull(s['pos'], u)
            t = _catmull(s['tgt'], u)
            cam.location = p
            q = (t - p).to_track_quat('-Z', 'Y')
            e = q.to_euler('XYZ', prev) if prev is not None else q.to_euler('XYZ')
            if s.get('roll'):
                e.rotate_axis('Z', math.radians(s['roll']))
            cam.rotation_euler = e
            prev = e.copy()
            cam.keyframe_insert("location", frame=f)
            cam.keyframe_insert("rotation_euler", frame=f)
            if isinstance(lens, tuple):
                cd.lens = lens[0] + (lens[1] - lens[0]) * u
                cd.keyframe_insert("lens", frame=f)
            if fstop:
                foc = s.get('focus', 'target')
                cd.dof.focus_distance = (t - p).length if foc == 'target' else float(foc)
                cd.dof.keyframe_insert("focus_distance", frame=f)
        if not isinstance(lens, tuple):
            cd.lens = lens
        m = scene.timeline_markers.new(s['name'], frame=f0)
        m.camera = cam
        ctx.shot_cams[s['name']] = cam
        ctx.cams[s['name']] = cam
    scene.camera = ctx.shot_cams[SHOTS[-1]['name']]
    scene.frame_start = 1
    scene.frame_end = frame_range(SHOTS[-1])[1]


def shot_at(frame):
    for s in SHOTS:
        f0, f1 = frame_range(s)
        if f0 <= frame <= f1:
            return s
    return SHOTS[-1]



