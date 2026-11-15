"""Birds: two small flocks of dark birds for scale and life (driver-animated, deterministic per frame).

  flock A circles the keep spires (seen in the spires shot and far away in the wide shots)
  flock B drifts north along the cliff edge west of the pool (seen in the hero / establishing shots)
"""
import bpy, math, random
from ..common import get_col, new_material

N_SPIRE = 10
N_VALLEY = 9


def _mat():
    m = bpy.data.materials.get("bird")
    if m:
        return m
    m, nt, out = new_material("bird")
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.012, 0.012, 0.016, 1), Roughness=0.6)
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


def _mesh(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.materials.append(_mat())
    return me


def _drv(obj, path, idx, expr):
    fc = obj.driver_add(path, idx)
    fc.driver.expression = expr
    return fc


def build(ctx):
    col = get_col("birds")
    rnd = random.Random(5)
    body = _mesh("bird_body", [(0, 0.24, 0.0), (0.05, 0.02, 0.0), (0, 0.02, 0.045), (-0.05, 0.02, 0.0), (0, 0.02, -0.04),
                               (0, -0.20, 0.0), (0.07, -0.30, 0.0), (-0.07, -0.30, 0.0)],
                 [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (5, 2, 1), (5, 3, 2), (5, 4, 3), (5, 1, 4), (5, 6, 7)])
    wing_r = _mesh("bird_wing_r", [(0.03, 0.10, 0), (0.03, -0.10, 0), (0.34, -0.12, 0), (0.62, -0.02, 0), (0.36, 0.09, 0)], [(0, 1, 2, 3, 4)])
    wing_l = _mesh("bird_wing_l", [(-0.03, 0.10, 0), (-0.36, 0.09, 0), (-0.62, -0.02, 0), (-0.34, -0.12, 0), (-0.03, -0.10, 0)], [(0, 1, 2, 3, 4)])

    def bird(i, scale, loc_exprs, rot_expr, flap_hz, phase):
        b = bpy.data.objects.new("bird_%02d" % i, body); col.objects.link(b)
        b.scale = (scale, scale, scale)
        b.rotation_mode = 'XYZ'
        for k in range(3):
            _drv(b, "location", k, loc_exprs[k])
        _drv(b, "rotation_euler", 2, rot_expr)
        w = flap_hz * math.tau / 24.0
        for nm, me, sgn in (("r", wing_r, -1), ("l", wing_l, 1)):
            o = bpy.data.objects.new("bird_%02d_%s" % (i, nm), me); col.objects.link(o)
            o.parent = b
            o.rotation_mode = 'XYZ'
            # flap with short glides: amplitude envelope stays between 0.15 and 1
            _drv(o, "rotation_euler", 1, "%d*(0.12+0.62*max(0.15, sin(frame*%.4f+%.3f)*0.5+0.6)*sin(frame*%.4f+%.3f))"
                 % (sgn, 0.021 + 0.004 * (i % 3), phase * 2.1, w, phase))
            o.visible_shadow = False
        b.visible_shadow = False

    # flock A: circling the keep spires
    for i in range(N_SPIRE):
        R = rnd.uniform(16.0, 44.0)
        w = rnd.choice((-1, 1)) * rnd.uniform(14.0, 19.0) / R / 24.0 * 1.0     # rad per frame (speed 14-19 m/s)
        p = rnd.uniform(0, math.tau)
        z0 = rnd.uniform(64.0, 90.0)
        cx, cy = 13.0 + rnd.uniform(-6, 6), 58.0 + rnd.uniform(-6, 6)
        loc = ("%.2f+%.2f*cos(frame*%.5f+%.3f)" % (cx, R, w, p),
               "%.2f+%.2f*sin(frame*%.5f+%.3f)" % (cy, R, w, p),
               "%.2f+2.2*sin(frame*0.017+%.3f)" % (z0, p * 1.7))
        rot = "frame*%.5f+%.3f%s" % (w, p, "" if w > 0 else "+3.14159")
        bird(i, rnd.uniform(1.0, 1.35), loc, rot, rnd.uniform(2.6, 3.4), p)

    # flock B: loose group gliding north past the cliff edge, west of the pool
    for j in range(N_VALLEY):
        i = N_SPIRE + j
        ox, oy, oz = rnd.uniform(-22, 22), rnd.uniform(-30, 30), rnd.uniform(-7, 9)
        p = rnd.uniform(0, math.tau)
        vx, vy = 0.052, 0.43                      # m per frame (about 10 m/s, heading NNE)
        loc = ("%.2f+%.4f*frame+3.0*sin(frame*0.011+%.3f)" % (-118.0 + ox, vx, p),
               "%.2f+%.4f*frame" % (-255.0 + oy, vy),
               "%.2f+0.010*frame+1.6*sin(frame*0.019+%.3f)" % (22.0 + oz, p * 1.3))
        rot = "%.4f+0.05*sin(frame*0.011+%.3f)" % (math.atan2(vy, vx) - math.pi / 2, p)
        bird(i, rnd.uniform(1.0, 1.4), loc, rot, rnd.uniform(2.4, 3.2), p)
    print("[birds] %d birds" % (N_SPIRE + N_VALLEY))
