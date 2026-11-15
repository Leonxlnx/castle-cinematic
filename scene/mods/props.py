"""PROPS module: lanterns, sunbeds, fire-pit lounge, planters and pots (collection "props").

Builds
  * lanterns     at layout.LANTERNS: black-metal + glass candle lanterns ("floor" ~0.9 m in two sizes, "small"
                 ~0.5 m), real instancing (2 shared meshes), each with ONE shadowless warm point light
                 (custom cutoff) whose energy flickers subtly (frame-driven simple-expression drivers).
                 Lanterns that would collide with a planter / pot are nudged just clear of it.
  * sunbeds      at layout.SUNBEDS (one shared mesh): teak frame, tufted cream cushion with raised back,
                 piping, head pillow and rolled towel; a round teak side table between each pair (glasses +
                 carafe or an unlit tea-light lantern).
  * lounge       L-shaped deep sectional on layout.SOFAS (dark base + frame, thick cream seat/back cushions with
                 sag variation, terracotta / grey throw pillows, folded knit blanket), 2 low side tables (one with
                 a tray, bottle and glasses), round stone fire pit at FIREPIT_POS (radius 2.2 m, warm LED
                 reveal at the plinth), lava-rock bed with glowing embers, ANIMATED FIRE (layered flame tongues:
                 instanced spindle meshes, additive alpha-cut emission whose noise rises with time, drivers
                 for height/sway), rising sparks, one flickering shadowless fire light.
  * planters     at layout.PLANTERS: 0.5 m stone troughs with moulded rim + soil at 0.45 m.
  * pots         two big terracotta pots (1.1 m) under MAPLES[2] and MAPLES[3] (soil at ~0.98 m).
  * stand-ins    temporary floors when 'terrace' is not built (main terrace only if 'water' is not built).

Tunable constants: LANTERN_W_FLOOR / LANTERN_W_SMALL (W), LANTERN_CUTOFF, FIRE_W, FIRE_CUTOFF, FLAME_GAIN,
FLICKER (relative amplitude), N_SPARKS.  All light energies are multiplied by layout.PRACTICAL_GAIN and all
emission strengths by layout.EMISSIVE_GAIN.
"""
import bpy, math, random
import numpy as np
from mathutils import Vector, Matrix

from ..common import MB, get_col, new_material, NT, drive_frame, xf, TAU
from .. import mats
from .. import layout as L

# ------------------------------------------------------------------------------------------- tunables
LANTERN_W_FLOOR = 55.0        # W per floor lantern (x PRACTICAL_GAIN)
LANTERN_W_SMALL = 22.0
LANTERN_CUTOFF = 9.0
LANTERN_COLOR = (1.0, 0.56, 0.26)
FLICKER = 1.0                 # 0 = steady lights
FIRE_W = 900.0                # fire point light (x PRACTICAL_GAIN)
FIRE_CUTOFF = 18.0
FIRE_COLOR = (1.0, 0.48, 0.18)
FLAME_GAIN = 0.62             # multiplier on flame emission (x EMISSIVE_GAIN)
FLAME_HEIGHT = 1.25           # multiplier on the flame tongue heights
N_SPARKS = 22

_R = random.Random(11)


# =========================================================================================== geometry helpers
def _axis_samples(h, r, seg, inner):
    r = min(r, h)
    a = np.linspace(-h, -(h - r), seg + 1)
    b = np.linspace(-(h - r), h - r, inner + 1)
    c = np.linspace(h - r, h, seg + 1)
    s = np.concatenate([a, b, c])
    return np.unique(np.round(s, 7))


def rbox(mb, c, size, r, mat, seg=2, inner=(1, 1, 1), deform=None, m=None):
    """Rounded box (welded, outward wound) centred at c with full size (sx, sy, sz) and edge radius r.
    deform(V, half) may modify the local vertex array (N,3) in place before placement."""
    half = np.array(size, dtype=np.float64) * 0.5
    r = min(r, float(half.min()) * 0.999)
    S = [_axis_samples(half[i], r, seg, inner[i]) for i in range(3)]
    keys = {}
    verts = []
    faces = []
    # (fixed axis k, sign, axis i, axis j) with e_i x e_j = sign * e_k
    spec = [(2, 1, 0, 1), (2, -1, 1, 0), (0, 1, 1, 2), (0, -1, 2, 1), (1, 1, 2, 0), (1, -1, 0, 2)]
    for k, sg, i, j in spec:
        Si, Sj = S[i], S[j]
        idx = np.empty((len(Si), len(Sj)), dtype=np.int64)
        for a, va in enumerate(Si):
            for b, vb in enumerate(Sj):
                p = [0.0, 0.0, 0.0]
                p[k] = sg * half[k]; p[i] = va; p[j] = vb
                key = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
                vi = keys.get(key)
                if vi is None:
                    vi = len(verts); keys[key] = vi; verts.append(p)
                idx[a, b] = vi
        for a in range(len(Si) - 1):
            for b in range(len(Sj) - 1):
                faces.append((idx[a, b], idx[a + 1, b], idx[a + 1, b + 1], idx[a, b + 1]))
    V = np.array(verts)
    inner_h = half - r
    C = np.clip(V, -inner_h, inner_h)
    D = V - C
    n = np.linalg.norm(D, axis=1)[:, None]
    n[n < 1e-12] = 1.0
    V = C + D / n * r
    if deform is not None:
        deform(V, half)
    V = V + np.array(c)
    mb.add(V, faces, mat, m=m)


def tube_loop(mb, pts, radius, mat, nseg=6, closed=True):
    """Round tube along a 3D polyline (piping, rings)."""
    P = [Vector(p) for p in pts]
    n = len(P)
    verts = []
    for i in range(n):
        if closed:
            t = (P[(i + 1) % n] - P[i - 1]).normalized()
        else:
            t = (P[min(i + 1, n - 1)] - P[max(i - 1, 0)]).normalized()
        up = Vector((0, 0, 1)) if abs(t.z) < 0.9 else Vector((1, 0, 0))
        n1 = t.cross(up).normalized(); n2 = t.cross(n1).normalized()
        for k in range(nseg):
            a = TAU * k / nseg
            q = P[i] + (n1 * math.cos(a) + n2 * math.sin(a)) * radius
            verts.append(tuple(q))
    faces = []
    segs = n if closed else n - 1
    for i in range(segs):
        i1 = (i + 1) % n
        for k in range(nseg):
            k1 = (k + 1) % nseg
            faces.append((i * nseg + k, i * nseg + k1, i1 * nseg + k1, i1 * nseg + k))
    # orientation check: first face normal must point away from the tube axis
    v = [Vector(verts[x]) for x in faces[0]]
    fn = (v[1] - v[0]).cross(v[3] - v[0])
    if fn.dot(v[0] - P[0]) < 0:
        faces = [tuple(reversed(f)) for f in faces]
    mb.add(verts, faces, mat)


def rrect_path(hx, hy, r, z, n=4):
    """Rounded rectangle outline (CCW) in the plane z."""
    pts = []
    for cx, cy, a0 in ((hx - r, -hy + r, -math.pi / 2), (hx - r, hy - r, 0.0), (-hx + r, hy - r, math.pi / 2),
                       (-hx + r, -hy + r, math.pi)):
        for k in range(n + 1):
            a = a0 + (math.pi / 2) * k / n
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a), z))
    return pts


def pillow(mb, c, sx, sy, thick, mat, m=None, n=12, pinch=0.06, sag=0.0):
    """Throw pillow: top/bottom surfaces that meet at the seam, pinched corners."""
    keys = {}; verts = []; faces = []
    us = np.linspace(-1, 1, n + 1)

    def vid(i, j, top):
        u, v = us[i], us[j]
        edge = (i in (0, n)) or (j in (0, n))
        key = (i, j, 0 if edge else (1 if top else -1))
        if key in keys:
            return keys[key]
        t = thick * 0.5 * (max(0.0, 1 - abs(u) ** 2.4) ** 0.55) * (max(0.0, 1 - abs(v) ** 2.4) ** 0.55)
        t *= (1.0 - sag * (1 - u * u) * (1 - v * v)) if top else 1.0
        x = u * sx * 0.5 * (1.0 - pinch * (1 - v * v) * 0.0 - pinch * v * v * 0.0)
        y = v * sy * 0.5
        # pinched (slightly concave) sides
        x *= 1.0 - pinch * (1 - v * v) * abs(u) ** 6
        y *= 1.0 - pinch * (1 - u * u) * abs(v) ** 6
        z = t if top else -t
        keys[key] = len(verts); verts.append((x, y, z))
        return keys[key]

    for i in range(n):
        for j in range(n):
            faces.append((vid(i, j, True), vid(i + 1, j, True), vid(i + 1, j + 1, True), vid(i, j + 1, True)))
            faces.append((vid(i, j + 1, False), vid(i + 1, j + 1, False), vid(i + 1, j, False), vid(i, j, False)))
    V = np.array(verts) + np.array(c)
    mb.add(V, faces, mat, m=m)


def _cushion_deform(puff=0.02, sag=0.0, side=0.01, tufts=None, tuft_depth=0.012, tuft_r=0.045):
    def f(V, half):
        hx, hy, hz = half
        x, y, z = V[:, 0], V[:, 1], V[:, 2]
        fx = np.clip(1 - (x / hx) ** 2, 0, 1); fy = np.clip(1 - (y / hy) ** 2, 0, 1)
        zt = np.clip(z / hz, -1, 1)
        V[:, 2] += puff * fx * fy * zt
        V[:, 2] -= sag * fx * fy * np.clip(zt, 0, 1)
        V[:, 0] += side * np.sign(x) * np.clip(np.abs(x) / hx, 0, 1) ** 4 * (1 - zt ** 2) * fy
        V[:, 1] += side * np.sign(y) * np.clip(np.abs(y) / hy, 0, 1) ** 4 * (1 - zt ** 2) * fx
        if tufts:
            top = np.clip(zt, 0, 1) ** 3
            for (tx, ty) in tufts:
                d2 = (x - tx) ** 2 + (y - ty) ** 2
                V[:, 2] -= tuft_depth * np.exp(-d2 / (tuft_r * tuft_r)) * top
    return f


# =========================================================================================== materials
def _time(nt, mul=1.0):
    v = nt.n('ShaderNodeValue')
    drive_frame(v.outputs[0], 'default_value', "frame/24*%.5f" % mul)
    return v.outputs[0]


@mats.material("lantern_glass2")
def _m_lantern_glass(name):
    """Clear lantern glass, lit from inside: fresnel reflection + transparency + warm glow around the flame."""
    m, nt, out = new_material(name)
    tc = nt.n('ShaderNodeTexCoord')
    z = nt.sep_xyz(tc.outputs['Object'])[2]
    glow = nt.math('ADD', nt.math('MULTIPLY', nt.math('EXPONENT', nt.math('MULTIPLY', nt.math('POWER',
              nt.math('DIVIDE', nt.math('SUBTRACT', z, 0.30), 0.16), 2.0), -1.0)), 2.2), 0.35)
    lw = nt.n('ShaderNodeLayerWeight', Blend=0.12)
    gl = nt.n('ShaderNodeBsdfGlossy', Color=(1, 1, 1, 1), Roughness=0.06)
    tr = nt.n('ShaderNodeBsdfTransparent', Color=(0.96, 0.90, 0.80, 1))
    f = nt.map_range(lw.outputs['Fresnel'], 0.0, 1.0, 0.12, 0.9)
    mix = nt.n('ShaderNodeMixShader', i0=f, i1=tr.outputs[0], i2=gl.outputs[0])
    em = nt.n('ShaderNodeEmission', Color=(1.0, 0.58, 0.26, 1), Strength=nt.math('MULTIPLY', glow, L.EMISSIVE_GAIN))
    add = nt.n('ShaderNodeAddShader', i0=mix.outputs[0], i1=em.outputs[0])
    nt.link(add.outputs[0], out.inputs['Surface'])
    m.surface_render_method = 'DITHERED'
    return m


@mats.material("candle_wax")
def _m_wax(name):
    m, nt, out = new_material(name)
    tc = nt.n('ShaderNodeTexCoord')
    z = nt.sep_xyz(tc.outputs['Object'])[2]
    glow = nt.map_range(z, 0.17, 0.25, 0.0, 3.0, interp='SMOOTHSTEP')
    b = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.86, 0.80, 0.66, 1), Roughness=0.45,
             Emission_Color=(1.0, 0.62, 0.30, 1), Emission_Strength=nt.math('MULTIPLY', glow, L.EMISSIVE_GAIN))
    b.inputs['Subsurface Weight'].default_value = 0.4
    nt.link(b.outputs[0], out.inputs['Surface'])
    return m


@mats.material("candle_flame")
def _m_candle_flame(name):
    m, nt, out = new_material(name)
    lw = nt.n('ShaderNodeLayerWeight', Blend=0.4)
    core = nt.math('SUBTRACT', 1.0, lw.outputs['Facing'])
    col = nt.ramp(core, [(0.0, (1.0, 0.35, 0.06, 1)), (0.6, (1.0, 0.70, 0.30, 1)), (1.0, (1.0, 0.93, 0.75, 1))])
    em = nt.n('ShaderNodeEmission', Color=col, Strength=nt.math('MULTIPLY', nt.map_range(core, 0, 1, 8.0, 40.0),
                                                                 L.EMISSIVE_GAIN))
    nt.link(em.outputs[0], out.inputs['Surface'])
    return m


@mats.material("teak")
def _m_teak(name):
    """Oiled teak: warm mid-brown, grain along the object's local X."""
    m, nt, out = new_material(name)
    tc = nt.n('ShaderNodeTexCoord')
    mp = nt.n('ShaderNodeMapping', Vector=tc.outputs['Object'])
    mp.inputs['Scale'].default_value = (0.6, 14.0, 14.0)
    g = nt.n('ShaderNodeTexNoise', Vector=mp.outputs[0], Scale=3.0, Detail=5.0, Distortion=0.5, Roughness=0.55)
    rings = nt.n('ShaderNodeTexWave', Vector=mp.outputs[0], Scale=0.9, Distortion=4.0, Detail=2.0)
    rings.wave_type = 'BANDS'
    v = nt.math('ADD', nt.math('MULTIPLY', g.outputs['Factor'], 0.7), nt.math('MULTIPLY', rings.outputs['Factor'], 0.3))
    col = nt.ramp(v, [(0.25, (0.105, 0.052, 0.024, 1)), (0.55, (0.20, 0.105, 0.050, 1)), (0.8, (0.26, 0.15, 0.075, 1))])
    b = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.42, Normal=nt.bump(v, 0.12, 0.002))
    b.inputs['Coat Weight'].default_value = 0.15
    nt.link(b.outputs[0], out.inputs['Surface'])
    return m


@mats.material("frame_dark")
def _m_frame_dark(name):
    """Powder-coated charcoal aluminium (sofa base / frames)."""
    m, nt, out = new_material(name)
    tc = nt.n('ShaderNodeTexCoord')
    n = nt.n('ShaderNodeTexNoise', Vector=tc.outputs['Object'], Scale=30.0, Detail=2.0)
    b = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.030, 0.028, 0.027, 1), Metallic=0.3, Roughness=0.52,
             Normal=nt.bump(n.outputs['Factor'], 0.05, 0.002))
    nt.link(b.outputs[0], out.inputs['Surface'])
    return m


def _fabric(name, color, sheen=0.35, weave=110.0, stripe=None):
    m, nt, out = new_material(name)
    tc = nt.n('ShaderNodeTexCoord')
    oc = tc.outputs['Object']
    w1 = nt.n('ShaderNodeTexNoise', Vector=oc, Scale=weave, Detail=2.0)
    big = nt.n('ShaderNodeTexNoise', Vector=oc, Scale=3.0, Detail=3.0)
    tone = nt.map_range(big.outputs['Factor'], 0.3, 0.7, 0.92, 1.06)
    col = nt.mix(1.0, color, nt.comb_xyz(tone, tone, tone), blend='MULTIPLY')
    if stripe is not None:
        s = nt.n('ShaderNodeTexWave', Vector=oc, Scale=stripe[0], Distortion=0.0)
        s.wave_type = 'BANDS'; s.bands_direction = 'Y'
        sm = nt.map_range(s.outputs['Factor'], 0.80, 0.86, 0.0, 1.0)
        col = nt.mix(sm, col, stripe[1])
    b = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.88, Normal=nt.bump(w1.outputs['Factor'], 0.25, 0.0015))
    b.inputs['Sheen Weight'].default_value = sheen
    b.inputs['Sheen Roughness'].default_value = 0.4
    b.inputs['Specular IOR Level'].default_value = 0.3
    nt.link(b.outputs[0], out.inputs['Surface'])
    return m


@mats.material("cushion_cream")
def _m_cushion(name):
    return _fabric(name, (0.66, 0.61, 0.53, 1))


@mats.material("fabric_terracotta")
def _m_terra_fab(name):
    return _fabric(name, (0.42, 0.17, 0.09, 1), weave=80.0)


@mats.material("fabric_grey")
def _m_grey_fab(name):
    return _fabric(name, (0.17, 0.165, 0.155, 1), weave=80.0)


@mats.material("fabric_piping")
def _m_piping(name):
    return _fabric(name, (0.50, 0.45, 0.38, 1), weave=150.0)


@mats.material("towel")
def _m_towel(name):
    return _fabric(name, (0.78, 0.76, 0.72, 1), sheen=0.6, weave=200.0, stripe=(9.0, (0.10, 0.17, 0.25, 1)))


@mats.material("knit")
def _m_knit(name):
    m, nt, out = new_material(name)
    tc = nt.n('ShaderNodeTexCoord')
    oc = tc.outputs['Object']
    w = nt.n('ShaderNodeTexWave', Vector=oc, Scale=45.0, Distortion=1.5, Detail=1.0)
    w.wave_type = 'BANDS'; w.bands_direction = 'X'
    b = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.29, 0.255, 0.215, 1), Roughness=0.95,
             Normal=nt.bump(w.outputs['Factor'], 0.5, 0.004))
    b.inputs['Sheen Weight'].default_value = 0.6
    nt.link(b.outputs[0], out.inputs['Surface'])
    return m


@mats.material("firepit_stone")
def _m_firepit_stone(name):
    """Smooth honed limestone / cast stone, sooted near the fire (object space: pit centre = origin)."""
    m, nt, out = new_material(name)
    tc = nt.n('ShaderNodeTexCoord')
    oc = tc.outputs['Object']
    xyz = nt.sep_xyz(oc)
    r = nt.vmath('LENGTH', nt.comb_xyz(xyz[0], xyz[1], 0.0))
    n1 = nt.n('ShaderNodeTexNoise', Vector=oc, Scale=1.2, Detail=5.0, Roughness=0.6)
    n2 = nt.n('ShaderNodeTexNoise', Vector=oc, Scale=25.0, Detail=3.0)
    col = nt.ramp(n1.outputs['Factor'], [(0.3, (0.36, 0.33, 0.29, 1)), (0.7, (0.47, 0.43, 0.38, 1))])
    soot = nt.math('MULTIPLY', nt.map_range(r, 1.95, 1.72, 0.0, 1.0, interp='SMOOTHSTEP'),
                   nt.map_range(n2.outputs['Factor'], 0.3, 0.7, 0.5, 1.0))
    col = nt.mix(nt.math('MULTIPLY', soot, 0.8), col, (0.03, 0.028, 0.026, 1))
    b = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.62,
             Normal=nt.bump(n2.outputs['Factor'], 0.12, 0.002))
    nt.link(b.outputs[0], out.inputs['Surface'])
    return m


@mats.material("led_warm")
def _m_led(name):
    m, nt, out = new_material(name)
    em = nt.n('ShaderNodeEmission', Color=(1.0, 0.58, 0.28, 1), Strength=5.0 * L.EMISSIVE_GAIN)
    nt.link(em.outputs[0], out.inputs['Surface'])
    return m


@mats.material("lava_rock")
def _m_lava(name):
    """Porous dark lava rock; embers glow in the cracks toward the pit centre, pulsing slowly."""
    m, nt, out = new_material(name)
    t = _time(nt)
    tc = nt.n('ShaderNodeTexCoord')
    oc = tc.outputs['Object']
    xyz = nt.sep_xyz(oc)
    r = nt.vmath('LENGTH', nt.comb_xyz(xyz[0], xyz[1], 0.0))
    por = nt.n('ShaderNodeTexVoronoi', Vector=oc, Scale=60.0)
    por.feature = 'F1'
    n = nt.n('ShaderNodeTexNoise', Vector=oc, Scale=14.0, Detail=3.0)
    col = nt.ramp(n.outputs['Factor'], [(0.3, (0.022, 0.020, 0.019, 1)), (0.7, (0.060, 0.045, 0.038, 1))])
    hot = nt.n('ShaderNodeTexNoise', noise_dimensions='4D', Vector=oc, W=nt.math('MULTIPLY', t, 0.35), Scale=6.0, Detail=2.0)
    mask = nt.map_range(r, 1.25, 0.35, 0.0, 1.0, interp='SMOOTHSTEP')
    crack = nt.map_range(por.outputs['Distance'], 0.0, 0.35, 1.0, 0.0)
    e = nt.math('MULTIPLY', nt.math('MULTIPLY', mask, nt.map_range(hot.outputs['Factor'], 0.42, 0.68, 0.0, 1.0)),
                nt.math('ADD', 0.25, crack))
    ecol = nt.ramp(e, [(0.0, (0.6, 0.06, 0.01, 1)), (0.6, (1.0, 0.25, 0.03, 1)), (1.0, (1.0, 0.5, 0.12, 1))])
    b = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.92,
             Normal=nt.bump(por.outputs['Distance'], 0.6, 0.01),
             Emission_Color=ecol, Emission_Strength=nt.math('MULTIPLY', nt.math('POWER', e, 1.3), 14.0 * L.EMISSIVE_GAIN))
    nt.link(b.outputs[0], out.inputs['Surface'])
    return m


@mats.material("fire_glass")
def _m_fire_glass(name):
    m, nt, out = new_material(name)
    b = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.85, 0.45, 0.12, 1), Roughness=0.08,
             Emission_Color=(1.0, 0.4, 0.08, 1), Emission_Strength=1.5 * L.EMISSIVE_GAIN)
    b.inputs['Coat Weight'].default_value = 0.5
    nt.link(b.outputs[0], out.inputs['Surface'])
    return m


@mats.material("fire_flame")
def _m_fire_flame(name):
    """Additive flame tongue: object space z = 0..1 along the tongue; alpha-cut emissive noise that rises and
    evolves with time; soft edges from the facing ratio, so no card edges are ever visible."""
    m, nt, out = new_material(name)
    t = _time(nt)
    tc = nt.n('ShaderNodeTexCoord')
    oc = tc.outputs['Object']
    oi = nt.n('ShaderNodeObjectInfo')
    rnd = nt.math('MULTIPLY', oi.outputs['Random'], 37.0)
    z = nt.sep_xyz(oc)[2]
    # distortion field (rises)
    mp = nt.n('ShaderNodeMapping', Vector=nt.vmath('SUBTRACT', oc, nt.comb_xyz(0.0, 0.0, nt.math('MULTIPLY', t, 1.7))))
    mp.inputs['Scale'].default_value = (3.0, 3.0, 1.6)
    dn = nt.n('ShaderNodeTexNoise', noise_dimensions='4D', Vector=mp.outputs[0], W=nt.math('ADD', nt.math('MULTIPLY', t, 0.9), rnd),
              Scale=1.4, Detail=2.0)
    dvec = nt.vmath('MULTIPLY', nt.vmath('SUBTRACT', dn.outputs['Color'], (0.5, 0.5, 0.5)),
                    nt.comb_xyz(nt.math('MULTIPLY', z, 0.5), nt.math('MULTIPLY', z, 0.5), nt.math('MULTIPLY', z, 0.3)))
    p2 = nt.vmath('ADD', oc, dvec)
    mp2 = nt.n('ShaderNodeMapping', Vector=nt.vmath('SUBTRACT', p2, nt.comb_xyz(0.0, 0.0, nt.math('MULTIPLY', t, 2.3))))
    mp2.inputs['Scale'].default_value = (4.5, 4.5, 2.4)
    n = nt.n('ShaderNodeTexNoise', noise_dimensions='4D', Vector=mp2.outputs[0], W=nt.math('ADD', nt.math('MULTIPLY', t, 1.3), rnd),
             Scale=1.3, Detail=3.0, Roughness=0.55)
    # fine licking detail, rising faster
    mp3 = nt.n('ShaderNodeMapping', Vector=nt.vmath('SUBTRACT', p2, nt.comb_xyz(0.0, 0.0, nt.math('MULTIPLY', t, 3.4))))
    mp3.inputs['Scale'].default_value = (11.0, 11.0, 4.5)
    nf = nt.n('ShaderNodeTexNoise', noise_dimensions='4D', Vector=mp3.outputs[0], W=nt.math('ADD', nt.math('MULTIPLY', t, 2.0), rnd),
              Scale=1.0, Detail=2.0)
    lw = nt.n('ShaderNodeLayerWeight', Blend=0.5)
    face = nt.math('SUBTRACT', 1.0, lw.outputs['Facing'])
    face = nt.math('POWER', nt.math('MAXIMUM', face, 0.0), 1.2)
    zz = nt.math('MINIMUM', nt.math('MAXIMUM', z, 0.0), 1.0)
    nn = nt.math('ADD', nt.math('MULTIPLY', n.outputs['Factor'], 0.75), nt.math('MULTIPLY', nf.outputs['Factor'], 0.35))
    dens = nt.math('ADD', nt.math('MULTIPLY', face, 0.62), nn)
    dens = nt.math('SUBTRACT', dens, nt.math('MULTIPLY', nt.math('POWER', zz, 1.1), 0.95))
    alpha = nt.map_range(dens, 0.50, 0.78, 0.0, 1.0, interp='SMOOTHSTEP')
    alpha = nt.math('MULTIPLY', alpha, nt.map_range(zz, 0.0, 0.08, 0.0, 1.0))
    heat = nt.math('MULTIPLY', nt.map_range(dens, 0.5, 1.1, 0.0, 1.0), nt.math('MULTIPLY', face, nt.map_range(zz, 0.0, 0.9, 1.0, 0.3)))
    heat = nt.math('MINIMUM', nt.math('MULTIPLY', heat, 1.3), 1.0)
    col = nt.ramp(heat, [(0.0, (0.80, 0.09, 0.012, 1)), (0.3, (1.0, 0.24, 0.03, 1)), (0.65, (1.0, 0.45, 0.10, 1)),
                         (1.0, (1.0, 0.70, 0.32, 1))])
    strength = nt.math('MULTIPLY', nt.math('ADD', 1.2, nt.math('MULTIPLY', nt.math('POWER', heat, 1.6), 20.0)), alpha)
    strength = nt.math('MULTIPLY', strength, FLAME_GAIN * L.EMISSIVE_GAIN)
    em = nt.n('ShaderNodeEmission', Color=col, Strength=strength)
    tr = nt.n('ShaderNodeBsdfTransparent', Color=(1, 1, 1, 1))
    add = nt.n('ShaderNodeAddShader', i0=tr.outputs[0], i1=em.outputs[0])
    nt.link(add.outputs[0], out.inputs['Surface'])
    m.surface_render_method = 'BLENDED'
    m.use_backface_culling = True
    try:
        m.use_transparency_overlap = True
    except Exception:
        pass
    return m


@mats.material("spark")
def _m_spark(name):
    m, nt, out = new_material(name)
    em = nt.n('ShaderNodeEmission', Color=(1.0, 0.45, 0.10, 1), Strength=30.0 * L.EMISSIVE_GAIN)
    nt.link(em.outputs[0], out.inputs['Surface'])
    return m


@mats.material("soil")
def _m_soil(name):
    m, nt, out = new_material(name)
    geo = nt.n('ShaderNodeNewGeometry')
    n = nt.n('ShaderNodeTexNoise', Vector=geo.outputs['Position'], Scale=18.0, Detail=5.0, Roughness=0.7)
    col = nt.ramp(n.outputs['Factor'], [(0.3, (0.020, 0.014, 0.010, 1)), (0.7, (0.060, 0.042, 0.030, 1))])
    b = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.95, Normal=nt.bump(n.outputs['Factor'], 0.8, 0.01))
    nt.link(b.outputs[0], out.inputs['Surface'])
    return m


@mats.material("terracotta")
def _m_terracotta(name):
    m, nt, out = new_material(name)
    geo = nt.n('ShaderNodeNewGeometry')
    P = geo.outputs['Position']
    n = nt.n('ShaderNodeTexNoise', Vector=P, Scale=2.5, Detail=5.0, Roughness=0.6)
    fine = nt.n('ShaderNodeTexNoise', Vector=P, Scale=40.0, Detail=3.0)
    col = nt.ramp(n.outputs['Factor'], [(0.3, (0.26, 0.10, 0.05, 1)), (0.7, (0.38, 0.16, 0.085, 1))])
    # lime bloom / weathering toward the bottom
    zl = nt.sep_xyz(nt.n('ShaderNodeTexCoord').outputs['Object'])[2]
    bloom = nt.math('MULTIPLY', nt.map_range(zl, 0.0, 0.5, 0.6, 0.0), nt.map_range(fine.outputs['Factor'], 0.4, 0.65, 0.0, 1.0))
    col = nt.mix(bloom, col, (0.42, 0.36, 0.30, 1))
    b = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.85, Normal=nt.bump(fine.outputs['Factor'], 0.2, 0.003))
    nt.link(b.outputs[0], out.inputs['Surface'])
    return m


@mats.material("glass_drink")
def _m_glass_drink(name):
    m, nt, out = new_material(name)
    lw = nt.n('ShaderNodeLayerWeight', Blend=0.15)
    gl = nt.n('ShaderNodeBsdfGlossy', Color=(1, 1, 1, 1), Roughness=0.02)
    tr = nt.n('ShaderNodeBsdfTransparent', Color=(0.97, 0.97, 0.95, 1))
    f = nt.map_range(lw.outputs['Fresnel'], 0.0, 1.0, 0.08, 1.0)
    mix = nt.n('ShaderNodeMixShader', i0=f, i1=tr.outputs[0], i2=gl.outputs[0])
    nt.link(mix.outputs[0], out.inputs['Surface'])
    m.surface_render_method = 'DITHERED'
    return m


@mats.material("wine")
def _m_wine(name):
    m, nt, out = new_material(name)
    b = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.18, 0.01, 0.02, 1), Roughness=0.05)
    nt.link(b.outputs[0], out.inputs['Surface'])
    return m


@mats.material("bottle_green")
def _m_bottle(name):
    m, nt, out = new_material(name)
    b = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.01, 0.035, 0.015, 1), Roughness=0.06)
    b.inputs['Specular IOR Level'].default_value = 0.8
    nt.link(b.outputs[0], out.inputs['Surface'])
    return m


# =========================================================================================== lanterns
_MESH = {}


def _lantern_mesh(kind):
    key = "lantern_" + kind
    if key in _MESH:
        return _MESH[key]
    mb = MB(key)
    if kind == "floor":
        w, hb, top = 0.15, 0.07, 0.64           # half width of the body, body bottom, body top
    else:
        w, hb, top = 0.14, 0.06, 0.52           # squat (scaled down to ~0.5 m by the object)
    M = "metal_dark"
    rbox(mb, (0, 0, 0.022), (2 * w + 0.03, 2 * w + 0.03, 0.044), 0.008, M)
    rbox(mb, (0, 0, 0.055), (2 * w - 0.01, 2 * w - 0.01, 0.026), 0.005, M)
    # frames + posts
    for z in (hb + 0.008, top - 0.008):
        for s in (-1, 1):
            rbox(mb, (0, s * (w - 0.012), z), (2 * w, 0.024, 0.018), 0.004, M)
            rbox(mb, (s * (w - 0.012), 0, z), (0.024, 2 * w, 0.018), 0.004, M)
    for sx in (-1, 1):
        for sy in (-1, 1):
            rbox(mb, (sx * (w - 0.012), sy * (w - 0.012), 0.5 * (hb + top)), (0.022, 0.022, top - hb), 0.004, M)
    # thin horizontal glazing bar at 1/3
    zb = hb + (top - hb) * 0.36
    for s in (-1, 1):
        rbox(mb, (0, s * (w - 0.012), zb), (2 * w - 0.02, 0.01, 0.01), 0.003, M)
        rbox(mb, (s * (w - 0.012), 0, zb), (0.01, 2 * w - 0.02, 0.01), 0.003, M)
    # glass panes (slightly inset)
    g = w - 0.014
    z0, z1 = hb + 0.017, top - 0.017
    mb.quad((-g + 0.01, -g, z0), (g - 0.01, -g, z0), (g - 0.01, -g, z1), (-g + 0.01, -g, z1), "lantern_glass2")
    mb.quad((g, -g + 0.01, z0), (g, g - 0.01, z0), (g, g - 0.01, z1), (g, -g + 0.01, z1), "lantern_glass2")
    mb.quad((g - 0.01, g, z0), (-g + 0.01, g, z0), (-g + 0.01, g, z1), (g - 0.01, g, z1), "lantern_glass2")
    mb.quad((-g, g - 0.01, z0), (-g, -g + 0.01, z0), (-g, -g + 0.01, z1), (-g, g - 0.01, z1), "lantern_glass2")
    # roof: square (4-segment lathe) hipped cap with a flared eave
    s2 = math.sqrt(2.0)
    rw = (w + 0.025) * s2
    prof = [(0.0, top), (rw, top), (rw, top + 0.018), (rw * 0.93, top + 0.03), (0.30 * rw, top + 0.13),
            (0.24 * rw, top + 0.15), (0.0, top + 0.15)]
    mb.lathe(prof, 4, M, a0=math.pi / 4, a1=math.pi / 4 + TAU)
    # finial knob + hanging ring
    kz = top + 0.15
    mb.lathe([(0.0, kz), (0.022, kz + 0.005), (0.03, kz + 0.025), (0.018, kz + 0.045), (0.008, kz + 0.05), (0.0, kz + 0.05)],
             10, M)
    rr = 0.045
    ring = [(rr * math.sin(TAU * k / 16), 0.0, kz + 0.05 + rr + rr * math.cos(TAU * k / 16) * 1.0) for k in range(16)]
    tube_loop(mb, ring, 0.006, M, nseg=6)
    # candle (pillar) + wick + flame
    cz = hb + 0.03
    cr = 0.042 if kind == "floor" else 0.048
    ch = 0.17 if kind == "floor" else 0.12
    mb.lathe([(0.0, cz), (cr, cz), (cr, cz + ch - 0.008), (cr - 0.006, cz + ch), (cr - 0.012, cz + ch - 0.004),
              (0.0, cz + ch - 0.006)], 16, "candle_wax")
    mb.lathe([(0.0, cz + ch - 0.006), (0.0015, cz + ch - 0.006), (0.0015, cz + ch + 0.012), (0.0, cz + ch + 0.012)],
             5, "metal_dark")
    fz = cz + ch + 0.006
    mb.lathe([(0.0, fz), (0.006, fz + 0.004), (0.0105, fz + 0.016), (0.0095, fz + 0.03), (0.005, fz + 0.046),
              (0.0, fz + 0.06)], 10, "candle_flame")
    me = mb.to_object(key + "_tmp", get_col("props"), smooth_angle=30.0, recalc=False)
    data = me.data
    bpy.data.objects.remove(me)
    data.name = key
    _MESH[key] = (data, fz + 0.025)
    return _MESH[key]


def _flicker_expr(e, k):
    if FLICKER <= 0:
        return "%.4f" % e
    a1, a2, a3 = 0.055 * FLICKER, 0.035 * FLICKER, 0.02 * FLICKER
    p1, p2, p3 = (k * 1.71) % 6.28, (k * 2.93) % 6.28, (k * 0.67) % 6.28
    return "%.4f*(1+%.3f*sin(frame*0.53+%.2f)+%.3f*sin(frame*1.37+%.2f)+%.3f*sin(frame*3.1+%.2f))" % (
        e, a1, p1, a2, p2, a3, p3)


def _add_light(col, name, loc, energy, color, cutoff, radius=0.03, expr=None, spec=1.0):
    ld = bpy.data.lights.new(name, 'POINT')
    ld.specular_factor = spec
    ld.energy = energy
    ld.color = color
    ld.use_shadow = False
    ld.use_custom_distance = True
    ld.cutoff_distance = cutoff
    ld.shadow_soft_size = radius
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    col.objects.link(ob)
    if expr:
        fc = ld.driver_add("energy")
        fc.driver.expression = expr
    return ob


def _keepouts():
    """Footprints (x0, x1, y0, y1) lanterns must stay clear of: planters, pots, sofas, tables."""
    k = [(p[0], p[1], p[2], p[3]) for p in L.PLANTERS]
    for mp in (L.MAPLES[2], L.MAPLES[3]):
        k.append((mp[0] - 0.68, mp[0] + 0.68, mp[1] - 0.68, mp[1] + 0.68))
    for s in L.SOFAS:
        k.append((s[0], s[1], s[2], s[3]))
    return k


def _nudge(x, y, half):
    for (x0, x1, y0, y1) in _keepouts():
        if x + half > x0 and x - half < x1 and y + half > y0 and y - half < y1:
            moves = [(x0 - half - 0.05 - x, 0), (x1 + half + 0.05 - x, 0), (0, y0 - half - 0.05 - y), (0, y1 + half + 0.05 - y)]
            dx, dy = min(moves, key=lambda d: abs(d[0]) + abs(d[1]))
            print("[props] lantern at (%.1f, %.1f) overlaps a planter/pot/sofa -> nudged by (%.2f, %.2f)" % (x, y, dx, dy))
            x += dx; y += dy
    return x, y


def _lanterns(col, lcol):
    rng = random.Random(5)
    n = 0
    for i, (x, y, z, kind) in enumerate(L.LANTERNS):
        if kind == "floor":
            data, fz = _lantern_mesh("floor")
            s = 1.0 if i % 3 else 1.1
            e = LANTERN_W_FLOOR * s * s
        else:
            data, fz = _lantern_mesh("small")
            s = 0.80 + 0.06 * (i % 2)
            e = LANTERN_W_SMALL
        x, y = _nudge(x, y, 0.19 * s)
        ob = bpy.data.objects.new("lantern_%02d" % i, data)
        ob.location = (x, y, z)
        ob.scale = (s, s, s)
        ob.rotation_euler = (0, 0, rng.uniform(-0.12, 0.12))
        col.objects.link(ob)
        _add_light(lcol, "lantern_light_%02d" % i, (x, y, z + fz * s), e * L.PRACTICAL_GAIN, LANTERN_COLOR,
                   LANTERN_CUTOFF, 0.03, _flicker_expr(e * L.PRACTICAL_GAIN, i))
        n += 1
    return n


# =========================================================================================== sunbeds
def _sunbed_mesh():
    if "sunbed" in _MESH:
        return _MESH["sunbed"]
    mb = MB("sunbed")
    T = "teak"
    Lh, Wh = 1.0, 0.375          # half length / half width;  +X = feet
    # legs
    for sx in (-1, 1):
        for sy in (-1, 1):
            rbox(mb, (sx * (Lh - 0.06), sy * (Wh - 0.045), 0.15), (0.06, 0.055, 0.30), 0.008, T)
    # side + end rails
    for sy in (-1, 1):
        rbox(mb, (0, sy * (Wh - 0.025), 0.245), (2 * Lh, 0.05, 0.11), 0.008, T)
    for sx in (-1, 1):
        rbox(mb, (sx * (Lh - 0.025), 0, 0.245), (0.05, 2 * Wh - 0.1, 0.11), 0.008, T)
    # slats (visible between rails)
    for k in range(13):
        x = -Lh + 0.12 + k * 0.145
        rbox(mb, (x, 0, 0.285), (0.09, 2 * Wh - 0.1, 0.018), 0.004, T)
    # backrest support (inclined panel) hinged at x = -0.33
    hinge = -0.33
    ang = math.radians(38)
    back_len = 0.66
    mb.push(Matrix.Translation((hinge, 0, 0.30)) @ Matrix.Rotation(-ang, 4, 'Y'))
    rbox(mb, (-back_len / 2, 0, -0.012), (back_len, 2 * Wh - 0.06, 0.024), 0.006, T)
    mb.pop()
    # strut under the backrest
    rbox(mb, (hinge - 0.38, 0, 0.36), (0.03, 2 * Wh - 0.12, 0.03), 0.006, T)
    # ---- cushion: seat part
    seat_len = Lh - 0.02 - hinge
    sx0 = hinge + seat_len / 2
    tufts = [(tx, ty) for tx in np.linspace(-seat_len / 2 + 0.17, seat_len / 2 - 0.17, 5) for ty in (-0.16, 0.16)]
    rbox(mb, (sx0, 0, 0.355), (seat_len, 2 * Wh - 0.03, 0.11), 0.035, "cushion_cream", seg=3, inner=(30, 12, 2),
         deform=_cushion_deform(puff=0.012, sag=0.006, side=0.006, tufts=tufts))
    for (tx, ty) in tufts:
        mb.lathe([(0.0, 0.395), (0.012, 0.397), (0.011, 0.402), (0.0, 0.404)], 8, "fabric_piping",
                 center=(sx0 + tx, ty))
    # piping around the top seam
    tube_loop(mb, [(p[0] + sx0, p[1], p[2]) for p in rrect_path(seat_len / 2 - 0.004, Wh - 0.019, 0.04, 0.395, 3)],
              0.006, "fabric_piping")
    # ---- cushion: back part
    mb.push(Matrix.Translation((hinge, 0, 0.30)) @ Matrix.Rotation(-ang, 4, 'Y'))
    btufts = [(tx, ty) for tx in (-0.15, -0.42) for ty in (-0.16, 0.16)]
    rbox(mb, (-back_len / 2 + 0.01, 0, 0.055), (back_len, 2 * Wh - 0.03, 0.11), 0.035, "cushion_cream", seg=3,
         inner=(14, 12, 2), deform=_cushion_deform(puff=0.012, sag=0.004, side=0.006,
                                                     tufts=[(t[0] + back_len / 2 - 0.01, t[1]) for t in btufts]))
    tube_loop(mb, [(p[0] - back_len / 2 + 0.01, p[1], p[2]) for p in rrect_path(back_len / 2 - 0.004, Wh - 0.019, 0.04, 0.095, 3)],
              0.006, "fabric_piping")
    # head pillow lying on the back cushion near the top
    pillow(mb, (-back_len + 0.17, 0, 0.165), 0.24, 0.5, 0.10, "cushion_cream", n=10, pinch=0.08)
    mb.pop()
    # rolled towel across the foot end
    zt = 0.405 + 0.055
    mb.push(Matrix.Translation((Lh - 0.26, 0, zt)) @ Matrix.Rotation(math.pi / 2, 4, 'X'))
    mb.lathe([(0.0, -0.27), (0.045, -0.27), (0.055, -0.262), (0.056, 0.0), (0.055, 0.262), (0.045, 0.27), (0.0, 0.27)],
             14, "towel")
    mb.pop()
    ob = mb.to_object("sunbed_tmp", get_col("props"), smooth_angle=40.0, recalc=False)
    data = ob.data; bpy.data.objects.remove(ob); data.name = "sunbed"
    _MESH["sunbed"] = data
    return data


def _side_table_mesh(kind):
    key = "sidetable_" + kind
    if key in _MESH:
        return _MESH[key]
    mb = MB(key)
    T = "teak"
    if kind == "sunbed":
        r, h = 0.23, 0.42
    else:
        r, h = 0.32, 0.46
    # turned teak top + dark metal pedestal
    mb.lathe([(0.0, h - 0.035), (r - 0.01, h - 0.035), (r, h - 0.025), (r, h - 0.008), (r - 0.008, h), (0.0, h)], 32, T)
    mb.lathe([(0.0, 0.0), (r * 0.62, 0.0), (r * 0.62, 0.012), (r * 0.55, 0.02), (0.03, 0.04), (0.025, h - 0.04),
              (0.06, h - 0.035), (0.0, h - 0.035)], 24, "frame_dark")
    ob = mb.to_object(key + "_tmp", get_col("props"), smooth_angle=35.0, recalc=False)
    data = ob.data; bpy.data.objects.remove(ob); data.name = key
    _MESH[key] = (data, h)
    return _MESH[key]


def _glass(mb, x, y, z, wine=True):
    # stemmed wine glass (outer + inner shell) with a little wine
    prof = [(0.0, z), (0.035, z), (0.036, z + 0.004), (0.006, z + 0.008), (0.004, z + 0.08), (0.02, z + 0.095),
            (0.04, z + 0.13), (0.042, z + 0.17), (0.044, z + 0.2), (0.0425, z + 0.2), (0.0405, z + 0.17),
            (0.038, z + 0.13), (0.018, z + 0.097), (0.0, z + 0.095)]
    mb.lathe(prof, 16, "glass_drink", center=(x, y))
    if wine:
        mb.lathe([(0.0, z + 0.097), (0.018, z + 0.099), (0.034, z + 0.122), (0.0, z + 0.124)], 14, "wine", center=(x, y))


def _bottle(mb, x, y, z):
    prof = [(0.0, z), (0.036, z), (0.038, z + 0.01), (0.038, z + 0.2), (0.03, z + 0.24), (0.014, z + 0.27),
            (0.0135, z + 0.33), (0.016, z + 0.335), (0.0, z + 0.34)]
    mb.lathe(prof, 18, "bottle_green", center=(x, y))


def _sunbeds(col):
    data = _sunbed_mesh()
    beds = L.SUNBEDS
    for i, (x, y, z, hd) in enumerate(beds):
        ob = bpy.data.objects.new("sunbed_%02d" % i, data)
        ob.location = (x, y, z); ob.rotation_euler = (0, 0, hd)
        col.objects.link(ob)
    # side tables between pairs (at the head end)
    tdata, th = _side_table_mesh("sunbed")
    mbd = MB("sunbed_table_deco")
    used = set()
    k = 0
    for i in range(len(beds)):
        for j in range(i + 1, len(beds)):
            if i in used or j in used:
                continue
            a, b = beds[i], beds[j]
            if abs(a[3] - b[3]) > 1e-3:
                continue
            dd = math.hypot(a[0] - b[0], a[1] - b[1])
            if dd > 3.3:
                continue
            used.update((i, j))
            hx, hy = -math.cos(a[3]), -math.sin(a[3])        # toward the head
            cx = 0.5 * (a[0] + b[0]) + hx * 0.62
            cy = 0.5 * (a[1] + b[1]) + hy * 0.62
            ob = bpy.data.objects.new("sunbed_table_%d" % k, tdata)
            ob.location = (cx, cy, a[2]); col.objects.link(ob)
            if k % 2 == 0:
                _glass(mbd, cx - 0.07, cy + 0.05, a[2] + th, True)
                _glass(mbd, cx + 0.08, cy - 0.04, a[2] + th, k % 4 == 0)
            else:
                ldata, fz = _lantern_mesh("small")
                lo = bpy.data.objects.new("sunbed_table_lantern_%d" % k, ldata)
                lo.location = (cx, cy, a[2] + th); lo.scale = (0.5, 0.5, 0.5); col.objects.link(lo)
            k += 1
    if mbd.v:
        mbd.to_object("sunbed_table_deco", col, recalc=False)


# =========================================================================================== lounge
def _run_frame(x0, x1, y0, y1, f):
    """Local frame for a sofa run: local +x along the run, local +y from the front edge toward the back."""
    fd = Vector((math.cos(f), math.sin(f)))
    b = -fd
    rz = math.atan2(-b.x, b.y)
    xd = Vector((math.cos(rz), math.sin(rz)))
    cs = [Vector(c) for c in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    fmax = max(c.dot(fd) for c in cs)
    front = [c for c in cs if abs(c.dot(fd) - fmax) < 1e-6]
    o = min(front, key=lambda c: c.dot(xd))
    length = max(c.dot(xd) for c in cs) - o.dot(xd)
    depth = fmax - min(c.dot(fd) for c in cs)
    return o, rz, length, depth


def _lounge(col, rng):
    LZ = L.LOUNGE_Z
    mb = MB("lounge_sofa")
    east, south = L.SOFAS[0], L.SOFAS[1]
    BASE_H, SEAT_T, BACK_T, FRAME_T = 0.24, 0.20, 0.28, 0.30
    o_e, rz_e, len_e, dep_e = _run_frame(*east)
    o_s, rz_s, len_s, dep_s = _run_frame(*south)
    # the east run continues to the back of the south run (corner module)
    ext_e = (Vector((south[0], south[2])) - o_e).dot(Vector((math.cos(rz_e), math.sin(rz_e))))
    len_e2 = max(len_e, ext_e)                  # run length incl. the corner square
    seat_d = dep_e - FRAME_T - BACK_T
    pal = ["fabric_terracotta", "fabric_grey", "fabric_terracotta", "cushion_cream", "fabric_grey"]

    def run(o, rz, length, depth, arm_start, arm_end, seat_x0, seat_x1, back_x0, back_x1, frame_x0, pillows, ki):
        mb.push(xf((o.x, o.y, LZ), rz=rz))
        # base plinth with a recessed toe kick
        rbox(mb, (0.5 * (frame_x0 + length), 0.5 * depth + 0.03, 0.03), (length - frame_x0 - 0.12, depth - 0.12, 0.06), 0.01,
             "frame_dark")
        rbox(mb, (0.5 * (frame_x0 + length), 0.5 * depth, 0.06 + (BASE_H - 0.06) / 2), (length - frame_x0, depth, BASE_H - 0.06),
             0.02, "frame_dark", inner=(4, 2, 1))
        # back frame
        rbox(mb, (0.5 * (frame_x0 + length), depth - FRAME_T / 2, BASE_H + 0.27), (length - frame_x0, FRAME_T, 0.54), 0.03,
             "frame_dark", inner=(4, 1, 1))
        # arms
        for ax, on in ((0.125, arm_start), (length - 0.125, arm_end)):
            if on:
                rbox(mb, (ax, 0.5 * (depth - FRAME_T), BASE_H + 0.19), (0.25, depth - FRAME_T, 0.38), 0.03, "frame_dark")
        # seat cushions
        n = max(1, int(round((seat_x1 - seat_x0) / 1.05)))
        w = (seat_x1 - seat_x0) / n
        for k in range(n):
            cx = seat_x0 + (k + 0.5) * w
            sag = 0.012 + 0.012 * rng.random()
            rbox(mb, (cx, seat_d / 2 + 0.01, BASE_H + SEAT_T / 2), (w - 0.012, seat_d - 0.01, SEAT_T), 0.055,
                 "cushion_cream", seg=3, inner=(8, 10, 2), deform=_cushion_deform(puff=0.022, sag=sag, side=0.012))
            # welt / piping along the top seam
            zp = BASE_H + SEAT_T - 0.03
            tube_loop(mb, [(p[0] + cx, p[1] + seat_d / 2 + 0.01, p[2]) for p in
                           rrect_path((w - 0.012) / 2 - 0.001, (seat_d - 0.01) / 2 - 0.001, 0.05, zp, 2)],
                      0.006, "fabric_piping", nseg=5)
        # back cushions (leaning back ~12 deg)
        n = max(1, int(round((back_x1 - back_x0) / 1.05)))
        w = (back_x1 - back_x0) / n
        for k in range(n):
            cx = back_x0 + (k + 0.5) * w
            yb = depth - FRAME_T - BACK_T / 2 + 0.02
            m = Matrix.Translation((cx, yb, BASE_H + SEAT_T + 0.0)) @ Matrix.Rotation(math.radians(-12), 4, 'X')
            mb.push(m)
            rbox(mb, (0, 0, 0.27), (w - 0.014, BACK_T, 0.56), 0.07, "cushion_cream", seg=3, inner=(8, 2, 8),
                 deform=_cushion_deform(puff=0.0, side=0.02, sag=0.0))
            mb.pop()
        # throw pillows leaning on the back cushions
        for (px, mat, yaw) in pillows:
            m = Matrix.Translation((px, depth - FRAME_T - BACK_T - 0.06, BASE_H + SEAT_T + 0.24)) @ \
                Matrix.Rotation(yaw, 4, 'Z') @ Matrix.Rotation(math.radians(-72), 4, 'X')
            pillow(mb, (0, 0, 0), 0.5, 0.48, 0.17, mat, m=m, n=12, pinch=0.1, sag=0.1)
        mb.pop()

    # east run: arm at its start (north end), seat to the south back cushions, frame across the corner
    run(o_e, rz_e, len_e2, dep_e, True, False, 0.25, len_e2 - (FRAME_T + BACK_T) + 0.0, 0.25, len_e2 - FRAME_T, 0.0,
        [(1.0, pal[0], 0.1), (1.5, pal[1], -0.12), (6.2, pal[3], 0.05), (9.6, pal[4], -0.1), (10.15, pal[2], 0.15)], 0)
    # south run: starts at the east seat's front line, arm at its far (west) end; back frame reaches the east frame
    run(o_s, rz_s, len_s, dep_s, False, True, 0.0, len_s - 0.25, -(dep_e - FRAME_T - BACK_T), len_s - 0.25,
        -(dep_e - FRAME_T), [(3.4, pal[1], 0.08), (8.6, pal[0], -0.1), (9.1, pal[1], 0.12)], 1)
    # folded knit blanket on the north end of the east sofa
    mb.push(xf((o_e.x, o_e.y, LZ), rz=rz_e))
    mb.push(Matrix.Translation((0.85, 0.75, BASE_H + SEAT_T + 0.035)) @ Matrix.Rotation(0.08, 4, 'Z'))
    rbox(mb, (0, 0, 0), (0.62, 0.48, 0.05), 0.02, "knit", seg=2, inner=(6, 6, 1),
         deform=_cushion_deform(puff=0.006, sag=0.0, side=0.004))
    rbox(mb, (0.02, 0.03, 0.045), (0.6, 0.46, 0.04), 0.018, "knit", seg=2, inner=(6, 6, 1),
         deform=_cushion_deform(puff=0.008, sag=0.0, side=0.003))
    # the free end of the blanket draped over the seat front
    rbox(mb, (0.0, -0.31, -0.05), (0.6, 0.06, 0.16), 0.025, "knit", seg=2, inner=(6, 1, 2))
    mb.pop(); mb.pop()
    mb.to_object("lounge_sofa", col, smooth_angle=50.0, recalc=False)

    # side tables
    tdata, th = _side_table_mesh("lounge")
    xe = Vector((math.cos(rz_e), math.sin(rz_e))); ye = Vector((-math.sin(rz_e), math.cos(rz_e)))
    p1 = o_e - xe * 0.42 + ye * (dep_e * 0.5)
    xs = Vector((math.cos(rz_s), math.sin(rz_s))); ys = Vector((-math.sin(rz_s), math.cos(rz_s)))
    p2 = o_s + xs * (len_s + 0.42) + ys * (dep_s * 0.5)
    mbd = MB("lounge_deco")
    for k, p in enumerate((p1, p2)):
        ob = bpy.data.objects.new("lounge_table_%d" % k, tdata)
        ob.location = (p.x, p.y, LZ); col.objects.link(ob)
    # tray with bottle + glasses on the first table
    z = LZ + th
    rbox(mbd, (p1.x, p1.y, z + 0.012), (0.42, 0.28, 0.024), 0.008, "teak")
    _bottle(mbd, p1.x - 0.1, p1.y + 0.02, z + 0.024)
    _glass(mbd, p1.x + 0.05, p1.y + 0.07, z + 0.024, True)
    _glass(mbd, p1.x + 0.11, p1.y - 0.06, z + 0.024, True)
    # small tea-light lantern (no light) on the second table
    ldata, fz = _lantern_mesh("small")
    lo = bpy.data.objects.new("lounge_table_lantern", ldata)
    lo.location = (p2.x, p2.y, LZ + th); lo.scale = (0.55, 0.55, 0.55); col.objects.link(lo)
    mbd.to_object("lounge_deco", col, recalc=False)


# =========================================================================================== fire pit
def _flame_mesh():
    if "flame" in _MESH:
        return _MESH["flame"]
    prof = []
    nz = 18
    for i in range(nz + 1):
        z = i / nz
        if z < 0.2:
            r = 0.30 * (z / 0.2) ** 0.45
        else:
            r = 0.30 * (1.0 - 0.72 * ((z - 0.2) / 0.8) ** 1.5)     # blunt top: the shader cuts the tips
        prof.append((max(r, 0.0), z))
    prof[0] = (0.0, 0.0)
    prof.append((0.0, 1.03))
    mb = MB("flame")
    mb.lathe(prof, 14, "fire_flame")
    ob = mb.to_object("flame_tmp", get_col("props"), smooth_angle=None, recalc=False)
    data = ob.data; bpy.data.objects.remove(ob); data.name = "flame"
    _MESH["flame"] = data
    return data


def _firepit(col, lcol, rng):
    cx, cy, cz = L.FIREPIT_POS
    R_OUT, R_IN, H, BED = 2.2, 1.78, 0.44, 0.28
    mb = MB("firepit")
    # ring: plinth recess with a warm LED strip, outer wall, wide cap, inner wall, bed floor
    prof = [(R_OUT - 0.10, 0.0), (R_OUT - 0.10, 0.055), (R_OUT - 0.005, 0.06), (R_OUT, 0.07), (R_OUT, H - 0.03),
            (R_OUT - 0.008, H - 0.008), (R_OUT - 0.03, H), (R_IN + 0.03, H), (R_IN + 0.008, H - 0.008),
            (R_IN, H - 0.03), (R_IN, BED), (0.0, BED)]
    mb.lathe(prof, 96, "firepit_stone")
    mb.lathe([(R_OUT - 0.095, 0.004), (R_OUT - 0.07, 0.006), (R_OUT - 0.07, 0.054), (R_OUT - 0.095, 0.054)], 96, "led_warm")
    # burner ring (dark steel) mostly hidden under the rocks
    mb.lathe([(0.0, BED), (0.95, BED), (0.95, BED + 0.04), (0.0, BED + 0.04)], 32, "metal_dark")
    # lava rocks (irregular, flattened icosahedra), denser + bigger toward the rim
    ico = _ico()
    nrock = 0
    for k in range(1250):
        a = rng.random() * TAU
        r = R_IN * 0.97 * math.sqrt(rng.random())
        s = 0.045 + 0.05 * rng.random()
        x, y = r * math.cos(a), r * math.sin(a)
        if r > R_IN - s * 0.9:
            continue
        z = BED + 0.02 + s * 0.4 + 0.03 * rng.random() + (0.04 if r > 0.9 else 0.0)
        vv = []
        for (px, py, pz) in ico[0]:
            j = 0.7 + 0.5 * rng.random()
            vv.append((px * s * j, py * s * j * (0.8 + 0.4 * rng.random()), pz * s * j * 0.65))
        m = xf((x, y, z), rz=rng.random() * TAU, rx=rng.uniform(-0.4, 0.4))
        mb.add(vv, ico[1], "fire_glass" if (k % 23 == 0 and r < 1.3) else "lava_rock", m=m)
        nrock += 1
    ob = mb.to_object("firepit", col, smooth_angle=35.0, recalc=False)
    ob.location = (cx, cy, cz)
    # flames: tall core cluster + shorter outer tongues on the burner ring
    fdata = _flame_mesh()
    tongues = []
    for k in range(4):           # tall core
        a = rng.random() * TAU; r = 0.3 * math.sqrt(rng.random())
        tongues.append((r * math.cos(a), r * math.sin(a), 0.95 + 0.6 * rng.random(), 1.0 + 0.35 * rng.random()))
    for k in range(9):           # medium tongues around the core
        a = TAU * k / 9 + rng.uniform(-0.3, 0.3); r = 0.35 + 0.35 * rng.random()
        tongues.append((r * math.cos(a), r * math.sin(a), 0.45 + 0.5 * rng.random() ** 1.5, 0.75 + 0.3 * rng.random()))
    for k in range(11):          # small licks along the burner ring
        a = TAU * k / 11 + rng.uniform(-0.25, 0.25); r = 0.7 + 0.25 * rng.random()
        tongues.append((r * math.cos(a), r * math.sin(a), 0.2 + 0.3 * rng.random() ** 2, 0.55 + 0.3 * rng.random()))
    tongues = [(x, y, h * FLAME_HEIGHT, w) for (x, y, h, w) in tongues]
    for k, (x, y, h, w) in enumerate(tongues):
        o = bpy.data.objects.new("fire_flame_%02d" % k, fdata)
        o.location = (cx + x, cy + y, cz + BED + 0.06)
        sx = w * (0.75 + 0.2 * rng.random()); sy = w * (0.75 + 0.2 * rng.random())
        o.scale = (sx, sy, h)
        o.rotation_euler = (0, 0, rng.random() * TAU)
        o.visible_shadow = False
        col.objects.link(o)
        p = [rng.random() * 6.28 for _ in range(6)]
        fc = o.driver_add("scale", 2)
        fc.driver.expression = "%.3f*(1+0.12*sin(frame*0.41+%.2f)+0.08*sin(frame*0.97+%.2f)+0.05*sin(frame*2.3+%.2f))" % (h, p[0], p[1], p[2])
        for ax, pp in ((0, p[3]), (1, p[4])):
            fc = o.driver_add("rotation_euler", ax)
            fc.driver.expression = "%.3f*sin(frame*%.3f+%.2f)" % (0.07 + 0.05 * (x * x + y * y), 0.21 + 0.07 * ax, pp)
    # sparks (tiny stretched emissive octahedra, driven upward, new direction every cycle)
    sm = MB("spark")
    sm.add([(0.006, 0, 0), (-0.006, 0, 0), (0, 0.006, 0), (0, -0.006, 0), (0, 0, 0.024), (0, 0, -0.012)],
           [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)], "spark")
    so = sm.to_object("spark_tmp", col, smooth_angle=None, recalc=False)
    sdata = so.data; bpy.data.objects.remove(so)
    for k in range(N_SPARKS):
        rate = 1.0 / (24.0 * (1.1 + 1.2 * rng.random()))       # cycles per frame
        ph = rng.random()
        x0 = rng.uniform(-0.5, 0.5); y0 = rng.uniform(-0.5, 0.5)
        H_ = 1.6 + 1.8 * rng.random()
        o = bpy.data.objects.new("fire_spark_%02d" % k, sdata)
        o.visible_shadow = False
        col.objects.link(o)
        life = "fmod(frame*%.5f+%.3f,1.0)" % (rate, ph)
        cyc = "floor(frame*%.5f+%.3f)" % (rate, ph)
        e = o.driver_add("location", 0)
        e.driver.expression = "%.3f+%.3f+0.5*sin(%s*7.31+%.2f)*%s+0.06*sin(frame*0.35+%.2f)" % (cx, x0, cyc, k * 1.3, life, k)
        e = o.driver_add("location", 1)
        e.driver.expression = "%.3f+%.3f+0.5*sin(%s*4.17+%.2f)*%s+0.06*cos(frame*0.31+%.2f)" % (cy, y0, cyc, k * 2.1, life, k)
        e = o.driver_add("location", 2)
        e.driver.expression = "%.3f+pow(%s,0.8)*%.3f" % (cz + BED + 0.25, life, H_)
        for ax in range(3):
            e = o.driver_add("scale", ax)
            e.driver.expression = "clamp((1-%s)*1.4,0,1)*clamp(%s*12,0,1)" % (life, life)
    # fire light (flickering, shadowless)
    p = [rng.random() * 6 for _ in range(4)]
    expr = "%.2f*(1+0.10*sin(frame*0.71+%.2f)+0.07*sin(frame*1.83+%.2f)+0.05*sin(frame*3.7+%.2f)+0.03*sin(frame*5.3+%.2f))" % (
        FIRE_W * L.PRACTICAL_GAIN, p[0], p[1], p[2], p[3])
    _add_light(lcol, "fire_light", (cx, cy, cz + 0.95), FIRE_W * L.PRACTICAL_GAIN, FIRE_COLOR, FIRE_CUTOFF, 0.45, expr,
               spec=0.35)
    return nrock


def _ico():
    t = (1.0 + 5 ** 0.5) / 2.0
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    n = (1 + t * t) ** 0.5
    v = [(a / n, b / n, c / n) for a, b, c in v]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7),
         (9, 8, 1)]
    # make sure faces are outward
    out = []
    for a, b, c in f:
        A, B, C = Vector(v[a]), Vector(v[b]), Vector(v[c])
        if (B - A).cross(C - A).dot(A + B + C) < 0:
            out.append((a, c, b))
        else:
            out.append((a, b, c))
    return v, out


# =========================================================================================== planters + pots
def _planters(col):
    mb = MB("planters")
    for (x0, x1, y0, y1, z0) in L.PLANTERS:
        path = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]          # CCW -> right = outward
        prof = [(0.0, z0), (0.012, z0 + 0.004), (0.012, z0 + 0.07), (0.0, z0 + 0.085), (0.0, z0 + 0.39),
                (0.018, z0 + 0.405), (0.03, z0 + 0.43), (0.032, z0 + 0.47), (0.022, z0 + 0.492), (0.0, z0 + 0.5),
                (-0.14, z0 + 0.5), (-0.152, z0 + 0.494), (-0.155, z0 + 0.48), (-0.155, z0 + 0.42)]
        mb.sweep(prof, path, "stone_trim", closed=True)
        mb.box(x0 + 0.15, x1 - 0.15, y0 + 0.15, y1 - 0.15, z0 + 0.25, z0 + 0.45, "soil", skip="-z")
    mb.to_object("planters", col, smooth_angle=40.0, recalc=False)
    # big terracotta pots under the maples
    mbp = MB("pots")
    prof = [(0.0, 0.0), (0.40, 0.0), (0.42, 0.02), (0.40, 0.05), (0.44, 0.09), (0.55, 0.32), (0.61, 0.6),
            (0.62, 0.84), (0.60, 0.98), (0.62, 1.0), (0.67, 1.02), (0.68, 1.07), (0.66, 1.1), (0.60, 1.1),
            (0.575, 1.08), (0.57, 1.0), (0.0, 0.98)]
    for mp in (L.MAPLES[2], L.MAPLES[3]):
        mbp.push(xf((mp[0], mp[1], mp[2])))
        mbp.lathe(prof[:-1] + [(0.57, 0.98)], 48, "terracotta")
        mbp.lathe([(0.0, 0.975), (0.571, 0.975), (0.571, 0.98), (0.0, 0.982)], 48, "soil")
        mbp.pop()
    ob = mbp.to_object("pots", col, smooth_angle=40.0, recalc=False)


# =========================================================================================== stand-ins
def _standins(ctx, col):
    if 'terrace' in ctx.mods:
        return
    mb = MB("props_standin_floor")
    mb.box(L.LOUNGE_X0, L.LOUNGE_X1, L.LOUNGE_Y0, L.LOUNGE_Y1, L.LOUNGE_Z - 0.3, L.LOUNGE_Z, "marble_wet")
    if 'water' not in ctx.mods:
        mb.box(-24.1, 44.0, L.TERRACE_Y0, 20.0, -0.3, 0.0, "marble_wet")
        mb.box(-46.0, -24.1, 8.9, 20.0, -0.3, 0.0, "marble_wet")
        mb.box(-46.0, 60.0, 20.0, 27.0, -0.3, L.PODIUM_Z, "marble_wet")
    mb.to_object("props_standin_floor", col)


# =========================================================================================== build
def build(ctx):
    col = get_col("props")
    lcol = get_col("props_lights", col)
    rng = random.Random(23)
    _standins(ctx, col)
    nl = _lanterns(col, lcol)
    _sunbeds(col)
    _lounge(col, rng)
    nrock = _firepit(col, lcol, rng)
    _planters(col)
    ntri = 0
    dg = bpy.context.evaluated_depsgraph_get()
    for ob in col.all_objects:
        if ob.type == 'MESH' and not ob.name.startswith("props_standin"):
            ob.data.calc_loop_triangles()
            ntri += len(ob.data.loop_triangles)
    nlights = sum(1 for o in lcol.objects if o.type == 'LIGHT')
    print("[props] lanterns %d, lights %d, lava rocks %d, props tris (instances counted) %d" % (nl, nlights, nrock, ntri))


