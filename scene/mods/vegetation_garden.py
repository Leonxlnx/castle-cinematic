"""Garden plants at the layout anchors: red Japanese maples and Italian cypresses.
(Planter shrubs live in vegetation_shrubs.py.)

All functions take the vegetation `env` (see vegetation.py): env.col (scene collection), env.height(x, y),
env.rng(seed) -> numpy Generator.
"""
import bpy, math
import numpy as np
from mathutils import Matrix, Vector
from .. import mats, layout as L
from ..common import add_gn_modifier
from . import vegetation_mats  # noqa  (registers materials)
from .vegetation_util import (Geo, points_object, hermite, gnarl, arclen, sample_poly, tube, basis_matrix,
                              frame_from_y, euler_from_axes, cam_ndc, cam_point, gn_instancer, gn_wind_mesh,
                              proto_collection, nrm, UP, smooth01, view_samples, ViewSet)

# ------------------------------------------------------------------------------------- tunables (maples)
MAPLE_LEAF_LEN = 0.085        # m, length of the central lobe (whole leaf is ~1.7x that)
MAPLE_HI_DIST = 6.0           # sprays seen closer than this by the hero views (inside the widened frame): 30-tri leaves
MAPLE_MID_DIST = 11.0         # ... 10-tri leaves
MAPLE_LOW_Z = 7.5             # m above the trunk base: lower crown (seen from the lounge) uses 5-tri star leaves,
                              # the upper crown 3-tri leaves in larger sprays
MAPLE_HERO_HEIGHT = 13.8      # the hero maple is grown taller than its anchor height so it overhangs the hero camera
MAPLE_HERO_HI_SPRAYS = 800    # caps for the hero tree (whole twigs are dropped when a cap is exceeded)
MAPLE_HERO_MID_SPRAYS = 700
MAPLE_HERO_FAR_SPRAYS = 1400
MAPLE_HERO_SPRAYS = 3600      # 3-tri sprays of the rest of the crown
MAPLE_FAR3_SCALE = 1.3
# caps for the other maples (index into layout.MAPLES): (max sprays, spray scale, 'far' | 'far3')
MAPLE_OTHER = {1: (1900, 1.25, 'far3'), 2: (420, 1.0, 'far'), 3: (450, 1.1, 'far3'), 4: (1200, 1.6, 'far3'),
               5: (900, 1.6, 'far3')}
MAPLE_FILL = (120, 55)        # extra leafy twigs grown into the top-right / bottom-right corner of the hero frame
MAPLE_WIND_AMP = 0.045        # m sway at twig tips
MAPLE_FLUTTER = 0.16          # rad leaf-spray flutter
MAPLE_POTS = True             # build pots under SMALL maples (h <= 5.5 m) that stand on paving (inside BUILT_AREAS)
# Foliage is pruned out of the central part of every view that comes this close (horizontally) to a maple trunk:
# all layout.CAMERAS and all shots of scene/mods/cameras.py (sampled along their paths).  Leaves may only show in
# the top-right / bottom-right corners of those frames and never closer than MAPLE_LENS_CLEAR to the lens.
MAPLE_VIEW_RANGE = 17.0
MAPLE_LENS_CLEAR = 1.7
# The hero maple grows extra limbs + leafy twigs into the corners of this view: the END of this shot of
# cameras.py if it exists, else this layout camera.
MAPLE_HERO_SHOT = "s7_hero"
MAPLE_HERO_CAM = "hero"
# reach limbs of the hero maple: (depth m, ndc x, ndc y, branch density) in the hero framing view
MAPLE_REACH = ((4.2, 0.93, 0.80, 1.3), (6.0, 0.80, 0.93, 1.3), (8.5, 0.72, 1.02, 1.2), (5.2, 1.06, 0.36, 1.2),
               (4.8, 1.03, -0.95, 1.2), (11.0, 0.92, 0.78, 1.0))


# =====================================================================================
# maple leaf + spray prototypes
# =====================================================================================
def maple_leaf(rng, hi=True, fold=0.10, droop=0.22):
    """Palmate leaf in the XY plane, petiole junction at the origin, central lobe tip at +Y (length 1), +Z = upper
    side.  UV = (lateral distance from the lobe midrib 0..1, radius).  hi: 7 lobes / 30 tris, else 5 lobes / 10."""
    if hi:
        angs = [-126, -80, -39, 0, 39, 80, 126]; lens = [0.36, 0.68, 0.92, 1.0, 0.92, 0.68, 0.36]
    else:
        angs = [-84, -41, 0, 41, 84]; lens = [0.62, 0.92, 1.0, 0.92, 0.62]
    n = len(angs)
    angs = [a + rng.uniform(-5, 5) for a in angs]
    lens = [l * rng.uniform(0.9, 1.08) for l in lens]
    fold *= rng.uniform(0.5, 1.5); droop *= rng.uniform(0.4, 1.6)
    twist = rng.uniform(-0.12, 0.12)

    def P(a, r, lat=0.0):
        a = math.radians(a)
        x, y = math.sin(a) * r, math.cos(a) * r
        z = fold * lat - droop * r * r + twist * x * r
        return (x, y, z)

    V = [(0.0, 0.0, 0.0)]; UV = [(0.0, 0.0)]; F = []
    sin_idx = []
    for i in range(n + 1):
        if i == 0:
            a = angs[0] - 30; r = 0.13
        elif i == n:
            a = angs[-1] + 30; r = 0.13
        else:
            a = 0.5 * (angs[i - 1] + angs[i])
            r = (0.27 if hi else 0.33) * rng.uniform(0.85, 1.15) * min(1.0, 0.45 + min(lens[i - 1], lens[i]))
        sin_idx.append(len(V)); V.append(P(a, r, 0.25)); UV.append((1.0, r))
    for i in range(n):
        a, l = angs[i], lens[i]
        if hi:
            w = math.degrees(math.atan2(0.145 * l, 0.46 * l))
            rs = 0.48 * l
            ia = len(V); V.append(P(a - w, rs, 0.16 * l)); UV.append((1.0, rs))
            it = len(V); V.append(P(a, l, 0.0)); UV.append((0.0, l))
            ib = len(V); V.append(P(a + w, rs, 0.16 * l)); UV.append((1.0, rs))
            F += [(0, ia, sin_idx[i]), (0, it, ia), (0, ib, it), (0, sin_idx[i + 1], ib)]
        else:
            it = len(V); V.append(P(a, l, 0.0)); UV.append((0.0, l))
            F += [(0, it, sin_idx[i]), (0, sin_idx[i + 1], it)]
    if hi:   # petiole
        k = len(V)
        V += [(-0.012, 0.0, 0.0), (0.012, 0.0, 0.0), (0.010, -0.42, 0.06), (-0.010, -0.42, 0.06)]
        UV += [(0.0, 0.0)] * 4
        F += [(k, k + 2, k + 1), (k, k + 3, k + 2)]
    return np.array(V), np.array(F), np.array(UV)


def maple_leaf_star(rng, fold=0.05, droop=0.22, three=False):
    """Cheap 5-triangle (or 3-triangle) leaf: one narrow triangle per lobe.  Same frame / UV as maple_leaf."""
    if three:
        angs = [-62, 0, 62]; lens = [0.80, 1.0, 0.80]
    else:
        angs = [-88, -43, 0, 43, 88]; lens = [0.60, 0.92, 1.0, 0.92, 0.60]
    droop *= rng.uniform(0.4, 1.6)
    V = []; F = []; UV = []
    for a, l in zip(angs, lens):
        a = math.radians(a + rng.uniform(-6, 6)); l *= rng.uniform(0.88, 1.1)
        d = np.array([math.sin(a), math.cos(a)]); p = np.array([math.cos(a), -math.sin(a)])
        w = (0.20 if three else 0.135) * (0.6 + 0.4 * l)
        c = d * 0.10
        k = len(V)
        V += [(c[0] + p[0] * w, c[1] + p[1] * w, fold), (c[0] - p[0] * w, c[1] - p[1] * w, fold),
              (d[0] * l, d[1] * l, -droop * l * l)]
        V[-3] = (V[-3][0] - d[0] * 0.16, V[-3][1] - d[1] * 0.16, fold)
        V[-2] = (V[-2][0] - d[0] * 0.16, V[-2][1] - d[1] * 0.16, fold)
        UV += [(1.0, 0.1), (1.0, 0.1), (0.0, l)]
        F.append((k + 1, k, k + 2))
    return np.array(V), np.array(F), np.array(UV)


def maple_spray(rng, kind, nleaf=7, leaf_len=MAPLE_LEAF_LEN, twig_len=0.27):
    """Twiglet along +Y (from the origin) with opposite leaf pairs; +Z = up.  Materials: 0 leaf, 1 bark.
    kind: 'hi' (30-tri leaves + petioles), 'mid' (10-tri leaves), 'far' (5-tri star leaves)."""
    g = Geo()
    tl = twig_len * rng.uniform(0.85, 1.15)
    sag = rng.uniform(0.02, 0.07) * (tl / 0.27) ** 2
    bend = rng.uniform(-0.04, 0.04) * (tl / 0.27)

    def tw(s):
        return np.array([bend * s * s, tl * s, -sag * s * s])
    if kind in ('far', 'far3'):
        tube(g, np.array([tw(0.0), tw(1.0)]), [0.004, 0.0015], sides=3, mat=1)
    else:
        tube(g, np.array([tw(0.0), tw(0.5), tw(1.0)]), [0.0032, 0.0024, 0.0012], sides=3, mat=1)
    npair = (nleaf - 1) // 2
    slots = []
    for k in range(npair):
        s = 0.26 + 0.74 * (k / max(npair - 1, 1))
        slots.append((s, +1)); slots.append((s, -1))
    slots.append((1.0, 0))
    if nleaf % 2 == 0:
        slots.append((0.55, int(rng.choice([-1, 1]))))
    for (s, side) in slots[:nleaf]:
        node = tw(s)
        yaw = side * rng.uniform(40, 75) + rng.uniform(-10, 10) if side else rng.uniform(-14, 14)
        pitch = rng.uniform(8, 42)
        roll = rng.uniform(-28, 28) + side * 10
        sc = leaf_len * rng.uniform(0.72, 1.12) * (1.0 if s > 0.4 else 0.9)
        Mr = Matrix.Rotation(math.radians(-yaw), 4, 'Z') @ Matrix.Rotation(math.radians(-pitch), 4, 'X') @ \
            Matrix.Rotation(math.radians(roll), 4, 'Y')
        ydir = np.array(Mr.to_3x3() @ Vector((0.0, 1.0, 0.0)))
        stalk = 0.42 if kind == 'hi' else rng.uniform(0.3, 0.9)
        pos = node + ydir * (stalk * sc) + np.array([0, 0, 0.004])
        M = np.array(Matrix.Translation(pos.tolist()) @ Mr @ Matrix.Scale(sc, 4))
        if kind in ('far', 'far3'):
            v, f, uv = maple_leaf_star(rng, three=(kind == 'far3'))
        else:
            v, f, uv = maple_leaf(rng, kind == 'hi')
        v = v @ M[:3, :3].T + M[:3, 3]
        g.add(v, f, uv=uv, mat=0, tint=rng.uniform())
    return g


PID = {'hi': (0, 8), 'mid': (8, 14), 'far': (14, 22), 'far3': (22, 28)}     # pid ranges per spray kind


def build_maple_protos(env):
    col = proto_collection("veg_proto_maple")
    leaf = mats.get("veg_maple_leaf"); bark = mats.get("veg_bark")
    rng = env.rng(11)
    info = []
    for kind in ('hi', 'mid', 'far', 'far3'):               # object names sort alphabetically = pid order
        a, b = PID[kind]
        for i in range(a, b):
            if kind == 'hi':
                g = maple_spray(rng, 'hi', nleaf=6 + (i % 2))
            elif kind == 'mid':
                g = maple_spray(rng, 'mid', nleaf=8 + (i % 2), leaf_len=MAPLE_LEAF_LEN * 1.05, twig_len=0.32)
            elif kind == 'far':
                g = maple_spray(rng, 'far', nleaf=11 + (i % 3), leaf_len=MAPLE_LEAF_LEN * 1.15, twig_len=0.44)
            else:
                g = maple_spray(rng, 'far3', nleaf=13 + (i % 3), leaf_len=MAPLE_LEAF_LEN * 1.25, twig_len=0.48)
            g.to_object("vmp_%02d_%s" % (i, kind), [leaf, bark], col); info.append(g.ntris)
    return col, info


# =====================================================================================
# maple skeleton
# =====================================================================================
_VIEWS = None        # ViewSet of the views near the tree that is being generated (set by build_maples)


def _clear_ok(P, margin=0.0):
    """True where foliage is allowed: outside the kept-clear part of all nearby views (see MAPLE_VIEW_RANGE)."""
    P = np.atleast_2d(P)
    if _VIEWS is None or len(_VIEWS) == 0:
        return np.ones(len(P), bool)
    f, x, y, d = _VIEWS.ndc(P)
    inside = (f > 0.4) & (np.abs(x) < 1.08 + margin) & (np.abs(y) < 1.10 + margin)
    top_right = (y + 1.4 * x) > (1.70 - margin)
    bot_right = (y - 1.07 * x) < (-1.77 + margin)
    bad = (inside & ~top_right & ~bot_right) | (d < MAPLE_LENS_CLEAR)
    return ~bad.any(axis=0)


def gen_maple(rng, H, R, reach=(), dens=1.0, clear=None, world_off=np.zeros(3), fill=None):
    """Returns (branches, sprays).  branches: list of (pts, radii, sway, sides); sprays: dict of arrays.
    Local coords: trunk base at the origin.  reach: extra limb targets (x, y, z, density) in local coords."""
    br = []
    sp_pos = []; sp_dir = []; sp_sway = []; sp_grp = []
    big = min(1.0, H / 11.0)
    r0 = 0.017 * H + 0.012
    hf = H * rng.uniform(0.13, 0.18)
    lean = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), 0.0])
    if len(reach):
        mean = np.mean(np.asarray(reach)[:, :3], axis=0); mean[2] = 0
        lean = nrm(mean) * 0.32
    fork = np.array([lean[0] * hf, lean[1] * hf, hf])
    tr = hermite((0, 0, -0.5), (0, 0, hf * 0.9), fork, (lean[0] * hf * 2.2, lean[1] * hf * 2.2, hf * 0.8), 9)
    tr = gnarl(tr, rng, 0.05 * hf, waves=1.2)
    s = np.linspace(0, 1, len(tr))
    trad = r0 * (0.86 + 0.75 * np.exp(-s * 5.0))
    br.append((tr, trad, np.zeros(len(tr)), 10, False))

    def ok_pt(p):
        return True if clear is None else bool(clear(p + world_off)[0])

    def add_sprays_along(pts, Lc, s0, s1, sway0, sway1, step, grp=-1):
        side = 1 if rng.uniform() < 0.5 else -1
        s_ = s0 + rng.uniform(0, step)
        while s_ < s1:
            p, t, _ = sample_poly(pts, Lc, s_)
            h = nrm(np.cross(t, UP) + 1e-6)
            d = nrm(t * rng.uniform(0.35, 0.75) + h * side * rng.uniform(0.6, 1.0) + UP * rng.uniform(-0.6, -0.05))
            sp_pos.append(p); sp_dir.append(d); sp_grp.append(grp)
            sp_sway.append(sway0 + (sway1 - sway0) * (s_ - s0) / max(s1 - s0, 1e-6))
            side = -side
            s_ += step * rng.uniform(0.7, 1.3)
        p, t, _ = sample_poly(pts, Lc, Lc[-1])
        sp_pos.append(p); sp_dir.append(nrm(t + UP * rng.uniform(-0.6, -0.15))); sp_sway.append(sway1)
        sp_grp.append(grp)

    def add_twig(p, t_par, side, sway0, length):
        h = nrm(np.cross(t_par, UP) + 1e-6)
        length *= 1.2
        d = nrm(t_par * rng.uniform(0.3, 0.75) + h * side * rng.uniform(0.6, 1.0) + UP * rng.uniform(-0.30, 0.18))
        e = p + d * length + UP * (-0.16 * length)
        pts = hermite(p, d * length, e, (d + UP * -0.4) * length, 2)
        if not (ok_pt(pts[-1]) and ok_pt(pts[1])):
            return
        rad = np.array([0.006, 0.0045, 0.0028]) * (0.8 + 0.4 * length)
        sw = sway0 + np.linspace(0, 0.3, 3)
        br.append((pts, rad, sw, 3, True))
        Lc = arclen(pts)
        add_sprays_along(pts, Lc, 0.08 * length, Lc[-1] * 0.97, sw[0], sw[-1], 0.13 / dens, grp=len(br) - 1)

    def add_sub(p, t_par, side, sway0, length, depth=0):
        h = nrm(np.cross(t_par, UP) + 1e-6)
        d0 = nrm(t_par * rng.uniform(0.4, 0.8) + h * side * rng.uniform(0.6, 0.95) + UP * rng.uniform(-0.12, 0.26))
        d1 = nrm(d0 * np.array([1, 1, 0]) + UP * rng.uniform(-0.42, -0.08))
        e = p + nrm(d0 + d1) * length
        nseg = max(4, int(length / 0.45))
        pts = hermite(p, d0 * length * 0.9, e, d1 * length * 0.9, nseg)
        pts = gnarl(pts, rng, 0.045 * length, waves=1.6)
        if clear is not None:
            okm = clear(pts + world_off)
            if not okm.all():
                k = int(np.argmin(okm))
                if k < 3:
                    return
                pts = pts[:k]
        Lc = arclen(pts); length = Lc[-1]
        s = Lc / max(length, 1e-6)
        rad = (0.004 + 0.0085 * length) * (1 - s) ** 0.8 + 0.0042
        sw = sway0 + 0.34 * s ** 1.5
        br.append((pts, rad, sw, 4 if length > 1.6 else 3, False))
        step = 0.46 / dens
        s_ = max(0.22 * length, 0.25) + rng.uniform(0, step)
        sd = side
        while s_ < length * 0.93:
            q, t, fi = sample_poly(pts, Lc, s_)
            fr = s_ / length
            tl = rng.uniform(0.5, 0.95) * (1 - 0.35 * fr) * (0.75 + 0.25 * big)
            if depth == 0 and length > 2.6 and fr < 0.6 and rng.uniform() < 0.3:
                add_sub(q, t, sd, sway0 + 0.34 * fr ** 1.5, length * rng.uniform(0.3, 0.45), depth + 1)
            else:
                add_twig(q, t, sd, sway0 + 0.34 * fr ** 1.5, tl)
            sd = -sd
            s_ += step * rng.uniform(0.7, 1.3)
        add_sprays_along(pts, Lc, length * 0.45, length, sw[len(sw) * 2 // 5], sw[-1], 0.15 / dens)

    # ---- scaffold limbs: three rings of targets on the crown envelope (wide umbrella)
    targets = []
    n_out = int(np.clip(round(2 + R / 1.4), 3, 8))
    n_mid = int(np.clip(round(1 + R / 1.6), 2, 6))
    n_top = 1 if R < 4.5 else 3
    az0 = rng.uniform(0, math.tau)
    for i in range(n_out):
        az = az0 + math.tau * i / n_out + rng.uniform(-0.22, 0.22)
        rho = rng.uniform(0.84, 1.0); z = H * rng.uniform(0.38, 0.56)
        targets.append((np.array([rho * R * math.cos(az), rho * R * math.sin(az), z]), 1.0))
    for i in range(n_mid):
        az = az0 + math.tau * (i + 0.5) / n_mid + rng.uniform(-0.3, 0.3)
        rho = rng.uniform(0.50, 0.72); z = H * rng.uniform(0.70, 0.84)
        targets.append((np.array([rho * R * math.cos(az), rho * R * math.sin(az), z]), 1.0))
    for i in range(n_top):
        az = rng.uniform(0, math.tau)
        rho = rng.uniform(0.08, 0.32); z = H * rng.uniform(0.92, 1.0)
        targets.append((np.array([rho * R * math.cos(az), rho * R * math.sin(az), z]), 1.0))
    for t in reach:
        targets.append((np.asarray(t[:3], float), t[3] if len(t) > 3 else 1.3))
    Lmax = max(np.linalg.norm(t[0] - fork) for t in targets)
    Ltr = arclen(tr)
    for (T, ldens) in targets:
        p0, tt, _ = sample_poly(tr, Ltr, Ltr[-1] * rng.uniform(0.72, 1.0))
        dv = T - p0; Ll = np.linalg.norm(dv)
        oh = nrm(dv * np.array([1, 1, 0]) + 1e-9)
        steep = np.clip(dv[2] / max(np.hypot(dv[0], dv[1]), 0.1), 0.2, 3.0)
        t0 = nrm(oh * 0.8 + UP * (0.35 + 0.45 * steep)) * Ll * 1.0
        t1 = nrm(oh * 1.0 + UP * rng.uniform(-0.25, 0.0)) * Ll * 0.85
        nseg = max(8, int(Ll / 0.5))
        pts = hermite(p0, t0, T, t1, nseg)
        pts = gnarl(pts, rng, 0.03 * Ll, waves=1.8)
        if clear is not None:
            okm = clear(pts + world_off)
            if not okm.all():
                k = int(np.argmin(okm))
                if k < 6:
                    continue
                pts = pts[:k]
        Lc = arclen(pts); Ll = Lc[-1]
        s = Lc / Ll
        rb = r0 * 0.6 * (Ll / Lmax) ** 0.5
        rad = rb * (1 - s) ** 0.75 + 0.011
        sw = 0.34 * s ** 1.7
        br.append((pts, rad, sw, 7 if rb > 0.07 else 5, False))
        step = (0.85 / ldens) * (0.7 + 0.3 * big)
        s_ = 0.24 * Ll + rng.uniform(0, step)
        side = 1 if rng.uniform() < 0.5 else -1
        while s_ < Ll * 0.97:
            q, t, _ = sample_poly(pts, Lc, s_)
            fr = s_ / Ll
            ln = float(np.clip(Ll * 0.46 * (1 - 0.6 * fr) * rng.uniform(0.7, 1.2), 0.7, 5.0))
            add_sub(q, t, side, 0.34 * fr ** 1.7, ln)
            side = -side
            s_ += step * rng.uniform(0.7, 1.3)
        add_sprays_along(pts, Lc, Ll * 0.86, Ll, sw[-3], sw[-1], 0.15 / dens)
    # ---- extra leafy twigs toward given fill points (hero framing): grown from the nearest limb / branch
    if fill is not None and len(fill):
        anchors = np.concatenate([b[0] for b in br if not b[4]])
        asway = np.concatenate([b[2] for b in br if not b[4]])
        for T in fill:
            dv = anchors - T
            dist = np.linalg.norm(dv, axis=1)
            score = dist + np.where(dv[:, 2] < -0.2, 1.5, 0.0)          # prefer anchors above (twigs hang)
            score[dist < 0.35] = 1e9
            k = int(np.argmin(score))
            if dist[k] > 3.2:
                continue
            A = anchors[k]; ln = dist[k]
            d = (T - A) / ln
            pts = hermite(A, nrm(d + UP * 0.5) * ln, T, nrm(d + UP * -0.9) * ln, max(3, int(ln / 0.3)))
            pts = gnarl(pts, rng, 0.04 * ln, waves=1.3)
            if clear is not None and not clear(pts[len(pts) // 2:] + world_off).all():
                continue
            Lc = arclen(pts)
            ss = Lc / Lc[-1]
            rad = (0.004 + 0.003 * ln) * (1 - ss) + 0.0028
            sw = np.clip(asway[k] + 0.4 * ss, 0, 1)
            br.append((pts, rad, sw, 3, True))
            add_sprays_along(pts, Lc, 0.25 * Lc[-1], Lc[-1], sw[1], sw[-1], 0.12, grp=len(br) - 1)
    sprays = dict(pos=np.array(sp_pos), dir=np.array(sp_dir), sway=np.array(sp_sway), grp=np.array(sp_grp))
    return br, sprays


def _pot(env, x, y, z, r=0.72, h=0.78):
    from ..common import MB
    mb = MB("veg_maple_pot")
    prof = [(r * 0.62, 0.0), (r * 0.70, 0.04), (r * 0.98, h * 0.82), (r * 1.06, h * 0.86), (r * 1.06, h),
            (r * 0.93, h), (r * 0.93, h - 0.06)]
    mb.lathe(prof, 28, "stone_trim", center=(0, 0))
    mb.lathe([(0.0, h - 0.07), (r * 0.93, h - 0.06)], 28, "veg_soil", center=(0, 0))
    ob = mb.to_object("veg_maple_pot", env.col)
    ob.location = (x, y, z)
    return ob


def _fill_points(rng, cam, n_top, n_bot):
    """World-space points inside the two foliage corners of the camera frame (see _clear_ok)."""
    out = []
    while len(out) < n_top:
        x = rng.uniform(0.45, 1.25); y = rng.uniform(0.15, 1.3)
        if y + 1.4 * x < 1.80:
            continue
        out.append(cam_point(cam, 3.0 + 6.0 * rng.uniform() ** 1.4, x, y))
    k = 0
    while k < n_bot:
        x = rng.uniform(0.68, 1.25); y = rng.uniform(-1.3, -0.55)
        if y - 1.07 * x > -1.86:
            continue
        out.append(cam_point(cam, 3.4 + 4.0 * rng.uniform(), x, y)); k += 1
    return np.array(out)


def build_maples(env):
    col = env.col
    proto_col, proto_tris = build_maple_protos(env)
    bark = mats.get("veg_bark")
    global _VIEWS
    all_views = view_samples()
    hero_end = [v for v in all_views if v[0].startswith(MAPLE_HERO_SHOT + "@")]
    cam0 = hero_end[-1][1:] if hero_end else L.CAMERAS[MAPLE_HERO_CAM]
    cam0 = (np.asarray(cam0[0], float), np.asarray(cam0[1], float), float(cam0[2]))
    cpos = cam0[0]
    print("[veg] maples: hero framing view = %s  pos %s lens %.0f" % (
        hero_end[-1][0] if hero_end else MAPLE_HERO_CAM, np.round(cpos, 1).tolist(), cam0[2]))
    P_all = []; E_all = []; S_all = []; PID_all = []; SW_all = []
    wood = Geo()
    stats = []
    for mi, (x, y, zb, H, R) in enumerate(L.MAPLES):
        rng = env.rng(100 + mi)
        in_built = any(a[0] <= x <= a[1] and a[2] <= y <= a[3] for a in L.BUILT_AREAS)
        base = np.array([x, y, zb if in_built else min(zb, env.height1(x, y, zb))])
        pot_h = 0.0
        if in_built and MAPLE_POTS and H <= 5.5:
            _pot(env, x, y, zb)
            pot_h = 0.72
        reach = []; fill = None
        near_views = [v for v in all_views if math.hypot(v[1][0] - x, v[1][1] - y) < MAPLE_VIEW_RANGE + R]
        _VIEWS = ViewSet(near_views)
        clear = _clear_ok if near_views else None
        if mi == 0:
            for (f, nx, ny, ld) in MAPLE_REACH:
                reach.append(tuple(cam_point(cam0, f, nx, ny) - base) + (ld,))
            H = max(H, MAPLE_HERO_HEIGHT)
            fill = _fill_points(rng, cam0, *MAPLE_FILL) - base
        branches, sp = gen_maple(rng, H, R, reach=reach, clear=clear, world_off=base, fill=fill)
        pos = sp["pos"] + base; d = sp["dir"]; grp = sp["grp"]
        keep = np.ones(len(pos), bool)
        if clear is not None:
            keep &= clear(pos) & clear(pos + d * 0.3, margin=-0.02)
        # clearance above paving (people / furniture) and above the pot rim
        for (x0, x1, y0, y1) in L.BUILT_AREAS:
            inb = (pos[:, 0] > x0 - 0.5) & (pos[:, 0] < x1 + 0.5) & (pos[:, 1] > y0 - 0.5) & (pos[:, 1] < y1 + 0.5)
            floor = L.LOUNGE_Z if (x0, x1, y0, y1) == L.BUILT_AREAS[1] else 0.0
            keep &= ~(inb & (pos[:, 2] < floor + (2.4 if H > 5.5 else 1.4)))
        keep &= pos[:, 2] > base[2] + pot_h + 0.5
        nraw = int(keep.sum())
        # ---- level of detail by distance to the hero camera
        dc = np.linalg.norm(pos - cpos, axis=1)
        if mi == 0:
            # closest distance at which a spray is seen inside a (widened) frame of the nearby views
            f_, x_, y_, d_ = _VIEWS.ndc(pos)
            vis = (f_ > 0.3) & (np.abs(x_) < 1.6) & (np.abs(y_) < 1.6)
            hero_like = np.linalg.norm(_VIEWS.pos - cpos, axis=1) < 12.0
            dh = np.where(vis & hero_like[:, None], d_, 1e9).min(axis=0)
            do = np.where(vis & ~hero_like[:, None], d_, 1e9).min(axis=0)
            kind = np.where(dh < MAPLE_HI_DIST, 0, np.where((dh < MAPLE_MID_DIST) | (do < 8.0), 1,
                            np.where(pos[:, 2] < base[2] + MAPLE_LOW_Z, 2, 3)))
            caps = {0: MAPLE_HERO_HI_SPRAYS, 1: MAPLE_HERO_MID_SPRAYS, 2: MAPLE_HERO_FAR_SPRAYS, 3: MAPLE_HERO_SPRAYS}
            names = {0: 'hi', 1: 'mid', 2: 'far', 3: 'far3'}; sscale = 1.0
        else:
            cap, sscale, nm = MAPLE_OTHER.get(mi, (int(26 * R * R), 1.0, 'far3'))
            kind = np.full(len(pos), 2); caps = {2: cap}; names = {2: nm}
        # ---- caps: drop whole twigs (wood + leaves) at random until each class fits
        for kd, cap in caps.items():
            sel = keep & (kind == kd)
            n = int(sel.sum())
            if n <= cap:
                continue
            gids = np.unique(grp[sel & (grp >= 0)])
            rng.shuffle(gids)
            for gi in gids:
                if n <= cap:
                    break
                m_ = sel & (grp == gi)
                n -= int(m_.sum()); keep &= ~m_
            sel = keep & (kind == kd)
            if n > cap:                                   # still too many (sprays that sit on branches)
                ids = np.nonzero(sel)[0]
                keep[rng.choice(ids, n - cap, replace=False)] = False
        alive = set(np.unique(grp[keep & (grp >= 0)]).tolist())   # twigs without any leaf left are removed too
        idx = np.nonzero(keep)[0]
        pos = pos[idx]; d = d[idx]; sw = sp["sway"][idx]; kind = kind[idx]
        n = len(pos)
        X = np.zeros((n, 3)); Y = np.zeros((n, 3)); Z = np.zeros((n, 3))
        rolls = rng.uniform(-0.45, 0.45, n)
        for i in range(n):
            X[i], Y[i], Z[i] = frame_from_y(d[i], UP, rolls[i])
        eul = euler_from_axes(X, Y, Z)
        pid = np.zeros(n, np.int32)
        for kd, nm in names.items():
            a, b = PID[nm]
            m_ = kind == kd
            pid[m_] = rng.integers(a, b, int(m_.sum()))
        scl = rng.uniform(0.85, 1.25, n) * sscale
        if mi == 0:
            scl = scl * np.where(kind == 3, MAPLE_FAR3_SCALE, 1.0)
        P_all.append(pos); E_all.append(eul); S_all.append(scl); PID_all.append(pid); SW_all.append(np.clip(sw, 0, 1))
        for bi, (pts, rad, sway, sides, is_twig) in enumerate(branches):
            if is_twig and bi not in alive:
                continue
            tube(wood, pts + base, rad, sides=sides, mat=0, sway=np.clip(sway, 0, 1))
        stats.append((nraw, n, int((kind == 0).sum()), int((kind == 1).sum())))
    wob = wood.to_object("veg_maple_wood", [bark], col)
    add_gn_modifier(wob, gn_wind_mesh("veg_gn_maple_wood", MAPLE_WIND_AMP), "wind")
    P = np.concatenate(P_all); S = np.concatenate(S_all)
    pts = points_object("veg_maple_leaves", P, col, rot=np.concatenate(E_all), scl=np.stack([S, S, S], 1),
                        pid=np.concatenate(PID_all).astype(np.int32), sway=np.concatenate(SW_all))
    g = gn_instancer("veg_gn_maple_leaves", proto_col, wind_amp=MAPLE_WIND_AMP, flutter=MAPLE_FLUTTER)
    add_gn_modifier(pts, g, "leaves")
    pid = np.concatenate(PID_all)
    tris = int(sum(proto_tris[p] for p in pid)) + wood.ntris
    print("[veg] maples: sprays per tree (raw, kept, hi, mid) %s  wood tris %d  total tris ~%d"
          % (stats, wood.ntris, tris))
    return tris


# =====================================================================================
# Italian cypresses
# =====================================================================================
CYPRESS_VARIANTS = 5
CYPRESS_TUFTS = 1500          # foliage sprays per tree
CYPRESS_WIDTH = 0.085         # max radius as a fraction of the height


def _lumps(rng, n=4):
    return [(int(rng.integers(2, 7)), rng.uniform(3.0, 11.0), rng.uniform(0, math.tau), rng.uniform(0.5, 1.0))
            for _ in range(n)]


def _lump_val(lumps, th, t):
    v = 0.0; w = 0.0
    for (k, kz, ph, a) in lumps:
        v = v + a * np.sin(k * th + kz * t * math.tau * 0.5 + ph); w += a
    return v / w


def cypress_proto(rng, H=10.0):
    """Flame-shaped cypress of height H: lumpy dark core + upright foliage sprays.  Mats: 0 foliage, 1 core, 2 bark."""
    g = Geo()
    Rm = H * CYPRESS_WIDTH * rng.uniform(0.88, 1.12)
    tw = rng.uniform(0.22, 0.34)                       # height fraction of the widest point
    t0 = 0.035                                          # foliage starts here (short bare trunk)
    lumps = _lumps(rng, 5)
    bend = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)]) * 0.012 * H
    top_pow = rng.uniform(1.35, 1.9)

    def axis(t):
        t = np.asarray(t)
        s = np.sin(t * math.pi * 0.5) ** 2
        return np.stack([bend[0] * s, bend[1] * s, t * H], axis=-1)

    def rad(t, th):
        t = np.asarray(t, float)
        lo = np.clip((t - t0) / (tw - t0), 0, 1) ** 0.55
        hi = np.clip(1 - np.clip((t - tw) / (1 - tw), 0, 1) ** top_pow, 0, 1) ** 0.8
        r = Rm * np.where(t < tw, 0.55 + 0.45 * lo, hi) * np.clip((t - t0) / 0.03, 0, 1) ** 0.5
        return r * (1 + 0.20 * _lump_val(lumps, th, t)) + 0.012

    # ---- core
    nz, ns = 26, 14
    ts = t0 + (1 - t0) * (np.linspace(0, 1, nz + 1) ** 0.9)
    th = np.arange(ns + 1) * math.tau / ns
    T, TH = np.meshgrid(ts, th, indexing='ij')
    Rr = rad(T, TH) * 0.94
    A = axis(T)
    V = A + np.stack([np.cos(TH) * Rr, np.sin(TH) * Rr, np.zeros_like(Rr)], -1)
    idx = np.arange((nz + 1) * (ns + 1)).reshape(nz + 1, ns + 1)
    Q = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
    g.quads(V.reshape(-1, 3), Q, mat=1, uv=np.stack([TH.ravel(), T.ravel() * H], 1))
    # ---- trunk
    tube(g, np.array([[0, 0, -0.4], [0, 0, 0.3], [0, 0, H * 0.3]]), [0.02 * H, 0.014 * H, 0.008 * H], sides=7, mat=2)
    # ---- foliage sprays
    n = CYPRESS_TUFTS
    # sample t with density ~ radius (surface area)
    tt = np.linspace(t0, 1.0, 400)
    w = rad(tt, 0.0) + 0.02 * Rm
    cdf = np.cumsum(w); cdf /= cdf[-1]
    t = np.interp(rng.uniform(0, 1, n), cdf, tt)
    thr = rng.uniform(0, math.tau, n)
    r = rad(t, thr)
    out = np.stack([np.cos(thr), np.sin(thr), np.zeros(n)], 1)
    P = axis(t) + out * (r * rng.uniform(0.88, 1.02, n))[:, None]
    for i in range(n):
        big = rng.uniform() < 0.07
        ln = rng.uniform(0.030, 0.052) * H * (1 - 0.3 * t[i]) * (1.4 if big else 1.0)
        wd = ln * rng.uniform(0.09, 0.14)
        d = nrm(UP * rng.uniform(1.0, 1.5) + out[i] * rng.uniform(0.12, 0.38 if not big else 0.6)
                + rng.normal(size=3) * 0.09)
        X, Y, Z = frame_from_y(d, out[i], rng.uniform(-0.5, 0.5))
        tint = rng.uniform()
        nb = 2
        for b in range(nb):
            a = b * math.pi / nb + rng.uniform(-0.3, 0.3)
            ca, sa = math.cos(a), math.sin(a)
            side = X * ca + Z * sa
            lean = Z * ca - X * sa
            bl = ln * rng.uniform(0.75, 1.1)
            tip = P[i] + Y * bl + lean * rng.uniform(-0.18, 0.18) * bl + side * rng.uniform(-0.1, 0.1) * bl
            mid = P[i] + Y * bl * 0.42 + lean * rng.uniform(-0.05, 0.05) * bl
            v = np.array([P[i] - Y * 0.05 * bl, mid - side * wd, mid + side * wd, tip])
            nv = nrm(out[i] * 0.75 + UP * 0.25 + lean * rng.uniform(-0.35, 0.35) + side * rng.uniform(-0.25, 0.25))
            g.add(v, [(0, 2, 1), (1, 2, 3)], uv=[(0.5, 0.0), (0.0, 0.42), (1.0, 0.42), (0.5, 1.0)], mat=0,
                  tint=tint, nrm=nv)
    return g


def build_cypresses(env):
    col = env.col
    pc = proto_collection("veg_proto_cypress")
    ms = [mats.get("veg_cypress"), mats.get("veg_cypress_core"), mats.get("veg_bark")]
    rng = env.rng(21)
    ntri = []
    for i in range(CYPRESS_VARIANTS):
        g = cypress_proto(rng)
        g.to_object("vcy_%02d" % i, ms, pc, custom_normals=True); ntri.append(g.ntris)
    A = np.array(L.CYPRESSES, float)
    n = len(A)
    pid = (np.arange(n) * 2 + rng.integers(0, 2, n)) % CYPRESS_VARIANTS
    sh = A[:, 3] / 10.0
    sw = sh * rng.uniform(0.9, 1.1, n)
    rot = np.zeros((n, 3)); rot[:, 2] = rng.uniform(0, math.tau, n)
    rot[:, 0] = rng.uniform(-0.012, 0.012, n); rot[:, 1] = rng.uniform(-0.012, 0.012, n)
    pts = points_object("veg_cypresses", A[:, :3], col, rot=rot, scl=np.stack([sw, sw, sh], 1), pid=pid.astype(np.int32))
    add_gn_modifier(pts, gn_instancer("veg_gn_cypress", pc), "cypress")
    tris = int(sum(ntri[p] for p in pid))
    print("[veg] cypresses: %d trees, proto tris %s, total ~%d" % (n, ntri, tris))
    return tris
