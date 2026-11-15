"""Planter shrubs: white blossom domes (hydrangea-like), red-leaf accents, green mounds, trailing ivy.

Planter boxes themselves are built by the props module; this file only needs layout.PLANTERS.
"""
import math
import numpy as np
from .. import mats, layout as L
from ..common import add_gn_modifier
from . import vegetation_mats  # noqa
from .vegetation_util import (Geo, points_object, hermite, arclen, sample_poly, tube, frame_from_y, gn_instancer,
                              proto_collection, nrm, UP)
from .vegetation_garden import maple_leaf, maple_leaf_star, _lumps, _lump_val

# ------------------------------------------------------------------------------------- tunables
PLANTER_SOIL_H = 0.45         # soil surface above the planter z_base
PLANTER_RIM_H = 0.53          # top of the planter wall above z_base (ivy hangs over it)
SHRUB_SPACING = 1.12          # m between shrub centres along a planter
SHRUB_HI_PLANTERS = (1, 2)    # indices into layout.PLANTERS that get the detailed (close-up) shrubs
SHRUB_MIX = (0.70, 0.14, 0.16)  # white blossom, red accent, green mound
IVY_STEP = 0.7                # m between trailing-ivy clumps along the planter edges
IVY_PROB = 0.6


def _ovate_leaf(rng, w=0.30, fold=0.07, droop=0.10):
    w *= rng.uniform(0.9, 1.2) * 1.25; fold *= rng.uniform(0.3, 1.6); droop *= rng.uniform(0.3, 1.8)
    V = np.array([(0, 0, 0), (-w, 0.32, fold), (-w * 0.82, 0.70, fold * 0.6 - droop * 0.4), (0, 1.0, -droop),
                  (w * 0.82, 0.70, fold * 0.6 - droop * 0.4), (w, 0.32, fold)])
    F = np.array([(0, 5, 4), (0, 4, 3), (0, 3, 2), (0, 2, 1)])
    UV = np.array([(0.5, 0.0), (0.0, 0.32), (0.1, 0.7), (0.5, 1.0), (0.9, 0.7), (1.0, 0.32)])
    return V, F, UV


def _ivy_leaf(rng):
    w = rng.uniform(0.50, 0.62)
    V = np.array([(0, 0, 0), (-w, 0.22, 0.03), (-w * 0.62, 0.66, 0.0), (0, 1.0, -0.06), (w * 0.62, 0.66, 0.0),
                  (w, 0.22, 0.03)])
    F = np.array([(0, 5, 4), (0, 4, 3), (0, 3, 2), (0, 2, 1)])
    UV = np.array([(0.5, 0.0), (0.0, 0.2), (0.2, 0.55), (0.5, 1.0), (0.8, 0.55), (1.0, 0.2)])
    return V, F, UV


def _place(g, V, F, UV, pos, ydir, up_hint, roll, scale, mat, tint, nblend=None):
    X, Y, Z = frame_from_y(ydir, up_hint, roll)
    M = np.stack([X, Y, Z], 1) * scale
    nv = None if nblend is None else nrm(Z * 0.55 + np.asarray(nblend) * 0.6)
    g.add(V @ M.T + pos, F, uv=UV, mat=mat, tint=tint, nrm=nv)


def _floret(rng, hi):
    a0 = rng.uniform(0, math.tau)
    V = [(0.0, 0.0, 0.0)]; F = []; UV = [(0.5, 0.0)]
    w = rng.uniform(0.34, 0.44)
    for k in range(4):
        a = a0 + k * math.pi / 2 + rng.uniform(-0.15, 0.15)
        c, s = math.cos(a), math.sin(a)
        l = rng.uniform(0.85, 1.1)
        i = len(V)
        if hi:
            pts = [(-w, 0.55 * l, 0.05), (0.0, l, -0.05), (w, 0.55 * l, 0.05)]
            UV += [(0.0, 0.55), (0.5, 1.0), (1.0, 0.55)]
            F += [(0, i + 2, i + 1), (0, i + 1, i)]
        else:
            pts = [(-w * 1.1, 0.8 * l, 0.0), (w * 1.1, 0.8 * l, 0.0)]
            UV += [(0.0, 0.8), (1.0, 0.8)]
            F += [(0, i + 1, i)]
        V += [(x * c - y * s, x * s + y * c, z) for (x, y, z) in pts]
    return np.array(V), np.array(F), np.array(UV)


def _flower_head(g, rng, centre, normal, rh, hi, mat=1):
    n = 14 if hi else 7
    hint = UP if abs(normal[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    X, Y, Z = frame_from_y(normal, hint, rng.uniform(0, 6.28))
    for k in range(n):
        zz = 1.0 - (k + 0.5) / n * 0.92                    # fibonacci cap around +Y (= normal)
        rr = math.sqrt(max(0.0, 1 - zz * zz))
        ph = k * 2.39996 + rng.uniform(-0.25, 0.25)
        d = nrm(Y * zz + X * rr * math.cos(ph) + Z * rr * math.sin(ph))
        p = centre + d * rh * rng.uniform(0.9, 1.06) - normal * rh * 0.35
        V, F, UV = _floret(rng, hi)
        s = rh * (0.62 if hi else 0.85) * rng.uniform(0.85, 1.15)
        fy = nrm(np.cross(d, rng.normal(size=3)))
        fx, fy, fz = frame_from_y(fy, d + rng.normal(size=3) * 0.18, 0.0)   # floret lies in the plane normal to d
        M = np.stack([fx, fy, fz], 1) * s
        g.add(V @ M.T + p, F, uv=UV, mat=mat, tint=rng.uniform(), nrm=nrm(d * 0.8 + normal * 0.45))


def shrub_proto(rng, kind, hi):
    """kind: 'white' | 'red' | 'green'.  Mats: 0 leaf, 1 petal, 2 core, 3 red leaf, 4 bark."""
    g = Geo()
    if kind == 'white':
        rx, ry, h = rng.uniform(0.58, 0.74), rng.uniform(0.50, 0.62), rng.uniform(0.55, 0.85)
    elif kind == 'red':
        rx, ry, h = rng.uniform(0.42, 0.55), rng.uniform(0.40, 0.50), rng.uniform(0.75, 1.15)
    else:
        rx, ry, h = rng.uniform(0.45, 0.62), rng.uniform(0.42, 0.55), rng.uniform(0.40, 0.60)
    lumps = _lumps(rng, 4)

    def surf(az, el):
        d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        k = 1 + 0.16 * float(_lump_val(lumps, az, el * 0.6))
        p = np.array([rx * d[0], ry * d[1], h * d[2]]) * k
        nn = nrm(np.array([d[0] / rx, d[1] / ry, d[2] / h]))
        return p, nn

    # ---- core (dark interior so one cannot see through the leaf shell)
    nz, ns = 6, 12
    V = []
    for i in range(nz + 1):
        el = (i / nz) * math.pi / 2
        for j in range(ns + 1):
            p, _ = surf(j * math.tau / ns, el)
            V.append(p * (0.42 if kind == 'red' else 0.86) - np.array([0, 0, 0.03]))
    idx = np.arange((nz + 1) * (ns + 1)).reshape(nz + 1, ns + 1)
    Q = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
    g.quads(np.array(V), Q, mat=5 if kind == 'red' else 2)
    # ---- a few stems
    for k in range(12 if kind == 'red' else 5):
        az = rng.uniform(0, math.tau)
        p, _ = surf(az, rng.uniform(0.3, 1.3))
        tube(g, hermite((rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), -0.05), (0, 0, 0.3), p * 0.9, p * 0.4, 4),
             [0.012, 0.01, 0.008, 0.006, 0.004], sides=4, mat=4)
    # ---- leaves
    if kind == 'white':
        nleaf = 300 if hi else 190; lsz = (0.12, 0.17)
    elif kind == 'red':
        nleaf = 520 if hi else 300; lsz = (0.055, 0.075)
    else:
        nleaf = 420 if hi else 260; lsz = (0.065, 0.095)
    for k in range(nleaf):
        az = rng.uniform(0, math.tau)
        el = math.asin(rng.uniform(0.0, 1.0) ** 0.8)
        p, nn = surf(az, el)
        p = p * (rng.uniform(0.45, 1.12) ** 0.6 if kind == 'red' else rng.uniform(0.90, 1.03))
        down = nrm(np.array([0, 0, -1.0]) + nn * nn[2] + 1e-6)
        tang = nrm(np.cross(nn, UP) + 1e-6)
        if kind == 'red':
            ydir = nrm(nn * 0.55 + down * rng.uniform(0.2, 0.9) + tang * rng.uniform(-0.6, 0.6))
            V, F, UV = maple_leaf_star(rng)
            _place(g, V, F, UV, p, ydir, nn + rng.normal(size=3) * 0.45, rng.uniform(-0.4, 0.4),
                   rng.uniform(*lsz), 3, rng.uniform())
        else:
            ydir = nrm(nn * rng.uniform(0.0, 0.3) + down * rng.uniform(0.2, 1.0) + tang * rng.uniform(-0.9, 0.9))
            V, F, UV = _ovate_leaf(rng, w=0.30 if kind == 'white' else 0.26)
            _place(g, V, F, UV, p, ydir, nn + rng.normal(size=3) * 0.22, rng.uniform(-0.25, 0.25),
                   rng.uniform(*lsz), 0, rng.uniform(), nblend=nn)
    # ---- flower heads
    if kind == 'white':
        want = 30 if hi else 20
        placed = []
        tries = 0
        while len(placed) < want and tries < 600:
            tries += 1
            az = rng.uniform(0, math.tau); el = math.asin(rng.uniform(0.12, 1.0))
            p, nn = surf(az, el)
            if any(np.linalg.norm(p - q) < 0.17 for q in placed):
                continue
            placed.append(p)
            rh = rng.uniform(0.062, 0.095)
            _flower_head(g, rng, p * 1.02 + nn * rh * 0.2, nrm(nn + UP * 0.35), rh, hi)
    return g


def ivy_proto(rng):
    """Trailing strands.  Origin on the planter rim (outer edge), +Y = outward, +X along the edge, +Z up.
    Mats: 0 ivy leaf, 1 bark."""
    g = Geo()
    for sidx in range(int(rng.integers(3, 6))):
        x0 = rng.uniform(-0.32, 0.32)
        ln = rng.uniform(0.16, 0.44)
        off = rng.uniform(0.035, 0.07)
        drift = rng.uniform(-0.12, 0.12)
        pts = np.array([(x0, -0.2, -0.10), (x0, -0.09, 0.03), (x0 + drift * 0.1, 0.0, 0.05),
                        (x0 + drift * 0.3, off, -0.03), (x0 + drift * 0.6, off * 1.05, -ln * 0.5),
                        (x0 + drift, off * 0.95, -ln)])
        pts[:, 0] += np.sin(np.linspace(0, 5, 6) + rng.uniform(0, 6)) * 0.012
        Lc = arclen(pts)
        dense = np.array([sample_poly(pts, Lc, s)[0] for s in np.linspace(0, Lc[-1], 12)])
        tube(g, dense, np.linspace(0.004, 0.0018, len(dense)), sides=3, mat=1)
        Ld = arclen(dense)
        s = 0.12 + rng.uniform(0, 0.03); side = 1
        while s < Ld[-1]:
            p, t, _ = sample_poly(dense, Ld, s)
            outward = np.array([0.0, 1.0, 0.0]) if p[1] > 0.02 else np.array([0.0, 0.3, 1.0])
            ydir = nrm(np.array([side * rng.uniform(0.5, 1.0), rng.uniform(0.0, 0.35), rng.uniform(-0.9, 0.1)]))
            V, F, UV = _ivy_leaf(rng)
            _place(g, V, F, UV, p + outward * 0.008, ydir, outward + rng.normal(size=3) * 0.25, 0.0,
                   rng.uniform(0.042, 0.068), 0, rng.uniform(), nblend=outward)
            side = -side
            s += rng.uniform(0.028, 0.055)
    return g


def build_shrubs(env):
    col = env.col
    pc = proto_collection("veg_proto_shrub")
    ms = [mats.get("veg_shrub_leaf"), mats.get("veg_petal"), mats.get("veg_shrub_core"), mats.get("veg_red_leaf"),
          mats.get("veg_bark"), mats.get("veg_red_core")]
    ivy_ms = [mats.get("veg_ivy_leaf"), mats.get("veg_bark")]
    rng = env.rng(31)
    ntri = []
    groups = {}

    def mk(kind, hi, count):
        ids = []
        for _ in range(count):
            i = len(ntri)
            g = shrub_proto(rng, kind, hi)
            g.to_object("vsh_%02d_%s%s" % (i, kind, "_hi" if hi else ""), ms, pc, custom_normals=True)
            ntri.append(g.ntris)
            ids.append(i)
        groups[(kind, hi)] = ids
    mk('white', True, 4); mk('white', False, 3); mk('red', True, 2); mk('red', False, 2)
    mk('green', True, 2); mk('green', False, 2)
    ivy_ids = []
    for k in range(6):
        i = len(ntri)
        g = ivy_proto(rng)
        g.to_object("vsh_%02d_ivy" % i, ivy_ms, pc, custom_normals=True); ntri.append(g.ntris); ivy_ids.append(i)
    P = []; ROT = []; SCL = []; PID = []
    for pi, (x0, x1, y0, y1, zb) in enumerate(L.PLANTERS):
        hi = pi in SHRUB_HI_PLANTERS
        along_x = (x1 - x0) >= (y1 - y0)
        length = (x1 - x0) if along_x else (y1 - y0)
        width = (y1 - y0) if along_x else (x1 - x0)
        cx = 0.5 * (x0 + x1); cy = 0.5 * (y0 + y1)
        n = max(1, int(round(length / SHRUB_SPACING)))
        step = length / n
        last = None
        for k in range(n):
            s = (k + 0.5) * step + rng.uniform(-0.12, 0.12) - length / 2
            u = rng.uniform()
            kind = 'white' if u < SHRUB_MIX[0] else ('red' if u < SHRUB_MIX[0] + SHRUB_MIX[1] else 'green')
            if kind == last and kind != 'white':
                kind = 'white'
            last = kind
            across = rng.uniform(-0.06, 0.06)
            p = (cx + s, cy + across) if along_x else (cx + across, cy + s)
            sc = rng.uniform(0.9, 1.12) * min(1.0, width / 1.25)
            P.append((p[0], p[1], zb + PLANTER_SOIL_H - 0.02))
            ROT.append((0.0, 0.0, (0.0 if along_x else math.pi / 2) + float(rng.choice([0.0, math.pi]))
                        + rng.uniform(-0.2, 0.2)))
            SCL.append((sc, sc, sc * rng.uniform(0.9, 1.15)))
            PID.append(int(rng.choice(groups[(kind, hi)])))
        for side in (-1, 1):                               # trailing ivy over both long edges
            s = rng.uniform(0.2, IVY_STEP)
            while s < length - 0.2:
                if rng.uniform() < IVY_PROB:
                    if along_x:
                        p = (x0 + s, y1 if side > 0 else y0); rz = 0.0 if side > 0 else math.pi
                    else:
                        p = (x1 if side > 0 else x0, y0 + s); rz = -math.pi / 2 if side > 0 else math.pi / 2
                    P.append((p[0], p[1], zb + PLANTER_RIM_H))
                    ROT.append((0.0, 0.0, rz))
                    sc = rng.uniform(0.85, 1.1)
                    SCL.append((sc, 1.0, min(sc, 1.0)))
                    PID.append(int(rng.choice(ivy_ids)))
                s += IVY_STEP * rng.uniform(0.6, 1.5)
    pts = points_object("veg_shrubs", np.array(P), col, rot=np.array(ROT), scl=np.array(SCL),
                        pid=np.array(PID, np.int32))
    add_gn_modifier(pts, gn_instancer("veg_gn_shrubs", pc), "shrubs")
    tris = int(sum(ntri[p] for p in PID))
    print("[veg] shrubs: %d instances, proto tris %s, total ~%d" % (len(PID), ntri, tris))
    return tris
