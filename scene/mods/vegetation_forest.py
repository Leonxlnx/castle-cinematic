"""Forest and wild vegetation on the natural terrain (instanced with geometry nodes, 3 LODs by distance).

  LOD0  (< LOD0_DIST from the built areas)   conifers with drooping needle-frond boughs + hanging curtains,
                                             broadleaf trees with a real branch skeleton + leaf-cluster cards
  LOD1  (< LOD1_DIST)                        same construction, a fraction of the cards
  LOD2  (to FOREST_RADIUS)                   very cheap cones / blobs
  small                                      bushes and grass tufts near the terraces

Placement uses env.height / env.forest_mask (landscape.height_at / forest_mask or the flat fallback), keeps out of
layout.BUILT_AREAS and never lets a tree rise into the sight lines from the standard cameras to the castle/terraces.
"""
import math
import numpy as np
from .. import mats, layout as L
from ..common import add_gn_modifier
from . import vegetation_mats, vegetation_fmats  # noqa
from .vegetation_util import (Geo, points_object, hermite, gnarl, arclen, sample_poly, tube, frame_from_y,
                              gn_instancer, proto_collection, nrm, UP, view_samples)

# ------------------------------------------------------------------------------------- tunables
FOREST_RADIUS = 1200.0          # m around FOREST_CENTER
FOREST_CENTER = (0.0, 20.0)
LOD0_DIST = 150.0               # distance from the bounding box of the built areas
LOD1_DIST = 500.0
SPACING = (7.5, 9.0, 12.5)      # candidate grid spacing per LOD zone (m)
TRI_BUDGET = (780_000, 700_000, 520_000)   # rendered triangles per LOD (trees are demoted / thinned to fit)
SMALL_BUDGET = (650, 4500)      # max bushes, max grass tufts
SMALL_DIST = 95.0               # bushes / grass only this close to the built areas
GARDEN_DIST = 30.0              # this close to the built areas: low planting + a few small specimen trees only
# sight lines from these layout cameras AND from every shot of scene/mods/cameras.py to the castle / terraces are
# kept free of trees
PROTECT_CAMS = ("hero", "wide", "firepit", "pool", "facade", "valley", "aerial", "cliff")
MIX = (0.80, 0.13, 0.07)        # conifer, broadleaf, bare


# =====================================================================================
# prototypes
# =====================================================================================
def _bough(g, rng, p0, az, Lb, elev0, sagk, tipup, lod, tint, dr=None):
    """One bough = a 'tent' of two steeply drooping needle-frond cards hanging from the bough spine
    (visible from the side as well as from above)."""
    o = np.array([math.cos(az), math.sin(az), 0.0]); td = np.array([-math.sin(az), math.cos(az), 0.0])
    nseg = 3 if lod == 0 else 1
    s = np.linspace(0.0, 1.0, nseg + 1)
    horiz = Lb * s * math.cos(elev0)
    dz = Lb * (math.sin(elev0) * s - sagk * s ** 2 + tipup * s ** 3.5)
    spine = p0 + o * horiz[:, None] + UP * dz[:, None]
    wh = Lb * rng.uniform(0.30, 0.40) * (1.0 - 0.5 * s)
    if dr is None:
        dr = rng.uniform(0.85, 1.2)
    drl = dr + rng.uniform(-0.15, 0.15); drr = dr + rng.uniform(-0.15, 0.15)
    sl = -td * math.cos(drl) - UP * math.sin(drl)
    sr = td * math.cos(drr) - UP * math.sin(drr)
    Lv = spine + sl * wh[:, None]; Rv = spine + sr * wh[:, None]
    V = np.concatenate([Lv, spine, Rv])
    n = nseg + 1
    UV = np.concatenate([np.stack([np.zeros(n), s], 1), np.stack([np.full(n, 0.5), s], 1), np.stack([np.ones(n), s], 1)])
    nc = nrm(o * 0.75 + UP * 0.45)
    N = np.concatenate([np.tile(nrm(nc - td * 0.55), (n, 1)), np.tile(nc, (n, 1)), np.tile(nrm(nc + td * 0.55), (n, 1))])
    Q = []
    for i in range(nseg):
        Q += [(i, n + i, n + i + 1, i + 1), (n + i, 2 * n + i, 2 * n + i + 1, n + i + 1)]
    g.quads(V, Q, uv=UV, mat=0, nrm=N, tint=tint)


def conifer(rng, H=20.0, lod=0, droop=1.0, slim=1.0):
    """Spruce / fir.  Mats: 0 frond card, 1 (unused), 2 bark, 3 core."""
    g = Geo()
    Rmax = H * rng.uniform(0.20, 0.245) * slim
    crown0 = rng.uniform(0.05, 0.15)
    nwh = 22 if lod == 0 else 9
    bend = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0]) * 0.012 * H
    tint = rng.uniform()

    def axis(t):
        return bend * math.sin(t * math.pi) + UP * (t * H)
    r0 = 0.0105 * H + 0.05
    ts = np.linspace(-0.03, 0.97, 7 if lod == 0 else 3)
    tube(g, np.array([axis(t) for t in ts]), r0 * (1.0 - 0.93 * np.clip(ts, 0, 1)) + (ts < 0) * r0 * 0.3,
         sides=7 if lod == 0 else 5, mat=2)
    for w in range(nwh):
        t = crown0 + (1 - crown0) * ((w + rng.uniform(0.25, 0.75)) / nwh) ** 0.92 * 0.96
        prof = (1 - t) / (1 - crown0)
        Lb = Rmax * (0.10 + 0.90 * prof ** 0.8)
        nb = 5 if lod == 0 else 4
        az0 = rng.uniform(0, math.tau)
        for b in range(nb):
            az = az0 + math.tau * b / nb + rng.uniform(-0.35, 0.35)
            if droop < 0.5:                                   # fir: boughs angled upward, flatter sprays
                elev0 = 0.50 - 0.40 * prof + rng.uniform(-0.1, 0.1)
                dr = rng.uniform(0.55, 0.85)
            else:
                elev0 = (0.55 - 0.75 * prof * droop) + rng.uniform(-0.12, 0.12)
                dr = None
            sagk = droop * (0.08 + 0.26 * prof) * rng.uniform(0.7, 1.2)
            tipup = 0.32 * droop * prof
            _bough(g, rng, axis(t), az, Lb * rng.uniform(0.72, 1.18), elev0, sagk, tipup, lod, tint, dr)
    for k in range(3):                                        # leader
        _bough(g, rng, axis(0.93), rng.uniform(0, math.tau), H * 0.075, 1.25, 0.0, 0.0, 1, tint, 0.5)
    # dark core so the crown is never see-through
    ns = 6
    ang = np.arange(ns) * math.tau / ns
    ring = np.stack([np.cos(ang), np.sin(ang), np.zeros(ns)], 1) * (Rmax * 0.33) + axis(crown0) + UP * 0.4
    mid = np.stack([np.cos(ang + 0.5), np.sin(ang + 0.5), np.zeros(ns)], 1) * (Rmax * 0.19) + axis(0.5)
    apex = axis(0.93)
    V = np.concatenate([ring, mid, apex[None, :]])
    F = []
    for i in range(ns):
        j = (i + 1) % ns
        F += [(i, j, ns + j), (i, ns + j, ns + i), (ns + i, ns + j, 2 * ns)]
    g.add(V, F, mat=3)
    return g


def _card_cluster(g, rng, c, size, centre, mat, tint, n=2):
    """n crossed quads around c; custom normals point away from the crown centre."""
    out = nrm(c - centre + UP * 0.3 * size)
    for k in range(n):
        a = rng.uniform(0, math.pi)
        ax = nrm(np.array([math.cos(a), math.sin(a), rng.uniform(-0.5, 0.5)]))
        ay = nrm(np.cross(ax, rng.normal(size=3)))
        h = size * 0.5 * rng.uniform(0.85, 1.15)
        V = np.array([c - ax * h - ay * h, c + ax * h - ay * h, c + ax * h + ay * h, c - ax * h + ay * h])
        g.quads(V, [(0, 1, 2, 3)], uv=[(0, 0), (1, 0), (1, 1), (0, 1)], mat=mat,
                nrm=np.tile(nrm(out + rng.normal(size=3) * 0.25), (4, 1)), tint=tint)


def broadleaf(rng, H=14.0, lod=0, leafy=1.0, card_mat=1):
    """Deciduous tree with a real branch skeleton.  Mats: 0 bark, 1 leaf cards.  leafy: fraction of cards kept."""
    g = Geo()
    tint = rng.uniform()
    centre = np.array([0.0, 0.0, H * 0.66])
    rc = H * rng.uniform(0.30, 0.38)
    r0 = 0.014 * H + 0.04
    hf = H * rng.uniform(0.28, 0.40)
    lean = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), 0.0])
    fork = lean * hf + UP * hf
    tr = gnarl(hermite((0, 0, -0.5), (0, 0, hf), fork, lean * hf + UP * hf, 5 if lod == 0 else 3), rng, 0.03 * hf)
    tube(g, tr, r0 * np.linspace(1.25, 0.8, len(tr)), sides=8 if lod == 0 else 5, mat=0)
    nl = int(rng.integers(4, 6))
    ends = []
    az0 = rng.uniform(0, math.tau)
    csize = H * (0.20 if lod == 0 else 0.30)

    def branch(p, d, ln, rad, level):
        e = p + nrm(d + UP * 0.25) * ln
        nseg = (4 if level == 1 else 3) if lod == 0 else 2
        pts = gnarl(hermite(p, d * ln, e, nrm(d + UP * 0.5) * ln, nseg), rng, 0.06 * ln)
        sides = (6 if level == 1 else (4 if level == 2 else 3)) if lod == 0 else (4 if level == 1 else 3)
        tube(g, pts, rad * np.linspace(1.0, 0.35, len(pts)), sides=sides, mat=0)
        maxlevel = 3 if lod == 0 else (2 if leafy < 0.5 else 1)
        if level < maxlevel:
            Lc = arclen(pts)
            nchild = 3 if level == 1 else int(rng.integers(2, 4))
            for k in range(nchild):
                q, t, _ = sample_poly(pts, Lc, Lc[-1] * (0.35 + 0.65 * (k + rng.uniform(0.2, 0.8)) / nchild))
                side = nrm(np.cross(t, rng.normal(size=3)))
                dd = nrm(t * rng.uniform(0.5, 0.9) + side * rng.uniform(0.5, 0.9) + UP * rng.uniform(0.0, 0.35))
                branch(q, dd, ln * rng.uniform(0.5, 0.68), rad * 0.55, level + 1)
            ends.append((pts[-1], level))
        else:
            ends.append((pts[-1], level)); ends.append((pts[len(pts) // 2], level))

    for i in range(nl):
        az = az0 + math.tau * i / nl + rng.uniform(-0.3, 0.3)
        up = rng.uniform(0.5, 1.2)
        d = nrm(np.array([math.cos(az), math.sin(az), up]))
        branch(fork, d, rc * rng.uniform(0.9, 1.25), r0 * 0.5, 1)
    for (p, level) in ends:
        if rng.uniform() > leafy:
            continue
        _card_cluster(g, rng, p + rng.normal(size=3) * 0.03 * H, csize * rng.uniform(0.8, 1.25), centre, card_mat, tint)
        if lod == 0 and leafy > 0.5:
            _card_cluster(g, rng, p + rng.normal(size=3) * 0.07 * H, csize * rng.uniform(0.7, 1.1), centre, card_mat,
                          tint, n=1)
    return g


def far_conifer(rng, H=20.0, sides=5):
    """LOD2: two overlapping cone tiers.  Mat 0.  UV v = height fraction."""
    g = Geo()
    R = H * rng.uniform(0.15, 0.20)
    ang = np.arange(sides) * math.tau / sides + rng.uniform(0, 1)
    c = np.stack([np.cos(ang), np.sin(ang), np.zeros(sides)], 1)
    lean = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0]) * 0.01 * H
    z = [0.10 * H, 0.52 * H, 0.42 * H, H]
    r = [R, R * 0.34, R * 0.66, 0.0]
    V = np.concatenate([c * r[0] + UP * z[0], c * r[1] + UP * z[1] + lean * 0.5, c * r[2] + UP * z[2] + lean * 0.5,
                        (lean + UP * z[3])[None, :], np.array([[0.0, 0.0, -1.5]])])
    N = np.concatenate([nrm(c + UP * 0.5), nrm(c + UP * 0.5), nrm(c + UP * 0.5), UP[None, :], -UP[None, :]])
    UVv = np.concatenate([np.full(sides, 0.1), np.full(sides, 0.52), np.full(sides, 0.42), [1.0], [0.0]])
    F = []
    for i in range(sides):
        j = (i + 1) % sides
        F += [(i, j, sides + j), (i, sides + j, sides + i), (2 * sides + i, 2 * sides + j, 3 * sides)]
    g.add(V, F, uv=np.stack([np.zeros(len(V)), UVv], 1), mat=0, nrm=N)
    # trunk stub (thin pyramid) so the tree touches the ground on slopes
    k = 3 * sides + 1
    g.add(np.concatenate([c[:3] * (0.02 * H) + UP * (0.11 * H), np.array([[0, 0, -2.0]])]),
          [(0, 1, 3), (1, 2, 3), (2, 0, 3)], mat=0, uv=np.zeros((4, 2)))
    return g


def far_broadleaf(rng, H=14.0, sides=5):
    g = Geo()
    R = H * rng.uniform(0.30, 0.38)
    ang = np.arange(sides) * math.tau / sides
    c = np.stack([np.cos(ang), np.sin(ang), np.zeros(sides)], 1)
    zs = [0.36, 0.60, 0.86]; rs = [0.75, 1.0, 0.62]
    rings = [c * (R * rr * rng.uniform(0.85, 1.15)) + UP * (H * zz) for zz, rr in zip(zs, rs)]
    V = np.concatenate(rings + [np.array([[0, 0, H * 1.0]]), np.array([[0, 0, H * 0.3]])])
    ctr = np.array([0, 0, H * 0.62])
    N = nrm(V - ctr)
    UVv = np.concatenate([np.full(sides, z_) for z_ in zs] + [[1.0], [0.3]])
    F = []
    for i in range(sides):
        j = (i + 1) % sides
        F += [(i, j, sides + j), (i, sides + j, sides + i), (sides + i, sides + j, 2 * sides + j),
              (sides + i, 2 * sides + j, 2 * sides + i), (2 * sides + i, 2 * sides + j, 3 * sides),
              (j, i, 3 * sides + 1)]
    g.add(V, F, uv=np.stack([np.zeros(len(V)), UVv], 1), mat=0, nrm=N)
    g.add(np.array([[0.035 * H, 0, H * 0.4], [-0.02 * H, 0.03 * H, H * 0.4], [-0.02 * H, -0.03 * H, H * 0.4], [0, 0, -2.0]]),
          [(0, 1, 3), (1, 2, 3), (2, 0, 3)], mat=1, uv=np.zeros((4, 2)))
    return g


def bush(rng, card_mat=1):
    """Unit bush (radius ~1, height ~1).  Mats: 0 core, 1 leaf cards."""
    g = Geo()
    sides = 6
    ang = np.arange(sides) * math.tau / sides
    c = np.stack([np.cos(ang), np.sin(ang), np.zeros(sides)], 1)
    V = np.concatenate([c * 0.60 + UP * -0.1, c * 0.48 + UP * 0.40, np.array([[0, 0, 0.66]])])
    F = []
    for i in range(sides):
        j = (i + 1) % sides
        F += [(i, j, sides + j), (i, sides + j, sides + i), (sides + i, sides + j, 2 * sides)]
    g.add(V, F, mat=0)
    tint = rng.uniform()
    centre = np.array([0, 0, 0.2])
    for k in range(16):
        az = rng.uniform(0, math.tau); el = math.asin(rng.uniform(0.05, 1.0))
        d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        p = d * np.array([0.85, 0.85, 0.8]) * rng.uniform(0.85, 1.1)
        _card_cluster(g, rng, p, rng.uniform(0.9, 1.3), centre, card_mat, tint, n=1)
    return g


def grass_tuft(rng):
    """Unit grass tuft (height 1, width ~1): three crossed cards flaring outward."""
    g = Geo()
    tint = rng.uniform()
    for k in range(3):
        a = k * math.pi / 3 + rng.uniform(-0.3, 0.3)
        ax = np.array([math.cos(a), math.sin(a), 0.0]); ay = np.array([-math.sin(a), math.cos(a), 0.0])
        lean = ay * rng.uniform(-0.25, 0.25)
        V = np.array([-ax * 0.32, ax * 0.32, ax * 0.6 + UP + lean, -ax * 0.6 + UP + lean]) + np.array([0, 0, -0.04])
        g.quads(V, [(0, 1, 2, 3)], uv=[(0, 0), (1, 0), (1, 1), (0, 1)], mat=0, nrm=np.tile(nrm(UP + ay * 0.2), (4, 1)),
                tint=tint)
    return g


class Protos:
    def __init__(self):
        self.col = proto_collection("veg_proto_forest")
        self.tris = []; self.height = []; self.groups = {}

    def add(self, key, geo, materials, H):
        i = len(self.tris)
        geo.to_object("vft_%02d_%s" % (i, key), materials, self.col, custom_normals=True)
        self.tris.append(geo.ntris); self.height.append(H)
        self.groups.setdefault(key, []).append(i)
        return i

    def avg(self, key):
        return float(np.mean([self.tris[i] for i in self.groups[key]]))


def build_protos(env):
    rng = env.rng(51)
    P = Protos()
    m = mats.get
    con = [m("veg_needle"), m("veg_needle"), m("veg_bark_far"), m("veg_conifer_core")]
    for lod in (0, 1):
        for k, (droop, slim) in enumerate(((1.0, 1.0), (1.15, 0.9), (0.8, 1.1), (0.3, 0.85), (1.0, 0.8))[:5 if lod == 0 else 4]):
            P.add("con%d" % lod, conifer(rng, 20.0, lod, droop, slim), con, 20.0)
    for lod in (0, 1):
        for k in range(3 if lod == 0 else 2):
            cm = m("veg_leafcard_autumn") if k == 1 else m("veg_leafcard")
            P.add("dec%d" % lod, broadleaf(rng, 14.0, lod, 1.0), [m("veg_bark_far"), cm], 14.0)
        for k in range(2 if lod == 0 else 1):
            P.add("bare%d" % lod, broadleaf(rng, 13.0, lod, 0.22), [m("veg_bark_far"), m("veg_leafcard_sparse")], 13.0)
    for k in range(4):
        P.add("con2", far_conifer(rng, 20.0), [m("veg_far_conifer")], 20.0)
    for k in range(2):
        P.add("dec2", far_broadleaf(rng, 14.0), [m("veg_far_broadleaf") if k == 0 else m("veg_far_broadleaf_autumn"),
                                               m("veg_bark_far")], 14.0)
    P.groups["bare2"] = P.groups["dec2"][1:]
    for k in range(3):
        P.add("bush", bush(rng), [m("veg_bush_core"), m("veg_leafcard_autumn") if k == 2 else m("veg_leafcard")], 1.0)
    for k in range(3):
        P.add("grass", grass_tuft(rng), [m("veg_grass")], 1.0)
    return P


# =====================================================================================
# placement
# =====================================================================================
def _rect_dist(x, y, r):
    dx = np.maximum(np.maximum(r[0] - x, x - r[1]), 0.0)
    dy = np.maximum(np.maximum(r[2] - y, y - r[3]), 0.0)
    return np.hypot(dx, dy)


def _built_bbox():
    B = np.array(L.BUILT_AREAS)
    return (B[:, 0].min(), B[:, 1].max(), B[:, 2].min(), B[:, 3].max())


def _built_dist(x, y):
    d = np.full(np.shape(x), 1e9)
    for r in L.BUILT_AREAS:
        d = np.minimum(d, _rect_dist(x, y, r))
    return d


def _wob(x, y, k=1.0):
    """Cheap smooth pseudo-noise in -1..1."""
    return (np.sin(x * 0.031 * k + 1.3) * np.cos(y * 0.027 * k + 0.4) + 0.6 * np.sin(x * 0.011 * k - y * 0.017 * k + 2.1)
            + 0.4 * np.sin(x * 0.071 * k + y * 0.053 * k)) / 2.0


def _jgrid(rng, x0, x1, y0, y1, sp):
    nx = max(1, int((x1 - x0) / sp)); ny = max(1, int((y1 - y0) / sp))
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    gx = gx.ravel().astype(float); gy = gy.ravel().astype(float)
    gx[(gy.astype(int) % 2) == 1] += 0.5
    x = x0 + (gx + rng.uniform(-0.38, 0.38, gx.shape)) * sp
    y = y0 + (gy + rng.uniform(-0.38, 0.38, gy.shape)) * sp
    return x, y


def _targets():
    T = []
    for x in np.arange(L.TERRACE_X0 + 4, L.TERRACE_X1, 14.0):
        for y in np.arange(L.TERRACE_Y0 + 4, L.TERRACE_Y1, 13.0):
            T.append((x, y, 0.6))
    for x in np.arange(L.LOUNGE_X0 + 3, L.LOUNGE_X1, 8.0):
        for y in np.arange(L.LOUNGE_Y0 + 3, L.LOUNGE_Y1, 8.0):
            T.append((x, y, L.LOUNGE_Z + 0.6))
    x0, x1, y0, y1 = L.CASTLE_BBOX
    for x in np.arange(x0 + 4, x1, 12.0):
        T.append((x, 24.0, L.PODIUM_Z + 1.5)); T.append((x, 30.0, 6.0))
    for y in np.arange(20.0, y1, 12.0):
        T.append((x1 - 2, y, 6.0)); T.append((x0 + 2, y, 6.0))
    return np.array(T)


def _sight_limit(x, y, r):
    """Highest z a tree at (x, y) with crown radius r may reach without entering a camera->target sight line."""
    T = _targets()
    P = np.stack([x, y], 1)
    allowed = np.full(len(x), 1e9)
    cams = [np.asarray(L.CAMERAS[cn][0], float) for cn in PROTECT_CAMS if cn in L.CAMERAS]
    cams += [v[1] for v in view_samples(3) if "@" in v[0]]
    for C in cams:
        d2 = T[:, :2] - C[:2]
        L2 = np.maximum((d2 ** 2).sum(1), 1e-6)
        for i0 in range(0, len(x), 4000):
            Pc = P[i0:i0 + 4000]
            s = np.clip(((Pc[:, None, :] - C[:2]) * d2[None, :, :]).sum(2) / L2[None, :], 0.0, 1.0)
            cl = C[:2] + s[:, :, None] * d2[None, :, :]
            dist = np.linalg.norm(Pc[:, None, :] - cl, axis=2)
            hit = (dist < r[i0:i0 + 4000, None] + 1.0) & (s > 0.02) & (s < 0.985)
            sz = C[2] + s * (T[None, :, 2] - C[2])
            allowed[i0:i0 + 4000] = np.minimum(allowed[i0:i0 + 4000], np.where(hit, sz, 1e9).min(axis=1))
        near = np.hypot(x - C[0], y - C[1]) < r + 7.0
        allowed = np.where(near, np.minimum(allowed, C[2] - 3.0), allowed)
    return allowed


def build_forest(env):
    col = env.col
    P = build_protos(env)
    rng = env.rng(61)
    bb = _built_bbox()
    cx, cy = FOREST_CENTER

    def zone_thr(x, y):
        w = _wob(x, y)
        return LOD0_DIST * (1 + 0.14 * w), LOD1_DIST * (1 + 0.10 * w)

    # ------------------------------------------------------------ tree candidates per LOD zone
    X = []; Y = []; Z = []; D = []; M = []
    for zi, sp in enumerate(SPACING):
        ext = (LOD0_DIST * 1.2, LOD1_DIST * 1.15, FOREST_RADIUS)[zi]
        if zi < 2:
            x, y = _jgrid(rng, bb[0] - ext, bb[1] + ext, bb[2] - ext, bb[3] + ext, sp)
        else:
            x, y = _jgrid(rng, cx - ext, cx + ext, cy - ext, cy + ext, sp)
        d = _rect_dist(x, y, bb)
        t0, t1 = zone_thr(x, y)
        if zi == 0:
            keep = d < t0
        elif zi == 1:
            keep = (d >= t0) & (d < t1)
        else:
            keep = (d >= t1) & (np.hypot(x - cx, y - cy) < FOREST_RADIUS)
        x, y, d = x[keep], y[keep], d[keep]
        m = env.forest_mask(x, y)
        z = env.height(x, y)
        ok = (rng.uniform(0, 1, len(x)) < m ** 0.7) & (z > L.VALLEY_Z + 1.0) & (_built_dist(x, y) > 6.0)
        X.append(x[ok]); Y.append(y[ok]); Z.append(z[ok]); D.append(d[ok]); M.append(m[ok])
    x = np.concatenate(X); y = np.concatenate(Y); z = np.concatenate(Z); d = np.concatenate(D); m = np.concatenate(M)
    n = len(x)
    db = _built_dist(x, y)
    # ------------------------------------------------------------ species + size
    u = rng.uniform(0, 1, n)
    sp_noise = _wob(x * 1.7 + 300, y * 1.7 - 120)
    p_con = np.clip(MIX[0] + 0.22 * sp_noise - 0.35 * np.clip(1 - db / 70.0, 0, 1), 0.2, 0.97)
    kind = np.where(u < p_con, 0, np.where(u < p_con + (1 - p_con) * MIX[1] / (MIX[1] + MIX[2]), 1, 2))
    v = rng.uniform(0, 1, n)
    H = np.where(kind == 0, 12.0 + 18.0 * v ** 1.25, np.where(kind == 1, 8.0 + 9.5 * v, 7.0 + 8.0 * v))
    H *= 0.72 + 0.28 * m
    garden = db < GARDEN_DIST
    drop = garden & (rng.uniform(0, 1, n) > 0.16)
    kind = np.where(garden, np.where(rng.uniform(0, 1, n) < 0.8, 1, 2), kind)
    H = np.where(garden, 5.5 + 3.5 * v, H)
    crown_r = np.where(kind == 0, 0.19 * H, 0.36 * H)
    # ------------------------------------------------------------ sight lines
    near_zone = d < LOD1_DIST * 1.3
    allowed = np.full(n, 1e9)
    if near_zone.any():
        allowed[near_zone] = _sight_limit(x[near_zone], y[near_zone], crown_r[near_zone])
    Hn = np.minimum(H, allowed - z - 0.6)
    shrink_ok = (Hn >= np.maximum(0.55 * H, 6.0))
    to_bush = (Hn < H) & ~shrink_ok & (Hn >= 1.3)
    drop |= (Hn < H) & ~shrink_ok
    H = np.where(shrink_ok, Hn, H)
    bush_x = x[to_bush & ~garden]; bush_y = y[to_bush & ~garden]; bush_h = np.minimum(Hn[to_bush & ~garden], 2.4)
    keep = ~drop
    x, y, z, d, H, kind = x[keep], y[keep], z[keep], d[keep], H[keep], kind[keep]
    n = len(x)
    # ------------------------------------------------------------ LOD assignment within the triangle budgets
    t0, t1 = zone_thr(x, y)
    lod = np.where(d < t0, 0, np.where(d < t1, 1, 2))
    names = ("con", "dec", "bare")

    def cost(lv):
        return np.array([P.avg(names[k] + str(lv)) for k in range(3)])
    for lv in (0, 1):
        idx = np.nonzero(lod == lv)[0]
        c = cost(lv)[kind[idx]]
        order = idx[np.argsort(d[idx])]
        cum = np.cumsum(cost(lv)[kind[order]])
        over = order[cum > TRI_BUDGET[lv]]
        lod[over] = lv + 1
    idx = np.nonzero(lod == 2)[0]
    c2 = cost(2)[kind[idx]]
    if c2.sum() > TRI_BUDGET[2]:
        kp = rng.uniform(0, 1, len(idx)) < TRI_BUDGET[2] / c2.sum()
        lod[idx[~kp]] = 9
    stats = {}
    total = 0
    for lv in (0, 1, 2):
        sel = lod == lv
        k = int(sel.sum())
        pid = np.zeros(k, np.int32)
        kk = kind[sel]
        for ki in range(3):
            mk = kk == ki
            pid[mk] = rng.choice(P.groups[names[ki] + str(lv)], int(mk.sum()))
        ph = np.array(P.height)[pid]
        sh = H[sel] / ph
        sw = sh * rng.uniform(0.85, 1.18, k) * (1.25 if lv == 2 else 1.0)
        rot = np.zeros((k, 3)); rot[:, 2] = rng.uniform(0, math.tau, k)
        rot[:, 0] = rng.uniform(-0.03, 0.03, k); rot[:, 1] = rng.uniform(-0.03, 0.03, k)
        pts = points_object("veg_forest_lod%d" % lv, np.stack([x[sel], y[sel], z[sel] - 0.15], 1), col, rot=rot,
                            scl=np.stack([sw, sw, sh], 1), pid=pid)
        add_gn_modifier(pts, gn_instancer("veg_gn_forest%d" % lv, P.col), "forest")
        if lv >= 1:
            pts.visible_shadow = False
        tr = int(np.array(P.tris)[pid].sum())
        stats["lod%d" % lv] = (k, tr); total += tr
    # ------------------------------------------------------------ bushes + grass near the terraces
    ext = SMALL_DIST + 5
    sx, sy = _jgrid(rng, bb[0] - ext, bb[1] + ext, bb[2] - ext, bb[3] + ext, 2.3)
    sd = _built_dist(sx, sy)
    ok = (sd > 1.6) & (sd < SMALL_DIST)
    sx, sy, sd = sx[ok], sy[ok], sd[ok]
    sz = env.height(sx, sy)
    slope = np.hypot(env.height(sx + 1.5, sy) - sz, env.height(sx, sy + 1.5) - sz) / 1.5
    ok = (sz > -60.0) & (slope < 0.75)
    sx, sy, sd, sz, slope = sx[ok], sy[ok], sd[ok], sz[ok], slope[ok]
    patch = 0.5 + 0.5 * _wob(sx * 4.0 + 50, sy * 4.0 + 80)
    fall = np.clip(1.15 - sd / SMALL_DIST, 0.0, 1.0)
    r1 = rng.uniform(0, 1, len(sx))
    is_bush = r1 < 0.045 * (0.4 + patch) * fall
    is_grass = (~is_bush) & (r1 < (0.045 + 0.42 * patch ** 1.5) * fall)
    bx = np.concatenate([sx[is_bush], bush_x]); by = np.concatenate([sy[is_bush], bush_y])
    bh = np.concatenate([rng.uniform(0.7, 2.0, int(is_bush.sum())), bush_h])
    if len(bx) > SMALL_BUDGET[0]:
        s_ = rng.choice(len(bx), SMALL_BUDGET[0], replace=False); bx, by, bh = bx[s_], by[s_], bh[s_]
    gx_, gy_ = sx[is_grass], sy[is_grass]
    if len(gx_) > SMALL_BUDGET[1]:
        s_ = rng.choice(len(gx_), SMALL_BUDGET[1], replace=False); gx_, gy_ = gx_[s_], gy_[s_]
    gh = rng.uniform(0.35, 0.95, len(gx_))
    nb, ng = len(bx), len(gx_)
    px = np.concatenate([bx, gx_]); py = np.concatenate([by, gy_])
    pz = env.height(px, py) - 0.03
    pid = np.concatenate([rng.choice(P.groups["bush"], nb), rng.choice(P.groups["grass"], ng)]).astype(np.int32)
    sh = np.concatenate([bh, gh]); sw = sh * np.concatenate([rng.uniform(0.9, 1.5, nb), rng.uniform(0.7, 1.3, ng)])
    rot = np.zeros((nb + ng, 3)); rot[:, 2] = rng.uniform(0, math.tau, nb + ng)
    pts = points_object("veg_forest_small", np.stack([px, py, pz], 1), col, rot=rot, scl=np.stack([sw, sw, sh], 1), pid=pid)
    add_gn_modifier(pts, gn_instancer("veg_gn_forest_small", P.col), "small")
    pts.visible_shadow = False
    tr = int(np.array(P.tris)[pid].sum())
    stats["bush/grass"] = (nb, ng, tr); total += tr
    print("[veg] forest: proto tris %s" % dict((k, [P.tris[i] for i in v]) for k, v in P.groups.items()))
    print("[veg] forest: %s  total ~%d tris" % (stats, total))
    return total
