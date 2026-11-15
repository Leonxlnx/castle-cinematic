"""Gothic architecture kit (used by castle.py / terrace.py).

Conventions
  * Facade local frame (see common.frame_xf): +X along the wall, +Z up, OUTSIDE = -Y, wall face at y = 0.
  * Plan outlines of buildings are CCW, so walking along an edge the outside is on the right.
  * Everything is added to an MB; winding is outward by construction (to_object(recalc=False, weld=True)).
"""
import math, random
from mathutils import Vector, Matrix
from .common import MB, pointed_arch, arch_apex, frame_xf, xf, circle_pts, rect_pts, TAU

STONE, TRIM, DARK = "stone", "stone_trim", "stone_dark"
SLATE, LEAD, METAL, GOLD, GLASS = "slate", "roof_lead", "metal_dark", "gold", "glass_warm"

RNG = random.Random(11)


# =====================================================================================
# windows
# =====================================================================================
def W(kind='arch', w=1.6, spring=2.2, sharp=1.0, depth=0.38, nx=2, nz=4, hood=True, sill=True,
      tracery=False, glass=GLASS, lit=None, n=9, frame=METAL, open=False):
    """Window spec.  kind: 'arch' | 'rect' | 'round'.  spring = jamb height (total height for rect).
    lit: None = random per window, or a fixed 0..1 value for the face attribute 'rnd' (<=0.14 = dark)."""
    return dict(kind=kind, w=w, spring=spring, sharp=sharp, depth=depth, nx=nx, nz=nz, hood=hood, sill=sill,
                tracery=tracery, glass=glass, lit=lit, n=n, frame=frame, open=open)


def win_height(s):
    if s['kind'] == 'arch':
        return s['spring'] + arch_apex(s['w'], s['sharp'])
    if s['kind'] == 'round':
        return s['w']
    return s['spring']


def window(mb, cx, zs, s, rng=RNG):
    """Adds reveal, glass, bars, tracery, hood and sill of one window whose sill is at (cx, zs) on the
    current facade.  Returns the opening outline [(x, z), ...] so the caller can cut the wall panel."""
    w = s['w']; kind = s['kind']; d = s['depth']
    arch = None
    if kind == 'arch':
        arch = pointed_arch(w, zs + s['spring'], n=s['n'], sharp=s['sharp'], x0=cx)
        outline = [(cx - w / 2, zs), (cx + w / 2, zs)] + arch
        ztop = zs + s['spring'] + arch_apex(w, s['sharp'])
    elif kind == 'round':
        r = w / 2; n = 20
        outline = [(cx + r * math.cos(-math.pi / 2 + TAU * i / n), zs + r + r * math.sin(-math.pi / 2 + TAU * i / n))
                   for i in range(n)]
        ztop = zs + w
    else:
        ztop = zs + s['spring']
        outline = [(cx - w / 2, zs), (cx + w / 2, zs), (cx + w / 2, ztop), (cx - w / 2, ztop)]
    step = min(0.09, w * 0.08)
    # stepped stone reveal
    mb.sweep_xz([(0.0, 0.0), (0.0, d * 0.35), (-step, d * 0.35), (-step, d)], outline, TRIM, closed=True, flip=True)
    if s.get('open'):
        # open arch (loggia / arcade): only the deep reveal, optional hood
        if s['hood'] and arch is not None:
            mb.sweep_xz([(0.03, 0.0), (0.03, -0.10), (0.19, -0.10), (0.25, 0.0)], arch, TRIM, closed=False)
        return outline
    # glass
    old = mb.rnd
    mb.rnd = rng.random() if s['lit'] is None else s['lit']
    x0, x1 = cx - w / 2 - 0.02, cx + w / 2 + 0.02
    mb.quad((x0, d, zs - 0.02), (x1, d, zs - 0.02), (x1, d, ztop + 0.02), (x0, d, ztop + 0.02), s['glass'])
    # drawn-back curtains (dark drapery against the lit room) on most of the larger windows
    if w >= 1.15 and kind != 'round' and rng.random() < 0.78:
        yc = d - 0.002
        for side in (-1, 1):
            cw = w * rng.uniform(0.13, 0.30)
            xe = cx + side * (w / 2 + 0.02); xi = xe - side * cw
            xa, xb = min(xe, xi), max(xe, xi)
            mb.quad((xa, yc, zs - 0.02), (xb, yc, zs - 0.02), (xb, yc, ztop + 0.02), (xa, yc, ztop + 0.02), "curtain")
        if rng.random() < 0.5:
            mb.quad((x0, yc, ztop - 0.34), (x1, yc, ztop - 0.34), (x1, yc, ztop + 0.02), (x0, yc, ztop + 0.02), "curtain")
    mb.rnd = old
    # metal frame + glazing bars
    fr = s['frame']
    mb.sweep_xz([(-step, d - 0.07), (-step - 0.055, d - 0.07), (-step - 0.055, d - 0.004)], outline, fr,
                closed=True, flip=True)
    bw = 0.02 if w < 1.0 else 0.026
    for i in range(1, s['nx']):
        x = cx - w / 2 + w * i / s['nx']
        mb.box(x - bw, x + bw, d - 0.06, d - 0.004, zs, ztop, fr)
    for j in range(1, s['nz']):
        z = zs + (ztop - zs) * j / s['nz']
        mb.box(cx - w / 2, cx + w / 2, d - 0.06, d - 0.004, z - bw, z + bw, fr)
    # stone tracery: two lights + oculus
    if s['tracery'] and kind == 'arch':
        y0, y1 = d - 0.17, d - 0.065
        bar = [(-0.045, y1), (-0.045, y0), (0.045, y0), (0.045, y1)]
        zsp = zs + s['spring']
        mb.box(cx - 0.05, cx + 0.05, y0, y1, zs, zsp + 0.02, TRIM)
        for side in (-1, 1):
            sub = pointed_arch(w / 2, zsp, n=7, sharp=1.0, x0=cx + side * w / 4)
            mb.sweep_xz(bar, sub, TRIM, closed=False)
        rr = 0.165 * w
        cz = zsp + 0.52 * w * (s['sharp'] ** 0.5)
        circ = [(cx + rr * math.cos(TAU * i / 14), cz + rr * math.sin(TAU * i / 14)) for i in range(14)]
        mb.sweep_xz(bar, circ, TRIM, closed=True)
    # engaged colonnettes with bases and capitals on the jambs of the larger arched windows
    if arch is not None and w >= 1.4 and s['spring'] >= 1.8:
        zsp = zs + s['spring']
        for side in (-1, 1):
            xs_ = cx + side * (w / 2 + 0.085)
            mb.lathe([(0.075, zs), (0.075, zs + 0.10), (0.05, zs + 0.17), (0.05, zsp - 0.26), (0.062, zsp - 0.22),
                      (0.085, zsp - 0.08), (0.085, zsp)], 6, TRIM, (xs_, -0.035))
    # hood mould / lintel
    if s['hood']:
        if arch is not None:
            mb.sweep_xz([(0.03, 0.0), (0.03, -0.10), (0.19, -0.10), (0.25, 0.0)], arch, TRIM, closed=False)
            zsp = zs + s['spring']
            for side in (-1, 1):
                xc = cx + side * (w / 2 + 0.14)
                mb.box(xc - 0.15, xc + 0.15, -0.14, -0.001, zsp - 0.24, zsp + 0.03, TRIM)
        elif kind == 'rect':
            mb.box(cx - w / 2 - 0.26, cx + w / 2 + 0.26, -0.12, -0.001, ztop + 0.10, ztop + 0.30, TRIM)
        else:
            mb.sweep_xz([(0.03, 0.0), (0.03, -0.09), (0.17, -0.09), (0.22, 0.0)], outline, TRIM, closed=True)
    if s['sill'] and kind != 'round':
        mb.box(cx - w / 2 - 0.22, cx + w / 2 + 0.22, -0.17, -0.001, zs - 0.24, zs - 0.001, TRIM)
    return outline


def facade(mb, p0, p1, z0, floors, bay=4.0, margin=0.8, wall=STONE, rng=RNG, skip=(), aprons=True):
    """Straight wall from plan point p0 to p1 (outside on the right), starting at height z0.
    floors: list of (height, spec, sill_height); spec = None | W(...) | callable(bay_index, n_bays) -> spec/None.
    skip: bay indices left blank on every floor.  Returns (z_top, n_bays, bay_width)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    mb.push(frame_xf((p0[0], p0[1], 0.0), (dx, dy)))
    nb = max(1, int(round((L - 2 * margin) / bay)))
    bw = (L - 2 * margin) / nb
    z = z0
    for (h, spec, sill) in floors:
        if margin > 1e-6:
            mb.quad((0, 0, z), (margin, 0, z), (margin, 0, z + h), (0, 0, z + h), wall)
            mb.quad((L - margin, 0, z), (L, 0, z), (L, 0, z + h), (L - margin, 0, z + h), wall)
        for i in range(nb):
            x0 = margin + i * bw; x1 = x0 + bw
            s = spec(i, nb) if callable(spec) else spec
            if s is None or i in skip:
                mb.quad((x0, 0, z), (x1, 0, z), (x1, 0, z + h), (x0, 0, z + h), wall)
            else:
                outline = window(mb, (x0 + x1) / 2, z + sill, s, rng)
                mb.poly_holes([(x0, z), (x1, z), (x1, z + h), (x0, z + h)], [outline], wall)
                if aprons and sill >= 0.8 and not s.get('open') and s['kind'] != 'round':
                    cxw = (x0 + x1) / 2
                    blind_arcade(mb, cxw - s['w'] / 2 - 0.22, cxw + s['w'] / 2 + 0.22, z + 0.17, sill - 0.45)
        z += h
    mb.pop()
    return z, nb, bw


def blind_arcade(mb, x0, x1, z, h, pitch=0.44, mat=TRIM):
    """Row of small blind pointed arches in low relief (gothic panelling) on the current facade."""
    if h < 0.25 or x1 - x0 < 0.5:
        return
    n = max(1, int(round((x1 - x0) / pitch)))
    p = (x1 - x0) / n
    mb.box(x0 - 0.03, x1 + 0.03, -0.055, -0.001, z - 0.05, z, mat)
    mb.box(x0 - 0.03, x1 + 0.03, -0.055, -0.001, z + h, z + h + 0.05, mat)
    bar = [(-0.022, -0.001), (-0.022, -0.05), (0.022, -0.05), (0.022, -0.001)]
    w = p - 0.06
    R = 0.8 * w
    top = math.sqrt(max(R * R - (R - w / 2) ** 2, 0.0))
    spring = max(0.05, h - top - 0.03)
    for i in range(n):
        xc = x0 + (i + 0.5) * p
        arch = pointed_arch(w, z + spring, n=3, sharp=0.8, x0=xc)
        mb.sweep_xz(bar, [(xc + w / 2, z)] + arch + [(xc - w / 2, z)], mat, closed=False)


# =====================================================================================
# mouldings (plan sweeps; profile goes: wall bottom -> out -> up -> back to wall)
# =====================================================================================
def plinth(mb, outline, z0, h=1.1, out=0.26, mat=DARK, closed=True):
    mb.sweep([(0.0, z0 - 0.15), (out, z0 - 0.15), (out, z0 + h - 0.22), (0.06, z0 + h), (0.0, z0 + h)], outline, mat, closed=closed)


def string_course(mb, outline, z, out=0.13, h=0.24, mat=TRIM, closed=True):
    mb.sweep([(0.0, z - h / 2), (out * 0.5, z - h / 2), (out, z - h / 2 + 0.07), (out, z + h / 2 - 0.05), (0.0, z + h / 2 + 0.04)],
             outline, mat, closed=closed)


def cornice(mb, outline, z, out=0.40, mat=TRIM, closed=True):
    """Projecting cornice whose top is at z (parapets sit on it)."""
    mb.sweep([(0.0, z - 0.62), (0.08, z - 0.62), (0.12, z - 0.50), (0.24, z - 0.42), (0.24, z - 0.30),
              (out, z - 0.16), (out, z), (0.0, z)], outline, mat, closed=closed)


def corbel_table(mb, p0, p1, z, out=0.36, step=0.72, mat=TRIM):
    """Row of two-step corbels under a cornice whose underside is at z."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    mb.push(frame_xf((p0[0], p0[1], 0.0), (dx, dy)))
    n = max(1, int(L / step))
    for i in range(n):
        x = (i + 0.5) * L / n
        mb.box(x - 0.13, x + 0.13, -out, -0.001, z - 0.26, z, mat, skip="+y")
        mb.box(x - 0.13, x + 0.13, -out * 0.55, -0.001, z - 0.50, z - 0.26, mat, skip="+y+z")
    mb.pop()


def crenellation(mb, p0, p1, z, out=0.40, t=0.42, base_h=0.55, mer_w=1.05, mer_h=0.95, gap=0.80, mat=STONE,
                 cap=TRIM, ext0=True, ext1=True):
    """Battlement along one edge; the parapet sits out over the cornice (y in [-out, -out+t])."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    mb.push(frame_xf((p0[0], p0[1], 0.0), (dx, dy)))
    xa = -out if ext0 else 0.0
    xb = L + out if ext1 else L
    y0, y1 = -out, -out + t
    mb.box(xa, xb, y0, y1, z, z + base_h, mat)
    span = xb - xa
    n = max(1, int((span + gap) / (mer_w + gap)))
    pitch = (span - mer_w) / max(1, n - 1) if n > 1 else 0.0
    for i in range(n):
        x = xa + i * pitch if n > 1 else xa + (span - mer_w) / 2
        zt = z + base_h + mer_h
        mb.box(x, x + mer_w, y0, y1, z + base_h, zt, mat, skip="-z")
        mb.box(x - 0.05, x + mer_w + 0.05, y0 - 0.05, y1 + 0.05, zt, zt + 0.09, cap)
        mb.frustum(rect_pts(x - 0.05, x + mer_w + 0.05, y0 - 0.05, y1 + 0.05), zt + 0.09,
                   rect_pts(x + 0.12, x + mer_w - 0.12, y0 + 0.14, y1 - 0.14), zt + 0.20, cap)
    mb.pop()


def low_parapet(mb, outline, z, h=0.9, out=0.38, t=0.36, mat=STONE, closed=True):
    """Plain parapet wall with a moulded coping, sitting out over the cornice."""
    mb.sweep([(out - t, z), (out, z), (out, z + h), (out + 0.05, z + h), (out + 0.05, z + h + 0.10),
              (out - t / 2, z + h + 0.18), (out - t - 0.05, z + h + 0.10), (out - t - 0.05, z + h), (out - t, z + h),
              (out - t, z)], outline, mat, closed=closed)


# =====================================================================================
# balustrade
# =====================================================================================
_BAL = None


def _baluster():
    global _BAL
    if _BAL is None:
        _BAL = MB("baluster")
        _BAL.lathe([(0.078, 0.0), (0.078, 0.045), (0.046, 0.075), (0.052, 0.14), (0.088, 0.25), (0.098, 0.32),
                    (0.074, 0.44), (0.043, 0.56), (0.048, 0.60), (0.074, 0.635), (0.074, 0.70), (0.056, 0.74)],
                   8, TRIM)
    return _BAL


def pier(mb, x, y, z, h=1.14, s=0.26, cap=True, mat=TRIM):
    mb.box(x - s, x + s, y - s, y + s, z, z + h, mat, skip="-z")
    mb.box(x - s - 0.04, x + s + 0.04, y - s - 0.04, y + s + 0.04, z, z + 0.14, mat, skip="-z")
    if cap:
        mb.box(x - s - 0.06, x + s + 0.06, y - s - 0.06, y + s + 0.06, z + h, z + h + 0.10, mat)
        mb.frustum(rect_pts(x - s - 0.06, x + s + 0.06, y - s - 0.06, y + s + 0.06), z + h + 0.10,
                   rect_pts(x - 0.07, x + 0.07, y - 0.07, y + 0.07), z + h + 0.24, mat)


def urn(mb, x, y, z, s=1.0, mat=TRIM):
    """Garden urn on a pier top."""
    mb.lathe([(0.10 * s, z), (0.16 * s, z + 0.03 * s), (0.16 * s, z + 0.08 * s), (0.06 * s, z + 0.14 * s), (0.07 * s, z + 0.22 * s),
              (0.20 * s, z + 0.34 * s), (0.25 * s, z + 0.50 * s), (0.22 * s, z + 0.62 * s), (0.15 * s, z + 0.68 * s),
              (0.18 * s, z + 0.74 * s), (0.20 * s, z + 0.76 * s), (0.14 * s, z + 0.76 * s)], 12, mat, (x, y))
    mb.lathe([(0.0, z + 0.70 * s), (0.14 * s, z + 0.70 * s)], 12, "stone_dark", (x, y))


def balustrade(mb, pts, z, closed=False, pier_every=3.4, end_piers=True, ball=False, urns=0):
    """Classic stone balustrade along the plan polyline pts at floor height z (centre line = path)."""
    n = len(pts)
    segs = n if closed else n - 1
    bal = _baluster()
    for i in range(segs):
        p0 = pts[i]; p1 = pts[(i + 1) % n]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        L = math.hypot(dx, dy)
        if L < 0.3:
            continue
        mb.push(frame_xf((p0[0], p0[1], 0.0), (dx, dy)))
        k = max(1, int(round(L / pier_every)))
        for j in range(k + 1):
            xp = L * j / k
            first, last = (j == 0), (j == k)
            if (first and (i > 0 or closed)):
                continue                       # shared corner already has a pier from the previous segment
            if (first or last) and not closed and not end_piers and ((i == 0 and first) or (i == segs - 1 and last)):
                continue
            pier(mb, xp, 0.0, z, cap=not (urns and (j + i) % urns == 0))
            if urns and (j + i) % urns == 0:
                mb.box(xp - 0.32, xp + 0.32, -0.32, 0.32, z + 1.14, z + 1.24, TRIM)
                urn(mb, xp, 0.0, z + 1.24)
            if ball:
                mb.lathe([(0.0, z + 1.36), (0.09, z + 1.40), (0.14, z + 1.50), (0.09, z + 1.60), (0.0, z + 1.64)],
                         10, TRIM, (xp, 0.0))
        for j in range(k):
            xa = L * j / k + 0.26; xb = L * (j + 1) / k - 0.26
            if xb - xa < 0.2:
                continue
            mb.box(xa, xb, -0.15, 0.15, z, z + 0.16, TRIM, skip="-z")
            mb.box(xa, xb, -0.17, 0.17, z + 0.90, z + 1.00, TRIM)
            mb.frustum(rect_pts(xa, xb, -0.17, 0.17), z + 1.00, rect_pts(xa, xb, -0.10, 0.10), z + 1.06, TRIM)
            nb = max(1, int((xb - xa) / 0.31))
            for b in range(nb):
                xbal = xa + (b + 0.5) * (xb - xa) / nb
                mb.stamp(bal, xf((xbal, 0.0, z + 0.16)))
        mb.pop()


# =====================================================================================
# buttress / pinnacle / finial
# =====================================================================================
def finial(mb, cx, cy, z, s=1.0, mat=GOLD):
    mb.lathe([(0.03 * s, z), (0.05 * s, z + 0.30 * s), (0.15 * s, z + 0.46 * s), (0.19 * s, z + 0.60 * s),
              (0.13 * s, z + 0.76 * s), (0.035 * s, z + 0.86 * s), (0.022 * s, z + 1.9 * s), (0.0, z + 2.05 * s)],
             10, mat, (cx, cy))
    return z + 2.05 * s


def pinnacle(mb, cx, cy, z, s=0.56, shaft=1.5, spire=2.6, mat=TRIM):
    """Square gothic pinnacle with crocketed spire.  Returns top z."""
    h = s / 2
    mb.box(cx - h, cx + h, cy - h, cy + h, z, z + shaft, mat, skip="-z")
    # little gablets on the four faces
    zt = z + shaft
    mb.box(cx - h - 0.06, cx + h + 0.06, cy - h - 0.06, cy + h + 0.06, zt, zt + 0.10, mat)
    zb = zt + 0.10
    mb.frustum(rect_pts(cx - h * 0.92, cx + h * 0.92, cy - h * 0.92, cy + h * 0.92), zb,
               rect_pts(cx - 0.04, cx + 0.04, cy - 0.04, cy + 0.04), zb + spire, mat)
    # crockets along the four arrises
    for k in range(1, 5):
        t = k / 5.2
        r = h * 0.92 * (1 - t) + 0.04 * t
        zc = zb + spire * t
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            mb.box(cx + sx * r - 0.045, cx + sx * r + 0.045, cy + sy * r - 0.045, cy + sy * r + 0.045, zc - 0.05, zc + 0.07, mat)
    # finial
    ztip = zb + spire
    mb.box(cx - 0.09, cx + 0.09, cy - 0.09, cy + 0.09, ztip - 0.02, ztip + 0.06, mat)
    mb.lathe([(0.0, ztip + 0.06), (0.085, ztip + 0.15), (0.0, ztip + 0.30)], 6, mat, (cx, cy))
    return ztip + 0.30


def buttress(mb, x, z0, z1, w=1.0, d1=1.25, d2=0.72, split=0.55, pinn=True, mat=STONE):
    """Two-stage buttress on the current facade at local x.  z1 = wall top."""
    zm = z0 + split * (z1 - z0)
    outline = [(0.0, z0 - 0.15), (-d1, z0 - 0.15), (-d1, zm), (-d2, zm + 0.9), (-d2, z1 - 1.0), (0.0, z1 + 0.25)]
    mb.extrude_yz(outline, x - w / 2, x + w / 2, mat)
    mb.box(x - w / 2 - 0.10, x + w / 2 + 0.10, -d1 - 0.10, -0.001, z0, z0 + 1.1, DARK, skip="-z+y")
    # weathering slabs
    mb.extrude_yz([(-d1 - 0.05, zm - 0.02), (-d2 + 0.02, zm + 0.93), (-d2 + 0.02, zm + 1.03), (-d1 - 0.05, zm + 0.10)],
                  x - w / 2 - 0.05, x + w / 2 + 0.05, TRIM)
    top = z1 + 0.25
    if pinn:
        top = pinnacle(mb, x, -d2 * 0.55, z1 - 0.8, s=min(0.62, w * 0.62), shaft=2.3, spire=2.9)
    return top


# =====================================================================================
# roofs
# =====================================================================================
def cone_roof(mb, cx, cy, z, r, h, segs=24, overhang=0.5, a0=0.0, fin=True, mat=SLATE, lucarnes=None):
    """Bell-cast conical / polygonal spire sitting on a wall top at z.  Returns tip z (incl. finial)."""
    R = r + overhang
    prof = [(r - 0.06, z - 0.30), (R, z - 0.28), (R, z - 0.20)]
    for k in range(1, 9):
        t = k / 8.0
        rr = 0.085 + (R - 0.085) * ((1 - t) ** 1.32)
        prof.append((rr, z - 0.20 + (h + 0.20) * t))
    mb.lathe(prof, segs, mat, (cx, cy), a0=a0, a1=a0 + TAU)
    zt = z + h
    mb.lathe([(0.20, zt - 0.55), (0.24, zt - 0.38), (0.12, zt - 0.05), (0.07, zt + 0.10)], 10, LEAD, (cx, cy))
    # small gabled spire lights (lucarnes) + lead bands for scale and richness
    if lucarnes is None:
        lucarnes = 4 if (r >= 2.6 and h >= 8.0) else 0
    for k in range(lucarnes):
        a = a0 + TAU * (k + 0.5) / lucarnes
        t = 0.16
        rr = 0.085 + (R - 0.085) * ((1 - t) ** 1.32)
        zl = z - 0.20 + (h + 0.20) * t
        dormer(mb, cx + (rr + 0.12) * math.cos(a), cy + (rr + 0.12) * math.sin(a), zl - 0.75, a + math.pi / 2,
               w=min(1.15, r * 0.34), h=1.05, depth=1.5)
    if h >= 6.0:
        for t in (0.42, 0.68):
            rr = 0.085 + (R - 0.085) * ((1 - t) ** 1.32)
            zb = z - 0.20 + (h + 0.20) * t
            mb.lathe([(rr + 0.01, zb - 0.07), (rr + 0.045, zb - 0.03), (rr + 0.04, zb + 0.05), (rr - 0.02, zb + 0.09)], segs, LEAD,
                     (cx, cy), a0=a0, a1=a0 + TAU)
    if fin:
        return finial(mb, cx, cy, zt + 0.05, s=1.0)
    return zt


def gable_roof(mb, x0, x1, y0, y1, z, rise, axis='y', overhang=0.45, mat=SLATE, crest=True):
    """Pitched roof over the rectangle; ridge along `axis` at height z + rise.  Eaves at z."""
    if axis == 'y':
        xm = (x0 + x1) / 2; half = (x1 - x0) / 2
        ze = z - overhang * rise / half
        mb.quad((x0 - overhang, y0, ze), (xm, y0, z + rise), (xm, y1, z + rise), (x0 - overhang, y1, ze), mat)
        mb.quad((x1 + overhang, y0, ze), (x1 + overhang, y1, ze), (xm, y1, z + rise), (xm, y0, z + rise), mat)
        mb.box(xm - 0.13, xm + 0.13, y0 - 0.04, y1 + 0.04, z + rise - 0.10, z + rise + 0.10, LEAD)
        if crest:
            n = int((y1 - y0) / 0.6)
            for i in range(n):
                y = y0 + (i + 0.5) * (y1 - y0) / n
                mb.box(xm - 0.02, xm + 0.02, y - 0.02, y + 0.02, z + rise + 0.10, z + rise + 0.48, METAL)
                mb.box(xm - 0.02, xm + 0.02, y - 0.13, y + 0.13, z + rise + 0.27, z + rise + 0.31, METAL)
    else:
        ym = (y0 + y1) / 2; half = (y1 - y0) / 2
        ze = z - overhang * rise / half
        mb.quad((x0, y0 - overhang, ze), (x1, y0 - overhang, ze), (x1, ym, z + rise), (x0, ym, z + rise), mat)
        mb.quad((x0, y1 + overhang, ze), (x0, ym, z + rise), (x1, ym, z + rise), (x1, y1 + overhang, ze), mat)
        mb.box(x0 - 0.04, x1 + 0.04, ym - 0.13, ym + 0.13, z + rise - 0.10, z + rise + 0.10, LEAD)
        if crest:
            n = int((x1 - x0) / 0.6)
            for i in range(n):
                x = x0 + (i + 0.5) * (x1 - x0) / n
                mb.box(x - 0.02, x + 0.02, ym - 0.02, ym + 0.02, z + rise + 0.10, z + rise + 0.48, METAL)
                mb.box(x - 0.13, x + 0.13, ym - 0.02, ym + 0.02, z + rise + 0.27, z + rise + 0.31, METAL)


def gable_wall(mb, p0, p1, z, rise, wall=STONE, oculus=True, rng=RNG, coping=True, tri=True):
    """Triangular gable end between plan points p0 -> p1 (outside on the right), base at z."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    mb.push(frame_xf((p0[0], p0[1], 0.0), (dx, dy)))
    tri_pts = [(0.0, z), (L, z), (L / 2, z + rise)]
    holes = []
    if tri and oculus and rise > 4.0:
        s = W('round', w=min(2.2, rise * 0.22), depth=0.3, nx=2, nz=2, hood=True, sill=False)
        holes.append(window(mb, L / 2, z + rise * 0.30, s, rng))
    if tri:
        mb.poly_holes(tri_pts, holes, wall)
    if coping:
        k = rise / (L / 2)
        ln = math.hypot(1.0, k)
        tx, tz = 0.42 * ln / k if k > 1e-6 else 0.42, 0.42 * ln   # thickness measured along x / z
        band = [(-0.30, z - 0.30 * k), (L / 2, z + rise + 0.02), (L + 0.30, z - 0.30 * k),
                (L + 0.30 + tx * 0.0, z - 0.30 * k + tz), (L / 2, z + rise + 0.02 + tz), (-0.30, z - 0.30 * k + tz)]
        mb.extrude_xz(band, -0.16, 0.34, TRIM)
        # crockets along both rakes
        nk = max(2, int(math.hypot(L / 2, rise) / 1.15))
        for side in (0, 1):
            for i in range(1, nk):
                t = i / nk
                xk = (-0.30 + (L / 2 + 0.30) * t) if side == 0 else (L + 0.30 - (L / 2 + 0.30) * t)
                zk = z - 0.30 * k + tz + (rise + 0.02 + 0.30 * k) * t
                mb.lathe([(0.0, zk - 0.05), (0.12, zk + 0.02), (0.14, zk + 0.14), (0.06, zk + 0.26), (0.0, zk + 0.36)], 6, TRIM, (xk, 0.09))
        # kneelers + apex finial
        for xk in (-0.30, L + 0.30):
            mb.box(xk - 0.36, xk + 0.36, -0.22, 0.40, z - 0.30 * k - 0.30, z - 0.30 * k + tz * 0.55, TRIM)
        pinnacle(mb, L / 2, 0.09, z + rise + tz * 0.9, s=0.5, shaft=0.7, spire=2.0)
    mb.pop()


def dormer(mb, cx, cy, z, ang, w=1.8, h=1.9, depth=2.6, rng=RNG):
    """Gabled dormer; front face at local y=0 looking toward local -Y, body runs back (+Y) into the roof."""
    mb.push(xf((cx, cy, z), rz=ang))
    s = W('arch', w=w * 0.52, spring=h * 0.56, sharp=0.9, depth=0.16, nx=2, nz=2, hood=False, sill=False)
    outline = window(mb, 0.0, 0.28, s, rng)
    gh = w * 0.62
    mb.poly_holes([(-w / 2, 0.0), (w / 2, 0.0), (w / 2, h), (0.0, h + gh), (-w / 2, h)], [outline], TRIM)
    mb.quad((-w / 2, 0, 0), (-w / 2, 0, h), (-w / 2, depth, h), (-w / 2, depth, 0), TRIM)
    mb.quad((w / 2, 0, 0), (w / 2, depth, 0), (w / 2, depth, h), (w / 2, 0, h), TRIM)
    ov = 0.18
    mb.quad((-w / 2 - ov, -ov, h - ov * gh / (w / 2)), (0.0, -ov, h + gh), (0.0, depth, h + gh), (-w / 2 - ov, depth, h - ov * gh / (w / 2)), SLATE)
    mb.quad((w / 2 + ov, -ov, h - ov * gh / (w / 2)), (w / 2 + ov, depth, h - ov * gh / (w / 2)), (0.0, depth, h + gh), (0.0, -ov, h + gh), SLATE)
    mb.box(-0.07, 0.07, -ov - 0.02, depth, h + gh - 0.05, h + gh + 0.08, LEAD)
    mb.lathe([(0.0, h + gh + 0.08), (0.07, h + gh + 0.2), (0.02, h + gh + 0.34), (0.0, h + gh + 0.7)], 6, GOLD, (0.0, -ov + 0.06))
    mb.pop()


def chimney(mb, cx, cy, z0, h, sx=1.4, sy=0.85, mat=STONE):
    mb.box(cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, z0, z0 + h, mat, skip="-z")
    mb.box(cx - sx / 2 - 0.10, cx + sx / 2 + 0.10, cy - sy / 2 - 0.10, cy + sy / 2 + 0.10, z0 + h - 0.55, z0 + h - 0.40, TRIM)
    mb.box(cx - sx / 2 - 0.14, cx + sx / 2 + 0.14, cy - sy / 2 - 0.14, cy + sy / 2 + 0.14, z0 + h, z0 + h + 0.16, TRIM)
    n = max(1, int(sx / 0.55))
    for i in range(n):
        x = cx - sx / 2 + (i + 0.5) * sx / n
        mb.lathe([(0.15, z0 + h + 0.16), (0.13, z0 + h + 0.55), (0.17, z0 + h + 0.60), (0.17, z0 + h + 0.68), (0.11, z0 + h + 0.68)],
                 8, "stone_dark", (x, cy))


# =====================================================================================
# towers
# =====================================================================================
SLIT = W('arch', w=0.62, spring=1.45, sharp=1.25, depth=0.30, nx=1, nz=3, hood=True, sill=True, n=5)


def round_tower(mb, cx, cy, z0, r, eave_z, roof_h, segs=24, flare=0.15, flare_h=4.5, storey=4.4,
                win=SLIT, win_step=3, top='cone', wall=STONE, rng=RNG, a0=0.0, sill=1.3, fin=True,
                roof_mat=SLATE, base_mat=DARK, win_from=0):
    """Round (or polygonal, e.g. segs=8) tower.  Returns the tip (x, y, z)."""
    da = TAU / segs

    def ring(rad):
        return [(cx + rad * math.cos(a0 + i * da), cy + rad * math.sin(a0 + i * da)) for i in range(segs)]

    P1 = ring(r)
    zs = z0
    if flare_h > 0:
        mb.frustum(ring(r * (1 + flare)), z0, P1, z0 + flare_h, base_mat, cap_top=False)
        mb.lathe([(r, z0 + flare_h - 0.12), (r + 0.11, z0 + flare_h - 0.08), (r + 0.11, z0 + flare_h + 0.08), (r, z0 + flare_h + 0.14)],
                 segs, TRIM, (cx, cy), a0=a0, a1=a0 + TAU)
        zs = z0 + flare_h
    nst = max(1, int(round((eave_z - zs) / storey)))
    sh = (eave_z - zs) / nst
    chord = 2 * r * math.sin(da / 2)
    for k in range(nst):
        za = zs + k * sh; zb = za + sh
        for i in range(segs):
            pa = P1[i]; pb = P1[(i + 1) % segs]
            mb.push(frame_xf((pa[0], pa[1], 0.0), (pb[0] - pa[0], pb[1] - pa[1])))
            has = win is not None and k >= win_from and ((i + k * 2) % win_step == 0) and win['w'] < chord - 0.16 \
                and win_height(win) + sill < sh - 0.3
            if has:
                outline = window(mb, chord / 2, za + sill, win, rng)
                mb.poly_holes([(0, za), (chord, za), (chord, zb), (0, zb)], [outline], wall)
            else:
                mb.quad((0, 0, za), (chord, 0, za), (chord, 0, zb), (0, 0, zb), wall)
            mb.pop()
        if k < nst - 1:
            mb.lathe([(r, zb - 0.10), (r + 0.09, zb - 0.07), (r + 0.11, zb + 0.05), (r, zb + 0.11)],
                     segs, TRIM, (cx, cy), a0=a0, a1=a0 + TAU)
    e = eave_z
    # blind arcade frieze under the tower head (one bay of little arches per facet)
    if chord > 0.9 and e - zs > 6.0:
        for i in range(segs):
            pa = P1[i]; pb = P1[(i + 1) % segs]
            mb.push(frame_xf((pa[0], pa[1], 0.0), (pb[0] - pa[0], pb[1] - pa[1])))
            blind_arcade(mb, 0.04, chord - 0.04, e - (2.45 if top == 'cone' else 2.75), 0.95, pitch=min(0.5, chord / 2.0))
            mb.pop()
    if top == 'cone':
        mb.lathe([(r, e - 0.95), (r + 0.10, e - 0.86), (r + 0.14, e - 0.62), (r + 0.30, e - 0.46), (r + 0.30, e - 0.22),
                  (r + 0.36, e - 0.12), (r + 0.36, e), (r, e)], segs, TRIM, (cx, cy), a0=a0, a1=a0 + TAU)
        tip = cone_roof(mb, cx, cy, e, r + 0.36, roof_h, segs=segs, overhang=0.34, a0=a0, fin=fin, mat=roof_mat)
    else:   # crenellated fighting platform with a recessed spire
        ro = r + 0.55
        mb.lathe([(r, e - 1.3), (r + 0.12, e - 1.2), (r + 0.18, e - 0.9), (ro - 0.1, e - 0.5), (ro, e - 0.3), (ro, e + 0.62),
                  (ro - 0.42, e + 0.62), (ro - 0.42, e)], segs, wall, (cx, cy), a0=a0, a1=a0 + TAU)
        mb.lathe([(0.0, e), (ro - 0.42, e)], segs, base_mat, (cx, cy), a0=a0, a1=a0 + TAU)   # deck
        Po = ring(ro); Pi = ring(ro - 0.42)
        for i in range(0, segs, 2):
            j = (i + 1) % segs
            poly = [Pi[i], Po[i], Po[j], Pi[j]]
            mb.prism(poly, e + 0.62, e + 1.55, wall, cap_top=True, mat_top=TRIM)
        # corbels under the platform
        for i in range(segs):
            a = a0 + (i + 0.5) * da
            mb.push(xf((cx + (r + 0.02) * math.cos(a), cy + (r + 0.02) * math.sin(a), 0.0), rz=a - math.pi / 2))
            mb.box(-0.13, 0.13, -0.42, 0.0, e - 0.62, e - 0.30, TRIM)
            mb.box(-0.13, 0.13, -0.22, 0.0, e - 0.92, e - 0.62, TRIM)
            mb.pop()
        ri = r * 0.62
        mb.lathe([(ri, e), (ri, e + 2.6), (ri + 0.2, e + 2.75), (ri + 0.2, e + 2.95)], segs, wall, (cx, cy), a0=a0, a1=a0 + TAU)
        tip = cone_roof(mb, cx, cy, e + 2.95, ri + 0.2, roof_h, segs=segs, overhang=0.3, a0=a0, fin=fin, mat=roof_mat)
    return (cx, cy, tip)


def bartizan(mb, cx, cy, z, r=1.25, h=5.2, roof_h=5.0, segs=12, rng=RNG, a0=0.0):
    """Corbelled corner turret hanging off a wall; its floor is at z."""
    mb.lathe([(0.10, z - 2.7), (r * 0.42, z - 2.0), (r * 0.48, z - 1.84), (r * 0.80, z - 0.95), (r * 0.86, z - 0.78),
              (r + 0.09, z - 0.12), (r + 0.09, z + 0.04), (r, z + 0.04)], segs, TRIM, (cx, cy), cap_bottom=True,
             a0=a0, a1=a0 + TAU)
    slit = W('arch', w=0.34, spring=0.95, sharp=1.2, depth=0.2, nx=1, nz=2, hood=False, sill=False, n=4)
    return round_tower(mb, cx, cy, z, r, z + h, roof_h, segs=segs, flare=0.0, flare_h=0.0, storey=h, win=slit,
                       win_step=2, sill=1.6, rng=rng, a0=a0)





# =====================================================================================
# sculpture
# =====================================================================================
def statue(mb, x, z, h=1.85, y=-0.36, canopy=True, mat=TRIM):
    """Robed figure on a moulded corbel under a gabled canopy, on the current facade at local x, corbel top z."""
    s = h / 1.85
    # corbel pedestal
    mb.lathe([(0.04, z - 0.55), (0.12, z - 0.42), (0.14, z - 0.30), (0.24, z - 0.12), (0.27, z - 0.04), (0.27, z)], 8, mat, (x, y), cap_top=True)
    # figure: robe, shoulders, head
    mb.lathe([(0.21 * s, z), (0.235 * s, z + 0.08 * s), (0.215 * s, z + 0.55 * s), (0.17 * s, z + 0.98 * s), (0.20 * s, z + 1.22 * s),
              (0.215 * s, z + 1.36 * s), (0.10 * s, z + 1.47 * s), (0.085 * s, z + 1.52 * s), (0.115 * s, z + 1.60 * s),
              (0.12 * s, z + 1.70 * s), (0.085 * s, z + 1.80 * s), (0.0, z + 1.85 * s)], 10, mat, (x, y))
    # arms / book block and a staff so the silhouette is not a skittle
    mb.box(x - 0.16 * s, x + 0.16 * s, y - 0.25 * s, y - 0.12 * s, z + 0.95 * s, z + 1.15 * s, mat)
    mb.box(x + 0.25 * s, x + 0.285 * s, y - 0.17 * s, y - 0.135 * s, z, z + 1.95 * s, mat)
    if canopy:
        zc = z + 1.98 * s
        mb.box(x - 0.36, x + 0.36, y - 0.36, -0.001, zc, zc + 0.10, mat)
        mb.extrude_xz([(x - 0.36, zc + 0.10), (x + 0.36, zc + 0.10), (x, zc + 0.62)], y - 0.36, -0.001, mat)
        pinnacle(mb, x, y - 0.12, zc + 0.5, s=0.2, shaft=0.25, spire=0.75, mat=mat)


def gargoyle(mb, x, z, length=1.35, mat=TRIM):
    """Projecting gargoyle on the current facade (neck, head, ears, folded wings)."""
    y0, y1 = -0.30, -0.30 - length
    mb.box(x - 0.16, x + 0.16, -0.34, -0.001, z - 0.20, z + 0.20, mat)
    mb.frustum(rect_pts(x - 0.14, x + 0.14, y1, y0), z - 0.16, rect_pts(x - 0.09, x + 0.09, y1 - 0.05, y0), z + 0.12, mat)
    # head + snout + ears
    mb.box(x - 0.13, x + 0.13, y1 - 0.30, y1 + 0.02, z - 0.10, z + 0.20, mat)
    mb.box(x - 0.08, x + 0.08, y1 - 0.48, y1 - 0.28, z - 0.10, z + 0.04, mat)
    for sx in (-1, 1):
        mb.frustum(rect_pts(x + sx * 0.11 - 0.035, x + sx * 0.11 + 0.035, y1 - 0.10, y1 - 0.02), z + 0.20,
                   rect_pts(x + sx * 0.13 - 0.01, x + sx * 0.13 + 0.01, y1 - 0.04, y1 - 0.02), z + 0.36, mat)
        # wing
        mb.extrude_yz([(y0 - 0.15, z + 0.10), (y0 - 0.75, z + 0.16), (y0 - 0.55, z + 0.62), (y0 - 0.25, z + 0.40)],
                      x + sx * 0.15 - 0.02, x + sx * 0.15 + 0.02, mat)


