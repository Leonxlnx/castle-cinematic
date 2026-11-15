"""Castle architecture: great hall with the giant glazed gothic window, wings, keep, towers, roofs,
banner, flags and architectural lighting (uplights, sconces, interior glow).

Everything is generated with the gothic kit (scene/gothic.py).  Coordinates follow scene/layout.py.
"""
import bpy, math, random
import numpy as np
from mathutils import Vector, Matrix
from .. import layout as L
from .. import mats
from ..common import MB, get_col, frame_xf, xf, rect_pts, circle_pts, pointed_arch, arch_apex, TAU, new_material, look_at
from .. import gothic as G
from ..gothic import W, STONE, TRIM, DARK, SLATE, LEAD, METAL, GOLD, GLASS

Z0 = L.PODIUM_Z          # castle ground-floor level
BASE = -3.6              # walls continue down to here (hidden in the ground / podium)
rng = random.Random(2026)
LIGHTS = []              # (kind, pos, energy, color, cutoff, target, spot_size, radius)

WARM = (1.0, 0.56, 0.26)
WARM2 = (1.0, 0.50, 0.20)


def _light(kind, pos, energy, color=WARM, cutoff=22.0, target=None, spot=1.5, radius=0.25, spec=1.0, shadow=False):
    LIGHTS.append((kind, tuple(pos), energy, color, cutoff, target, spot, radius, spec, shadow))


def _wpos(mb, p):
    v = mb.M @ Vector(p)
    return (v.x, v.y, v.z)


# =====================================================================================
# generic rectangular block
# =====================================================================================
def rect_block(mb, x0, x1, y0, y1, z0, heights, specs, bay=4.6, margin=0.8, parapet='crenel',
               parapet_sides='SENW', butt='', skip=None, wall=STONE, strings=True, deck=True, pinn=True,
               uplight='', uplight_e=520.0, balconies=None, quoin='SW SE NE NW', statues='', corner_pinn=''):
    """specs: dict side -> list of (spec, sill) per floor | None (blank wall) | False (no wall at all).
    Returns wall top z."""
    skip = skip or {}
    o = rect_pts(x0, x1, y0, y1)
    sides = {'S': (o[0], o[1]), 'E': (o[1], o[2]), 'N': (o[2], o[3]), 'W': (o[3], o[0])}
    H = sum(heights); zt = z0 + H
    for side, (p0, p1) in sides.items():
        sp = specs.get(side)
        if sp is False:
            continue
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        Ls = math.hypot(dx, dy)
        if sp is None:
            mb.push(frame_xf((p0[0], p0[1], 0.0), (dx, dy)))
            mb.quad((0, 0, z0), (Ls, 0, z0), (Ls, 0, zt), (0, 0, zt), wall)
            mb.pop()
            continue
        floors = [(h, s[0], s[1]) for h, s in zip(heights, sp)]
        _, nb, bw = G.facade(mb, p0, p1, z0, floors, bay=bay, margin=margin, wall=wall, rng=rng, skip=skip.get(side, ()))
        mb.push(frame_xf((p0[0], p0[1], 0.0), (dx, dy)))
        if side in butt:
            for i in range(nb + 1):
                G.buttress(mb, margin + i * bw, z0, zt, pinn=pinn)
        else:
            for i in range(1, nb):
                spout(mb, margin + i * bw, zt - 0.95)
        if side in statues and len(heights) > 1:
            zs_ = z0 + heights[0] + 0.75
            for i in range(1, nb):
                if side in butt:
                    G.statue(mb, margin + i * bw, z0 + 3.1, y=-1.25 - 0.36)
                else:
                    G.statue(mb, margin + i * bw, zs_)
        if balconies and side in balconies:
            fi, bays_, bwid = balconies[side]
            zb_ = z0 + sum(heights[:fi])
            for i in bays_:
                if i < nb:
                    balcony(mb, margin + (i + 0.5) * bw, zb_ + 0.62, w=bwid)
        if side in uplight:
            for i in range(nb + 1):
                xl = margin + i * bw
                yl = -2.0 if side in butt else -0.9
                _light('SPOT', _wpos(mb, (xl + (0.0 if side not in butt else bw * 0.5), yl, z0 + 0.35)), uplight_e, WARM, 26.0,
                       _wpos(mb, (xl + (0.0 if side not in butt else bw * 0.5), 0.6, z0 + 14.0)), 1.45)
        mb.pop()
    mb.prism(o, BASE, z0, DARK, cap_top=False)
    G.plinth(mb, o, z0)
    if quoin:
        quoins(mb, x0, x1, y0, y1, z0, zt, quoin)
    if strings:
        z = z0
        for h in heights[:-1]:
            z += h
            G.string_course(mb, o, z)
    G.cornice(mb, o, zt)
    for side, (p0, p1) in sides.items():
        if specs.get(side) is False:
            continue
        G.corbel_table(mb, p0, p1, zt - 0.62)
        if parapet == 'crenel' and side in parapet_sides:
            G.crenellation(mb, p0, p1, zt)
    if parapet == 'low':
        G.low_parapet(mb, o, zt)
    cs_ = {'SW': (x0 - 0.2, y0 - 0.2), 'SE': (x1 + 0.2, y0 - 0.2), 'NE': (x1 + 0.2, y1 + 0.2), 'NW': (x0 - 0.2, y1 + 0.2)}
    for nm in corner_pinn.split():
        px, py = cs_[nm]
        G.pinnacle(mb, px, py, zt + (0.5 if parapet == 'crenel' else 0.9), s=0.62, shaft=2.2, spire=3.4)
    if deck:
        mb.quad((x0, y0, zt + 0.02), (x1, y0, zt + 0.02), (x1, y1, zt + 0.02), (x0, y1, zt + 0.02), LEAD)
    return zt


def quoins(mb, x0, x1, y0, y1, z0, zt, corners='SW SE NE NW'):
    """Alternating long/short dressed corner stones."""
    cs = {'SW': (x0, y0, 1, 1), 'SE': (x1, y0, -1, 1), 'NE': (x1, y1, -1, -1), 'NW': (x0, y1, 1, -1)}
    for name in corners.split():
        cx, cy, ox, oy = cs[name]
        z = z0 + 1.12; k = 0
        while z < zt - 1.0:
            lx, ly = (0.98, 0.52) if k % 2 == 0 else (0.52, 0.98)
            xa, xb = sorted((cx - ox * 0.035, cx + ox * lx)); ya, yb = sorted((cy - oy * 0.035, cy + oy * ly))
            mb.box(xa, xb, ya, yb, z + 0.012, z + 0.408, TRIM)
            z += 0.42; k += 1


def spout(mb, x, z):
    """Small stone water spout / gargoyle stub on the current facade under the cornice."""
    G.gargoyle(mb, x, z + 0.2)


def balcony(mb, x, z, w=3.0, depth=1.25):
    """Stone balcony on corbels with a balustrade, on the current facade (floor level z)."""
    mb.box(x - w / 2, x + w / 2, -depth, -0.001, z - 0.26, z, TRIM)
    mb.sweep([(0.0, z - 0.26), (0.05, z - 0.26), (0.09, z - 0.18), (0.09, z - 0.04), (0.0, z)],
             [(x - w / 2, -0.001), (x - w / 2, -depth), (x + w / 2, -depth), (x + w / 2, -0.001)], TRIM, closed=False)
    n = max(2, int(w / 1.1))
    for i in range(n + 1):
        xc = x - w / 2 + 0.2 + (w - 0.4) * i / n
        mb.extrude_yz([(-0.001, z - 0.26), (-depth + 0.15, z - 0.26), (-depth + 0.15, z - 0.42), (-0.55, z - 0.62), (-0.001, z - 1.05)],
                      xc - 0.11, xc + 0.11, TRIM)
    G.balustrade(mb, [(x - w / 2 + 0.2, -0.22), (x - w / 2 + 0.2, -depth + 0.2), (x + w / 2 - 0.2, -depth + 0.2), (x + w / 2 - 0.2, -0.22)],
                 z, pier_every=2.4, end_piers=False)


def hip_roof(mb, x0, x1, y0, y1, z, tx0, tx1, ty0, ty1, zt, mat=SLATE, crest=True):
    """Steep hipped roof from the base rectangle at z to the top rectangle at zt (explicit slate UVs)."""
    b = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    t = [(tx0, ty0), (tx1, ty0), (tx1, ty1), (tx0, ty1)]
    for i in range(4):
        j = (i + 1) % 4
        a0, a1, b1, b0 = b[i], b[j], t[j], t[i]
        le = math.hypot(a1[0] - a0[0], a1[1] - a0[1])
        mid_b = ((a0[0] + a1[0]) / 2, (a0[1] + a1[1]) / 2); mid_t = ((b0[0] + b1[0]) / 2, (b0[1] + b1[1]) / 2)
        sl = math.sqrt((mid_t[0] - mid_b[0]) ** 2 + (mid_t[1] - mid_b[1]) ** 2 + (zt - z) ** 2)
        lt = math.hypot(b1[0] - b0[0], b1[1] - b0[1])
        uv = ((0, 0), (le, 0), ((le + lt) / 2, sl), ((le - lt) / 2, sl))
        mb.quad((a0[0], a0[1], z), (a1[0], a1[1], z), (b1[0], b1[1], zt), (b0[0], b0[1], zt), mat, uv=uv)
    mb.box(tx0 - 0.1, tx1 + 0.1, ty0 - 0.1, ty1 + 0.1, zt - 0.06, zt + 0.12, LEAD)
    if crest:
        n = max(2, int((tx1 - tx0) / 0.6))
        ym = (ty0 + ty1) / 2
        for i in range(n):
            x = tx0 + (i + 0.5) * (tx1 - tx0) / n
            mb.box(x - 0.02, x + 0.02, ym - 0.02, ym + 0.02, zt + 0.12, zt + 0.58, METAL)
            mb.box(x - 0.14, x + 0.14, ym - 0.02, ym + 0.02, zt + 0.32, zt + 0.36, METAL)
    # lead hips
    for i in range(4):
        a, c = b[i], t[i]
        v = Vector((c[0] - a[0], c[1] - a[1], zt - z)); ln = v.length
        mb.push(Matrix.Translation(Vector((a[0], a[1], z))) @ v.to_track_quat('Z', 'Y').to_matrix().to_4x4())
        mb.box(-0.09, 0.09, -0.09, 0.09, 0.0, ln, LEAD)
        mb.pop()


def sconce(mb, x, z, energy=110.0):
    """Wall lantern on the current facade (local frame) + its light."""
    mb.box(x - 0.05, x + 0.05, -0.30, -0.001, z + 0.62, z + 0.70, METAL)
    mb.box(x - 0.03, x + 0.03, -0.30, -0.24, z + 0.40, z + 0.70, METAL)
    # cage
    for sx in (-0.15, 0.15):
        for sy in (-0.42, -0.12):
            mb.box(x + sx - 0.012, x + sx + 0.012, sy - 0.012, sy + 0.012, z - 0.10, z + 0.40, METAL)
    mb.box(x - 0.17, x + 0.17, -0.44, -0.10, z - 0.13, z - 0.10, METAL)
    mb.frustum(rect_pts(x - 0.19, x + 0.19, -0.46, -0.08), z + 0.40, rect_pts(x - 0.04, x + 0.04, -0.31, -0.23), z + 0.55, METAL)
    mb.box(x - 0.135, x + 0.135, -0.405, -0.135, z - 0.09, z + 0.39, "lamp_glass")
    _light('POINT', _wpos(mb, (x, -0.75, z + 0.15)), energy, WARM, 12.0, radius=0.12)


# =====================================================================================
# great hall
# =====================================================================================
def great_window(mb, cx, zs, w, spring, sharp, d):
    """The hero window: deep moulded gothic arch, bronze mullion grid, intersecting tracery, glazed doors."""
    R = sharp * w
    zsp = zs + spring
    arch = pointed_arch(w, zsp, n=16, sharp=sharp, x0=cx)
    outline = [(cx - w / 2, zs), (cx + w / 2, zs)] + arch
    ztop = zsp + arch_apex(w, sharp)
    # three receding orders
    mb.sweep_xz([(0.0, 0.0), (0.0, 0.28), (-0.20, 0.28), (-0.20, 0.36), (-0.26, 0.42), (-0.26, 0.66), (-0.44, 0.66),
                 (-0.44, 0.74), (-0.50, 0.80), (-0.50, d)], outline, TRIM, closed=True, flip=True)
    mb.sweep_xz([(0.06, 0.0), (0.06, -0.18), (0.20, -0.22), (0.36, -0.16), (0.46, 0.0)], arch, TRIM, closed=False)
    for side in (-1, 1):
        xc = cx + side * (w / 2 + 0.26)
        mb.box(xc - 0.27, xc + 0.27, -0.24, -0.001, zsp - 0.5, zsp + 0.05, TRIM)
        # engaged shafts with capitals + bases in the jamb
        for k, off in enumerate((0.10, 0.34)):
            xs = cx + side * (w / 2 - off + 0.02)
            ys = 0.14 + k * 0.38
            mb.lathe([(0.11, zs), (0.11, zs + 0.25), (0.075, zs + 0.34), (0.075, zsp - 0.5), (0.10, zsp - 0.42),
                      (0.14, zsp - 0.2), (0.14, zsp - 0.05)], 10, TRIM, (xs, ys))
    # clear glass
    hw = w / 2 - 0.50
    mb.quad((cx - w / 2, d, zs), (cx + w / 2, d, zs), (cx + w / 2, d, ztop + 0.05), (cx - w / 2, d, ztop + 0.05), "glass_clear")

    def arch_z(a):   # height of the inner arch line above x = cx +/- a
        a = min(abs(a), hw)
        c = hw - (sharp * 2 * hw)          # inner arch built from the inner width
        Ri = sharp * 2 * hw
        return zsp + math.sqrt(max(Ri * Ri - (a - c) ** 2, 0.0))

    y0, y1 = d - 0.16, d - 0.02
    nm = 8
    step = 2 * hw / nm
    # perimeter frame
    inner = [(cx - hw, zs), (cx + hw, zs)] + pointed_arch(2 * hw, zsp, n=16, sharp=sharp, x0=cx)
    mb.sweep_xz([(0.0, y1), (0.0, y0), (-0.10, y0), (-0.10, y1)], inner, METAL, closed=True, flip=True)
    # mullions (every second one heavier)
    for i in range(1, nm):
        a = -hw + i * step
        t = 0.055 if i % 2 == 0 else 0.035
        mb.box(cx + a - t, cx + a + t, y0, y1, zs, arch_z(a) + 0.05, METAL)
    # transoms
    nz = int(spring / 2.25)
    for j in range(1, nz + 1):
        z = zs + j * spring / nz
        t = 0.05 if j in (2, nz) else 0.032
        mb.box(cx - hw, cx + hw, y0, y1, z - t, z + t, METAL)
    # intersecting tracery in the head
    Ri = sharp * 2 * hw
    cxl = cx - hw + Ri; cxr = cx + hw - Ri
    bar = [(-0.04, y1), (-0.04, y0), (0.04, y0), (0.04, y1)]
    for i in range(1, nm):
        m = cx - hw + i * step
        # arc bending left (parallel to the right side of the main arch)
        ce = (cxl - (m - Ri)) / (2 * Ri)
        if -1 < ce < 1:
            th = math.acos(ce)
            pts = [(m - Ri + Ri * math.cos(th * k / 12), zsp + Ri * math.sin(th * k / 12)) for k in range(13)]
            mb.sweep_xz(bar, pts, METAL, closed=False)
        ce = ((m + Ri) - cxr) / (2 * Ri)
        if -1 < ce < 1:
            th = math.acos(ce)
            pts = [(m + Ri - Ri * math.cos(th * k / 12), zsp + Ri * math.sin(th * k / 12)) for k in range(13)]
            mb.sweep_xz(bar, pts[::-1], METAL, closed=False)
    # glazed entrance doors in the bottom register: heavier bronze portal frame
    dz = zs + spring / nz * 2
    mb.box(cx - 2 * step - 0.09, cx - 2 * step + 0.09, y0 - 0.06, y1, zs, dz, METAL)
    mb.box(cx + 2 * step - 0.09, cx + 2 * step + 0.09, y0 - 0.06, y1, zs, dz, METAL)
    mb.box(cx - 2 * step - 0.09, cx + 2 * step + 0.09, y0 - 0.06, y1, dz - 0.09, dz + 0.09, METAL)
    for sx in (-0.12, 0.12):
        mb.box(cx + sx - 0.025, cx + sx + 0.025, y0 - 0.14, y0 - 0.06, zs + 0.9, zs + 2.1, GOLD)   # door pulls
    return outline, ztop


def hall_interior(mb, x0, x1, y0, y1, z0, zc):
    """What you see through the great window: a tall, warmly lit hall."""
    # floor, back wall, side walls, ceiling (normals face inward)
    mb.quad((x0, y0, z0 + 0.02), (x1, y0, z0 + 0.02), (x1, y1, z0 + 0.02), (x0, y1, z0 + 0.02), "marble_wet")
    mb.quad((x0, y1, z0), (x1, y1, z0), (x1, y1, zc), (x0, y1, zc), "interior_glow")
    mb.quad((x0, y0, z0), (x0, y1, z0), (x0, y1, zc), (x0, y0, zc), "interior_glow")
    mb.quad((x1, y1, z0), (x1, y0, z0), (x1, y0, zc), (x1, y1, zc), "interior_glow")
    mb.quad((x0, y0, zc), (x0, y1, zc), (x1, y1, zc), (x1, y0, zc), "wood_dark")
    # tall stained-glass window glowing at the far end of the hall
    xm_ = (x0 + x1) / 2
    sg = [(xm_ - 3.6, z0 + 6.5), (xm_ + 3.6, z0 + 6.5)] + pointed_arch(7.2, z0 + 16.0, n=12, sharp=1.0, x0=xm_)
    mb.poly_holes(sg, [], "stained_glass", to3d=lambda a, b: (a, y1 - 0.06, b), normal=(0, -1, 0))
    fr = [(xm_ - 3.6, z0 + 6.5), (xm_ + 3.6, z0 + 6.5)] + pointed_arch(7.2, z0 + 16.0, n=12, sharp=1.0, x0=xm_)
    mb.push(xf((0.0, y1 - 0.06, 0.0)))
    mb.sweep_xz([(0.0, 0.0), (0.0, -0.30), (0.35, -0.30), (0.35, 0.0)], fr, TRIM, closed=True)
    mb.pop()
    # galleries on both sides + across the back
    for zg in (z0 + 6.2, z0 + 12.6, z0 + 19.0):
        for (ga, gb) in ((x0, x0 + 3.6), (x1 - 3.6, x1)):
            mb.box(ga, gb, y0 + 1.0, y1, zg, zg + 0.35, "wood_dark")
            xr = gb if ga == x0 else ga
            mb.box(xr - 0.05, xr + 0.05, y0 + 1.0, y1, zg + 0.35, zg + 1.35, METAL)
            mb.box(xr - 0.03, xr + 0.03, y0 + 1.0, y1, zg + 0.02, zg + 0.10, "lamp_glow")
        mb.box(x0, x1, y1 - 3.0, y1, zg, zg + 0.35, "wood_dark")
        mb.box(x0 + 3.6, x1 - 3.6, y1 - 3.05, y1 - 2.95, zg + 0.02, zg + 0.10, "lamp_glow")
    # columns
    for yc in np.linspace(y0 + 4.0, y1 - 4.0, 4):
        for xc in (x0 + 3.9, x1 - 3.9):
            mb.lathe([(0.55, z0), (0.55, z0 + 0.5), (0.38, z0 + 0.7), (0.38, zc - 1.2), (0.6, zc - 0.5), (0.6, zc)], 12, TRIM, (xc, yc))
    # chandeliers
    xm = (x0 + x1) / 2
    for yc in (y0 + 5.0, y0 + 11.5, y0 + 18.0):
        zc2 = z0 + 15.5
        mb.box(xm - 0.03, xm + 0.03, yc - 0.03, yc + 0.03, zc2 + 0.6, zc, METAL)
        for (rr, zz, nb) in ((2.0, zc2, 16), (1.3, zc2 + 0.9, 12), (0.7, zc2 + 1.7, 8)):
            mb.lathe([(rr - 0.04, zz - 0.04), (rr + 0.04, zz - 0.04), (rr + 0.04, zz + 0.04), (rr - 0.04, zz + 0.04), (rr - 0.04, zz - 0.04)],
                     24, GOLD, (xm, yc))
            for k in range(nb):
                a = TAU * k / nb
                bx, by = xm + rr * math.cos(a), yc + rr * math.sin(a)
                mb.box(bx - 0.05, bx + 0.05, by - 0.05, by + 0.05, zz + 0.04, zz + 0.30, "lamp_glow")


def build_hall(mb, mbr):
    x0, x1, y0, y1 = -9.0, 13.0, 22.0, 46.0
    ze, rise = 28.0, 14.0
    Lw = x1 - x0
    # ---- south gable facade (hero)
    mb.push(frame_xf((x0, y0, 0.0), (1.0, 0.0)))
    cx = Lw / 2
    outline, ztop = great_window(mb, cx, Z0, 11.0, 18.5, 1.0, 1.25)
    holes = [outline]
    lan1 = W('arch', w=1.15, spring=3.9, sharp=1.2, depth=0.42, nx=2, nz=5, tracery=False, lit=0.8)
    lan2 = W('arch', w=1.15, spring=3.2, sharp=1.2, depth=0.42, nx=2, nz=4, tracery=False, lit=0.55)
    for xs in (3.15, Lw - 3.15):
        holes.append(G.window(mb, xs, Z0 + 7.2, lan1, rng))
        holes.append(G.window(mb, xs, Z0 + 15.6, lan2, rng))
        sconce(mb, xs, Z0 + 3.2, energy=160.0)
        G.statue(mb, xs, Z0 + 12.9, h=1.9)
        G.statue(mb, xs - 1.25, Z0 + 4.6, h=1.75); G.statue(mb, xs + 1.25, Z0 + 4.6, h=1.75)
        G.gargoyle(mb, xs, ze - 1.0, length=1.6)
    rose = W('round', w=2.6, depth=0.45, nx=4, nz=4, hood=True, sill=False, lit=0.9)
    holes.append(G.window(mb, cx, 33.6, rose, rng))
    gl = W('arch', w=0.9, spring=2.4, sharp=1.2, depth=0.35, nx=1, nz=3, lit=0.75, hood=True)
    for xs in (cx - 3.7, cx + 3.7):
        holes.append(G.window(mb, xs, 28.9, gl, rng))
    topw = W('arch', w=0.75, spring=1.5, sharp=1.2, depth=0.3, nx=1, nz=2, lit=0.95, hood=True)
    holes.append(G.window(mb, cx, 37.3, topw, rng))
    mb.poly_holes([(0, Z0), (Lw, Z0), (Lw, ze), (cx, ze + rise), (0, ze)], holes, STONE)
    G.gable_wall(mb, (0, 0), (Lw, 0), ze, rise, tri=False)       # coping + apex pinnacle (local frame)
    # gothic panelling (blind arcades) on the wall flanks either side of the great window
    for (xa, xb) in ((1.15, cx - 6.05), (cx + 6.05, Lw - 1.15)):
        for zb_ in (Z0 + 6.62, Z0 + 13.42, ze - 2.0):
            G.blind_arcade(mb, xa, xb, zb_, 1.15, pitch=0.6)
    # string courses on the hall front (between the turrets and the window)
    for zc in (Z0 + 6.4, Z0 + 13.2, ze):
        for (xa, xb) in ((0.0, cx - 5.96), (cx + 5.96, Lw)):
            mb.sweep([(0.0, zc - 0.12), (0.07, zc - 0.12), (0.13, zc - 0.05), (0.13, zc + 0.07), (0.0, zc + 0.15)],
                     [(xa, 0.0), (xb, 0.0)], TRIM, closed=False)
    # door spill + facade wash lights
    _light('POINT', _wpos(mb, (cx, -2.5, Z0 + 2.6)), 900.0, WARM, 26.0, radius=1.2, spec=0.0)
    _light('SPOT', _wpos(mb, (1.6, -1.2, Z0 + 0.3)), 1800.0, WARM, 40.0, _wpos(mb, (2.6, 0.5, Z0 + 20)), 1.3)
    _light('SPOT', _wpos(mb, (Lw - 1.6, -1.2, Z0 + 0.3)), 1800.0, WARM, 40.0, _wpos(mb, (Lw - 2.6, 0.5, Z0 + 20)), 1.3)
    mb.pop()
    mb.prism(rect_pts(x0, x1, y0, y1), BASE, Z0, DARK, cap_top=False)
    G.plinth(mb, [(x0, y0), (x0 + 5.0, y0)], Z0, closed=False)
    G.plinth(mb, [(x1 - 5.0, y0), (x1, y0)], Z0, closed=False)
    # ---- side walls with clerestory lancets + buttresses
    clere = W('arch', w=1.7, spring=4.4, sharp=1.1, depth=0.4, nx=2, nz=6, tracery=True)
    for (p0, p1) in (((x0, y1), (x0, y0)), ((x1, y0), (x1, y1))):
        floors = [(17.6, None, 0.0), (8.0, clere, 1.0)]
        _, nb, bw = G.facade(mb, p0, p1, Z0, floors, bay=4.6, margin=0.8, rng=rng)
        mb.push(frame_xf((p0[0], p0[1], 0.0), (p1[0] - p0[0], p1[1] - p0[1])))
        for i in range(nb + 1):
            G.buttress(mb, 0.8 + i * bw, Z0, ze, w=0.9, d1=1.3, d2=0.75, split=0.72)
        Ls = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        G.cornice(mb, [(0.0, 0.0), (Ls, 0.0)], ze, closed=False)
        G.string_course(mb, [(0.0, 0.0), (Ls, 0.0)], Z0 + 17.6, closed=False)
        mb.pop()
        G.corbel_table(mb, p0, p1, ze - 0.62)
    # ---- roof
    G.gable_roof(mbr, x0, x1, y0 + 0.2, y1, ze, rise, axis='y', overhang=0.35)
    for yd in (27.5, 33.5, 39.5):
        G.dormer(mbr, x0 + 3.2, yd, ze + rise * 3.2 / 11.0 - 0.35, -math.pi / 2, rng=rng)
        G.dormer(mbr, x1 - 3.2, yd, ze + rise * 3.2 / 11.0 - 0.35, math.pi / 2, rng=rng)
    # ---- interior
    hall_interior(mb, x0 + 0.6, x1 - 0.6, y0 + 1.3, y1 - 1.5, Z0, ze - 0.8)
    # ---- flanking octagonal stair turrets
    for nm in ("hall_turret_w", "hall_turret_e"):
        tx, ty, tr, te, th = L.TOWERS[nm]
        G.round_tower(mb, tx, ty, BASE, tr, te, th, segs=8, flare=0.0, flare_h=0.0, storey=4.6, win=G.SLIT,
                      win_step=2, a0=math.pi / 8, rng=rng, sill=1.5)
        # blind arcade + pinnacles at the turret head
        for k in range(8):
            a = math.pi / 8 + TAU * k / 8
            G.pinnacle(mb, tx + (tr + 0.25) * math.cos(a), ty + (tr + 0.25) * math.sin(a), te - 0.3, s=0.42, shaft=1.3, spire=2.2)


# =====================================================================================
# wings, keep, towers
# =====================================================================================
def build_west_wing(mb, mbr):
    x0, x1, y0, y1 = -46.0, -9.0, 27.0, 46.0
    heights = [6.2, 5.6, 4.0]
    arc = W('arch', w=3.3, spring=2.7, sharp=0.85, depth=0.95, open=True, hood=True, n=10)
    f1 = W('arch', w=1.9, spring=2.45, sharp=1.0, depth=0.4, nx=2, nz=4, tracery=True)
    f2 = W('rect', w=1.25, spring=1.7, depth=0.32, nx=2, nz=2)
    g0 = W('arch', w=1.9, spring=2.6, sharp=1.0, depth=0.4, nx=2, nz=4)
    specs = {'S': [(arc, 0.0), (f1, 0.95), (f2, 1.1)],
             'W': [(g0, 1.0), (f1, 0.95), (f2, 1.1)],
             'N': [(g0, 1.0), (f1, 0.95), (f2, 1.1)],
             'E': False}
    zt = rect_block(mb, x0, x1, y0, y1, Z0, heights, specs, bay=4.6, parapet='crenel', butt='', uplight='W',
                    balconies={'S': (1, [1, 5], 3.0)}, quoin='SW NW', statues='SW', corner_pinn='SW NW')
    # loggia behind the arcade
    zf, zc = Z0, Z0 + 6.0
    yb = y0 + 4.4
    mb.quad((x0 + 0.9, y0 + 0.95, zf + 0.02), (x1 - 0.5, y0 + 0.95, zf + 0.02), (x1 - 0.5, yb, zf + 0.02), (x0 + 0.9, yb, zf + 0.02), "marble_wet")
    mb.quad((x0 + 0.9, yb, zf), (x1 - 0.5, yb, zf), (x1 - 0.5, yb, zc), (x0 + 0.9, yb, zc), STONE)
    mb.quad((x0 + 0.9, y0 + 0.95, zc), (x0 + 0.9, yb, zc), (x1 - 0.5, yb, zc), (x1 - 0.5, y0 + 0.95, zc), "wood_dark")
    nb = 8; bw = (x1 - x0 - 1.6) / nb
    for i in range(nb):
        xc = x0 + 0.8 + (i + 0.5) * bw
        # glazed doors on the back wall of the loggia
        mb.rnd = rng.uniform(0.3, 1.0)
        mb.quad((xc - 1.0, yb - 0.02, zf), (xc + 1.0, yb - 0.02, zf), (xc + 1.0, yb - 0.02, zf + 4.2), (xc - 1.0, yb - 0.02, zf + 4.2), GLASS)
        mb.rnd = 0.0
        mb.box(xc - 1.12, xc - 1.0, yb - 0.12, yb, zf, zf + 4.34, METAL); mb.box(xc + 1.0, xc + 1.12, yb - 0.12, yb, zf, zf + 4.34, METAL)
        mb.box(xc - 1.12, xc + 1.12, yb - 0.12, yb, zf + 4.2, zf + 4.34, METAL); mb.box(xc - 0.03, xc + 0.03, yb - 0.1, yb, zf, zf + 4.2, METAL)
        # hanging lantern in every arch
        mb.box(xc - 0.012, xc + 0.012, y0 + 2.2 - 0.012, y0 + 2.2 + 0.012, zc - 1.3, zc, METAL)
        mb.box(xc - 0.17, xc + 0.17, y0 + 2.03, y0 + 2.37, zc - 1.85, zc - 1.3, "lamp_glass")
        mb.frustum(rect_pts(xc - 0.2, xc + 0.2, y0 + 2.0, y0 + 2.4), zc - 1.3, rect_pts(xc - 0.03, xc + 0.03, y0 + 2.17, y0 + 2.23), zc - 1.1, METAL)
        if i % 2 == 0:
            _light('POINT', (xc + bw * 0.5, y0 + 2.2, zc - 1.6), 420.0, WARM, 16.0, radius=0.3)
    # roof
    zr = zt + 0.25
    G.gable_roof(mbr, x0 + 1.1, x1, y0 + 1.1, y1 - 1.1, zr, 8.2, axis='x', overhang=0.25)
    G.gable_wall(mbr, (x0 + 1.1, y1 - 1.1), (x0 + 1.1, y0 + 1.1), zr, 8.2, rng=rng)
    for i in range(1, nb, 2):
        xc = x0 + 0.8 + (i + 0.5) * bw
        G.dormer(mbr, xc, y0 + 1.1 + 2.6, zr + 8.2 * 2.6 / 8.4 - 0.3, 0.0, rng=rng)
    for xc in (-40.0, -27.0, -15.0):
        G.chimney(mbr, xc, 36.5 + 1.8, zr + 5.5, 5.2)
    # uplights between the arches
    for i in range(0, nb + 1, 1):
        xl = x0 + 0.8 + i * bw
        _light('SPOT', (xl, y0 - 0.8, Z0 + 0.3), 480.0, WARM, 24.0, (xl, y0 + 0.5, Z0 + 14.0), 1.4)


def build_east_wing(mb, mbr):
    x0, x1, y0, y1 = 31.0, 62.0, 27.0, 45.0
    heights = [5.8, 5.4, 5.0, 4.0]
    g0 = W('arch', w=2.0, spring=2.7, sharp=1.0, depth=0.42, nx=2, nz=5, tracery=True)
    f1 = W('arch', w=1.9, spring=2.4, sharp=1.0, depth=0.4, nx=2, nz=4, tracery=True)
    f2 = W('arch', w=1.5, spring=2.2, sharp=1.0, depth=0.36, nx=2, nz=4)
    f3 = W('rect', w=1.15, spring=1.55, depth=0.3, nx=2, nz=2)
    fl = [(g0, 0.9), (f1, 0.9), (f2, 0.9), (f3, 1.2)]
    specs = {'S': fl, 'E': fl, 'N': fl, 'W': None}
    zt = rect_block(mb, x0, x1, y0, y1, Z0, heights, specs, bay=4.6, parapet='low', butt='S', uplight='E', uplight_e=560.0,
                    balconies={'S': (1, [1, 4], 3.0), 'E': (1, [1, 2], 2.8)}, statues='SE', corner_pinn='SW SE NE NW')
    zr = zt + 0.3
    G.gable_roof(mbr, x0 + 0.9, x1, y0 + 0.9, y1 - 0.9, zr, 9.4, axis='x', overhang=0.2)
    G.gable_wall(mb, (x1, y0), (x1, y1), zt, 9.9, rng=rng)
    for xc in (36.0, 45.6, 55.2):
        G.dormer(mbr, xc, y0 + 0.9 + 2.5, zr + 9.4 * 2.5 / 8.1 - 0.3, 0.0, rng=rng)
    for xc in (40.5, 51.0):
        G.chimney(mbr, xc, 36.0 + 1.6, zr + 6.6, 5.4)


def build_east_tower(mb, mbr):
    x0, x1, y0, y1 = 17.0, 31.0, 15.0, 31.0
    z0 = 0.0
    heights = [6.0, 5.6, 5.6, 5.6, 5.6, 5.6]
    g0 = W('arch', w=2.2, spring=2.6, sharp=1.0, depth=0.5, nx=2, nz=5, tracery=True)
    f1 = W('arch', w=1.6, spring=2.4, sharp=1.05, depth=0.4, nx=2, nz=4)
    f2 = W('arch', w=1.8, spring=2.5, sharp=1.0, depth=0.4, nx=2, nz=4, tracery=True)
    f3 = W('rect', w=1.1, spring=1.6, depth=0.3, nx=2, nz=2)
    fl = [(g0, 0.9), (f1, 0.95), (f1, 0.95), (f1, 0.95), (f2, 0.95), (f3, 1.3)]

    def mid_blank(spec):
        return lambda i, n: None if i == n // 2 else spec
    fl_s = [(g0, 0.9), (f1, 0.95), (f1, 0.95), (mid_blank(f1), 0.95), (mid_blank(f2), 0.95), (mid_blank(f3), 1.3)]
    specs = {'S': fl_s, 'E': fl, 'N': None, 'W': fl}
    zt = rect_block(mb, x0, x1, y0, y1, z0, heights, specs, bay=4.4, parapet='crenel', uplight='SW', uplight_e=620.0,
                    balconies={'S': (1, [1], 3.2), 'W': (1, [1], 3.2)}, statues='SWE')
    # corner bartizans
    for (bx, by) in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        G.bartizan(mb, bx, by, zt - 4.4, r=1.55, h=6.6, roof_h=5.6, rng=rng)
    # steep hipped roof with cresting, dormers and a chimney
    zr = zt + 0.3
    hip_roof(mbr, x0 + 1.2, x1 - 1.2, y0 + 1.2, y1 - 1.2, zr, x0 + 5.6, x1 - 5.6, 22.3, 23.7, zr + 12.8)
    G.dormer(mbr, 24.0, y0 + 1.2 + 1.9, zr + 12.8 * 1.9 / 5.9 - 0.25, 0.0, rng=rng)
    G.dormer(mbr, x0 + 1.2 + 1.6, 23.0, zr + 12.8 * 1.6 / 4.4 - 0.25, -math.pi / 2, rng=rng)
    G.dormer(mbr, x1 - 1.2 - 1.6, 23.0, zr + 12.8 * 1.6 / 4.4 - 0.25, math.pi / 2, rng=rng)
    G.finial(mbr, x0 + 5.8, 23.0, zr + 12.9, s=1.3); G.finial(mbr, x1 - 5.8, 23.0, zr + 12.9, s=1.3)
    # door on the south face flanked by sconces
    mb.push(frame_xf((x0, y0, 0.0), (1.0, 0.0)))
    for xs in (2.75, 11.25):
        sconce(mb, xs, z0 + 3.0, energy=140.0)
    mb.pop()
    return zt


def build_keep(mb, mbr):
    x0, x1, y0, y1 = -6.0, 32.0, 46.0, 72.0
    hf = 5.16
    heights = [hf] * 10
    f1 = W('arch', w=1.7, spring=2.3, sharp=1.0, depth=0.4, nx=2, nz=4, tracery=True)
    f0 = W('arch', w=1.5, spring=2.3, sharp=1.0, depth=0.4, nx=2, nz=4, tracery=True)
    ft = W('arch', w=1.25, spring=2.5, sharp=1.2, depth=0.36, nx=2, nz=4, lit=0.92)
    up = [(f0, 0.9), (f1, 0.9), (f1, 0.9), (ft, 0.8)]
    fl_s = [(None, 0.0)] * 6 + up
    fl_o = [(None, 0.0)] + [(f0, 0.9)] * 5 + up
    specs = {'S': fl_s, 'E': fl_o, 'N': fl_o, 'W': fl_o}
    zt = rect_block(mb, x0, x1, y0, y1, Z0, heights, specs, bay=4.75, parapet='crenel', strings=True)
    # corner towers
    for nm, st in (("keep_sw", 'cone'), ("keep_se", 'cone')):
        tx, ty, tr, te, th = L.TOWERS[nm]
        G.round_tower(mb, tx, ty, Z0, tr, te + 4.0, th, segs=20, flare=0.0, flare_h=0.0, storey=4.8, win=G.SLIT,
                      win_step=3, top=st, rng=rng, win_from=6)
    for (bx, by) in ((x0, y1), (x1, y1)):
        G.bartizan(mb, bx, by, zt - 5.0, r=1.8, h=8.4, roof_h=7.0, rng=rng)
    tx, ty, tr, te, th = L.TOWERS["keep_n"]
    tip_n = G.round_tower(mb, tx, ty, BASE, tr, te + 2.0, th, segs=24, flare=0.08, flare_h=6.0, storey=4.9, win=G.SLIT,
                          win_step=3, top='crenel', rng=rng, win_from=5)
    # big steep hipped roof + glazed lantern fleche (tallest point of the castle)
    zr = zt + 0.3
    hip_roof(mbr, x0 + 1.3, x1 - 1.3, y0 + 1.3, y1 - 1.3, zr, 7.0, 19.0, 57.6, 60.4, zr + 12.4)
    for xc in (2.0, 13.0, 24.0):
        G.dormer(mbr, xc, y0 + 1.3 + 2.4, zr + 12.4 * 2.4 / 10.3 - 0.3, 0.0, w=2.0, h=2.1, rng=rng)
    for yc in (53.0, 65.0):
        G.dormer(mbr, x0 + 1.3 + 2.4, yc, zr + 12.4 * 2.4 / 12.3 - 0.3, -math.pi / 2, w=2.0, h=2.1, rng=rng)
        G.dormer(mbr, x1 - 1.3 - 2.4, yc, zr + 12.4 * 2.4 / 12.3 - 0.3, math.pi / 2, w=2.0, h=2.1, rng=rng)
    lant = W('arch', w=0.95, spring=2.9, sharp=1.2, depth=0.22, nx=2, nz=4, hood=False, sill=False, lit=0.97)
    tip_f = G.round_tower(mbr, 13.0, 59.0, zr + 11.2, 2.1, zr + 17.4, 14.5, segs=8, flare=0.0, flare_h=0.0, storey=6.2,
                          win=lant, win_step=1, a0=math.pi / 8, rng=rng, sill=1.2, wall=TRIM)
    for k in range(8):
        a = math.pi / 8 + TAU * k / 8
        G.pinnacle(mbr, 13.0 + 2.35 * math.cos(a), 59.0 + 2.35 * math.sin(a), zr + 17.0, s=0.4, shaft=1.2, spire=2.4)
    G.chimney(mbr, 0.5, 59.0, zr + 7.0, 7.5); G.chimney(mbr, 25.5, 59.0, zr + 7.0, 7.5)
    return tip_n, tip_f


def build_towers(mb):
    tips = {}
    tx, ty, tr, te, th = L.TOWERS["west_round"]
    tips["west_round"] = G.round_tower(mb, tx, ty, BASE, tr, te, th, segs=24, storey=4.6, win=G.SLIT, win_step=3,
                                       top='cone', rng=rng, flare_h=7.5, flare=0.12)
    tx, ty, tr, te, th = L.TOWERS["west_slim"]
    tips["west_slim"] = G.round_tower(mb, tx, ty, Z0, tr, te, th, segs=20, storey=4.7, win=G.SLIT, win_step=3,
                                      top='cone', rng=rng, flare=0.0, flare_h=0.0, win_from=4)
    tx, ty, tr, te, th = L.TOWERS["ne_tower"]
    tips["ne_tower"] = G.round_tower(mb, tx, ty, BASE, tr, te, th, segs=24, storey=4.6, win=G.SLIT, win_step=3,
                                     top='crenel', rng=rng, flare_h=7.0, flare=0.1)
    # bastion rotunda on the cliff edge + gallery to the west wing
    tx, ty, tr, te, th = L.TOWERS["bastion"]
    bw_ = W('arch', w=1.45, spring=2.7, sharp=1.0, depth=0.5, nx=2, nz=4, tracery=False)
    tips["bastion"] = G.round_tower(mb, tx, ty, L.MASONRY_BOTTOM_Z - 2.0, tr, te + 1.0, 8.5, segs=24, storey=6.3, win=bw_,
                                    win_step=2, top='crenel', rng=rng, flare_h=Z0 - (L.MASONRY_BOTTOM_Z - 2.0), flare=0.16, sill=1.2)
    gal = W('arch', w=2.3, spring=2.6, sharp=0.95, depth=0.45, nx=2, nz=4)
    gup = W('arch', w=1.5, spring=2.2, sharp=1.0, depth=0.36, nx=2, nz=4)
    rect_block(mb, -52.0, -45.5, 31.5, 40.5, Z0, [6.2, 5.0], {'S': [(gal, 0.6), (gup, 0.9)], 'N': [(gal, 0.6), (gup, 0.9)], 'E': False, 'W': False},
               bay=3.2, margin=0.5, parapet='crenel', parapet_sides='SN')
    return tips


# =====================================================================================
# banner + flags
# =====================================================================================
@mats.material("banner_cloth")
def _m_banner(name):
    m, nt, out = new_material(name)
    uv = nt.n('ShaderNodeTexCoord').outputs['UV']
    u, v, _ = nt.sep_xyz(uv)
    # gold border: |u-0.5| > 0.44  (u,v normalised on the banner)
    edge = nt.math('GREATER_THAN', nt.math('ABSOLUTE', nt.math('SUBTRACT', u, 0.5)), 0.435)
    edge2 = nt.math('LESS_THAN', nt.math('ABSOLUTE', nt.math('SUBTRACT', nt.math('ABSOLUTE', nt.math('SUBTRACT', u, 0.5)), 0.39)), 0.008)
    gold = nt.math('MAXIMUM', edge, edge2)
    geo = nt.n('ShaderNodeNewGeometry')
    weave = nt.n('ShaderNodeTexNoise', Vector=geo.outputs['Position'], Scale=70.0, Detail=2.0)
    col = nt.mix(gold, (0.010, 0.010, 0.014, 1), (0.80, 0.50, 0.16, 1))
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=nt.mixf(gold, 0.72, 0.38), Metallic=nt.mixf(gold, 0.0, 0.85),
                Normal=nt.bump(weave.outputs['Factor'], 0.12, 0.002))
    bsdf.inputs['Sheen Weight'].default_value = 0.35
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


def _grid_mesh(name, w, h, nx, ny, shape_fn=None):
    """Grid in the local XY plane (x across, y up from -h to 0), UVs normalised."""
    xs = np.linspace(0, w, nx + 1); ys = np.linspace(-h, 0, ny + 1)
    X, Y = np.meshgrid(xs, ys)
    if shape_fn is not None:
        Y = shape_fn(X, Y)
    verts = np.stack([X, Y, np.zeros_like(X)], axis=-1).reshape(-1, 3)
    idx = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], axis=-1).reshape(-1, 4)
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], quads.tolist())
    uvl = me.uv_layers.new(name="UVMap")
    lv = np.empty(len(me.loops), dtype=np.int32); me.loops.foreach_get("vertex_index", lv)
    U = (np.stack([X / w, (Y - Y.min()) / (Y.max() - Y.min() + 1e-9)], axis=-1).reshape(-1, 2))[lv]
    uvl.data.foreach_set("uv", U.astype(np.float32).ravel())
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    return me, verts


def _emblem_mesh(height):
    """Heraldic fleur-de-lis from a system symbol font; falls back to a simple cross if unavailable."""
    try:
        font = bpy.data.fonts.load(r"C:\Windows\Fonts\seguisym.ttf")
        cu = bpy.data.curves.new("emblem_txt", 'FONT')
        cu.body = "\u269c"; cu.font = font; cu.align_x = 'CENTER'; cu.align_y = 'CENTER'
        cu.size = height * 1.25; cu.extrude = 0.012
        ob = bpy.data.objects.new("emblem_txt", cu)
        bpy.context.scene.collection.objects.link(ob)
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
        bpy.data.objects.remove(ob)
        if len(me.vertices) > 30:
            return me
    except Exception as e:
        print("[castle] emblem font fallback:", e)
    mb = MB("emblem")
    mb.box(-0.12 * height, 0.12 * height, -0.01, 0.01, -0.5 * height, 0.5 * height, GOLD)
    mb.box(-0.36 * height, 0.36 * height, -0.01, 0.01, 0.05 * height, 0.25 * height, GOLD)
    o = mb.to_object("emblem_tmp", bpy.context.scene.collection, recalc=False)
    me = o.data; bpy.data.objects.remove(o)
    return me


def _wave(obj, height, width, speed, vg=None, narrow=1.2, axis='x', offset=0.0):
    md = obj.modifiers.new("wave", 'WAVE')
    md.use_x = (axis == 'x'); md.use_y = (axis == 'y'); md.use_cyclic = True
    md.height = height; md.width = width; md.narrowness = narrow; md.speed = speed
    md.time_offset = -200.0 + offset; md.lifetime = 0.0; md.damping_time = 0.0
    if vg:
        md.vertex_group = vg
    return md


def make_banner(col, x_center, y_wall, z_top, w=3.6, h=12.5, name="banner"):
    """Hanging swallow-tail banner on a south-facing wall (hangs in the XZ plane, front = -Y)."""
    def tail(X, Y):
        cut = 1.3 * (1.0 - np.abs(X / w - 0.5) * 2.0)            # V notch at the bottom
        k = (Y - Y.max()) / (Y.min() - Y.max())                   # 0 top .. 1 bottom
        return Y + cut * k
    me, verts = _grid_mesh(name, w, h, 14, 44, tail)
    me.materials.append(mats.get("banner_cloth"))
    ob = bpy.data.objects.new(name, me); col.objects.link(ob)
    ob.location = (x_center - w / 2, y_wall - 0.45, z_top)
    ob.rotation_euler = (math.pi / 2, 0, 0)
    vg = ob.vertex_groups.new(name="sway")
    for i, v in enumerate(verts):
        vg.add([i], float(min(1.0, (-v[1] / h) ** 1.3)), 'REPLACE')
    _wave(ob, 0.10, 2.6, 0.045, "sway", narrow=0.9, axis='y')
    _wave(ob, 0.035, 0.9, 0.06, "sway", narrow=1.6, axis='x', offset=37.0)
    # emblem: baked into the banner's local frame + same modifiers -> moves exactly with the cloth
    em = _emblem_mesh(w * 0.62)
    em.materials.clear(); em.materials.append(mats.get(GOLD))
    ne = len(em.vertices)
    co = np.empty(ne * 3, dtype=np.float64); em.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    co[:, 0] += w / 2; co[:, 1] += -h * 0.36; co[:, 2] += 0.03
    em.vertices.foreach_set("co", co.ravel()); em.update()
    eo = bpy.data.objects.new(name + "_emblem", em); col.objects.link(eo)
    eo.location = ob.location
    eo.rotation_euler = (math.pi / 2, 0, 0)
    evg = eo.vertex_groups.new(name="sway")
    wts = np.clip((-co[:, 1] / h), 0.0, 1.0) ** 1.3
    for i in range(ne):
        evg.add([i], float(wts[i]), 'REPLACE')
    _wave(eo, 0.10, 2.6, 0.045, "sway", narrow=0.9, axis='y')
    _wave(eo, 0.035, 0.9, 0.06, "sway", narrow=1.6, axis='x', offset=37.0)
    # hanging rod + brackets
    mb = MB(name + "_rod")
    mb.push(xf((x_center, y_wall, z_top)))
    mb.box(-w / 2 - 0.35, w / 2 + 0.35, -0.50, -0.40, 0.02, 0.12, METAL)
    for sx in (-w / 2 - 0.2, w / 2 + 0.2):
        mb.box(sx - 0.04, sx + 0.04, -0.5, 0.0, 0.03, 0.11, METAL)
        mb.lathe([(0.0, 0.0), (0.09, 0.04), (0.12, 0.12), (0.07, 0.2), (0.0, 0.24)], 8, GOLD, (sx + (0.22 if sx > 0 else -0.22), -0.45))
    mb.pop()
    mb.to_object(name + "_rod", col, recalc=False)
    return ob


def make_flag(col, tip, w=3.4, h=2.0, pole=4.2, name="flag", heading=0.0):
    """Flag pole on a spire tip (x, y, z) with a streaming animated flag."""
    x, y, z = tip
    mb = MB(name + "_pole")
    mb.lathe([(0.05, z - 0.6), (0.035, z + pole)], 8, METAL, (x, y))
    mb.lathe([(0.0, z + pole), (0.09, z + pole + 0.08), (0.11, z + pole + 0.18), (0.06, z + pole + 0.28), (0.0, z + pole + 0.32)], 8, GOLD, (x, y))
    mb.to_object(name + "_pole", col, recalc=False)
    me, verts = _grid_mesh(name, w, h, 26, 12)
    me.materials.append(mats.get("banner_cloth"))
    ob = bpy.data.objects.new(name, me); col.objects.link(ob)
    ob.location = (x + 0.05 * math.cos(heading), y + 0.05 * math.sin(heading), z + pole - 0.05)
    ob.rotation_euler = (math.pi / 2, 0, heading)
    vg = ob.vertex_groups.new(name="sway")
    for i, v in enumerate(verts):
        vg.add([i], float(min(1.0, (v[0] / w) ** 0.7)), 'REPLACE')
    _wave(ob, 0.22, 1.5, 0.11, "sway", narrow=1.5, axis='x')
    _wave(ob, 0.10, 0.8, 0.16, "sway", narrow=2.2, axis='x', offset=53.0)
    _wave(ob, 0.08, 2.4, 0.05, "sway", narrow=0.8, axis='y', offset=11.0)
    return ob


# =====================================================================================
# build
# =====================================================================================
FLOOD = 1.0     # global multiplier for the facade floodlights (tuned at integration)
SHADOW_FLOODS = True   # the few hero floodlights below cast (soft) shadows; all other practicals never do


def floodlights():
    """Architectural floodlighting: warm spots on the terraces and roofs washing the facades and towers."""
    def fl(pos, tgt, energy, spot=1.15, cutoff=60.0, color=WARM, shadow=False):
        _light('SPOT', pos, energy * FLOOD, color, cutoff, tgt, spot, 0.6, 0.0, shadow)
    # great hall front (from the terrace, either side of the stair)
    fl((-9.5, 8.0, 0.5), (-2.0, 22.0, 24.0), 26000.0, shadow=True)
    fl((13.5, 8.0, 0.5), (6.0, 22.0, 24.0), 26000.0, shadow=True)
    fl((2.0, 4.0, 0.5), (2.0, 22.0, 36.0), 22000.0, spot=0.8, cutoff=70.0)
    # west wing + west round tower
    for x in (-42.0, -36.5, -22.0, -16.0):
        fl((x, 21.2, Z0 + 0.15), (x, 27.0, 15.0), 5200.0, spot=1.5, cutoff=30.0)
    fl((-30.0, 17.0, 0.5), (-30.0, 25.0, 34.0), 26000.0, spot=0.75, cutoff=70.0, shadow=True)
    fl((-24.0, 21.5, 3.0), (-20.5, 38.0, 46.0), 18000.0, spot=0.6, cutoff=70.0)
    # east tower (south + west faces) and east wing
    fl((20.0, 3.0, 0.5), (22.0, 15.0, 24.0), 22000.0, spot=1.0, shadow=True)
    fl((28.0, 3.0, 0.5), (26.0, 15.0, 24.0), 22000.0, spot=1.0, shadow=True)
    fl((15.2, 12.5, 0.5), (17.0, 22.0, 26.0), 9000.0, spot=1.0, cutoff=45.0)
    for x in (34.2, 39.1, 44.0, 48.9, 53.8, 58.7):
        fl((x, 21.2, Z0 + 0.15), (x, 27.0, 17.0), 4600.0, spot=1.5, cutoff=32.0)
    fl((67.0, 20.0, 0.3), (62.0, 36.0, 18.0), 14000.0, spot=1.2, cutoff=45.0)
    # upper works: keep walls, spires, lantern (lit from the lower roofs)
    fl((2.0, 34.0, 43.5), (6.0, 46.0, 54.0), 14000.0, spot=1.3, cutoff=40.0)
    fl((2.0, 34.0, 43.5), (22.0, 46.0, 54.0), 14000.0, spot=1.3, cutoff=45.0)
    fl((-12.0, 40.0, 22.0), (-6.0, 56.0, 46.0), 16000.0, spot=1.1, cutoff=50.0)
    fl((38.0, 40.0, 26.0), (32.0, 56.0, 46.0), 16000.0, spot=1.1, cutoff=50.0)
    fl((6.0, 54.0, 56.5), (13.0, 59.0, 78.0), 9000.0, spot=0.7, cutoff=40.0)
    fl((20.0, 54.0, 56.5), (13.0, 59.0, 78.0), 9000.0, spot=0.7, cutoff=40.0)
    fl((13.0, 62.0, 56.5), (13.0, 72.0, 74.0), 9000.0, spot=0.8, cutoff=40.0)
    fl((24.0, 20.0, 36.0), (24.0, 23.0, 50.0), 5000.0, spot=1.2, cutoff=25.0)
    # bastion + gallery (from the lower west terrace) and the keep towers' west sides
    fl((-52.0, 22.0, -1.0), (-56.0, 31.0, 12.0), 12000.0, spot=1.2, cutoff=40.0)
    fl((-46.5, 24.0, -1.0), (-46.0, 34.0, 12.0), 8000.0, spot=1.2, cutoff=35.0)
    fl((52.0, 38.0, 24.0), (52.0, 47.0, 40.0), 9000.0, spot=1.0, cutoff=40.0)


def build(ctx):
    col = get_col("castle")
    LIGHTS.clear()
    floodlights()
    mb = MB("castle_walls")
    mbr = MB("castle_roofs")
    build_hall(mb, mbr)
    build_west_wing(mb, mbr)
    build_east_wing(mb, mbr)
    zt_et = build_east_tower(mb, mbr)
    tip_n, tip_f = build_keep(mb, mbr)
    tips = build_towers(mb)
    walls = mb.to_object("castle_walls", col, recalc=False, weld=True)
    roofs = mbr.to_object("castle_roofs", col, recalc=False, weld=True)
    print("[castle] walls faces=%d  roofs faces=%d" % (len(walls.data.polygons), len(roofs.data.polygons)))

    # banner on the east tower's south face, flags on the tallest spires
    make_banner(col, 24.0, 15.0, 30.6)
    make_banner(col, 13.0, 46.0, 52.6, w=3.2, h=10.0, name="banner_keep")
    make_banner(col, -27.5, 27.0, 16.8, w=2.4, h=7.0, name="banner_west")
    make_flag(col, tips["west_round"], name="flag_west", heading=0.25)
    make_flag(col, tip_f, w=4.0, h=2.3, pole=5.0, name="flag_keep", heading=0.25)
    make_flag(col, tips["bastion"], w=2.8, h=1.7, pole=3.6, name="flag_bastion", heading=0.25)

    # lights
    lc = get_col("castle_lights", col)
    for i, (kind, pos, energy, color, cutoff, target, spot, radius, spec, shadow) in enumerate(LIGHTS):
        ld = bpy.data.lights.new("castle_l%03d" % i, kind)
        ld.energy = energy * L.PRACTICAL_GAIN
        ld.color = color
        ld.use_shadow = bool(shadow and SHADOW_FLOODS)
        ld.use_custom_distance = True; ld.cutoff_distance = cutoff
        ld.shadow_soft_size = radius
        ld.specular_factor = spec
        if ld.use_shadow:
            ld.shadow_soft_size = 0.35
            try:
                ld.use_shadow_jitter = True; ld.shadow_jitter_overblur = 12.0; ld.shadow_filter_radius = 2.0
            except Exception as e:
                print('[castle] shadow props:', e)
        if kind == 'SPOT':
            ld.spot_size = spot; ld.spot_blend = 0.85
        lo = bpy.data.objects.new(ld.name, ld); lc.objects.link(lo)
        lo.location = pos
        if target is not None:
            look_at(lo, target)
    print("[castle] lights=%d" % len(LIGHTS))











