"""Terraces: wet marble main terrace, infinity-pool basin, grand stair, podium, lounge terrace,
lower west terrace with lookout pavilion, balustrades, arcaded retaining walls on the cliff line.

Water surfaces, lanterns, furniture and plants are built by other modules (water / props / vegetation).
"""
import bpy, math, random
from mathutils import Vector
from .. import layout as L
from .. import mats
from ..common import MB, get_col, frame_xf, xf, rect_pts, circle_pts, TAU, new_material, look_at
from .. import gothic as G
from ..gothic import W, STONE, TRIM, DARK, SLATE, LEAD, METAL, GOLD

FLOOR = "marble_wet"
ZB = L.MASONRY_BOTTOM_Z - 1.5       # bottom of the masonry substructure
rng = random.Random(99)
LIGHTS = []
WARM = (1.0, 0.60, 0.30)

# outer face of pool walls
PX0 = L.POOL_X0 - 0.9
PY0 = L.POOL_Y0 - 0.9               # = cliff line y = -50
PY1 = L.POOL_Y1 + 0.9
ZW = L.POOL_WATER_Z
LIP = ZW - 0.015                    # top of the infinity walls (just under the water surface)

# cliff-line path (outside on the RIGHT when walking it): from the NW end round the SW corner to the SE
EDGE = [(-60.0, 24.0), (-62.0, 16.0), (-72.0, 10.0), (-70.0, 0.0), (-58.0, -6.0), (-50.0, -20.0), (-47.5, -50.0),
        (PX0, -50.0), (L.POOL_X1 + 0.6, -50.0), (14.0, -50.0), (16.0, -70.0), (48.0, -70.0)]
EDGE_TOP = [L.LOWER_Z, L.LOWER_Z, L.LOWER_Z, L.LOWER_Z, L.LOWER_Z, L.LOWER_Z, L.LOWER_Z, LIP, 0.0, L.LOUNGE_Z, L.LOUNGE_Z]


@mats.material("channel_water")
def _m_channel(name):
    m, nt, out = new_material(name)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.008, 0.035, 0.04, 1), Roughness=0.03)
    bsdf.inputs['Specular IOR Level'].default_value = 0.8
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


@mats.material("soil_dark")
def _m_soil(name):
    m, nt, out = new_material(name)
    geo = nt.n('ShaderNodeNewGeometry')
    n = nt.n('ShaderNodeTexNoise', Vector=geo.outputs['Position'], Scale=18.0, Detail=4.0)
    col = nt.ramp(n.outputs['Factor'], [(0.3, (0.02, 0.014, 0.009, 1)), (0.7, (0.06, 0.04, 0.025, 1))])
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.95, Normal=nt.bump(n.outputs['Factor'], 0.6, 0.02))
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


def _flat(mb, poly, z, mat, holes=()):
    mb.poly_holes(poly, list(holes), mat, to3d=lambda a, b: (a, b, z), normal=(0, 0, 1))


def _vwall(mb, p0, p1, z0, z1, mat):
    """Vertical quad from p0 to p1 (outside on the right)."""
    mb.quad((p0[0], p0[1], z0), (p1[0], p1[1], z0), (p1[0], p1[1], z1), (p0[0], p0[1], z1), mat)


def _light(kind, pos, energy, color=WARM, cutoff=18.0, target=None, spot=1.4, radius=0.2, spec=1.0):
    LIGHTS.append((kind, tuple(pos), energy, color, cutoff, target, spot, radius, spec))


# =====================================================================================
def floors(mb):
    # main terrace (z = 0)
    main = [(L.POOL_X1, -50.0), (14.0, -50.0), (14.0, -46.0), (44.0, -46.0), (44.0, 20.0), (-46.0, 20.0), (-46.0, PY1), (L.POOL_X1, PY1)]
    _flat(mb, main, 0.0, FLOOR)
    # lounge (z = LOUNGE_Z) with a round tree pit for the big maple
    lz = L.LOUNGE_Z
    mx, my = L.MAPLES[0][0], L.MAPLES[0][1]
    pit = circle_pts(1.25, 20, mx, my)
    _flat(mb, [(14.0, -50.0), (16.0, -70.0), (48.0, -70.0), (48.0, -46.0), (14.0, -46.0)], lz, FLOOR, holes=[pit])
    _flat(mb, pit, lz - 0.04, "soil_dark")
    mb.lathe([(1.25, lz - 0.1), (1.25, lz + 0.14), (1.43, lz + 0.14), (1.47, lz + 0.08), (1.47, lz - 0.1)], 20, TRIM, (mx, my))
    # step faces between main terrace and lounge
    _vwall(mb, (14.0, -46.0), (44.0, -46.0), lz - 0.1, 0.0, TRIM)          # faces south
    _vwall(mb, (14.0, -50.0), (14.0, -46.0), lz - 0.1, 0.0, TRIM)          # faces east
    _vwall(mb, (44.0, -46.0), (44.0, 20.0), -3.5, 0.0, DARK)               # east edge of the main terrace
    _vwall(mb, (48.0, -70.0), (48.0, -46.0), -3.5, lz, DARK)               # east edge of the lounge
    _vwall(mb, (48.0, -46.0), (44.0, -46.0), -3.5, lz, DARK)
    # lower west terrace (z = LOWER_Z)
    zl = L.LOWER_Z
    lower = [(PX0, -50.0), (-47.5, -50.0), (-50.0, -20.0), (-58.0, -6.0), (-70.0, 0.0), (-72.0, 10.0), (-62.0, 16.0),
             (-60.0, 24.0), (-63.0, 31.5), (-46.0, 31.5), (-46.0, PY1), (PX0, PY1)]
    _flat(mb, lower, zl, FLOOR)
    # catch channel along the pool's west wall
    mb.quad((PX0 - 1.0, -50.0, zl + 0.015), (PX0, -50.0, zl + 0.015), (PX0, PY1, zl + 0.015), (PX0 - 1.0, PY1, zl + 0.015), "channel_water")
    mb.box(PX0 - 1.12, PX0 - 1.0, -49.9, PY1, zl - 0.05, zl + 0.07, TRIM)
    # walls between the lower terrace and the upper levels
    _vwall(mb, (-46.0, 20.0), (-46.0, PY1), zl - 0.1, 0.0, STONE)           # north deck, faces west
    _vwall(mb, (-46.0, 31.5), (-46.0, 20.0), zl - 0.1, L.PODIUM_Z, STONE)   # podium / west wing base
    _vwall(mb, (-46.0, PY1), (PX0, PY1), zl - 0.1, 0.0, STONE)
    # podium (z = PODIUM_Z)
    zp = L.PODIUM_Z
    for (x0, x1, y0, y1) in ((-46.0, -9.0, 20.0, 27.3), (-9.0, 13.0, 20.0, 22.3), (13.0, 17.0, 20.0, 46.0),
                             (31.0, 64.0, 20.0, 27.3), (17.0, 31.0, 31.0, 46.0)):
        mb.quad((x0, y0, zp), (x1, y0, zp), (x1, y1, zp), (x0, y1, zp), FLOOR)
    # podium front + end walls
    for (xa, xb) in ((-46.0, L.STAIR_X0 - 0.7), (L.STAIR_X1 + 0.7, 17.0), (31.0, 64.0)):
        _vwall(mb, (xa, 20.0), (xb, 20.0), -0.1, zp, STONE)
        mb.sweep([(0.0, zp - 0.30), (0.06, zp - 0.30), (0.12, zp - 0.22), (0.12, zp - 0.02), (0.0, zp + 0.03)],
                 [(xa, 20.0), (xb, 20.0)], TRIM, closed=False)
        mb.sweep([(0.0, -0.1), (0.14, -0.1), (0.14, 0.42), (0.04, 0.55), (0.0, 0.55)], [(xa, 20.0), (xb, 20.0)], DARK, closed=False)
    _vwall(mb, (64.0, 20.0), (64.0, 27.3), -3.5, zp, STONE)
    _vwall(mb, (44.0, 20.0), (64.0, 20.0), -3.5, -0.1, DARK)


def pool(mb):
    x0, x1, y0, y1 = L.POOL_X0, L.POOL_X1, L.POOL_Y0, L.POOL_Y1
    zf = ZW - L.POOL_DEPTH
    tile = "pool_tile"
    mb.quad((x0, y0, zf), (x1, y0, zf), (x1, y1, zf), (x0, y1, zf), tile)
    mb.quad((x0, y0, zf), (x0, y1, zf), (x0, y1, LIP), (x0, y0, LIP), tile)          # west inner (faces +x)
    mb.quad((x1, y1, zf), (x1, y0, zf), (x1, y0, 0.0), (x1, y1, 0.0), tile)          # east inner
    mb.quad((x1, y0, zf), (x0, y0, zf), (x0, y0, LIP), (x1, y0, LIP), tile)          # south inner
    mb.quad((x0, y1, zf), (x1, y1, zf), (x1, y1, 0.0), (x0, y1, 0.0), tile)          # north inner
    # infinity walls: west + south (top just under the water), notch for the hero spill
    zl = L.LOWER_Z
    mb.quad((PX0, y0 - 0.9, LIP), (x0, y0 - 0.9, LIP), (x0, y1, LIP), (PX0, y1, LIP), TRIM)                     # west top
    _vwall(mb, (PX0, PY1), (PX0, -50.0), zl - 0.1, LIP, TRIM)                                                    # west outer face
    sx0, sx1 = L.SPILL_X0, L.SPILL_X1
    notch = LIP - 0.07
    for (xa, xb, zt) in ((x0, sx0, LIP), (sx0, sx1, notch), (sx1, x1 + 0.6, LIP)):
        mb.quad((xa, PY0, zt), (xb, PY0, zt), (xb, y0, zt), (xa, y0, zt), TRIM)                                  # south top
    for xs in (sx0, sx1):
        mb.quad((xs, PY0, notch), (xs, y0, notch), (xs, y0, LIP), (xs, PY0, LIP), TRIM)
    # east + north coping (slightly proud, bull-nosed look)
    mb.box(x1 - 0.02, x1 + 0.62, y0 - 0.9, y1 + 0.9, -0.12, 0.022, TRIM, skip="-z")
    mb.box(x0 - 0.9, x1 + 0.62, y1 - 0.02, y1 + 0.9, -0.12, 0.022, TRIM, skip="-z")
    # entry steps at the north end
    for k in range(4):
        mb.box(-40.5, -29.0, y1 - 0.45 * (k + 1), y1 - 0.45 * k + 0.001, zf, ZW - 0.36 * (k + 1) + 0.24, tile)
    # corbelled stone gutter under the south overflow (both sides of the notch)
    for (xa, xb) in ((PX0, sx0 - 0.4), (sx1 + 0.4, x1 + 0.6)):
        mb.box(xa, xb, PY0 - 0.95, PY0, -1.80, -1.66, TRIM)
        mb.box(xa, xb, PY0 - 0.95, PY0 - 0.83, -1.66, -1.36, TRIM)
        mb.box(xa, xa + 0.12, PY0 - 0.95, PY0, -1.66, -1.36, TRIM); mb.box(xb - 0.12, xb, PY0 - 0.95, PY0, -1.66, -1.36, TRIM)
        mb.quad((xa + 0.1, PY0 - 0.85, -1.47), (xb - 0.1, PY0 - 0.85, -1.47), (xb - 0.1, PY0 - 0.001, -1.47), (xa + 0.1, PY0 - 0.001, -1.47), "channel_water")
        n = max(2, int((xb - xa) / 1.3))
        for i in range(n + 1):
            xc = xa + 0.2 + (xb - xa - 0.4) * i / n
            mb.box(xc - 0.15, xc + 0.15, PY0 - 0.8, PY0, -2.25, -1.80, TRIM)
            mb.box(xc - 0.15, xc + 0.15, PY0 - 0.45, PY0, -2.65, -2.25, TRIM)
    # soft turquoise spill light on the surroundings of the pool (no specular so no fake highlights)
    for yc in (-42.0, -30.0, -18.0, -6.0, 4.0):
        _light('POINT', (-34.7, yc, 0.9), 260.0, (0.25, 0.95, 1.0), 13.0, radius=2.0, spec=0.0)


def stairs(mb):
    # grand stair: 16 steps from the terrace up to the podium
    n = 16
    y0, y1 = L.STAIR_Y0, L.STAIR_Y1
    d = (y1 - y0) / n; h = L.PODIUM_Z / n
    prof = [(y0, -0.06)]
    for k in range(n):
        prof.append((y0 + k * d, (k + 1) * h)); prof.append((y0 + (k + 1) * d, (k + 1) * h))
    prof.append((y1 + 0.02, -0.06))
    mb.extrude_yz(prof, L.STAIR_X0, L.STAIR_X1, FLOOR)
    # nosings
    for k in range(n):
        mb.box(L.STAIR_X0, L.STAIR_X1, y0 + k * d - 0.035, y0 + k * d + 0.02, (k + 1) * h - 0.045, (k + 1) * h + 0.004, TRIM)
    # cheek walls with moulded coping, end piers and ball finials
    for (xa, xb) in ((L.STAIR_X0 - 0.75, L.STAIR_X0), (L.STAIR_X1, L.STAIR_X1 + 0.75)):
        mb.extrude_yz([(y0 - 0.5, -0.06), (y0 - 0.5, 0.95), (y1 + 0.45, L.PODIUM_Z + 0.95), (y1 + 0.45, -0.06)], xa, xb, TRIM)
        mb.extrude_yz([(y0 - 0.56, 0.93), (y0 - 0.56, 1.05), (y1 + 0.5, L.PODIUM_Z + 1.05), (y1 + 0.5, L.PODIUM_Z + 0.93)], xa - 0.07, xb + 0.07, TRIM)
        xm = (xa + xb) / 2
        for (yy, zz) in ((y0 - 0.95, 0.0), (y1 + 0.9, L.PODIUM_Z)):
            G.pier(mb, xm, yy, zz, h=1.3, s=0.42)
            mb.lathe([(0.0, zz + 1.52), (0.12, zz + 1.56), (0.22, zz + 1.70), (0.24, zz + 1.82), (0.18, zz + 1.96), (0.0, zz + 2.04)], 12, TRIM, (xm, yy))
    # stair from the main terrace down to the lounge
    lz = L.LOUNGE_Z
    n2 = 8; d2 = 0.42; h2 = -lz / n2
    ya = -46.0 - n2 * d2
    prof = [(-46.0, 0.0), (-46.0, lz - 0.06), (ya, lz - 0.06)]
    for k in range(n2):
        prof.append((ya + k * d2, lz + (k + 1) * h2)); prof.append((ya + (k + 1) * d2, lz + (k + 1) * h2))
    mb.extrude_yz(prof[:-1], L.LOUNGE_STAIR_X0, L.LOUNGE_STAIR_X1, FLOOR)
    for xs in (L.LOUNGE_STAIR_X0 - 0.3, L.LOUNGE_STAIR_X1 + 0.3):
        mb.extrude_yz([(-46.0, lz - 0.06), (ya - 0.3, lz - 0.06), (ya - 0.3, lz + 0.5), (-46.0, 0.5)], xs - 0.3, xs + 0.3, TRIM)
        G.pier(mb, xs, ya - 0.75, lz, h=0.9, s=0.36)


def balustrades(mb):
    # main terrace: south edge, the corner above the lounge, both sides of the lounge stair, east edge
    G.balustrade(mb, [(L.POOL_X1 + 0.85, -49.68), (13.7, -49.68), (13.7, -45.7), (L.LOUNGE_STAIR_X0 - 0.75, -45.7)], 0.0, urns=2)
    G.balustrade(mb, [(L.LOUNGE_STAIR_X1 + 0.75, -45.7), (43.7, -45.7), (43.7, 19.7)], 0.0, urns=2)
    # north deck (west edge) above the lower terrace
    G.balustrade(mb, [(-45.7, PY1 + 0.3), (-45.7, 19.7)], 0.0)
    # lounge: cliff edges west + south, then the east side
    lz = L.LOUNGE_Z
    G.balustrade(mb, [(14.42, -50.4), (16.32, -69.68), (47.7, -69.68), (47.7, -46.4)], lz, urns=2)
    # podium front
    zp = L.PODIUM_Z
    G.balustrade(mb, [(-45.7, 20.3), (L.STAIR_X0 - 1.35, 20.3)], zp, ball=True)
    G.balustrade(mb, [(31.3, 20.3), (63.7, 20.3), (63.7, 27.0)], zp, ball=True)
    # lower west terrace: along the cliff edge round the lookout spur
    zl = L.LOWER_Z
    G.balustrade(mb, [(-59.72, 24.0), (-61.72, 16.25), (-71.66, 10.3), (-69.72, 0.3), (-57.78, -5.72), (-49.72, -19.9),
                      (-47.2, -49.68), (PX0 - 0.3, -49.68)], zl)


def retaining_walls(mb):
    """Arcaded masonry substructure on the cliff line."""
    arch = W('arch', w=4.2, spring=6.4, sharp=0.8, depth=1.9, open=True, hood=True, n=10)
    for i in range(len(EDGE) - 1):
        p0, p1 = EDGE[i], EDGE[i + 1]
        top = EDGE_TOP[i]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        Ls = math.hypot(dx, dy)
        H = top - ZB
        if Ls < 9.0:
            _vwall(mb, p0, p1, ZB, top, DARK)
            nb, bw, margin = 0, 0.0, 0.0
        else:
            margin = 1.2
            _, nb, bw = G.facade(mb, p0, p1, ZB, [(H, arch, 2.4)], bay=7.0, margin=margin, wall=DARK, rng=rng)
        mb.push(frame_xf((p0[0], p0[1], 0.0), (dx, dy)))
        if nb:
            mb.quad((0, 1.9, ZB), (Ls, 1.9, ZB), (Ls, 1.9, top - 0.5), (0, 1.9, top - 0.5), DARK)     # back of the recesses
            for k in range(nb + 1):
                G.buttress(mb, margin + k * bw, ZB, top - 0.2, w=1.7, d1=2.0, d2=1.0, split=0.62, pinn=False, mat=DARK)
            for k in range(nb):
                # warm light glowing out of every blind arch + a small visible luminaire on its sill
                xc = margin + (k + 0.5) * bw
                v = mb.M @ Vector((xc, 0.75, ZB + 3.3))
                _light('POINT', (v.x, v.y, v.z), 420.0, (1.0, 0.50, 0.20), 10.0, radius=0.5, spec=0.0)
                mb.box(xc - 0.12, xc + 0.12, 1.3, 1.5, ZB + 2.42, ZB + 2.50, "lamp_glow")
        # moulded band under the top
        mb.sweep([(0.0, top - 0.95), (0.10, top - 0.95), (0.20, top - 0.80), (0.20, top - 0.55), (0.0, top - 0.5)],
                 [(0.0, 0.0), (Ls, 0.0)], TRIM, closed=False)
        mb.pop()
    # cool floodlights on the hero waterfall (mounted on the wall either side of the spill, aimed down the fall)
    # (hung well in front of the wall and aimed almost straight down the fall, so the masonry only gets grazing light)
    xm_ = (L.SPILL_X0 + L.SPILL_X1) / 2
    # two distant floods out in front of the cliff: even light over the whole upper fall, no hot spots
    for dx_ in (-9.0, 9.0):
        _light('SPOT', (xm_ + dx_, PY0 - 34.0, -28.0), 62000.0, (0.80, 0.90, 1.0), 260.0, (xm_, PY0 - 4.0, -75.0), 0.95, 1.0, 0.0)
    # warm wash on the rock below the masonry so the cliff under the castle glows instead of going black
    for i in range(len(EDGE) - 1):
        p0, p1 = EDGE[i], EDGE[i + 1]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        Ls = math.hypot(dx, dy)
        if Ls < 15.0:
            continue
        nx_, ny_ = dy / Ls, -dx / Ls            # outward (right of travel)
        n = max(1, int(Ls / 24.0))
        for k in range(n):
            f = (k + 0.5) / n
            px, py = p0[0] + dx * f + nx_ * 2.6, p0[1] + dy * f + ny_ * 2.6
            _light('SPOT', (px, py, ZB + 1.5), 52000.0, (1.0, 0.55, 0.26), 110.0, (px + nx_ * 9.0, py + ny_ * 9.0, ZB - 70.0), 0.95, 0.5, 0.0)


def lookout(mb):
    """Small round tempietto on the lookout spur + its lights."""
    cx, cy = -65.5, 6.2
    z = L.LOWER_Z
    mb.lathe([(4.2, z - 0.05), (4.2, z + 0.16), (3.85, z + 0.16), (3.85, z + 0.32), (3.5, z + 0.32), (3.5, z + 0.48), (0.0, z + 0.48)], 32, TRIM, (cx, cy))
    zb = z + 0.48
    hc = 3.9
    for k in range(8):
        a = TAU * (k + 0.5) / 8
        px, py = cx + 2.9 * math.cos(a), cy + 2.9 * math.sin(a)
        mb.lathe([(0.30, zb), (0.30, zb + 0.14), (0.22, zb + 0.24), (0.185, zb + 0.5), (0.17, zb + hc - 0.5), (0.2, zb + hc - 0.4),
                  (0.3, zb + hc - 0.12), (0.32, zb + hc)], 14, TRIM, (px, py))
    zt = zb + hc
    mb.lathe([(2.55, zt), (3.3, zt), (3.3, zt + 0.22), (3.42, zt + 0.30), (3.42, zt + 0.62), (3.56, zt + 0.72), (3.56, zt + 0.86), (2.55, zt + 0.86), (2.55, zt)],
             32, TRIM, (cx, cy))
    mb.lathe([(0.0, zt + 0.02), (2.56, zt + 0.02)], 32, "wood_dark", (cx, cy))
    G.cone_roof(mb, cx, cy, zt + 0.86, 3.3, 3.6, segs=32, overhang=0.3)
    # hanging lantern
    mb.box(cx - 0.012, cx + 0.012, cy - 0.012, cy + 0.012, zt - 1.0, zt, METAL)
    mb.box(cx - 0.2, cx + 0.2, cy - 0.2, cy + 0.2, zt - 1.6, zt - 1.0, "lamp_glass")
    _light('POINT', (cx, cy, zt - 1.3), 600.0, WARM, 14.0, radius=0.3)
    # in-ground uplights along the lower terrace wall + path lights
    for (lx, ly) in ((-52.5, -12.0), (-49.0, -30.0), (-48.0, -44.0), (-60.0, 20.0), (-52.0, 26.0)):
        mb.lathe([(0.0, z + 0.03), (0.13, z + 0.03), (0.16, z + 0.0)], 12, "lamp_glow", (lx, ly))
        _light('POINT', (lx, ly, z + 0.5), 160.0, WARM, 9.0, radius=0.2)


def uplight_fixtures(mb):
    """Small in-ground luminaires in front of the castle so facade uplights have a visible source."""
    pts = [(x, 21.0, L.PODIUM_Z) for x in (-7.4, 11.4)]
    pts += [(-45.2 + i * 4.425, 26.2, L.PODIUM_Z) for i in range(9)]
    pts += [(17.8 + i * 4.13, 14.1, 0.0) for i in range(4)]
    pts += [(31.8 + i * 4.9, 25.0, L.PODIUM_Z) for i in range(7)]
    for (x, y, z) in pts:
        mb.lathe([(0.0, z + 0.035), (0.11, z + 0.035), (0.15, z + 0.004), (0.17, z + 0.004)], 12, "lamp_glow", (x, y))


def build(ctx):
    col = get_col("terrace")
    LIGHTS.clear()
    mb = MB("terrace")
    floors(mb)
    pool(mb)
    stairs(mb)
    balustrades(mb)
    retaining_walls(mb)
    lookout(mb)
    uplight_fixtures(mb)
    obj = mb.to_object("terrace", col, recalc=False, weld=False)
    print("[terrace] faces=%d" % len(obj.data.polygons))
    lc = get_col("terrace_lights", col)
    for i, (kind, pos, energy, color, cutoff, target, spot, radius, spec) in enumerate(LIGHTS):
        ld = bpy.data.lights.new("terrace_l%03d" % i, kind)
        ld.energy = energy * L.PRACTICAL_GAIN
        ld.color = color
        ld.use_shadow = False
        ld.use_custom_distance = True; ld.cutoff_distance = cutoff
        ld.shadow_soft_size = radius
        ld.specular_factor = spec
        if kind == 'SPOT':
            ld.spot_size = spot; ld.spot_blend = 0.9
        lo = bpy.data.objects.new(ld.name, ld); lc.objects.link(lo)
        lo.location = pos
        if target is not None:
            look_at(lo, target)






