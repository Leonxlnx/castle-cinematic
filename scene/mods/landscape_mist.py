"""Mist / low clouds and distant conifer cones of the landscape module (materials + builders).

Mist comes in three flavours, all alpha-blended emission sheets that fade out where they come close to
the ground: every mist vertex carries the terrain height below it (point attribute "ground", taken from
the terrain grid = the rendered surface), so no textures are needed and the mist survives a .blend save.

    ls_mist         valley fog banks: a stack of horizontal sheets (z -334 .. -262) that all evaluate the
                    SAME drifting noise field with a threshold that rises with height -> dome shaped banks.
                    The sheets only exist over low ground (cut out of the terrain grid), so they cost little.
    ls_mist_shell   distant mist / low clouds between the ridges: concentric vertical cylinder shells around
                    the castle (radius 1.9 .. 14 km) slicing one 3D noise field; density clings to the ground.
    ls_mist_plume   spray cloud where the hero waterfall dissolves (small discs at the cliff foot).
    ls_trees        distant conifer cones.
"""
import math
import numpy as np
import bpy
from .. import mats, atmo, layout as L
from ..common import new_material
from . import landscape_field as F
from .landscape_mats import _noise, _time


# (z, max alpha, threshold)
VALLEY_LAYERS = [(-333.0, 0.50, 0.46), (-324.0, 0.50, 0.49), (-312.0, 0.50, 0.52), (-297.0, 0.48, 0.555),
                 (-279.0, 0.45, 0.595), (-258.0, 0.40, 0.64)]
VALLEY_R = 7000.0
# (radius, max alpha, threshold)
SHELLS = [(1900.0, 0.32, 0.585), (2700.0, 0.42, 0.57), (3800.0, 0.48, 0.56), (5300.0, 0.50, 0.555),
          (7400.0, 0.50, 0.555), (10300.0, 0.50, 0.56), (14300.0, 0.46, 0.57)]
SHELL_Z0, SHELL_Z1 = -345.0, 620.0
PLUME_LAYERS = [(-262.0, 0.55, 0.30), (-250.0, 0.60, 0.33), (-238.0, 0.60, 0.37), (-226.0, 0.55, 0.42),
                (-214.0, 0.45, 0.48), (-200.0, 0.32, 0.55)]
PLUME_CENTER = (F.WATERFALL_C[0], F.WATERFALL_C[1] - 14.0)
N_TREES = 188000


# =====================================================================================
# height images
# =====================================================================================
def _sample_polar(tg, x, y):
    """Bilinear lookup of the terrain grid (the rendered surface) at world x, y."""
    radii = tg['radii']; theta = tg['theta']; H = tg['H']
    r = np.hypot(x - F.CX, y - F.CY)
    a = np.arctan2(y - F.CY, x - F.CX) % (2.0 * np.pi)
    i = np.clip(np.searchsorted(radii, r) - 1, 0, len(radii) - 2)
    fr = np.clip((r - radii[i]) / (radii[i + 1] - radii[i]), 0.0, 1.0)
    th = np.concatenate([theta, [theta[0] + 2.0 * np.pi]])
    a = np.where(a < theta[0], a + 2.0 * np.pi, a)
    j = np.clip(np.searchsorted(th, a) - 1, 0, len(theta) - 1)
    fa = np.clip((a - th[j]) / (th[j + 1] - th[j]), 0.0, 1.0)
    j1 = (j + 1) % len(theta)
    h0 = H[i, j] * (1 - fa) + H[i, j1] * fa
    h1 = H[i + 1, j] * (1 - fa) + H[i + 1, j1] * fa
    return h0 * (1 - fr) + h1 * fr


# =====================================================================================
# materials
# =====================================================================================
def _mist_colour(nt, thick):
    """Emission colour of mist: cool grey, glowing warm toward the sun, brighter where it is thick."""
    geo = nt.n('ShaderNodeNewGeometry')
    sd = L.sun_dir()
    d = nt.vmath('DOT_PRODUCT', geo.outputs['Incoming'], (-sd[0], -sd[1], -sd[2]))
    sun_amt = nt.math('POWER', nt.map_range(d, -0.3, 1.0, 0.0, 1.0), 2.4)
    c = nt.mix(sun_amt, (0.21, 0.225, 0.29, 1.0), (1.25, 0.70, 0.34, 1.0))
    return nt.mix(1.0, c, nt.ramp(thick, [(0.0, (0.72, 0.72, 0.76, 1)), (1.0, (1.18, 1.15, 1.10, 1))]),
                  blend='MULTIPLY')


def _finish_mist(m, nt, out, colour, alpha, fog_mul=0.8):
    em = nt.n('ShaderNodeEmission', Color=colour, Strength=1.0)
    sh = atmo.fog(nt, em.outputs[0], density_mul=fog_mul)
    tr = nt.n('ShaderNodeBsdfTransparent')
    mix = nt.n('ShaderNodeMixShader', i0=alpha, i1=tr.outputs[0], i2=sh)
    nt.link(mix.outputs[0], out.inputs['Surface'])
    m.surface_render_method = 'BLENDED'
    m.use_backface_culling = False
    for k, v in (("use_transparency_overlap", True), ("use_transparent_shadow", False)):
        try:
            setattr(m, k, v)
        except Exception:
            pass
    return m


def _mist_common(nt):
    geo = nt.n('ShaderNodeNewGeometry')
    pos = geo.outputs['Position']
    z = nt.sep_xyz(pos)[2]
    dist = nt.n('ShaderNodeCameraData').outputs['View Distance']
    oc = nt.n('ShaderNodeSeparateColor', Color=nt.n('ShaderNodeObjectInfo').outputs['Color'])
    clear = nt.math('SUBTRACT', z, nt.attr('ground').outputs['Factor'])
    return geo, pos, z, dist, oc.outputs[0], oc.outputs[1], clear, 1.0


@mats.material("ls_mist")
def _m_mist(name):
    """Valley fog banks.  Object colour: R = max alpha of the layer, G = density threshold."""
    m, nt, out = new_material(name)
    geo, pos, z, dist, o_alpha, o_thr, clear, inside = _mist_common(nt)
    t = _time(nt)
    sx = nt.sep_xyz(pos)
    # same field for every layer (xy only); slow drift + slow evolution along the 3rd noise axis
    p = nt.comb_xyz(nt.math('ADD', sx[0], nt.math('MULTIPLY', t, 0.75)),
                    nt.math('ADD', sx[1], nt.math('MULTIPLY', t, 0.35)), nt.math('MULTIPLY', t, 6.0))
    n1 = _noise(nt, p, 1250.0, 3.0, 0.52)
    n2 = _noise(nt, p, 210.0, 2.0, 0.55)
    d = nt.math('ADD', nt.math('MULTIPLY', n1, 0.74), nt.math('MULTIPLY', n2, 0.26))
    d = nt.math('ADD', d, nt.map_range(clear, 0.0, 110.0, 0.10, -0.07))        # hugs hollows
    dens = nt.map_range(d, o_thr, nt.math('ADD', o_thr, 0.16), 0.0, 1.0, interp='SMOOTHSTEP')
    fade = nt.map_range(clear, 2.0, 30.0, 0.0, 1.0, interp='SMOOTHSTEP')
    a = nt.math('MULTIPLY', nt.math('MULTIPLY', dens, fade), o_alpha)
    a = nt.math('MULTIPLY', a, nt.math('MULTIPLY', inside, nt.map_range(dist, 40.0, 220.0, 0.0, 1.0)))
    return _finish_mist(m, nt, out, _mist_colour(nt, dens), a)


@mats.material("ls_mist_shell")
def _m_mist_shell(name):
    """Distant mist on vertical cylinder shells around the castle."""
    m, nt, out = new_material(name)
    geo, pos, z, dist, o_alpha, o_thr, clear, inside = _mist_common(nt)
    t = _time(nt)
    sx = nt.sep_xyz(pos)
    p = nt.comb_xyz(nt.math('ADD', sx[0], nt.math('MULTIPLY', t, 1.6)),
                    nt.math('ADD', sx[1], nt.math('MULTIPLY', t, 0.7)),
                    nt.math('ADD', nt.math('MULTIPLY', sx[2], 7.0), nt.math('MULTIPLY', t, 5.0)))
    n1 = _noise(nt, p, 1700.0, 3.0, 0.55)
    d = nt.math('ADD', n1, nt.map_range(clear, 0.0, 380.0, 0.10, -0.17))       # clings to the ground
    dens = nt.map_range(d, o_thr, nt.math('ADD', o_thr, 0.22), 0.0, 1.0, interp='SMOOTHSTEP')
    fade = nt.map_range(clear, 3.0, 60.0, 0.0, 1.0, interp='SMOOTHSTEP')
    top = nt.map_range(z, SHELL_Z1 - 160.0, SHELL_Z1, 1.0, 0.0)
    facing = nt.math('ABSOLUTE', nt.vmath('DOT_PRODUCT', geo.outputs['Incoming'], geo.outputs['True Normal']))
    facing = nt.map_range(facing, 0.25, 0.55, 0.0, 1.0)
    a = nt.math('MULTIPLY', nt.math('MULTIPLY', dens, fade), nt.math('MULTIPLY', o_alpha, top))
    a = nt.math('MULTIPLY', a, nt.math('MULTIPLY', nt.math('MULTIPLY', inside, facing),
                                       nt.map_range(dist, 500.0, 1400.0, 0.0, 1.0)))
    return _finish_mist(m, nt, out, _mist_colour(nt, dens), a, fog_mul=0.7)


@mats.material("ls_mist_plume")
def _m_mist_plume(name):
    """Spray / mist cloud where the hero waterfall dissolves."""
    m, nt, out = new_material(name)
    geo, pos, z, dist, o_alpha, o_thr, clear, inside = _mist_common(nt)
    t = _time(nt)
    oi = nt.n('ShaderNodeObjectInfo')
    r = nt.vmath('LENGTH', nt.vmath('SUBTRACT', pos, oi.outputs['Location']))
    radial = nt.map_range(r, 10.0, 105.0, 1.0, 0.0, interp='SMOOTHSTEP')
    p = nt.vmath('ADD', pos, nt.comb_xyz(nt.math('MULTIPLY', t, 1.2), nt.math('MULTIPLY', t, -0.8),
                                         nt.math('MULTIPLY', t, 2.5)))
    n1 = _noise(nt, p, 46.0, 3.0, 0.58)
    d = nt.math('ADD', n1, nt.math('MULTIPLY', radial, 0.45))
    dens = nt.map_range(d, o_thr, nt.math('ADD', o_thr, 0.22), 0.0, 1.0, interp='SMOOTHSTEP')
    fade = nt.map_range(clear, 0.5, 9.0, 0.0, 1.0, interp='SMOOTHSTEP')
    a = nt.math('MULTIPLY', nt.math('MULTIPLY', dens, fade), nt.math('MULTIPLY', o_alpha, radial))
    a = nt.math('MULTIPLY', a, nt.map_range(dist, 15.0, 60.0, 0.0, 1.0))
    return _finish_mist(m, nt, out, _mist_colour(nt, dens), a, fog_mul=0.5)


@mats.material("ls_trees")
def _m_trees(name):
    """Distant conifer cones: point attribute "tree" = (random, height fraction, -, -)."""
    m, nt, out = new_material(name)
    tr = nt.n('ShaderNodeSeparateColor', Color=nt.attr('tree').outputs['Color'])
    rnd, hf = tr.outputs[0], tr.outputs[1]
    col = nt.mix(rnd, (0.006, 0.014, 0.007, 1), (0.018, 0.032, 0.013, 1))
    col = nt.mix(1.0, col, nt.ramp(hf, [(0.0, (0.35, 0.38, 0.35, 1)), (1.0, (1.25, 1.25, 1.1, 1))]),
                 blend='MULTIPLY')
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.9)
    bsdf.inputs['Specular IOR Level'].default_value = 0.15
    from .landscape_mats import FOG_TERRAIN          # same haze as the ground they stand on
    nt.link(atmo.fog(nt, bsdf.outputs[0], density_mul=FOG_TERRAIN), out.inputs['Surface'])
    return m


# =====================================================================================
# builders
# =====================================================================================
def _disc(name, cx, cy, z, radius, nseg, nring, col, material, tg, new_mesh_object, grid_quads, add_attr):
    rr = radius * (np.arange(nring + 1) / nring) ** 1.3
    th = np.arange(nseg) / nseg * 2.0 * np.pi
    X = rr[:, None] * np.cos(th)[None, :]; Y = rr[:, None] * np.sin(th)[None, :]
    co = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    quads = grid_quads(nring + 1, nseg, True)
    ob = new_mesh_object(name, co, quads, np.arange(len(quads) // 4) * 4, col, material)
    add_attr(ob.data, "ground", _sample_polar(tg, cx + X.ravel(), cy + Y.ravel()))
    ob.location = (cx, cy, z)
    ob.visible_shadow = False
    return ob


def _valley_grid(tg):
    """Coarse sub-grid of the terrain grid for the valley sheets: positions + conservative ground height
    (max of the terrain around every coarse vertex, so the mist never cuts through a bump)."""
    X = tg['X']; Y = tg['Y']; H = tg['H']; radii = tg['radii']
    SR, SC = 3, 8
    i_hi = int(np.searchsorted(radii, VALLEY_R))
    i_hi -= (i_hi % SR)
    Hs = H[:i_hi + 1]
    m = Hs.copy()
    for k in range(1, SC + 1):
        m = np.maximum(m, np.roll(Hs, -k, axis=1)); m = np.maximum(m, np.roll(Hs, k, axis=1))
    mr = m.copy()
    for k in range(1, SR + 1):
        mr[:-k] = np.maximum(mr[:-k], m[k:]); mr[k:] = np.maximum(mr[k:], m[:-k])
    return X[:i_hi + 1:SR, ::SC], Y[:i_hi + 1:SR, ::SC], mr[::SR, ::SC]


def _valley_sheet(name, z, vg, col, new_mesh_object, add_attr):
    """Horizontal sheet at height z that only exists over ground lower than z."""
    Xc, Yc, G = vg
    nrc, ncc = Xc.shape
    cl = (z - G) > 1.5
    clw = np.concatenate([cl, cl[:, :1]], axis=1)
    keep = clw[:-1, :-1] | clw[1:, :-1] | clw[:-1, 1:] | clw[1:, 1:]
    ii, jj = np.nonzero(keep)
    j1 = (jj + 1) % ncc
    quads = np.stack([ii * ncc + jj, (ii + 1) * ncc + jj, (ii + 1) * ncc + j1, ii * ncc + j1], axis=1)
    used = np.unique(quads)
    remap = np.full(nrc * ncc, -1, dtype=np.int64); remap[used] = np.arange(len(used))
    co = np.stack([Xc.ravel()[used], Yc.ravel()[used], np.full(len(used), z)], axis=1)
    ob = new_mesh_object(name, co, remap[quads].ravel(), np.arange(len(quads)) * 4, col, "ls_mist")
    add_attr(ob.data, "ground", G.ravel()[used])
    ob.visible_shadow = False
    return ob


def _shell(name, radius, tg, col, new_mesh_object, grid_quads, add_attr):
    nseg = 768
    th = np.arange(nseg) / nseg * 2.0 * np.pi
    dth = 2.0 * np.pi / nseg
    g = None
    for o in (-0.5, -0.25, 0.0, 0.25, 0.5):               # conservative ground height per column
        gg = _sample_polar(tg, F.CX + radius * np.cos(th + o * dth), F.CY + radius * np.sin(th + o * dth))
        g = gg if g is None else np.maximum(g, gg)
    zz = np.array([SHELL_Z0, SHELL_Z1])
    X = F.CX + radius * np.cos(th)[None, :] + 0.0 * zz[:, None]
    Y = F.CY + radius * np.sin(th)[None, :] + 0.0 * zz[:, None]
    Z = zz[:, None] + 0.0 * th[None, :]
    co = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    quads = grid_quads(2, nseg, True)
    ob = new_mesh_object(name, co, quads, np.arange(len(quads) // 4) * 4, col, "ls_mist_shell")
    add_attr(ob.data, "ground", np.concatenate([g, g]))
    ob.visible_shadow = False
    return ob


def build_mist(col, tg, new_mesh_object, grid_quads, add_attr):
    k = 0
    vg = _valley_grid(tg)
    for (z, alpha, thr) in VALLEY_LAYERS:
        ob = _valley_sheet("landscape_mist_%02d" % k, z, vg, col, new_mesh_object, add_attr)
        ob.color = (alpha, thr, 0.0, 1.0); k += 1
    for (radius, alpha, thr) in SHELLS:
        ob = _shell("landscape_mist_%02d" % k, radius, tg, col, new_mesh_object, grid_quads, add_attr)
        ob.color = (alpha, thr, 0.0, 1.0); k += 1
    for (z, alpha, thr) in PLUME_LAYERS:
        ob = _disc("landscape_mist_%02d" % k, PLUME_CENTER[0], PLUME_CENTER[1], z, 110.0, 64, 20, col,
                   "ls_mist_plume", tg, new_mesh_object, grid_quads, add_attr)
        ob.color = (alpha, thr, 0.0, 1.0); k += 1
    return k


def build_trees(col, tg, new_mesh_object, add_attr):
    """Distant conifer cones, placed on the terrain grid cells (so they sit exactly on the rendered mesh)."""
    rs = np.random.RandomState(777)
    X = tg['X']; Y = tg['Y']; H = tg['H']; fm = tg['fm']; radii = tg['radii']; theta = tg['theta']
    nr, nc = H.shape
    i_lo = int(np.searchsorted(radii, 800.0)); i_hi = int(np.searchsorted(radii, 9500.0))
    sl = slice(i_lo, i_hi)
    r = radii[sl]
    # visibility from the castle area (eye a little above the terrace): hidden cells get few trees,
    # cells right at a horizon line get more (they make the ridge silhouettes)
    elev = (H[:i_hi] - 25.0) / radii[:i_hi, None]
    runmax = np.maximum.accumulate(elev, axis=0)
    vis = elev >= runmax - 0.004
    vis_s = vis[sl]
    ahead = np.ones_like(vis_s)
    for k in (2, 4, 7):
        ahead &= np.roll(vis, -k, axis=0)[sl]
    sil = vis_s & ~ahead
    dth = (np.roll(theta, -1) - theta) % (2.0 * np.pi)
    area = (np.gradient(radii)[sl, None]) * (r[:, None] * dth[None, :])
    dens = F.smoothstep(820.0, 1000.0, r) * np.minimum(1.0, (2300.0 / r)) ** 1.5 * (1.0 - F.smoothstep(5500.0, 8000.0, r))
    w = fm[sl] * dens[:, None] * area
    w = w * np.where(vis_s, 1.0, 0.10) * np.where(sil, 4.0, 1.0)
    w = w * np.where(H[sl] < F.VALLEY_Z + 14.0, 2.2, 1.0)                 # shores / islands
    p = (w / w.sum()).ravel()
    cnt = rs.multinomial(N_TREES, p)
    idx = np.repeat(np.arange(p.size), cnt)
    ci = idx // nc + i_lo; cj = idx % nc
    cj1 = (cj + 1) % nc
    ci1 = np.minimum(ci + 1, nr - 1)
    u = rs.rand(len(idx)); v = rs.rand(len(idx))

    def bil(A):
        return (A[ci, cj] * (1 - u) + A[ci1, cj] * u) * (1 - v) + (A[ci, cj1] * (1 - u) + A[ci1, cj1] * u) * v
    tx = bil(X); ty = bil(Y); tz = bil(H)
    q = np.hypot(tx - F.CX, ty - F.CY)
    far = q > tg.get('r_split', 1e9) - 5.0          # on the far mesh: take the exact height (finer than the grid)
    if far.any():
        tz[far] = F.height_at(tx[far], ty[far]) - 2.0
    scale = np.clip(q / 2600.0, 1.0, 1.45)          # near real size: long lenses look at them
    ht = (19.0 + 17.0 * rs.rand(len(idx)) ** 1.5) * scale
    rad = ht * (0.15 + 0.06 * rs.rand(len(idx)))
    rot = rs.rand(len(idx)) * 2.0 * np.pi
    n = len(idx)
    co = np.empty((n, 5, 3), dtype=np.float32)
    co[:, 0, 0] = tx; co[:, 0, 1] = ty; co[:, 0, 2] = tz + ht
    for k in range(4):
        a = rot + k * (np.pi / 2.0)
        co[:, 1 + k, 0] = tx + rad * np.cos(a); co[:, 1 + k, 1] = ty + rad * np.sin(a); co[:, 1 + k, 2] = tz - 1.5
    base = np.arange(n)[:, None] * 5
    tri = np.concatenate([base, base + 1, base + 2, base, base + 2, base + 3,
                          base, base + 3, base + 4, base, base + 4, base + 1], axis=1).reshape(-1)
    ob = new_mesh_object("landscape_trees", co.reshape(-1, 3), tri, np.arange(n * 4) * 3, col, "ls_trees")
    me = ob.data
    at = me.attributes.new("sharp_face", 'BOOLEAN', 'FACE')
    at.data.foreach_set("value", np.ones(n * 4, dtype=bool))
    ta = np.zeros((n, 5, 4), dtype=np.float32)
    ta[:, :, 0] = rs.rand(n)[:, None]; ta[:, 0, 1] = 1.0; ta[:, :, 3] = 1.0
    add_attr(me, "tree", ta.reshape(-1, 4), 'FLOAT_COLOR')
    return ob, n
