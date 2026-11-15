"""Forest materials (alpha-cut needle fronds, leaf cards, grass, far cones).  All fogged with atmo.fog."""
import bpy, math
from .. import mats, atmo
from ..common import NT, new_material
from .vegetation_mats import _two_sided, _alpha, _inst_noise, _true_normal


def _finish(m, nt, out, bsdf_sock, alpha=None, fog=True):
    sh = bsdf_sock
    if fog:
        sh = atmo.fog(nt, sh)
    if alpha is not None:
        tr = nt.n('ShaderNodeBsdfTransparent', Color=(1, 1, 1, 1))
        sh = nt.n('ShaderNodeMixShader', i0=alpha, i1=tr.outputs[0], i2=sh).outputs[0]
        _alpha(m)
    nt.link(sh, out.inputs['Surface'])
    m.use_backface_culling = False
    return m


def _needle(name, curtain=False):
    """Alpha-cut conifer foliage card.
    frond:   UV u across (0.5 = spine), v along (0 trunk .. 1 tip)  -> herringbone of side twigs
    curtain: UV u along the bough, v downward (0 at the spine .. 1 bottom) -> hanging strands"""
    m, nt, out = new_material(name)
    tc = nt.n('ShaderNodeTexCoord')
    geo = nt.n('ShaderNodeNewGeometry')
    uv = nt.sep_xyz(tc.outputs['UV'])
    u, v = uv[0], uv[1]
    patch, rnd, oi = _inst_noise(nt, 0.02)
    tint = nt.attr('tint').outputs['Factor']
    nz = nt.n('ShaderNodeTexNoise', Vector=tc.outputs['UV'], Scale=42.0, Detail=2.0, Roughness=0.6).outputs['Factor']
    jit = nt.math('SUBTRACT', nz, 0.5)
    if not curtain:
        x = nt.math('MULTIPLY', nt.math('ABSOLUTE', nt.math('SUBTRACT', u, 0.5)), 2.0)
        t = nt.math('FRACT', nt.math('SUBTRACT', nt.math('MULTIPLY', v, 9.0), nt.math('MULTIPLY', x, 2.6)))
        duty = nt.math('ADD', nt.math('SUBTRACT', 0.80, nt.math('MULTIPLY', x, 0.42)), nt.math('MULTIPLY', jit, 0.55))
        stripe = nt.math('LESS_THAN', t, duty)
        centre = nt.math('LESS_THAN', x, 0.09)
        lim = nt.math('ADD', nt.math('SUBTRACT', 1.0, nt.math('MULTIPLY', nt.math('POWER', v, 3.0), 0.75)),
                      nt.math('MULTIPLY', jit, 0.35))
        outline = nt.math('LESS_THAN', x, lim)
        alpha = nt.math('MULTIPLY', nt.math('MAXIMUM', stripe, centre), outline)
        tipness = nt.math('MULTIPLY', x, x)
    else:
        us = nt.math('MULTIPLY', u, 11.0)
        cell = nt.math('FLOOR', us)
        ln = nt.n('ShaderNodeTexWhiteNoise', noise_dimensions='1D', W=nt.math('ADD', cell, nt.math('MULTIPLY', rnd, 37.0)))
        length = nt.math('ADD', 0.35, nt.math('MULTIPLY', ln.outputs['Value'], 0.65))
        vv = nt.math('DIVIDE', v, length)
        tt = nt.math('ABSOLUTE', nt.math('SUBTRACT', nt.math('FRACT', us), 0.5))
        wid = nt.math('ADD', nt.math('MULTIPLY', nt.math('SUBTRACT', 1.0, vv), 0.42), nt.math('MULTIPLY', jit, 0.30))
        alpha = nt.math('MULTIPLY', nt.math('LESS_THAN', tt, wid), nt.math('LESS_THAN', vv, 1.0))
        tipness = nt.math('MULTIPLY', vv, vv)
    h = nt.math('ADD', nt.math('MULTIPLY', tint, 0.5), nt.math('MULTIPLY', nt.map_range(patch, 0.3, 0.7), 0.5))
    col = nt.mix(h, (0.014, 0.034, 0.018, 1.0), (0.036, 0.066, 0.022, 1.0))
    col = nt.mix(nt.math('MULTIPLY', tipness, 0.6), col, nt.mix(1.0, col, (1.9, 1.8, 1.2, 1.0), blend='MULTIPLY'))
    col = nt.mix(nt.map_range(v, 0.0, 0.25, 0.6, 0.0), col, (0.004, 0.008, 0.005, 1.0))        # dark near the trunk
    col = nt.mix(nt.math('MULTIPLY', nt.math('GREATER_THAN', rnd, 0.9), 0.5), col, (0.05, 0.045, 0.012, 1.0))
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.6, Normal=_true_normal(nt, geo))
    bsdf.inputs['Specular IOR Level'].default_value = 0.2
    return _finish(m, nt, out, bsdf.outputs[0], alpha)


@mats.material("veg_needle")
def _m_needle(name):
    return _needle(name, False)


@mats.material("veg_needle_curtain")
def _m_needle_curtain(name):
    return _needle(name, True)


@mats.material("veg_conifer_core")
def _m_conifer_core(name):
    m, nt, out = new_material(name)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.008, 0.016, 0.010, 1), Roughness=0.9)
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    return _finish(m, nt, out, bsdf.outputs[0])


@mats.material("veg_far_conifer")
def _m_far_conifer(name):
    """LOD2 cones: UV v = height fraction; darker at the bottom, per-tree tone, coarse noise 'boughs'."""
    m, nt, out = new_material(name)
    tc = nt.n('ShaderNodeTexCoord')
    geo = nt.n('ShaderNodeNewGeometry')
    v = nt.sep_xyz(tc.outputs['UV'])[1]
    patch, rnd, oi = _inst_noise(nt, 0.012)
    n = nt.n('ShaderNodeTexNoise', Vector=geo.outputs['Position'], Scale=0.55, Detail=2.0, Roughness=0.6)
    h = nt.math('ADD', nt.math('MULTIPLY', rnd, 0.45), nt.math('MULTIPLY', nt.map_range(patch, 0.3, 0.7), 0.55))
    col = nt.mix(h, (0.013, 0.030, 0.017, 1.0), (0.034, 0.060, 0.022, 1.0))
    col = nt.mix(nt.math('MULTIPLY', nt.math('GREATER_THAN', rnd, 0.92), 0.6), col, (0.07, 0.06, 0.015, 1.0))
    shade = nt.math('MULTIPLY', nt.map_range(v, 0.0, 0.8, 0.45, 1.1), nt.map_range(n.outputs['Factor'], 0.3, 0.7, 0.6, 1.2))
    col = nt.mix(1.0, col, nt.n('ShaderNodeCombineColor', Red=shade, Green=shade, Blue=shade).outputs[0], blend='MULTIPLY')
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.75)
    bsdf.inputs['Specular IOR Level'].default_value = 0.1
    return _finish(m, nt, out, bsdf.outputs[0])


def _leafcard(name, c_a, c_b, c_c, scale=15.0, fill=0.66, autumn=0.0):
    """Leaf-cluster card.  UV 0..1 square; alpha = voronoi leaf blobs fading out toward the card rim.
    Colour: per-leaf (voronoi cell) variation between c_a, c_b and (rarely) c_c; per-instance drift."""
    m, nt, out = new_material(name)
    tc = nt.n('ShaderNodeTexCoord')
    geo = nt.n('ShaderNodeNewGeometry')
    patch, rnd, oi = _inst_noise(nt, 0.03)
    tint = nt.attr('tint').outputs['Factor']
    uvo = nt.vmath('ADD', tc.outputs['UV'], nt.comb_xyz(nt.math('MULTIPLY', tint, 13.0), nt.math('MULTIPLY', tint, 7.0), 0.0))
    vor = nt.n('ShaderNodeTexVoronoi', Vector=uvo, Scale=scale, Randomness=1.0)
    vor.feature = 'F1'
    d = vor.outputs['Distance']
    uv = nt.sep_xyz(tc.outputs['UV'])
    rx = nt.math('SUBTRACT', uv[0], 0.5); ry = nt.math('SUBTRACT', uv[1], 0.5)
    r2 = nt.math('MULTIPLY', nt.math('ADD', nt.math('MULTIPLY', rx, rx), nt.math('MULTIPLY', ry, ry)), 4.0)
    cellr = nt.n('ShaderNodeSeparateColor', Color=vor.outputs['Color'])
    thr = nt.math('MULTIPLY', fill, nt.math('SUBTRACT', 1.2, nt.math('MULTIPLY', nt.math('MULTIPLY', r2, r2), 1.0)))
    thr = nt.math('MULTIPLY', thr, nt.map_range(cellr.outputs[1], 0.0, 1.0, 0.55, 1.25))
    alpha = nt.math('LESS_THAN', d, thr)
    h = nt.math('ADD', nt.math('MULTIPLY', cellr.outputs[0], 0.55), nt.math('MULTIPLY', nt.map_range(patch, 0.3, 0.7), 0.45))
    col = nt.mix(h, c_a, c_b)
    rare = nt.math('GREATER_THAN', nt.math('ADD', nt.math('MULTIPLY', cellr.outputs[2], 0.5),
                                           nt.math('MULTIPLY', rnd, 0.5)), 1.0 - 0.45 * autumn - 0.12)
    col = nt.mix(rare, col, c_c)
    col = nt.mix(1.0, col, nt.ramp(d, [(0.0, (0.7, 0.7, 0.7, 1)), (0.35, (1.15, 1.15, 1.1, 1))]), blend='MULTIPLY')
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.55, Normal=_true_normal(nt, geo))
    bsdf.inputs['Specular IOR Level'].default_value = 0.25
    return _finish(m, nt, out, bsdf.outputs[0], alpha)


@mats.material("veg_leafcard")
def _m_leafcard(name):
    return _leafcard(name, (0.018, 0.042, 0.010, 1), (0.050, 0.090, 0.018, 1), (0.16, 0.11, 0.015, 1))


@mats.material("veg_leafcard_autumn")
def _m_leafcard_autumn(name):
    return _leafcard(name, (0.11, 0.050, 0.010, 1), (0.26, 0.15, 0.018, 1), (0.30, 0.035, 0.010, 1), fill=0.60, autumn=1.0)


@mats.material("veg_leafcard_sparse")
def _m_leafcard_sparse(name):
    return _leafcard(name, (0.09, 0.045, 0.012, 1), (0.20, 0.11, 0.02, 1), (0.25, 0.04, 0.010, 1), scale=17.0, fill=0.30,
                     autumn=1.0)


@mats.material("veg_grass")
def _m_grass(name):
    """Grass tuft card: UV u across, v up.  Blades = skewed stripes of random length."""
    m, nt, out = new_material(name)
    tc = nt.n('ShaderNodeTexCoord')
    geo = nt.n('ShaderNodeNewGeometry')
    uv = nt.sep_xyz(tc.outputs['UV'])
    u, v = uv[0], uv[1]
    patch, rnd, oi = _inst_noise(nt, 0.05)
    tint = nt.attr('tint').outputs['Factor']
    # fan: blades lean outward with height
    uu = nt.math('ADD', nt.math('MULTIPLY', nt.math('SUBTRACT', u, 0.5), nt.math('SUBTRACT', 1.0, nt.math('MULTIPLY', v, 0.45))), 0.5)
    us = nt.math('ADD', nt.math('MULTIPLY', uu, 13.0), nt.math('MULTIPLY', tint, 5.0))
    cell = nt.math('FLOOR', us)
    wn = nt.n('ShaderNodeTexWhiteNoise', noise_dimensions='1D', W=nt.math('ADD', cell, nt.math('MULTIPLY', rnd, 91.0)))
    length = nt.math('ADD', 0.35, nt.math('MULTIPLY', wn.outputs['Value'], 0.65))
    vv = nt.math('DIVIDE', v, length)
    bend = nt.math('MULTIPLY', nt.math('MULTIPLY', vv, vv), nt.math('SUBTRACT', wn.outputs['Value'], 0.5))
    tt = nt.math('ABSOLUTE', nt.math('SUBTRACT', nt.math('FRACT', nt.math('ADD', us, nt.math('MULTIPLY', bend, 0.0))), 0.5))
    wid = nt.math('MULTIPLY', nt.math('SUBTRACT', 1.0, nt.math('MULTIPLY', vv, vv)), 0.30)
    alpha = nt.math('MULTIPLY', nt.math('LESS_THAN', tt, wid), nt.math('LESS_THAN', vv, 1.0))
    h = nt.math('ADD', nt.math('MULTIPLY', wn.outputs['Value'], 0.5), nt.math('MULTIPLY', nt.map_range(patch, 0.3, 0.7), 0.5))
    col = nt.mix(h, (0.030, 0.050, 0.012, 1.0), (0.13, 0.11, 0.035, 1.0))
    col = nt.mix(1.0, col, nt.ramp(v, [(0.0, (0.45, 0.45, 0.45, 1)), (0.7, (1.1, 1.1, 1.0, 1))]), blend='MULTIPLY')
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.6, Normal=_true_normal(nt, geo))
    bsdf.inputs['Specular IOR Level'].default_value = 0.2
    return _finish(m, nt, out, bsdf.outputs[0], alpha)


@mats.material("veg_bush_core")
def _m_bush_core(name):
    m, nt, out = new_material(name)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.012, 0.024, 0.008, 1), Roughness=0.9)
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    return _finish(m, nt, out, bsdf.outputs[0])


def _far_broadleaf(name, c1, c2):
    m, nt, out = new_material(name)
    tc = nt.n('ShaderNodeTexCoord')
    geo = nt.n('ShaderNodeNewGeometry')
    v = nt.sep_xyz(tc.outputs['UV'])[1]
    patch, rnd, oi = _inst_noise(nt, 0.012)
    n = nt.n('ShaderNodeTexNoise', Vector=geo.outputs['Position'], Scale=0.8, Detail=2.0, Roughness=0.6)
    h = nt.math('ADD', nt.math('MULTIPLY', rnd, 0.5), nt.math('MULTIPLY', n.outputs['Factor'], 0.5))
    col = nt.mix(h, c1, c2)
    shade = nt.math('MULTIPLY', nt.map_range(v, 0.3, 0.9, 0.5, 1.1), nt.map_range(n.outputs['Factor'], 0.3, 0.7, 0.6, 1.2))
    col = nt.mix(1.0, col, nt.n('ShaderNodeCombineColor', Red=shade, Green=shade, Blue=shade).outputs[0], blend='MULTIPLY')
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.7)
    bsdf.inputs['Specular IOR Level'].default_value = 0.1
    return _finish(m, nt, out, bsdf.outputs[0])


@mats.material("veg_far_broadleaf")
def _m_far_broadleaf(name):
    return _far_broadleaf(name, (0.014, 0.032, 0.009, 1.0), (0.045, 0.070, 0.016, 1.0))


@mats.material("veg_far_broadleaf_autumn")
def _m_far_broadleaf_autumn(name):
    return _far_broadleaf(name, (0.10, 0.05, 0.012, 1.0), (0.24, 0.12, 0.02, 1.0))
