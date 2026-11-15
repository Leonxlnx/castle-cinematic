"""Procedural materials of the vegetation module (registered in scene.mats under the prefix 'veg_')."""
import bpy, math
from .. import mats, atmo
from ..common import NT, new_material


def _two_sided(m, thickness_out=None, nt=None, out=None):
    m.use_backface_culling = False
    try:
        m.thickness_mode = 'SLAB'
        m.use_thickness_from_shadow = False
    except Exception:
        pass
    if nt is not None and out is not None:
        try:
            nt.set(out.inputs['Thickness'], 0.0 if thickness_out is None else thickness_out)
        except Exception:
            pass


def _alpha(m):
    m.surface_render_method = 'DITHERED'
    try:
        m.use_transparent_shadow = True
    except Exception:
        pass


def _inst_noise(nt, scale=0.5):
    """Spatially coherent 0..1 variation per instance (works for GN instances: uses the instance origin)."""
    oi = nt.n('ShaderNodeObjectInfo')
    n = nt.n('ShaderNodeTexNoise', Vector=oi.outputs['Location'], Scale=scale, Detail=1.0, Roughness=0.5)
    return n.outputs['Factor'], oi.outputs['Random'], oi


def _true_normal(nt, geo):
    """Shading normal un-flipped on back faces (for cards that carry custom 'volume' normals)."""
    n = geo.outputs['Normal']
    flip = nt.vmath('SCALE', n, scale=-1.0)
    mx = nt.n('ShaderNodeMix', data_type='VECTOR')
    nt.set(mx.inputs[0], geo.outputs['Backfacing']); nt.set(mx.inputs[4], n); nt.set(mx.inputs[5], flip)
    return mx.outputs[1]


# =====================================================================================
# bark
# =====================================================================================
def _bark(name, c1, c2, scale=18.0, fog=False, stretch=6.0):
    m, nt, out = new_material(name)
    tc = nt.n('ShaderNodeTexCoord')
    mp = nt.n('ShaderNodeMapping', Vector=tc.outputs['UV'])
    mp.inputs['Scale'].default_value = (stretch, 1.0, 1.0)
    n1 = nt.n('ShaderNodeTexNoise', Vector=mp.outputs[0], Scale=scale, Detail=4.0, Roughness=0.62, Distortion=0.4)
    geo = nt.n('ShaderNodeNewGeometry')
    n2 = nt.n('ShaderNodeTexNoise', Vector=geo.outputs['Position'], Scale=1.3, Detail=2.0)
    col = nt.ramp(n1.outputs['Factor'], [(0.28, c1), (0.72, c2)])
    col = nt.mix(nt.map_range(n2.outputs['Factor'], 0.35, 0.7), col, (0.05, 0.062, 0.035, 1.0), blend='MIX')   # lichen
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.82,
                Normal=nt.bump(n1.outputs['Factor'], strength=0.55, distance=0.01))
    bsdf.inputs['Specular IOR Level'].default_value = 0.25
    sh = bsdf.outputs[0]
    if fog:
        sh = atmo.fog(nt, sh)
    nt.link(sh, out.inputs['Surface'])
    return m


@mats.material("veg_bark")
def _m_bark(name):
    return _bark(name, (0.022, 0.017, 0.013, 1), (0.095, 0.078, 0.062, 1))


@mats.material("veg_bark_far")
def _m_bark_far(name):
    return _bark(name, (0.030, 0.022, 0.016, 1), (0.11, 0.085, 0.062, 1), scale=9.0, fog=True)


# =====================================================================================
# maple leaves (real geometry, translucent)
# =====================================================================================
MAPLE_BACKLIGHT = 0.55     # strength of the fake sun-through-leaf glow


def _maple_leaf(name, stops, trans_boost=1.0):
    m, nt, out = new_material(name)
    tc = nt.n('ShaderNodeTexCoord')
    geo = nt.n('ShaderNodeNewGeometry')
    uv = nt.sep_xyz(tc.outputs['UV'])
    ulat, rad = uv[0], uv[1]
    patch, rnd, oi = _inst_noise(nt, 0.55)
    tint = nt.attr('tint').outputs['Factor']
    h = nt.math('ADD', nt.math('ADD', nt.math('MULTIPLY', nt.map_range(patch, 0.25, 0.75), 0.42),
                               nt.math('MULTIPLY', tint, 0.40)), nt.math('MULTIPLY', rnd, 0.18))
    col = nt.ramp(h, stops)
    # per-leaf value variation (some leaves much darker)
    val = nt.map_range(nt.math('FRACT', nt.math('ADD', nt.math('MULTIPLY', tint, 7.13), nt.math('MULTIPLY', rnd, 3.7))),
                       0.0, 1.0, 0.45, 1.15)
    col = nt.mix(1.0, col, nt.n('ShaderNodeCombineColor', Red=val, Green=val, Blue=val).outputs[0], blend='MULTIPLY')
    # veins: slightly darker midribs, lighter rim
    vein = nt.map_range(ulat, 0.0, 0.16, 1.0, 0.0, interp='SMOOTHSTEP')
    col = nt.mix(nt.math('MULTIPLY', vein, 0.45), col, (0.06, 0.004, 0.006, 1.0))
    rim = nt.map_range(ulat, 0.75, 1.0, 0.0, 1.0)
    col = nt.mix(nt.math('MULTIPLY', rim, 0.25), col, nt.mix(1.0, col, (1.5, 1.35, 1.2, 1.0), blend='MULTIPLY'))
    # underside is paler / duller
    back = geo.outputs['Backfacing']
    col_s = nt.mix(nt.math('MULTIPLY', back, 0.35), col, nt.mix(0.5, col, (0.30, 0.10, 0.08, 1.0)))
    rough = nt.mixf(back, 0.38, 0.62)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col_s, Roughness=rough)
    bsdf.inputs['Specular IOR Level'].default_value = 0.45
    tcol = nt.mix(1.0, col, (2.4 * trans_boost, 2.2 * trans_boost, 1.6 * trans_boost, 1.0), blend='MULTIPLY')
    tcol = nt.mix(0.10, tcol, (1.0, 0.10, 0.02, 1.0), blend='ADD')
    tr = nt.n('ShaderNodeBsdfTranslucent', Color=tcol)
    mix = nt.n('ShaderNodeMixShader', i0=0.36, i1=bsdf.outputs[0], i2=tr.outputs[0])
    # fake through-light: leaves seen against the sun glow (EEVEE's translucency alone is too weak at dusk)
    from .. import layout as _L
    sd = _L.sun_dir()
    toward = nt.vmath('DOT_PRODUCT', geo.outputs['Incoming'], (-sd[0], -sd[1], -sd[2]))
    glow = nt.math('POWER', nt.map_range(toward, 0.15, 1.0, 0.0, 1.0), 1.6)
    glow = nt.math('MULTIPLY', glow, nt.map_range(val, 0.45, 1.15, 0.25, 1.0))
    em = nt.n('ShaderNodeEmission', Color=nt.mix(0.3, tcol, (1.0, 0.10, 0.02, 1.0)),
              Strength=nt.math('MULTIPLY', glow, MAPLE_BACKLIGHT))
    mix = nt.n('ShaderNodeAddShader', i0=mix.outputs[0], i1=em.outputs[0])
    nt.link(mix.outputs[0], out.inputs['Surface'])
    _two_sided(m, None, nt, out)
    return m


@mats.material("veg_maple_leaf")
def _m_maple_leaf(name):
    return _maple_leaf(name, [(0.0, (0.075, 0.003, 0.008, 1)), (0.35, (0.24, 0.006, 0.012, 1)),
                              (0.68, (0.46, 0.016, 0.010, 1)), (1.0, (0.60, 0.07, 0.012, 1))])


# =====================================================================================
# debug ground (isolated test builds only)
# =====================================================================================
@mats.material("veg_dbg_ground")
def _m_dbg_ground(name):
    m, nt, out = new_material(name)
    geo = nt.n('ShaderNodeNewGeometry')
    n = nt.n('ShaderNodeTexNoise', Vector=geo.outputs['Position'], Scale=0.3, Detail=4.0)
    col = nt.ramp(n.outputs['Factor'], [(0.3, (0.035, 0.045, 0.022, 1)), (0.7, (0.075, 0.07, 0.04, 1))])
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.9)
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


@mats.material("veg_soil")
def _m_soil(name):
    m, nt, out = new_material(name)
    geo = nt.n('ShaderNodeNewGeometry')
    n = nt.n('ShaderNodeTexNoise', Vector=geo.outputs['Position'], Scale=22.0, Detail=4.0, Roughness=0.7)
    col = nt.ramp(n.outputs['Factor'], [(0.3, (0.012, 0.009, 0.006, 1)), (0.75, (0.045, 0.032, 0.02, 1))])
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.95,
                Normal=nt.bump(n.outputs['Factor'], strength=0.8, distance=0.02))
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


# =====================================================================================
# cypress / shrub foliage (opaque geometry leaves)
# =====================================================================================
def _foliage(name, c_dark, c_light, rough=0.6, trans=0.0, trans_col=(0.2, 0.5, 0.05, 1.0), spec=0.3, vein=False,
             patch_scale=0.8, fog=False, unflip=True):
    """Leaf material for geometry foliage.  UV = (across 0..1, along 0..1: 0 base, 1 tip); attribute 'tint'."""
    m, nt, out = new_material(name)
    tc = nt.n('ShaderNodeTexCoord')
    geo = nt.n('ShaderNodeNewGeometry')
    uv = nt.sep_xyz(tc.outputs['UV'])
    patch, rnd, oi = _inst_noise(nt, patch_scale)
    tint = nt.attr('tint').outputs['Factor']
    h = nt.math('ADD', nt.math('MULTIPLY', tint, 0.7), nt.math('MULTIPLY', nt.map_range(patch, 0.3, 0.7), 0.3))
    col = nt.mix(h, c_dark, c_light)
    # darker toward the base (inside of the plant), lighter tips
    col = nt.mix(1.0, col, nt.ramp(uv[1], [(0.0, (0.45, 0.45, 0.45, 1)), (0.6, (1.0, 1.0, 1.0, 1)), (1.0, (1.25, 1.25, 1.1, 1))]),
                 blend='MULTIPLY')
    if vein:
        v = nt.map_range(nt.math('ABSOLUTE', nt.math('SUBTRACT', uv[0], 0.5)), 0.0, 0.07, 1.0, 0.0)
        col = nt.mix(nt.math('MULTIPLY', v, 0.5), col, nt.mix(1.0, col, (1.9, 1.9, 1.3, 1.0), blend='MULTIPLY'))
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=rough)
    bsdf.inputs['Specular IOR Level'].default_value = spec
    if unflip:
        nt.link(_true_normal(nt, geo), bsdf.inputs['Normal'])
    sh = bsdf.outputs[0]
    if trans > 0:
        tr = nt.n('ShaderNodeBsdfTranslucent', Color=nt.mix(0.5, col, trans_col))
        sh = nt.n('ShaderNodeMixShader', i0=trans, i1=sh, i2=tr.outputs[0]).outputs[0]
    if fog:
        sh = atmo.fog(nt, sh)
    nt.link(sh, out.inputs['Surface'])
    _two_sided(m, None, nt, out)
    return m


@mats.material("veg_cypress")
def _m_cypress(name):
    return _foliage(name, (0.010, 0.024, 0.009, 1), (0.032, 0.066, 0.019, 1), rough=0.62, spec=0.25)


@mats.material("veg_cypress_core")
def _m_cypress_core(name):
    m, nt, out = new_material(name)
    geo = nt.n('ShaderNodeNewGeometry')
    mp = nt.n('ShaderNodeMapping', Vector=geo.outputs['Position'])
    mp.inputs['Scale'].default_value = (9.0, 9.0, 1.6)
    n = nt.n('ShaderNodeTexNoise', Vector=mp.outputs[0], Scale=1.0, Detail=3.0, Roughness=0.6)
    col = nt.ramp(n.outputs['Factor'], [(0.3, (0.004, 0.009, 0.004, 1)), (0.75, (0.014, 0.030, 0.010, 1))])
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.8,
                Normal=nt.bump(n.outputs['Factor'], strength=0.9, distance=0.05))
    bsdf.inputs['Specular IOR Level'].default_value = 0.1
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


@mats.material("veg_shrub_leaf")
def _m_shrub_leaf(name):
    return _foliage(name, (0.016, 0.045, 0.010, 1), (0.055, 0.125, 0.022, 1), rough=0.42, spec=0.5, vein=True,
                    trans=0.2, trans_col=(0.25, 0.55, 0.04, 1.0), unflip=True)


@mats.material("veg_ivy_leaf")
def _m_ivy_leaf(name):
    return _foliage(name, (0.012, 0.040, 0.010, 1), (0.04, 0.105, 0.024, 1), rough=0.35, spec=0.5, vein=True,
                    trans=0.2, trans_col=(0.2, 0.5, 0.05, 1.0), unflip=True)


@mats.material("veg_shrub_core")
def _m_shrub_core(name):
    m, nt, out = new_material(name)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.007, 0.016, 0.005, 1), Roughness=0.9)
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


@mats.material("veg_petal")
def _m_petal(name):
    """White hydrangea florets.  UV = (across, along 0 centre .. 1 petal tip); 'tint' per floret."""
    m, nt, out = new_material(name)
    tc = nt.n('ShaderNodeTexCoord')
    geo = nt.n('ShaderNodeNewGeometry')
    uv = nt.sep_xyz(tc.outputs['UV'])
    tint = nt.attr('tint').outputs['Factor']
    base = nt.mix(tint, (0.80, 0.80, 0.72, 1.0), (0.70, 0.76, 0.62, 1.0))
    col = nt.mix(nt.map_range(uv[1], 0.0, 0.4, 0.8, 0.0), base, (0.30, 0.42, 0.10, 1.0))
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.55, Normal=_true_normal(nt, geo))
    bsdf.inputs['Specular IOR Level'].default_value = 0.3
    tr = nt.n('ShaderNodeBsdfTranslucent', Color=(0.85, 0.85, 0.72, 1.0))
    mix = nt.n('ShaderNodeMixShader', i0=0.3, i1=bsdf.outputs[0], i2=tr.outputs[0])
    nt.link(mix.outputs[0], out.inputs['Surface'])
    _two_sided(m, None, nt, out)
    return m


@mats.material("veg_red_leaf")
def _m_red_leaf(name):
    return _maple_leaf(name, [(0.0, (0.11, 0.004, 0.012, 1)), (0.4, (0.30, 0.010, 0.014, 1)),
                              (0.75, (0.50, 0.03, 0.010, 1)), (1.0, (0.58, 0.09, 0.012, 1))])


@mats.material("veg_red_core")
def _m_red_core(name):
    m, nt, out = new_material(name)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.030, 0.004, 0.005, 1), Roughness=0.9)
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m
