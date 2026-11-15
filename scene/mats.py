"""Material registry.  All materials are procedural (no image files) and tuned for EEVEE.

    from scene import mats
    mats.get("stone")                      # lazily creates + caches

Modules may register their own materials:

    @mats.material("my_rock")
    def _my_rock(name):
        m, nt, out = new_material(name)
        ...
        return m

UV convention: MB meshes carry UVs in METERS (u along the wall / v up, or x/y on floors), so brick/tile
textures use Scale=1 and real-world sizes.
"""
import bpy, math
from .common import NT, new_material

_REG = {}


def material(name):
    def deco(fn):
        _REG[name] = fn
        return fn
    return deco


def get(name):
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    if name not in _REG:
        raise KeyError("unknown material %r (registered: %s)" % (name, sorted(_REG)))
    m = _REG[name](name)
    m.name = name
    return m


def names():
    return sorted(_REG)


# =====================================================================================
# shared building blocks
# =====================================================================================
def _coords(nt):
    tc = nt.n('ShaderNodeTexCoord')
    geo = nt.n('ShaderNodeNewGeometry')
    return tc.outputs['UV'], geo.outputs['Position'], geo


def _weathering(nt, pos, strength=1.0):
    """Returns a 0..1 grime factor: large blotches + vertical rain streaks + damp base."""
    big = nt.n('ShaderNodeTexNoise', Vector=pos, Scale=0.11, Detail=4.0, Roughness=0.6)
    sx = nt.n('ShaderNodeMapping', Vector=pos)
    sx.inputs['Scale'].default_value = (1.3, 1.3, 0.07)
    streak = nt.n('ShaderNodeTexNoise', Vector=sx.outputs[0], Scale=1.0, Detail=4.0, Roughness=0.6)
    sx2 = nt.n('ShaderNodeMapping', Vector=pos)
    sx2.inputs['Scale'].default_value = (5.5, 5.5, 0.17)
    streak2 = nt.n('ShaderNodeTexNoise', Vector=sx2.outputs[0], Scale=1.0, Detail=3.0, Roughness=0.6)
    s = nt.map_range(streak.outputs['Factor'], 0.50, 0.74, 0.0, 1.0)
    s2 = nt.map_range(streak2.outputs['Factor'], 0.54, 0.76, 0.0, 1.0)
    b = nt.map_range(big.outputs['Factor'], 0.35, 0.75, 0.0, 1.0)
    g = nt.math('MAXIMUM', nt.math('MAXIMUM', nt.math('MULTIPLY', s, 0.85), nt.math('MULTIPLY', s2, 0.55)),
                nt.math('MULTIPLY', b, 0.7))
    # damp, dirty base course (first metres above the ground)
    z = nt.sep_xyz(pos)[2]
    base = nt.math('MULTIPLY', nt.map_range(z, 0.0, 8.0, 0.55, 0.0), nt.map_range(big.outputs['Factor'], 0.2, 0.8, 0.5, 1.2))
    g = nt.math('MAXIMUM', g, base)
    return nt.math('MULTIPLY', g, strength, clamp=True)


def _stone(name, c1, c2, mortar, brick_w=0.86, row_h=0.37, mortar_size=0.012, rough=0.82,
           grime=1.0, bump=0.55, moss=0.0):
    m, nt, out = new_material(name)
    uv, pos, geo = _coords(nt)
    brick = nt.n('ShaderNodeTexBrick', Vector=uv, Color1=c1, Color2=c2, Mortar=mortar, Scale=1.0,
                 Mortar_Size=mortar_size, Mortar_Smooth=0.15, Bias=-0.1, Brick_Width=brick_w, Row_Height=row_h)
    brick.offset = 0.5
    # second brick lookup with other colours = per-block tone variation that is not just 2 colours
    brick2 = nt.n('ShaderNodeTexBrick', Vector=uv, Color1=(0.84, 0.84, 0.85, 1), Color2=(1.08, 1.07, 1.05, 1),
                  Mortar=(1, 1, 1, 1), Scale=1.0, Mortar_Size=0.0, Bias=0.25, Brick_Width=brick_w * 2.0,
                  Row_Height=row_h)
    brick2.offset = 0.5
    col = nt.mix(1.0, brick.outputs['Color'], brick2.outputs['Color'], blend='MULTIPLY')
    # soft dirt halo along the joints: makes every block read at a distance, and gives a bevelled look
    brickd = nt.n('ShaderNodeTexBrick', Vector=uv, Color1=(1, 1, 1, 1), Color2=(1, 1, 1, 1), Mortar=(0, 0, 0, 1), Scale=1.0,
                  Mortar_Size=max(0.03, mortar_size * 4.5), Mortar_Smooth=1.0, Bias=0.0, Brick_Width=brick_w, Row_Height=row_h)
    brickd.offset = 0.5
    edge = nt.n('ShaderNodeSeparateColor', Color=brickd.outputs['Color']).outputs[0]
    col = nt.mix(nt.map_range(edge, 0.0, 1.0, 0.30, 0.0), col, (0.10, 0.09, 0.075, 1.0))
    # large warm / cool tonal drift across the walls
    tone = nt.n('ShaderNodeTexNoise', Vector=pos, Scale=0.036, Detail=2.0, Roughness=0.5)
    col = nt.mix(0.75, col, nt.ramp(tone.outputs['Factor'], [(0.3, (0.90, 0.93, 0.99, 1)), (0.7, (1.10, 1.02, 0.87, 1))]),
                 blend='MULTIPLY')
    # grain + fine speckle (tooling marks, pores)
    grain = nt.n('ShaderNodeTexNoise', Vector=pos, Scale=9.0, Detail=5.0, Roughness=0.7)
    col = nt.mix(0.35, col, nt.ramp(grain.outputs['Factor'], [(0.25, (0.55, 0.55, 0.55, 1)), (0.75, (1.25, 1.2, 1.15, 1))]),
                 blend='MULTIPLY')
    speck = nt.n('ShaderNodeTexNoise', Vector=pos, Scale=46.0, Detail=3.0, Roughness=0.75)
    col = nt.mix(0.5, col, nt.ramp(speck.outputs['Factor'], [(0.3, (0.80, 0.80, 0.80, 1)), (0.7, (1.16, 1.15, 1.13, 1))]),
                 blend='MULTIPLY')
    # weathering: rain streaks, blotches, damp base
    w = _weathering(nt, pos, grime)
    col = nt.mix(w, col, (0.085, 0.075, 0.062, 1.0), blend='MIX')
    # pale lichen / ochre stains
    ln = nt.n('ShaderNodeTexNoise', Vector=pos, Scale=1.9, Detail=5.0, Roughness=0.7)
    lf = nt.math('MULTIPLY', nt.map_range(ln.outputs['Factor'], 0.60, 0.72), 0.32)
    col = nt.mix(lf, col, (0.46, 0.42, 0.24, 1.0))
    if moss > 0:
        mn = nt.n('ShaderNodeTexNoise', Vector=pos, Scale=0.45, Detail=5.0, Roughness=0.65)
        mf = nt.math('MULTIPLY', nt.map_range(mn.outputs['Factor'], 0.5, 0.7), moss, clamp=True)
        col = nt.mix(mf, col, (0.045, 0.07, 0.03, 1.0))
    # bump: mortar recess + bevelled block edges + block face unevenness + grain + pores
    h = nt.math('ADD', nt.math('MULTIPLY', brick.outputs['Factor'], -1.0),
                nt.math('MULTIPLY', grain.outputs['Factor'], 0.35))
    h = nt.math('ADD', h, nt.math('MULTIPLY', edge, 0.6))
    h = nt.math('ADD', h, nt.math('MULTIPLY', speck.outputs['Factor'], 0.22))
    nrm = nt.bump(h, strength=bump, distance=0.02)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=rough, Normal=nrm)
    bsdf.inputs['Specular IOR Level'].default_value = 0.35
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


@material("stone")
def _m_stone(name):
    """Main castle ashlar: warm light limestone blocks."""
    return _stone(name, (0.585, 0.525, 0.44, 1), (0.50, 0.445, 0.375, 1), (0.34, 0.305, 0.26, 1), brick_w=1.05, row_h=0.42, mortar_size=0.008, grime=0.85, bump=0.4)


@material("stone_dark")
def _m_stone_dark(name):
    """Plinths, retaining walls, lower substructure: larger, greyer, damp + mossy blocks."""
    return _stone(name, (0.36, 0.32, 0.27, 1), (0.29, 0.26, 0.225, 1), (0.17, 0.155, 0.135, 1),
                  brick_w=1.35, row_h=0.56, mortar_size=0.014, rough=0.88, grime=1.1, bump=0.7, moss=0.6)


@material("stone_trim")
def _m_stone_trim(name):
    """Dressed stone for mouldings, frames, tracery, balustrades, copings: smooth, pale, few joints."""
    return _stone(name, (0.60, 0.545, 0.47, 1), (0.53, 0.48, 0.41, 1), (0.36, 0.33, 0.29, 1),
                  brick_w=2.4, row_h=1.2, mortar_size=0.006, rough=0.7, grime=0.7, bump=0.25)


@material("slate")
def _m_slate(name):
    """Blue-grey roof slates in overlapping rows."""
    m, nt, out = new_material(name)
    uv, pos, geo = _coords(nt)
    brick = nt.n('ShaderNodeTexBrick', Vector=uv, Color1=(0.060, 0.072, 0.100, 1), Color2=(0.135, 0.150, 0.190, 1),
                 Mortar=(0.012, 0.014, 0.018, 1), Scale=1.0, Mortar_Size=0.008, Mortar_Smooth=0.1, Bias=0.0,
                 Brick_Width=0.30, Row_Height=0.20)
    brick.offset = 0.5
    # sawtooth per row -> overlapping shingle relief
    v = nt.sep_xyz(uv)[1]
    saw = nt.math('FRACT', nt.math('DIVIDE', v, 0.20))
    big = nt.n('ShaderNodeTexNoise', Vector=pos, Scale=0.35, Detail=4.0, Roughness=0.6)
    col = nt.mix(0.5, brick.outputs['Color'],
                 nt.ramp(big.outputs['Factor'], [(0.3, (0.6, 0.62, 0.66, 1)), (0.7, (1.25, 1.2, 1.15, 1))]), blend='MULTIPLY')
    # decorative courses of lighter slate every ~2.6 m and lichen freckles
    band = nt.math('LESS_THAN', nt.math('FRACT', nt.math('DIVIDE', v, 2.6)), 0.155)
    col = nt.mix(nt.math('MULTIPLY', band, 0.55), col, (0.20, 0.215, 0.25, 1))
    fre = nt.n('ShaderNodeTexNoise', Vector=pos, Scale=6.0, Detail=4.0, Roughness=0.7)
    col = nt.mix(nt.math('MULTIPLY', nt.map_range(fre.outputs['Factor'], 0.62, 0.74), 0.5), col, (0.16, 0.17, 0.11, 1))
    h = nt.math('ADD', nt.math('MULTIPLY', saw, -0.8), nt.math('MULTIPLY', brick.outputs['Factor'], -0.6))
    nrm = nt.bump(h, strength=0.6, distance=0.02)
    rough = nt.map_range(big.outputs['Factor'], 0.3, 0.7, 0.34, 0.55)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=rough, Normal=nrm)
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


@material("roof_lead")
def _m_lead(name):
    m, nt, out = new_material(name)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.16, 0.17, 0.19, 1), Metallic=0.85, Roughness=0.48)
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


@material("metal_dark")
def _m_metal_dark(name):
    """Dark bronze / blackened steel: window mullions, lantern frames, railings."""
    m, nt, out = new_material(name)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.022, 0.020, 0.018, 1), Metallic=0.9, Roughness=0.42)
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


@material("gold")
def _m_gold(name):
    m, nt, out = new_material(name)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.95, 0.62, 0.22, 1), Metallic=1.0, Roughness=0.28)
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


@material("marble_wet")
def _m_marble_wet(name):
    """Polished, rain-wet terrace paving: large slabs, soft veins, puddle-like roughness variation."""
    m, nt, out = new_material(name)
    uv, pos, geo = _coords(nt)
    tile = nt.n('ShaderNodeTexBrick', Vector=uv, Color1=(0.47, 0.425, 0.365, 1), Color2=(0.385, 0.35, 0.305, 1),
                Mortar=(0.10, 0.09, 0.08, 1), Scale=1.0, Mortar_Size=0.006, Mortar_Smooth=0.2, Bias=0.0,
                Brick_Width=1.6, Row_Height=0.8)
    tile.offset = 0.5
    # palace paving: every other slab (checker on the 1.6 x 0.8 bond) in a darker, greyer marble
    su = nt.sep_xyz(uv)
    chk = nt.math('MODULO', nt.math('ADD', nt.math('FLOOR', nt.math('DIVIDE', nt.math('ADD', su[0], su[1]), 3.2)),
                                    nt.math('FLOOR', nt.math('DIVIDE', nt.math('SUBTRACT', su[0], su[1]), 3.2))), 2.0)
    chk = nt.math('ABSOLUTE', chk)
    base_col = nt.mix(nt.math('MULTIPLY', chk, 0.55), tile.outputs['Color'], (0.235, 0.225, 0.225, 1))
    vein_n = nt.n('ShaderNodeTexNoise', Vector=pos, Scale=0.55, Detail=7.0, Roughness=0.62, Distortion=1.6)
    vein = nt.map_range(nt.math('ABSOLUTE', nt.math('SUBTRACT', vein_n.outputs['Factor'], 0.5)), 0.0, 0.04, 1.0, 0.0)
    vein2_n = nt.n('ShaderNodeTexNoise', Vector=pos, Scale=2.3, Detail=5.0, Roughness=0.6, Distortion=2.2)
    vein2 = nt.map_range(nt.math('ABSOLUTE', nt.math('SUBTRACT', vein2_n.outputs['Factor'], 0.5)), 0.0, 0.02, 1.0, 0.0)
    col = nt.mix(nt.math('MULTIPLY', vein, 0.55), base_col, (0.17, 0.14, 0.115, 1))
    col = nt.mix(nt.math('MULTIPLY', vein2, 0.30), col, (0.78, 0.74, 0.66, 1))
    cloud = nt.n('ShaderNodeTexNoise', Vector=pos, Scale=0.22, Detail=3.0, Roughness=0.5)
    col = nt.mix(0.35, col, nt.ramp(cloud.outputs['Factor'], [(0.3, (0.75, 0.75, 0.76, 1)), (0.7, (1.12, 1.1, 1.06, 1))]),
                 blend='MULTIPLY')
    # wetness: puddles are darker + mirror-like
    wet_n = nt.n('ShaderNodeTexNoise', Vector=pos, Scale=0.16, Detail=4.0, Roughness=0.55)
    wet = nt.map_range(wet_n.outputs['Factor'], 0.38, 0.62, 0.0, 1.0, interp='SMOOTHSTEP')
    col = nt.mix(nt.math('MULTIPLY', wet, 0.45), col, (0.0, 0.0, 0.0, 1), blend='MIX')
    rough = nt.mixf(wet, 0.20, 0.035)
    h = nt.math('MULTIPLY', tile.outputs['Factor'], -1.0)
    nrm = nt.bump(h, strength=nt.mixf(wet, 0.5, 0.05), distance=0.004)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=rough, Normal=nrm)
    bsdf.inputs['Specular IOR Level'].default_value = 0.6
    bsdf.inputs['Coat Weight'].default_value = 0.0
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


@material("glass_warm")
def _m_glass_warm(name):
    """Lit window pane (opaque): warm interior glow that varies per window (face attribute 'rnd'),
    with blocky interior structure, plus a glossy surface that reflects the sky."""
    m, nt, out = new_material(name)
    uv, pos, geo = _coords(nt)
    rnd = nt.attr('rnd').outputs['Factor']
    hue = nt.ramp(nt.math('FRACT', nt.math('MULTIPLY', rnd, 7.31)),
                  [(0.0, (1.0, 0.36, 0.08, 1)), (0.5, (1.0, 0.50, 0.15, 1)), (1.0, (1.0, 0.66, 0.30, 1))])
    # interior seen through the glass: soft pools of light, darker toward the top (lamps sit low),
    # with faint vertical drapery folds - no hard shapes
    soft = nt.n('ShaderNodeTexNoise', Vector=uv, Scale=0.42, Detail=2.0, Roughness=0.45)
    fold_c = nt.n('ShaderNodeMapping', Vector=uv)
    fold_c.inputs['Scale'].default_value = (7.0, 0.35, 1.0)
    fold = nt.n('ShaderNodeTexNoise', Vector=fold_c.outputs[0], Scale=1.0, Detail=2.0)
    vgrad = nt.n('ShaderNodeTexNoise', Vector=uv, Scale=0.16, Detail=1.0)
    inner = nt.math('MULTIPLY', nt.map_range(soft.outputs['Factor'], 0.28, 0.72, 0.5, 1.3),
                    nt.map_range(fold.outputs['Factor'], 0.35, 0.65, 0.82, 1.1))
    inner = nt.math('MULTIPLY', inner, nt.map_range(vgrad.outputs['Factor'], 0.3, 0.7, 0.75, 1.15))
    # per window brightness; ~14% of the windows are dark
    lit = nt.math('GREATER_THAN', rnd, 0.14)
    bright = nt.map_range(nt.math('FRACT', nt.math('MULTIPLY', rnd, 3.17)), 0.0, 1.0, 0.65, 1.9)
    strength = nt.math('MULTIPLY', nt.math('MULTIPLY', bright, inner), lit)
    strength = nt.math('ADD', strength, 0.02)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.01, 0.012, 0.016, 1), Roughness=0.06,
                Emission_Color=hue, Emission_Strength=strength)
    bsdf.inputs['Specular IOR Level'].default_value = 0.8
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


@material("glass_clear")
def _m_glass_clear(name):
    """See-through pane for the great window / glazed pavilions: mostly transparent, fresnel sky reflection."""
    m, nt, out = new_material(name)
    fres = nt.n('ShaderNodeFresnel', IOR=1.5)
    f = nt.map_range(fres.outputs[0], 0.0, 1.0, 0.10, 1.0)
    glossy = nt.n('ShaderNodeBsdfGlossy', Color=(1, 1, 1, 1), Roughness=0.02)
    transp = nt.n('ShaderNodeBsdfTransparent', Color=(0.93, 0.90, 0.86, 1))
    mix = nt.n('ShaderNodeMixShader', i0=f, i1=transp.outputs[0], i2=glossy.outputs[0])
    nt.link(mix.outputs[0], out.inputs['Surface'])
    m.surface_render_method = 'BLENDED'
    try:
        m.use_transparency_overlap = False
    except Exception:
        pass
    return m


@material("interior_glow")
def _m_interior(name):
    """Emissive interior surfaces seen through the great window (warm, brighter low down)."""
    m, nt, out = new_material(name)
    uv, pos, geo = _coords(nt)
    z = nt.sep_xyz(pos)[2]
    grad = nt.map_range(z, 2.0, 34.0, 1.0, 0.25)
    soft = nt.n('ShaderNodeTexNoise', Vector=pos, Scale=0.09, Detail=2.0, Roughness=0.4)
    # vertical pilaster rhythm so the walls read as architecture, not as a flat glow
    u = nt.sep_xyz(uv)[0]
    pil = nt.map_range(nt.math('ABSOLUTE', nt.math('SUBTRACT', nt.math('FRACT', nt.math('DIVIDE', u, 3.2)), 0.5)), 0.38, 0.46, 1.0, 0.45)
    s = nt.math('MULTIPLY', nt.math('MULTIPLY', nt.math('MULTIPLY', grad, pil), nt.map_range(soft.outputs['Factor'], 0.3, 0.7, 0.7, 1.2)), 1.5)
    em = nt.n('ShaderNodeEmission', Color=(1.0, 0.50, 0.17, 1), Strength=s)
    nt.link(em.outputs[0], out.inputs['Surface'])
    return m


@material("lamp_glow")
def _m_lamp_glow(name):
    """Small very bright warm emitter: candle flames, bulbs, chandelier lights."""
    m, nt, out = new_material(name)
    em = nt.n('ShaderNodeEmission', Color=(1.0, 0.58, 0.22, 1), Strength=9.0)
    nt.link(em.outputs[0], out.inputs['Surface'])
    return m


@material("lamp_glass")
def _m_lamp_glass(name):
    """Frosted lantern glass lit from inside."""
    m, nt, out = new_material(name)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.9, 0.7, 0.45, 1), Roughness=0.25,
                Emission_Color=(1.0, 0.52, 0.18, 1), Emission_Strength=2.2)
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


@material("pool_tile")
def _m_pool_tile(name):
    """Glowing turquoise pool mosaic (underwater lighting is faked with emission)."""
    m, nt, out = new_material(name)
    uv, pos, geo = _coords(nt)
    tile = nt.n('ShaderNodeTexBrick', Vector=uv, Color1=(0.03, 0.42, 0.50, 1), Color2=(0.05, 0.55, 0.60, 1),
                Mortar=(0.02, 0.18, 0.22, 1), Scale=1.0, Mortar_Size=0.01, Brick_Width=0.25, Row_Height=0.25)
    tile.offset = 0.0
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=tile.outputs['Color'], Roughness=0.2,
                Emission_Color=tile.outputs['Color'], Emission_Strength=2.2)
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


@material("fabric_black")
def _m_fabric_black(name):
    m, nt, out = new_material(name)
    uv, pos, geo = _coords(nt)
    weave = nt.n('ShaderNodeTexNoise', Vector=pos, Scale=60.0, Detail=2.0)
    nrm = nt.bump(weave.outputs['Factor'], strength=0.15, distance=0.002)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.012, 0.012, 0.016, 1), Roughness=0.7, Normal=nrm)
    bsdf.inputs['Sheen Weight'].default_value = 0.4
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


@material("fabric_cream")
def _m_fabric_cream(name):
    m, nt, out = new_material(name)
    uv, pos, geo = _coords(nt)
    weave = nt.n('ShaderNodeTexNoise', Vector=pos, Scale=90.0, Detail=2.0)
    nrm = nt.bump(weave.outputs['Factor'], strength=0.2, distance=0.002)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.62, 0.57, 0.50, 1), Roughness=0.85, Normal=nrm)
    bsdf.inputs['Sheen Weight'].default_value = 0.3
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


@material("wood_dark")
def _m_wood_dark(name):
    m, nt, out = new_material(name)
    uv, pos, geo = _coords(nt)
    mp = nt.n('ShaderNodeMapping', Vector=pos)
    mp.inputs['Scale'].default_value = (1.0, 1.0, 12.0)
    grain = nt.n('ShaderNodeTexNoise', Vector=mp.outputs[0], Scale=3.0, Detail=4.0, Distortion=0.6)
    col = nt.ramp(grain.outputs['Factor'], [(0.3, (0.05, 0.028, 0.014, 1)), (0.7, (0.13, 0.075, 0.035, 1))])
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.45,
                Normal=nt.bump(grain.outputs['Factor'], 0.15, 0.003))
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m









@material("curtain")
def _m_curtain(name):
    """Heavy drapery seen against a lit room: dark, with vertical folds; colour varies per window ('rnd')."""
    m, nt, out = new_material(name)
    uv, pos, geo = _coords(nt)
    rnd = nt.attr('rnd').outputs['Factor']
    u = nt.sep_xyz(uv)[0]
    fold = nt.math('SINE', nt.math('MULTIPLY', u, 46.0))
    fold2 = nt.n('ShaderNodeTexNoise', Vector=uv, Scale=3.0, Detail=2.0)
    f = nt.map_range(nt.math('ADD', fold, nt.math('MULTIPLY', fold2.outputs['Factor'], 1.2)), -1.0, 2.2, 0.35, 1.0)
    base = nt.ramp(nt.math('FRACT', nt.math('MULTIPLY', rnd, 5.77)),
                   [(0.0, (0.22, 0.035, 0.03, 1)), (0.35, (0.30, 0.20, 0.10, 1)), (0.7, (0.10, 0.11, 0.16, 1)), (1.0, (0.33, 0.26, 0.17, 1))],
                   interp='CONSTANT')
    col = nt.mix(1.0, base, nt.n('ShaderNodeCombineColor', Red=f, Green=f, Blue=f).outputs[0], blend='MULTIPLY')
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.85, Emission_Color=col,
                Emission_Strength=nt.math('MULTIPLY', nt.math('GREATER_THAN', rnd, 0.14), 0.9))
    bsdf.inputs['Sheen Weight'].default_value = 0.3
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


@material("stained_glass")
def _m_stained(name):
    """Glowing stained glass (seen at the far end of the great hall): jewel-toned panes with dark leading."""
    m, nt, out = new_material(name)
    uv, pos, geo = _coords(nt)
    cells = nt.n('ShaderNodeTexVoronoi', Vector=uv, Scale=3.6, Randomness=0.9)
    cells.feature = 'F1'
    edge = nt.n('ShaderNodeTexVoronoi', Vector=uv, Scale=3.6, Randomness=0.9)
    edge.feature = 'DISTANCE_TO_EDGE'
    hue = nt.n('ShaderNodeSeparateColor', Color=cells.outputs['Color']).outputs[0]
    col = nt.ramp(hue, [(0.0, (0.70, 0.01, 0.02, 1)), (0.24, (0.02, 0.06, 0.75, 1)), (0.46, (0.90, 0.42, 0.02, 1)),
                        (0.62, (0.01, 0.38, 0.12, 1)), (0.78, (0.34, 0.02, 0.52, 1)), (0.92, (0.90, 0.72, 0.30, 1))], interp='CONSTANT')
    lead = nt.map_range(edge.outputs['Distance'], 0.012, 0.03, 0.0, 1.0)
    # stone mullions every 1.15 m and a transom rhythm
    su = nt.sep_xyz(uv)
    mul = nt.map_range(nt.math('ABSOLUTE', nt.math('SUBTRACT', nt.math('FRACT', nt.math('DIVIDE', su[0], 1.15)), 0.5)), 0.44, 0.47, 1.0, 0.0)
    tra = nt.map_range(nt.math('ABSOLUTE', nt.math('SUBTRACT', nt.math('FRACT', nt.math('DIVIDE', su[1], 2.3)), 0.5)), 0.46, 0.48, 1.0, 0.0)
    glow = nt.n('ShaderNodeTexNoise', Vector=uv, Scale=0.3, Detail=1.0)
    s = nt.math('MULTIPLY', nt.math('MULTIPLY', nt.math('MULTIPLY', lead, mul), tra),
                nt.map_range(glow.outputs['Factor'], 0.3, 0.7, 0.9, 2.3))
    em = nt.n('ShaderNodeEmission', Color=col, Strength=s)
    nt.link(em.outputs[0], out.inputs['Surface'])
    return m




