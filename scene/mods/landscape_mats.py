"""Materials of the landscape module (registered in scene.mats under the names ls_*).

    ls_terrain   heightfield: meadow / forest carpet / rock / scree / snow / shore, driven by slope, height
                 and the point attribute "lmask" (R forest, G shore, B talus).  Alpine zone: multi-scale
                 ridged relief, fall-line ribs / couloirs, strata, snow broken up by slope / aspect / altitude.
                 Every detail scale fades out before it becomes sub-pixel (no shimmer).
    ls_cliff     hero cliff rock: bedding planes, vertical joints, blocky fracture, stains, ledge vegetation
                 (attribute "cmask": R cavity, G waterfall wetness, B depth)
    ls_water     lake: rippled mirror with wind patches
(mist + distant trees: see landscape_mist.py).  All of them end in atmo.fog().

Tunables: FOG_TERRAIN (how much aerial perspective the terrain gets), the colour ramps and bump heights below.
"""
import bpy, math, os
from .. import mats, atmo, layout as L
from ..common import new_material, drive_frame
from . import landscape_field as F

FOG_TERRAIN = 0.55        # haze density multiplier for terrain (1 = shared default; lower keeps contrast)
FOG_CLIFF = 0.8


def _noise(nt, vec, wavelength, detail=3.0, rough=0.55, w=None, **kw):
    n = nt.n('ShaderNodeTexNoise', Vector=vec, Scale=1.0 / wavelength, Detail=detail, Roughness=rough, **kw)
    if w is not None:
        n.noise_dimensions = '4D'
        nt.set(n.inputs['W'], w)
    return n.outputs['Factor']


def _time(nt, expr="frame/24"):
    v = nt.n('ShaderNodeValue')
    drive_frame(v.outputs[0], 'default_value', expr)
    return v.outputs[0]


def _ridge(nt, n):
    """0..1 noise -> ridged 0..1 (1 on the crease lines)."""
    return nt.math('SUBTRACT', 1.0, nt.math('ABSOLUTE', nt.math('SUBTRACT', nt.math('MULTIPLY', n, 2.0), 1.0)))


def _mul(nt, *a):
    r = a[0]
    for b in a[1:]:
        r = nt.math('MULTIPLY', r, b)
    return r


def _add(nt, *a):
    r = a[0]
    for b in a[1:]:
        r = nt.math('ADD', r, b)
    return r


def _steps(nt, v, n):
    """Quantise v*n into soft steps.  Returns (stepped value in units of steps, line mask at the step edges,
    integer step index)."""
    t = nt.math('MULTIPLY', v, n)
    fl = nt.math('FLOOR', t)
    fr = nt.math('FRACT', t)
    st = nt.math('ADD', fl, nt.map_range(fr, 0.80, 1.0, 0.0, 1.0, interp='SMOOTHSTEP'))
    edge = nt.math('MINIMUM', fr, nt.math('SUBTRACT', 1.0, fr))
    line = nt.map_range(edge, 0.0, 0.13, 1.0, 0.0)
    return st, line, fl


# =====================================================================================
@mats.material("ls_terrain")
def _m_terrain(name):
    m, nt, out = new_material(name)
    if os.environ.get("LS_PLAIN"):          # timing aid: flat material (to measure what the shading costs)
        bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.08, 0.08, 0.07, 1), Roughness=0.9)
        nt.link(atmo.fog(nt, bsdf.outputs[0], density_mul=FOG_TERRAIN), out.inputs['Surface'])
        return m
    geo = nt.n('ShaderNodeNewGeometry')
    pos = geo.outputs['Position']
    sx = nt.sep_xyz(pos)
    z = sx[2]
    nxyz = nt.sep_xyz(geo.outputs['Normal'])
    nz = nxyz[2]
    dist = nt.n('ShaderNodeCameraData').outputs['View Distance']
    lm = nt.n('ShaderNodeSeparateColor', Color=nt.attr('lmask').outputs['Color'])
    a_forest, a_shore, a_talus = lm.outputs[0], lm.outputs[1], lm.outputs[2]

    # ---- distance fades (a detail scale is switched off before it gets smaller than ~2 pixels)
    near = nt.map_range(dist, 60.0, 400.0, 1.0, 0.0)
    f9 = nt.map_range(dist, 1800.0, 4500.0, 1.0, 0.0)            # ~9 m detail (tree crowns)
    f20 = nt.map_range(dist, 6000.0, 14000.0, 1.0, 0.0)          # ~20 m detail
    f60 = nt.map_range(dist, 16000.0, 34000.0, 1.0, 0.25)        # ~60 m detail
    farf = nt.map_range(dist, 2500.0, 9000.0, 0.0, 1.0)

    # ---- noises
    n_big = _noise(nt, pos, 520.0, 3.0)
    n_med = _noise(nt, pos, 42.0, 4.0, 0.6)
    n_fine = _noise(nt, pos, 2.6, 3.0, 0.6)
    n_grain = _noise(nt, pos, 0.22, 2.0, 0.6)
    rA = _ridge(nt, _noise(nt, pos, 330.0, 4.0, 0.62))            # crags 330 .. 20 m
    rB = _ridge(nt, _noise(nt, pos, 46.0, 3.0, 0.62))             # crags 46 .. 6 m
    # fall-line streaks: a coordinate that is constant along the line of steepest descent
    tl = nt.math('MAXIMUM', nt.math('SQRT', nt.math('ADD', nt.math('POWER', nxyz[0], 2.0),
                                                    nt.math('POWER', nxyz[1], 2.0))), 0.02)
    ucoord = nt.math('DIVIDE', nt.math('SUBTRACT', nt.math('MULTIPLY', sx[0], nxyz[1]),
                                       nt.math('MULTIPLY', sx[1], nxyz[0])), tl)
    sN = _noise(nt, nt.comb_xyz(ucoord, nt.math('MULTIPLY', z, 0.16), nt.math('MULTIPLY', n_big, 300.0)),
                95.0, 3.0, 0.6)
    ribs = _ridge(nt, sN)                                          # 1 on rib crests

    # ---- masks
    nzj = _add(nt, nz, nt.math('MULTIPLY', nt.math('SUBTRACT', n_med, 0.5), 0.12),
               nt.math('MULTIPLY', nt.math('SUBTRACT', rA, 0.5), 0.20))
    steep = nt.map_range(nzj, 0.60, 0.80, 1.0, 0.0, interp='SMOOTHSTEP')
    zj = _add(nt, z, nt.math('MULTIPLY', nt.math('SUBTRACT', n_big, 0.5), 480.0),
              nt.math('MULTIPLY', nt.math('SUBTRACT', rA, 0.5), 260.0))
    alpine = nt.map_range(zj, F.TREELINE_Z - 60.0, F.TREELINE_Z + 260.0, 0.0, 1.0)
    # snow: altitude, slope limit that relaxes with altitude, more on north-east aspects, couloirs on steep
    # faces, rock ribs showing through
    aspect = nt.math('ADD', nt.math('MULTIPLY', nxyz[0], 0.45), nt.math('MULTIPLY', nxyz[1], 0.80))
    snow_h = nt.map_range(nt.math('ADD', zj, nt.math('MULTIPLY', aspect, 170.0)),
                          F.SNOWLINE_Z - 130.0, F.SNOWLINE_Z + 200.0, 0.0, 1.0, interp='SMOOTHSTEP')
    snow_lim = nt.map_range(zj, F.SNOWLINE_Z, F.SNOWLINE_Z + 900.0, 0.74, 0.46)
    snow_s = nt.map_range(nt.math('SUBTRACT', nzj, snow_lim), -0.02, 0.10, 0.0, 1.0, interp='SMOOTHSTEP')
    gully = nt.map_range(sN, 0.56, 0.70, 0.0, 1.0, interp='SMOOTHSTEP')
    snow = nt.math('MAXIMUM', nt.math('MULTIPLY', snow_h, snow_s), _mul(nt, snow_h, gully, 0.92))
    rib_rock = _mul(nt, nt.map_range(ribs, 0.72, 0.95, 0.0, 1.0), nt.map_range(nzj, 0.90, 0.60, 0.0, 1.0), 0.85)
    snow = nt.math('MULTIPLY', snow, nt.math('SUBTRACT', 1.0, rib_rock))
    snow = nt.math('MULTIPLY', snow, nt.map_range(rB, 0.80, 0.97, 1.0, nt.mixf(f20, 1.0, 0.35)))  # small outcrops
    fj = nt.math('ADD', a_forest, nt.math('MULTIPLY', nt.math('SUBTRACT', n_med, 0.5), 0.55))
    forest = nt.map_range(fj, 0.36, 0.60, 0.0, 1.0, interp='SMOOTHSTEP')
    forest = nt.math('MULTIPLY', forest, nt.math('SUBTRACT', 1.0, steep))
    # avalanche chutes / gullies cut light stripes through the forest on the mountain flanks
    chute = _mul(nt, nt.map_range(sN, 0.60, 0.72, 0.0, 1.0, interp='SMOOTHSTEP'),
                 nt.map_range(z, -120.0, 140.0, 0.0, 1.0), nt.map_range(nzj, 0.94, 0.80, 0.0, 1.0))
    forest = nt.math('MULTIPLY', forest, nt.math('SUBTRACT', 1.0, nt.math('MULTIPLY', chute, 0.85)))
    # krummholz / dwarf pine patches above the tree line
    krum = _mul(nt, nt.map_range(zj, F.TREELINE_Z - 40.0, F.TREELINE_Z + 300.0, 1.0, 0.0),
                nt.map_range(zj, F.TREELINE_Z - 150.0, F.TREELINE_Z - 20.0, 0.0, 1.0),
                nt.map_range(nt.math('ADD', n_med, nt.math('MULTIPLY', rA, 0.4)), 0.55, 0.75, 0.0, 1.0),
                nt.math('SUBTRACT', 1.0, steep))

    # ---- colours
    grass = nt.ramp(n_fine, [(0.25, (0.030, 0.052, 0.014, 1)), (0.55, (0.058, 0.080, 0.022, 1)),
                             (0.8, (0.095, 0.100, 0.034, 1))])
    grass = nt.mix(nt.map_range(n_med, 0.3, 0.7, 0.0, 0.6), grass, (0.105, 0.088, 0.040, 1))
    grass = nt.mix(nt.math('MULTIPLY', near, 0.55), grass,
                   nt.mix(n_grain, (0.022, 0.034, 0.010, 1), (0.105, 0.115, 0.040, 1)))
    grass = nt.mix(alpine, grass, nt.mix(rA, (0.060, 0.058, 0.036, 1), (0.130, 0.110, 0.066, 1)))
    # conifer carpet: crowns from a voronoi (near), clump texture further away
    vor = nt.n('ShaderNodeTexVoronoi', Vector=pos, Scale=1.0 / 9.0, Randomness=1.0)
    crown = nt.map_range(vor.outputs['Distance'], 0.0, 0.62, 1.0, 0.0)
    crown = nt.mixf(f9, 0.55, crown)
    fcol = nt.mix(crown, (0.004, 0.009, 0.005, 1), (0.022, 0.042, 0.016, 1))
    clump = nt.math('ADD', nt.math('MULTIPLY', n_med, 0.6), nt.math('MULTIPLY', rB, 0.4))
    fcol = nt.mix(1.0, fcol, nt.ramp(clump, [(0.25, (0.45, 0.48, 0.45, 1)), (0.5, (0.95, 0.95, 0.90, 1)),
                                             (0.75, (1.45, 1.40, 1.15, 1))]), blend='MULTIPLY')
    fcol = nt.mix(nt.map_range(n_big, 0.3, 0.7, 0.0, 0.5), fcol, (0.034, 0.042, 0.014, 1))
    q = nt.math('SQRT', nt.math('ADD', nt.math('POWER', nt.math('SUBTRACT', sx[0], F.CX), 2.0),
                                nt.math('POWER', nt.math('SUBTRACT', sx[1], F.CY), 2.0)))
    canf = nt.map_range(q, 720.0, 1000.0, 0.0, 1.0, interp='SMOOTHSTEP')
    floorc = nt.mix(n_fine, (0.020, 0.025, 0.011, 1), (0.052, 0.046, 0.024, 1))
    floorc = nt.mix(nt.map_range(n_med, 0.35, 0.65, 0.0, 0.7), floorc, (0.030, 0.045, 0.014, 1))
    fcol = nt.mix(canf, floorc, fcol)
    # rock: strata bands (warped along z), fracture tone from the ridged fields
    strat = _noise(nt, nt.comb_xyz(nt.math('MULTIPLY', n_big, 60.0), nt.math('MULTIPLY', rA, 25.0), z), 34.0, 3.0, 0.7)
    tone = _add(nt, nt.math('MULTIPLY', strat, 0.45), nt.math('MULTIPLY', rA, 0.30), nt.math('MULTIPLY', rB, 0.25))
    rock = nt.ramp(tone, [(0.22, (0.034, 0.032, 0.031, 1)), (0.42, (0.088, 0.080, 0.074, 1)),
                          (0.60, (0.150, 0.136, 0.120, 1)), (0.80, (0.250, 0.226, 0.196, 1))])
    rock = nt.mix(nt.map_range(n_big, 0.45, 0.75, 0.0, 0.35), rock, (0.150, 0.110, 0.078, 1))
    scree = nt.mix(nt.mixf(near, rB, n_fine), (0.080, 0.076, 0.070, 1), (0.180, 0.168, 0.150, 1))
    shorec = nt.mix(n_fine, (0.030, 0.027, 0.022, 1), (0.070, 0.062, 0.050, 1))
    snowc = nt.mix(nt.math('ADD', nt.math('MULTIPLY', rA, 0.6), nt.math('MULTIPLY', rB, 0.4)),
                   (0.56, 0.61, 0.72, 1), (0.90, 0.90, 0.92, 1))
    krumc = nt.mix(rB, (0.008, 0.016, 0.008, 1), (0.030, 0.045, 0.020, 1))

    col = nt.mix(nt.math('MULTIPLY', a_talus, nt.map_range(n_med, 0.3, 0.6, 0.35, 1.0)), grass, scree)
    # scree fans below steep rock in the alpine zone
    fan = _mul(nt, alpine, nt.map_range(nzj, 0.70, 0.84, 1.0, 0.0), nt.map_range(sN, 0.35, 0.6, 1.0, 0.0))
    col = nt.mix(fan, col, scree)
    col = nt.mix(krum, col, krumc)
    col = nt.mix(forest, col, fcol)
    col = nt.mix(steep, col, rock)
    col = nt.mix(nt.math('MULTIPLY', a_shore, nt.math('SUBTRACT', 1.0, forest)), col, shorec)
    col = nt.mix(snow, col, snowc)

    # ---- relief (heights in metres, one bump node)
    rocky = nt.math('MAXIMUM', nt.math('MAXIMUM', steep, alpine), nt.math('MULTIPLY', farf, 0.45))
    h_can = _mul(nt, nt.math('ADD', nt.math('MULTIPLY', crown, nt.mixf(f9, 0.0, 5.0)),
                             nt.math('MULTIPLY', clump, nt.mixf(farf, 3.0, 9.0))), forest, canf)
    h_near = nt.math('MULTIPLY', nt.math('ADD', nt.math('MULTIPLY', n_fine, 0.35),
                                         nt.math('MULTIPLY', n_grain, 0.06)), near)
    h_rock = _mul(nt, _add(nt, nt.math('MULTIPLY', rA, 42.0),
                           _mul(nt, rB, 9.0, f20),
                           _mul(nt, ribs, 11.0, f60, nt.map_range(nzj, 0.95, 0.70, 0.0, 1.0),
                                nt.map_range(n_big, 0.3, 0.7, 0.45, 1.0)),
                           _mul(nt, strat, 10.0, f60, steep)), rocky)
    h_snow = _mul(nt, snow, nt.math('ADD', nt.math('MULTIPLY', n_med, 1.5), nt.math('MULTIPLY', gully, -4.0)), f20)
    hsum = _add(nt, h_can, h_near, h_rock, h_snow)
    nrm = nt.bump(hsum, strength=1.0, distance=1.0)
    rough = nt.mixf(snow, 0.92, 0.58)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=rough, Normal=nrm)
    bsdf.inputs['Specular IOR Level'].default_value = 0.22
    nt.link(atmo.fog(nt, bsdf.outputs[0], density_mul=FOG_TERRAIN), out.inputs['Surface'])
    return m


# =====================================================================================
@mats.material("ls_water")
def _m_water(name):
    m, nt, out = new_material(name)
    geo = nt.n('ShaderNodeNewGeometry')
    pos = geo.outputs['Position']
    dist = nt.n('ShaderNodeCameraData').outputs['View Distance']
    t = _time(nt)
    # slow wind drift
    drift = nt.comb_xyz(nt.math('MULTIPLY', t, 0.9), nt.math('MULTIPLY', t, -0.35), 0.0)
    p = nt.vmath('ADD', pos, drift)
    patch = _noise(nt, p, 330.0, 3.0, 0.55, w=nt.math('MULTIPLY', t, 0.015))
    calm = nt.map_range(patch, 0.38, 0.62, 0.0, 1.0, interp='SMOOTHSTEP')          # 0 = glassy, 1 = ruffled
    # wind streaks (elongated along the wind) break up the big patches
    mp = nt.n('ShaderNodeMapping', Vector=p)
    mp.inputs['Rotation'].default_value = (0.0, 0.0, math.radians(-21.0))
    mp.inputs['Scale'].default_value = (1.0 / 420.0, 1.0 / 38.0, 1.0)
    streak = nt.n('ShaderNodeTexNoise', Vector=mp.outputs[0], Scale=1.0, Detail=2.0, Roughness=0.5).outputs['Factor']
    calm = nt.math('MULTIPLY', calm, nt.map_range(streak, 0.35, 0.65, 0.55, 1.0), clamp=True)
    rip1 = _noise(nt, p, 2.4, 2.0, 0.5, w=nt.math('MULTIPLY', t, 0.55))
    rip2 = _noise(nt, p, 14.0, 3.0, 0.55, w=nt.math('MULTIPLY', t, 0.16))
    rip3 = _noise(nt, p, 70.0, 2.0, 0.5, w=nt.math('MULTIPLY', t, 0.05))
    h = nt.math('ADD', nt.math('MULTIPLY', rip1, nt.mixf(calm, 0.004, 0.03)),
                nt.math('ADD', nt.math('MULTIPLY', rip2, nt.mixf(calm, 0.02, 0.10)), nt.math('MULTIPLY', rip3, 0.12)))
    nrm = nt.bump(h, strength=nt.map_range(dist, 300.0, 5000.0, 1.0, 0.45), distance=1.0)
    rough = nt.math('ADD', nt.mixf(calm, 0.012, 0.10), nt.map_range(dist, 800.0, 9000.0, 0.0, 0.08))
    col = nt.mix(calm, (0.0035, 0.010, 0.014, 1), (0.006, 0.016, 0.020, 1))
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=rough, IOR=1.333, Normal=nrm)
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
    nt.link(atmo.fog(nt, bsdf.outputs[0], density_mul=0.8), out.inputs['Surface'])
    return m


# =====================================================================================
@mats.material("ls_cliff")
def _m_cliff(name):
    """Layered limestone: beds (coarse + fine) cut by vertical joints into blocks, every block with its own
    tone and relief; seams and joints as thin dark lines; stains, moss on ledges.  The fine systems fade
    out with distance so nothing shimmers."""
    m, nt, out = new_material(name)
    if os.environ.get("LS_PLAIN"):
        bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.08, 0.08, 0.07, 1), Roughness=0.9)
        nt.link(atmo.fog(nt, bsdf.outputs[0], density_mul=FOG_CLIFF), out.inputs['Surface'])
        return m
    geo = nt.n('ShaderNodeNewGeometry')
    pos = geo.outputs['Position']
    sx = nt.sep_xyz(pos)
    z = sx[2]
    nz = nt.sep_xyz(geo.outputs['Normal'])[2]
    dist = nt.n('ShaderNodeCameraData').outputs['View Distance']
    cm = nt.n('ShaderNodeSeparateColor', Color=nt.attr('cmask').outputs['Color'])
    cav, wet_a, depth = cm.outputs[0], cm.outputs[1], cm.outputs[2]
    fA = nt.map_range(dist, 90.0, 300.0, 1.0, 0.0)               # 0.2 - 0.6 m detail
    fB = nt.map_range(dist, 300.0, 900.0, 1.0, 0.0)              # 1 - 3 m detail
    dry = nt.math('SUBTRACT', 1.0, wet_a)

    n1 = _noise(nt, pos, 16.0, 3.0, 0.6)
    n2 = _noise(nt, pos, 1.7, 4.0, 0.62)
    n3 = _noise(nt, pos, 0.28, 3.0, 0.6)
    n_big = _noise(nt, pos, 75.0, 3.0, 0.55)
    # bedding coordinate: same dip / warp idea as the geometry beds
    zs = _add(nt, z, nt.math('MULTIPLY', n1, 5.0), nt.math('MULTIPLY', sx[0], 0.04), nt.math('MULTIPLY', sx[1], 0.03))
    bedv = nt.comb_xyz(nt.math('MULTIPLY', sx[0], 0.035), nt.math('MULTIPLY', sx[1], 0.035), zs)
    b_c = _noise(nt, bedv, 9.0, 2.0, 0.5)                         # coarse beds  (~1.3 m)
    b_f = _noise(nt, bedv, 2.3, 2.0, 0.5)                         # fine beds    (~0.3 m)
    st_c, line_c, i_c = _steps(nt, b_c, 7.0)
    st_f, line_f, i_f = _steps(nt, b_f, 7.0)
    # vertical joints: noise that hardly changes with z, shifted from bed to bed so blocks are staggered
    jv = nt.comb_xyz(nt.math('ADD', sx[0], nt.math('MULTIPLY', i_c, 3.7)),
                     nt.math('ADD', sx[1], nt.math('MULTIPLY', i_c, 2.3)), nt.math('MULTIPLY', z, 0.06))
    j_c = _noise(nt, jv, 5.0, 2.0, 0.5)
    st_j, line_j, i_j = _steps(nt, j_c, 6.0)
    blk = nt.n('ShaderNodeTexWhiteNoise', Vector=nt.comb_xyz(i_c, i_j, 0.0))
    blk.noise_dimensions = '3D'
    brnd = blk.outputs['Value']
    blk2 = nt.n('ShaderNodeTexWhiteNoise', Vector=nt.comb_xyz(i_c, 3.0, 7.0))
    blk2.noise_dimensions = '3D'
    bed_rnd = blk2.outputs['Value']

    # ---- colour
    tone = _add(nt, nt.math('MULTIPLY', bed_rnd, 0.40), nt.math('MULTIPLY', n2, 0.35), nt.math('MULTIPLY', brnd, 0.25))
    col = nt.ramp(tone, [(0.20, (0.046, 0.043, 0.040, 1)), (0.42, (0.100, 0.093, 0.085, 1)),
                         (0.60, (0.160, 0.148, 0.132, 1)), (0.82, (0.265, 0.244, 0.214, 1))])
    col = nt.mix(nt.map_range(n_big, 0.40, 0.72, 0.0, 0.40), col, (0.150, 0.112, 0.080, 1))      # ochre zones
    # seams between beds and joints between blocks: thin, dark, irregular
    lines = nt.math('MAXIMUM', nt.math('MULTIPLY', line_c, 0.85),
                    nt.math('MAXIMUM', _mul(nt, line_f, fA, 0.6), _mul(nt, line_j, fB, 0.75)))
    lines = _mul(nt, lines, nt.map_range(n2, 0.30, 0.55, 0.25, 1.0), dry)
    col = nt.mix(nt.math('MULTIPLY', lines, 0.72), col, (0.016, 0.015, 0.014, 1))
    # cavities darker, protruding parts lighter (from the displacement)
    col = nt.mix(1.0, col, nt.ramp(cav, [(0.22, (0.34, 0.34, 0.36, 1)), (0.48, (0.82, 0.82, 0.82, 1)),
                                         (0.72, (1.18, 1.16, 1.12, 1))]), blend='MULTIPLY')
    # stains: dark wet streaks + pale lime runs down the face
    mp = nt.n('ShaderNodeMapping', Vector=pos)
    mp.inputs['Scale'].default_value = (1.0 / 3.2, 1.0 / 3.2, 1.0 / 85.0)
    stn = nt.n('ShaderNodeTexNoise', Vector=mp.outputs[0], Scale=1.0, Detail=3.0, Roughness=0.6).outputs['Factor']
    streak = nt.map_range(stn, 0.55, 0.74, 0.0, 1.0, interp='SMOOTHSTEP')
    pale = nt.map_range(stn, 0.40, 0.26, 0.0, 0.5, interp='SMOOTHSTEP')
    wet = nt.math('MAXIMUM', nt.math('MULTIPLY', streak, 0.5),
                  nt.math('MULTIPLY', wet_a, nt.map_range(n1, 0.2, 0.7, 0.75, 1.0)), clamp=True)
    col = nt.mix(pale, col, (0.30, 0.28, 0.25, 1))
    col = nt.mix(nt.math('MULTIPLY', wet, 0.62), col, (0.012, 0.012, 0.012, 1))
    # vegetation: moss / grass on ledges and gentle parts, and creeping along the big seams
    up = nt.math('ADD', nz, nt.math('MULTIPLY', nt.math('SUBTRACT', n2, 0.5), 0.35))
    moss = nt.math('MULTIPLY', nt.map_range(up, 0.34, 0.68, 0.0, 1.0, interp='SMOOTHSTEP'),
                   nt.map_range(n1, 0.25, 0.55, 0.35, 1.0))
    moss = nt.math('MAXIMUM', moss, _mul(nt, line_c, nt.map_range(n1, 0.5, 0.7, 0.0, 0.8), dry))
    mossc = nt.mix(n3, (0.016, 0.032, 0.010, 1), (0.060, 0.082, 0.024, 1))
    col = nt.mix(moss, col, mossc)
    lich = _mul(nt, nt.map_range(_noise(nt, pos, 0.9, 2.0, 0.7), 0.62, 0.75, 0.0, 0.5), dry, fB)
    col = nt.mix(lich, col, (0.21, 0.20, 0.12, 1))

    # ---- relief (metres): beds step in and out, blocks have their own depth, seams are recessed
    fC = nt.map_range(dist, 40.0, 140.0, 1.0, 0.0)               # centimetre detail, only close up
    n4 = _noise(nt, pos, 0.07, 2.0, 0.6)
    pit = _mul(nt, nt.map_range(n3, 0.60, 0.72, 0.0, 1.0), fA, dry)
    col = nt.mix(nt.math('MULTIPLY', pit, 0.45), col, (0.020, 0.018, 0.016, 1))
    col = nt.mix(nt.math('MULTIPLY', fC, 0.30), col,
                 nt.mix(1.0, col, nt.ramp(n4, [(0.3, (0.70, 0.70, 0.70, 1)), (0.7, (1.30, 1.28, 1.22, 1))]),
                        blend='MULTIPLY'))
    h = _add(nt, _mul(nt, bed_rnd, 0.75),
             _mul(nt, brnd, 0.38, fB),
             _mul(nt, nt.math('FRACT', nt.math('MULTIPLY', st_f, 0.37)), 0.24, fA),
             nt.math('MULTIPLY', lines, -0.20),
             nt.math('MULTIPLY', n2, 0.34),
             _mul(nt, n3, 0.11, fA),
             _mul(nt, pit, -0.06),
             _mul(nt, n4, 0.018, fC))
    nrm = nt.bump(h, strength=1.0, distance=1.0)
    rough = nt.mixf(wet, 0.88, 0.45)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=rough, Normal=nrm)
    bsdf.inputs['Specular IOR Level'].default_value = 0.35
    nt.link(atmo.fog(nt, bsdf.outputs[0], density_mul=FOG_CLIFF), out.inputs['Surface'])
    return m
