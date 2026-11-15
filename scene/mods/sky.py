"""SKY module: procedural sunset world shader (gradient + sun + three animated cloud layers) and the sun lamp.

What it builds
  * World "SunsetSky": ONE Background shader, no image textures, no volumes.
      - clear-sky gradient: blue-violet zenith -> mauve / salmon / orange mid sky (ramp on the angle to the sun)
        -> haze band on the horizon that reproduces scene/atmo.py's haze exactly
        (mix(HAZE_AWAY, HAZE_TOWARD_SUN, sun_amt) * HAZE_STRENGTH), continued below the horizon (slowly
        darkening) so there is never a black ground.  The sun glow is added on top (brighter toward the sun).
      - sun: wide / medium / tight glow lobes + a small very bright disc.  The disc is CAMERA ONLY
        (Light Path > Is Camera Ray): lighting and reflections get the sun from the lamp, the world probe
        only sees the sky clamped to PROBE_MAX, so there is no second sun and no fireflies.
      - clouds: three layers on horizontal planes   P = dir.xy / (max(dir.z, 0) + PLANE_K)
        (cells get smaller and flatter toward the horizon, finite extent, nothing stretches to infinity)
          cirrus : high thin veil sheets with a fibrous (stretched, warped fBm) texture, slow drift
          deck   : THE hero layer.  Domain-warped fBm eroded by Voronoi billows (stratocumulus puffs), coverage
                   threshold modulated by a very low frequency noise (big clear gaps and big masses).
          scud   : a few dark low shreds near the horizon on the sun side, fastest drift
        Noise Detail is driven per pixel (LOD): only as many octaves as a final-frame pixel can resolve on
        the cloud plane, so the distance never shimmers.
        Fake lighting of the deck: the density is sampled a second time, offset toward the sun azimuth;
        the difference is a relief term (slopes of the cloud base that face the low sun light up), thick
        cores go dark, thin edges forward-scatter (gold lining near the sun).  Looking toward the sun the
        relief is contrasty, looking away the deck is front lit (soft pink, grey bases).  All colours come
        from ramps on the angle to the sun (gold -> fiery orange -> salmon -> dusty pink / slate).
      - animation: a Value node driven by  frame/24  moves every layer along its own wind vector and
        slowly evolves the shapes through the third noise dimension (deterministic, no loops, no popping).
  * Sun lamp "Sun" (collection "sky"): the only shadow casting light of the film.
  * World / EEVEE light settings: probe_resolution 2048, sun_threshold 0 + use_sun_shadow False (no automatic
    world-sun extraction).

Public:  build(ctx),  SKY (dict with the key colours / sun values for other modules).
Tunables: the constants below.  Colours were tuned against AgX (Medium High Contrast), exposure 0.
"""
import bpy, math
from mathutils import Vector
from .. import layout as L
from ..common import NT, get_col

# ------------------------------------------------------------------ tunables
# sun lamp
SUN_COLOR = (1.0, 0.50, 0.25)
SUN_STRENGTH = 3.2            # W/m2
SUN_ANGLE_DEG = 2.0           # soft shadows
# sun in the sky
DISC_RADIUS_DEG = 0.50        # visual disc (bloom is added by the compositor)
DISC_VALUE = (17.0, 10.5, 4.6)
PROBE_MAX = 2.6               # clamp of what lights / reflections see (camera sees the unclamped sky)

# clear sky colours (scene linear)
ZENITH = (0.058, 0.084, 0.215)
HAZE_K = 15.0                 # haze band thickness: weight = exp(-HAZE_K * sin(elevation))
CLOUD_HAZE_K = 30.0           # clouds sink into the haze a little later than the clear sky

# cloud projection
PLANE_K = 0.10                # horizon compression: plane radius at the horizon = 1/PLANE_K layer heights
PIXEL_RAD = 0.0003            # angular size of a final-frame pixel (LOD of the noise detail) - 4K final

# main deck
A_SCALE = 1.45                # cells per layer height
A_COVER = 0.66                # approximate coverage 0..1
A_COVER_VAR = 0.75            # how strongly the very large scale noise opens gaps / builds masses
A_EDGE = 0.045                # softness of the cloud edge (in noise units)
A_WARP = 0.55
A_TRANS = 0.20                # density range over which thin cloud lets the sun glow through
A_RIM_COLOR = (1.5, 0.42, 0.012)
A_DETAIL = 7.5                # max fBm detail (reached only high in the sky, LOD reduces it toward the horizon)
A_LAC = 2.2
A_ROUGH = 0.60                # fBm roughness (higher = more ragged)
A_BILLOW = 0.16               # strength of the cellular (Voronoi) billow erosion
A_BILLOW_SCALE = 4.5          # billow cells per base noise cell
A_LIGHT_OFFSET = 0.04         # offset of the lighting sample toward the sun (noise units)
A_RELIEF = 6.5                # gain of the directional relief lighting
A_WIND = (0.0052, 0.0024)     # drift in layer heights per second (toward ENE)
A_EVOLVE = 0.0020             # shape evolution per second
# cirrus
C_SCALE = 7.0                 # streak widths per layer height
C_STRETCH = 0.16              # streak aspect (smaller = longer fibres)
C_WARP = 3.0                  # curl of the streaks
C_WIND = (0.0016, -0.0006)
C_EVOLVE = 0.0012
C_ALPHA = 0.50
# scud
S_SCALE = 3.2
S_WIND = (0.016, 0.006)
S_EVOLVE = 0.006

FPS = 24.0


def _mul3(c, f):
    return (c[0] * f, c[1] * f, c[2] * f)


SKY = {
    "zenith": ZENITH,
    "horizon_sun": _mul3(L.HAZE_TOWARD_SUN, L.HAZE_STRENGTH),
    "horizon_away": _mul3(L.HAZE_AWAY, L.HAZE_STRENGTH),
    "sun_color": SUN_COLOR,
    "sun_strength": SUN_STRENGTH,
    "sun_dir": L.sun_dir(),
    "sun_angle_deg": SUN_ANGLE_DEG,
}


# ====================================================================================== world shader
def _build_world(scene):
    w = bpy.data.worlds.new("SunsetSky")
    w.use_nodes = True
    scene.world = w
    nt = NT(w.node_tree, clear=True)
    out = nt.n('ShaderNodeOutputWorld')
    M = nt.math

    def sstep(x, e0, e1):
        return nt.map_range(x, e0, e1, 0.0, 1.0, interp='SMOOTHSTEP')

    def cscale(c, f):
        return nt.vmath('SCALE', c, scale=f)

    def cadd(a, b):
        return nt.vmath('ADD', a, b)

    def noise(vec, scale, detail, rough=0.5, lac=2.0, dist=0.0):
        return nt.n('ShaderNodeTexNoise', noise_dimensions='3D', Vector=vec, Scale=scale, Detail=detail,
                    Roughness=rough, Lacunarity=lac, Distortion=dist)

    S = L.sun_dir()
    az = math.radians(L.SUN_AZIMUTH_DEG)
    SAZ = (math.cos(az), math.sin(az), 0.0)

    # ---------------------------------------------------------------- direction terms
    tc = nt.n('ShaderNodeTexCoord')
    d = nt.vmath('NORMALIZE', tc.outputs['Generated'])
    dx, dy, dz = nt.sep_xyz(d)
    zc = M('MAXIMUM', dz, 0.0)
    zneg = M('MAXIMUM', M('MULTIPLY', dz, -1.0), 0.0)
    mu = nt.vmath('DOT_PRODUCT', d, S)
    mu01 = M('MULTIPLY_ADD', mu, 0.5, 0.5)
    mup = M('MAXIMUM', mu, 0.0)

    # time in seconds (frame driven)
    tnode = nt.t.nodes.new('ShaderNodeValue'); tnode.label = "time_s"
    fc = tnode.outputs[0].driver_add('default_value')
    fc.driver.type = 'SCRIPTED'
    fc.driver.expression = "frame/%g" % FPS
    tsec = tnode.outputs[0]

    # ---------------------------------------------------------------- haze colour (identical to atmo.py)
    sun_amt = M('POWER', nt.map_range(mu, -0.2, 1.0, 0.0, 1.0), 2.2)
    haze = nt.mix(sun_amt, _mul3(L.HAZE_AWAY, L.HAZE_STRENGTH), _mul3(L.HAZE_TOWARD_SUN, L.HAZE_STRENGTH))

    # ---------------------------------------------------------------- clear sky gradient
    mid = nt.ramp(mu01, [(0.00, (0.200, 0.125, 0.205, 1)),
                         (0.45, (0.140, 0.125, 0.250, 1)),
                         (0.72, (0.230, 0.150, 0.255, 1)),
                         (0.88, (0.550, 0.200, 0.140, 1)),
                         (1.00, (1.000, 0.320, 0.060, 1))])
    up = M('POWER', M('SUBTRACT', 1.0, zc), 3.0)
    sky = nt.mix(up, ZENITH, mid)
    hz = M('EXPONENT', M('MULTIPLY', zc, -HAZE_K))
    hzc = M('EXPONENT', M('MULTIPLY', zc, -CLOUD_HAZE_K))
    sky = nt.mix(hz, sky, haze)

    # ---------------------------------------------------------------- cloud plane + LOD
    inv = M('DIVIDE', 1.0, M('ADD', zc, PLANE_K))
    P = nt.vmath('SCALE', nt.comb_xyz(dx, dy, 0.0), scale=inv)
    lod2 = M('MULTIPLY', M('LOGARITHM', M('ADD', zc, PLANE_K), 2.0), 2.0)

    def detail(dmax, f0, lac=2.0, margin=0.3):
        """Per-pixel noise Detail: as many octaves as the pixel footprint on the cloud plane can resolve."""
        k = 1.0 / math.log2(lac)
        c = (math.log2(1.0 / (2.2 * PIXEL_RAD * f0)) - margin) * k
        return M('MINIMUM', M('MAXIMUM', M('MULTIPLY_ADD', lod2, k, c), 1.0), dmax)

    def layer_coords(scale, wind, evolve, seed):
        drift = nt.vmath('SCALE', (wind[0] * scale, wind[1] * scale, evolve), scale=tsec)
        return cadd(cadd(nt.vmath('SCALE', P, scale=scale), drift), (seed[0], seed[1], seed[2]))

    hfade = sstep(dz, 0.0, 0.035)

    # ================================================================ CIRRUS (high, thin)
    pc = layer_coords(C_SCALE, C_WIND, C_EVOLVE, (31.7, 12.3, 5.0))
    cw = noise(pc, 0.20, 2.0)
    pcw = cadd(pc, cscale(nt.vmath('SUBTRACT', cw.outputs['Color'], (0.5, 0.5, 0.5)), C_WARP))
    ca = math.radians(L.SUN_AZIMUTH_DEG - 62.0)
    rot = nt.n('ShaderNodeMapping', Vector=pcw)
    rot.inputs['Rotation'].default_value = (0.0, 0.0, ca)
    rot.inputs['Scale'].default_value = (C_STRETCH, 1.0, 1.0)
    cn = noise(rot.outputs[0], 1.0, detail(4.0, C_SCALE), rough=0.62)
    cmask = noise(pc, 0.07, 2.0)
    # thin veil sheets (soft patches) with a fibrous texture
    sheet = sstep(cmask.outputs['Factor'], 0.44, 0.74)
    fibre = M('MULTIPLY_ADD', sstep(cn.outputs['Factor'], 0.36, 0.78), 0.8, 0.2)
    cir = M('MULTIPLY', M('MULTIPLY', M('MULTIPLY', sheet, fibre), C_ALPHA), hfade)
    ccol = nt.ramp(mu01, [(0.00, (0.20, 0.12, 0.19, 1)),
                          (0.50, (0.34, 0.14, 0.19, 1)),
                          (0.78, (0.62, 0.17, 0.13, 1)),
                          (0.93, (1.10, 0.28, 0.04, 1)),
                          (1.00, (2.40, 0.90, 0.05, 1))])
    ccol = nt.mix(hzc, ccol, haze)
    col = nt.mix(cir, sky, ccol)

    # ================================================================ MAIN DECK
    pa = layer_coords(A_SCALE, A_WIND, A_EVOLVE, (0.0, 0.0, 0.0))
    wn = noise(pa, 0.5, 1.0)
    warp = cscale(nt.vmath('SUBTRACT', wn.outputs['Color'], (0.5, 0.5, 0.5)), A_WARP)
    paw = cadd(pa, warp)
    detA = detail(A_DETAIL, A_SCALE, A_LAC)
    hi_fade = sstep(dz, 0.02, 0.16)
    detV = detail(1.5, A_SCALE * A_BILLOW_SCALE, 2.3)
    sun_off = cadd(paw, _mul3(SAZ, A_LIGHT_OFFSET))

    # billow strength varies over the sky: puffy cellular regions next to smoother stratiform ones
    billow = M('MULTIPLY', M('MULTIPLY', hi_fade, A_BILLOW), M('MULTIPLY_ADD', wn.outputs['Factor'], 1.5, 0.25))

    def density(vec):
        f = noise(vec, 1.0, detA, rough=A_ROUGH, lac=A_LAC).outputs['Factor']
        v = nt.n('ShaderNodeTexVoronoi', voronoi_dimensions='2D', feature='F1', normalize=True, Vector=vec,
                 Scale=A_BILLOW_SCALE, Detail=detV, Roughness=0.5, Lacunarity=2.3).outputs['Distance']
        # billows: cell centres are denser, cell borders are eroded
        return M('ADD', f, M('MULTIPLY', M('SUBTRACT', 0.30, v), billow))

    n1 = density(paw)
    n2 = density(sun_off)
    big = noise(pa, 0.27, 1.0).outputs['Factor']
    # threshold: fBm is ~N(0.5, 0.11); coverage c -> threshold 0.5 - 0.11 * probit(c)  (linearised)
    thr0 = 0.5 - 0.28 * (A_COVER - 0.5)
    thr = M('SUBTRACT', thr0 + 0.5 * A_COVER_VAR, M('MULTIPLY', big, A_COVER_VAR))
    e = M('SUBTRACT', n1, thr)
    a_alpha = M('MULTIPLY', sstep(e, 0.0, A_EDGE), hfade)
    # looking toward the sun we mostly see shadowed slopes + glowing thin parts (contrasty relief);
    # looking away the deck is front lit: broad soft light on the edges, grey bases, little fine contrast
    wf = sstep(mu, 0.60, -0.45)
    gain = nt.mixf(wf, A_RELIEF, A_RELIEF * 0.7)
    relief = M('MULTIPLY_ADD', M('SUBTRACT', n1, n2), gain, 0.5, clamp=True)
    core = sstep(e, 0.04, 0.22)
    lit_t = M('MULTIPLY', M('POWER', sstep(relief, 0.38, 1.0), 1.4), M('MULTIPLY_ADD', core, -0.75, 1.0))
    lit_f = M('MULTIPLY', M('MULTIPLY_ADD', relief, 0.7, 0.3), M('MULTIPLY_ADD', core, -0.7, 1.0))
    lit = nt.mixf(wf, lit_t, lit_f)
    lit = M('MULTIPLY', lit, M('MULTIPLY_ADD', big, 0.9, 0.55))      # large scale brightness variation
    a_dark = nt.ramp(mu01, [(0.00, (0.046, 0.046, 0.076, 1)),
                            (0.30, (0.048, 0.046, 0.076, 1)),
                            (0.60, (0.046, 0.043, 0.068, 1)),
                            (0.85, (0.042, 0.038, 0.058, 1)),
                            (1.00, (0.060, 0.040, 0.050, 1))])
    a_lit = nt.ramp(mu01, [(0.00, (0.30, 0.13, 0.17, 1)),
                           (0.40, (0.44, 0.13, 0.17, 1)),
                           (0.65, (0.68, 0.12, 0.09, 1)),
                           (0.85, (1.10, 0.15, 0.00, 1)),
                           (1.00, (1.60, 0.36, 0.00, 1))])
    acol = cadd(a_dark, cscale(a_lit, lit))
    # silver / gold lining: thin edges forward-scatter the sun
    trans = M('SUBTRACT', 1.0, sstep(e, 0.0, A_TRANS))
    rim = M('MULTIPLY', M('POWER', trans, 2.0), M('POWER', mu01, 10.0))
    rim = M('MULTIPLY', rim, M('MULTIPLY_ADD', relief, 0.6, 0.4))
    acol = cadd(acol, cscale(A_RIM_COLOR, rim))
    acol = nt.mix(hzc, acol, haze)
    col = nt.mix(a_alpha, col, acol)

    # ================================================================ SCUD (low, dark, near the horizon)
    ps = layer_coords(S_SCALE, S_WIND, S_EVOLVE, (7.1, 3.3, 9.0))
    sn = noise(ps, 1.0, detail(3.0, S_SCALE), rough=0.55).outputs['Factor']
    sband = M('MULTIPLY', sstep(dz, 0.015, 0.07), M('SUBTRACT', 1.0, sstep(dz, 0.08, 0.24)))
    smask = noise(ps, 0.16, 1.0).outputs['Factor']
    s_alpha = M('MULTIPLY', M('MULTIPLY', sstep(sn, 0.54, 0.63), sstep(smask, 0.42, 0.6)), sband)
    s_alpha = M('MULTIPLY', s_alpha, M('MULTIPLY_ADD', wf, -0.8, 0.9))
    s_thin = M('SUBTRACT', 1.0, sstep(sn, 0.55, 0.66))
    scol = nt.ramp(mu01, [(0.00, (0.036, 0.038, 0.060, 1)),
                          (0.60, (0.042, 0.038, 0.058, 1)),
                          (0.88, (0.060, 0.036, 0.038, 1)),
                          (1.00, (0.120, 0.050, 0.030, 1))])
    scol = cadd(scol, cscale((1.8, 0.6, 0.02), M('MULTIPLY', M('POWER', s_thin, 2.0), M('POWER', mu01, 16.0))))
    scol = nt.mix(M('MULTIPLY', hzc, 0.8), scol, haze)
    col = nt.mix(s_alpha, col, scol)

    # ---------------------------------------------------------------- below the horizon: haze, slowly darkening
    ground = cscale(haze, M('SUBTRACT', 1.0, M('MULTIPLY', sstep(zneg, 0.0, 0.7), 0.55)))
    col = nt.mix(sstep(dz, -0.02, 0.0), ground, col)

    # ---------------------------------------------------------------- sun glow (in front of everything, veiled by cloud)
    occ = M('MAXIMUM', a_alpha, M('MAXIMUM', s_alpha, M('MULTIPLY', cir, 0.5)))
    veil = M('SUBTRACT', 1.0, M('MULTIPLY', occ, 0.86))
    gfade = M('EXPONENT', M('MULTIPLY', zneg, -7.0))
    g_wide = M('MULTIPLY', M('POWER', mup, 9.0), M('EXPONENT', M('MULTIPLY', zc, -7.0)))
    g_mid = M('POWER', mup, 70.0)
    g_tight = M('MULTIPLY', M('POWER', mup, 900.0), veil)
    glow = cadd(cadd(cscale((0.75, 0.16, 0.0), g_wide), cscale((1.6, 0.6, 0.03), M('MULTIPLY', g_mid, M('MULTIPLY_ADD', veil, 0.8, 0.2)))),
                cscale((6.0, 3.0, 0.6), g_tight))
    col = cadd(col, cscale(glow, gfade))

    # disc: camera only (lights / reflections get the sun from the lamp)
    cr0 = math.cos(math.radians(DISC_RADIUS_DEG * 1.5)); cr1 = math.cos(math.radians(DISC_RADIUS_DEG * 0.6))
    disc = M('MULTIPLY', sstep(mu, cr0, cr1), M('MULTIPLY_ADD', veil, 0.92, 0.08))
    col_cam = cadd(col, cscale(DISC_VALUE, disc))
    col_probe = nt.vmath('MINIMUM', col, (PROBE_MAX, PROBE_MAX, PROBE_MAX))
    lp = nt.n('ShaderNodeLightPath')
    final = nt.mix(lp.outputs['Is Camera Ray'], col_probe, col_cam)

    bg = nt.n('ShaderNodeBackground', Color=final, Strength=1.0)
    nt.link(bg.outputs[0], out.inputs['Surface'])

    # ---------------------------------------------------------------- world lighting settings (EEVEE)
    w.probe_resolution = '2048'
    w.sun_threshold = 0.0          # no automatic "world sun" extraction: the lamp below is the only sun
    w.use_sun_shadow = False
    return w


# ====================================================================================== sun lamp
def _build_sun(scene, col):
    sd = L.sun_dir()
    sun = bpy.data.lights.new("Sun", 'SUN')
    sun.color = SUN_COLOR
    sun.energy = SUN_STRENGTH
    sun.angle = math.radians(SUN_ANGLE_DEG)
    sun.use_shadow = True
    sun.shadow_maximum_resolution = 0.004
    sun.shadow_filter_radius = 1.5
    sun.use_shadow_jitter = False
    so = bpy.data.objects.new("Sun", sun)
    col.objects.link(so)
    so.rotation_euler = Vector(sd).to_track_quat('Z', 'Y').to_euler()
    so.location = (0.0, 0.0, 200.0)
    return so


def build(ctx):
    scene = ctx.scene
    col = get_col("sky")
    _build_world(scene)
    _build_sun(scene, col)



