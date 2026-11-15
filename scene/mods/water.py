"""WATER module: infinity-pool water, overflow films and the hero waterfall down the cliff.

Builds (collection "water"):
  * pool_water      glassy turquoise water sheet over the pool rectangle (POOL_X0..X1, POOL_Y0..Y1) at
                    POOL_WATER_Z, with frame-driven gentle ripples, fresnel mirror reflections and raytraced
                    refraction of the glowing tiled basin (built by terrace.py).
  * pool_overflow   thin clinging water films over the WEST and SOUTH infinity edges, down the outer wall face
                    into the catch gutter ~1.2 m below (gutter geometry = terrace.py).
  * fall_*          the hero waterfall leaving the 7 m outlet (SPILL_X0..X1 at SPILL_Y) and falling to
                    SPILL_BOTTOM_Z: flow sheet over the lip, three layered curved veils + breakaway strands with
                    streak textures that scroll at the physical fall speed (time-of-flight UVs, v = sqrt(2 g h)
                    with mild air drag), foam at the lip, soft spray / mist puffs near the top, where the fall
                    grazes the rock, and a mist cloud where it dissolves at the bottom.
  * stand-ins       (only in isolated builds) basin + terrace floor when 'terrace' is not built, a rough cliff
                    face when 'landscape' is not built.

The waterfall path adapts to whatever rock / masonry exists when this module runs (landscape + terrace are
built before water): horizontal rays find the face behind the fall, and the fall is pushed out so it never
cuts into the rock (it then clings / grazes the face, which is where the extra spray is placed).

API for the integrator:
    add_waterfall(x, y, z_top, z_bottom, width, facing_deg, name, ...) -> list of objects
        (x, y, z_top) = centre of the lip, facing_deg = horizontal direction the water leaves toward
        (degrees CCW from +X; -90 = toward the south), width = lip width in metres.

Tunable constants: see the block below (WATER_MODE, RIPPLE_*, FALL_*, MIST_*).
"""
import bpy, math, random
import numpy as np
from mathutils import Vector

from ..common import MB, get_col, new_material, NT, drive_frame
from .. import mats, atmo
from .. import layout as L

# ------------------------------------------------------------------------------------------- tunables
WATER_MODE = "refract"        # "refract" (Principled transmission + raytraced refraction) | "fresnel"
RIPPLE_STRENGTH = 0.18        # bump strength of the pool ripples
RIPPLE_SPEED = 1.0            # time multiplier of the ripple animation
POOL_TINT = (0.62, 0.93, 0.90)
INF_WALL_T = 0.9              # thickness of the pool wall the overflow film runs over
INF_DROP = 1.2                # depth of the catch gutter below the water line

G = 9.81
FALL_V0 = 1.9                 # m/s horizontal speed of the water leaving the lip
FALL_VT = 42.0                # m/s terminal velocity (air drag) of the falling water mass
FALL_CLEAR = 0.9              # m minimum clearance between the core sheet and the rock behind it
FALL_SPREAD = 0.06           # lateral spread (m of width per m of fall, grows ~ sqrt)
FALL_STREAK_SPEED = 1.0       # 1.0 = streaks move at the physical speed
FALL_BRIGHT = 1.0             # albedo multiplier of the white water
MIST_DENSITY = 1.0            # opacity multiplier of all spray/mist puffs
FALL_METHOD = 'BLENDED'       # render method of the fall sheets ('BLENDED' = smooth, 'DITHERED' = sorted)
SOUTH_GUTTER_WATER_Z = -1.5   # water level in the corbelled south gutter (terrace.py builds the stone)
WEST_CHANNEL_WATER_Z = L.LOWER_Z - 0.06   # water level in the west catch channel (1.0 m wide)
WEST_CHANNEL_W = 1.0

_RNG = random.Random(7)


# ------------------------------------------------------------------------------------------- helpers
def _time(nt, mul=1.0):
    """Value socket = scene time in seconds * mul (frame driven)."""
    v = nt.n('ShaderNodeValue')
    drive_frame(v.outputs[0], 'default_value', "frame/24*%.5f" % mul)
    return v.outputs[0]


def _mesh_from_grid(name, P, attrs, mat, col, faces=None, shadow=False):
    """P: (nv, nu, 3) vertex grid -> quad mesh.  attrs: {name: (nv, nu) float array} point attributes."""
    nv, nu, _ = P.shape
    verts = P.reshape(-1, 3)
    cen = verts.mean(axis=0)               # object origin at the centroid (BLENDED sorting)
    verts = verts - cen
    if faces is None:
        idx = np.arange(nv * nu).reshape(nv, nu)
        a = idx[:-1, :-1].ravel(); b = idx[:-1, 1:].ravel(); c = idx[1:, 1:].ravel(); d = idx[1:, :-1].ravel()
        faces = np.stack([a, b, c, d], axis=1)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts)); me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    nf = len(faces)
    me.loops.add(nf * 4); me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    me.polygons.add(nf)
    me.polygons.foreach_set("loop_start", (np.arange(nf) * 4).astype(np.int32))
    me.update(calc_edges=True)
    for k, arr in attrs.items():
        at = me.attributes.new(k, 'FLOAT', 'POINT')
        at.data.foreach_set("value", np.asarray(arr, dtype=np.float32).ravel())
    me.polygons.foreach_set("use_smooth", np.ones(nf, dtype=bool))
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    ob.location = tuple(cen)
    col.objects.link(ob)
    if not shadow:
        ob.visible_shadow = False
    return ob


def _smoothstep_np(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# =========================================================================================== materials
def _ripple_normal(nt, t, strength=RIPPLE_STRENGTH):
    geo = nt.n('ShaderNodeNewGeometry')
    P = geo.outputs['Position']
    o1 = nt.comb_xyz(nt.math('MULTIPLY', t, 0.09), nt.math('MULTIPLY', t, 0.04), 0.0)
    n1 = nt.n('ShaderNodeTexNoise', noise_dimensions='4D', Vector=nt.vmath('ADD', P, o1),
              W=nt.math('MULTIPLY', t, 0.16), Scale=0.45, Detail=2.0, Roughness=0.45)
    o2 = nt.comb_xyz(nt.math('MULTIPLY', t, -0.05), nt.math('MULTIPLY', t, 0.13), 0.0)
    n2 = nt.n('ShaderNodeTexNoise', noise_dimensions='4D', Vector=nt.vmath('ADD', P, o2),
              W=nt.math('MULTIPLY', t, 0.31), Scale=1.4, Detail=2.0, Roughness=0.5)
    n3 = nt.n('ShaderNodeTexNoise', noise_dimensions='4D', Vector=P,
              W=nt.math('MULTIPLY', t, 0.6), Scale=4.5, Detail=1.0, Roughness=0.5)
    h = nt.math('ADD', nt.math('MULTIPLY', n1.outputs['Factor'], 0.62),
                nt.math('ADD', nt.math('MULTIPLY', n2.outputs['Factor'], 0.28),
                        nt.math('MULTIPLY', n3.outputs['Factor'], 0.10)))
    return nt.bump(h, strength=strength, distance=0.06)


@mats.material("water_pool")
def _m_water_pool(name):
    m, nt, out = new_material(name)
    t = _time(nt, RIPPLE_SPEED)
    nrm = _ripple_normal(nt, t)
    if WATER_MODE == "refract":
        bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=POOL_TINT, Roughness=0.012, IOR=1.333,
                    Transmission_Weight=1.0, Normal=nrm,
                    Emission_Color=(0.03, 0.72, 0.80, 1.0), Emission_Strength=0.36 * L.EMISSIVE_GAIN)   # luminous pool
        bsdf.inputs['Specular IOR Level'].default_value = 0.5
        nt.link(bsdf.outputs[0], out.inputs['Surface'])
        m.surface_render_method = 'DITHERED'
        m.use_raytrace_refraction = True
        try:
            out.inputs['Thickness'].default_value = 0.0
        except Exception:
            pass
    else:
        fres = nt.n('ShaderNodeFresnel', IOR=1.333, Normal=nrm)
        glossy = nt.n('ShaderNodeBsdfGlossy', Color=(1, 1, 1, 1), Roughness=0.015, Normal=nrm)
        transp = nt.n('ShaderNodeBsdfTransparent', Color=POOL_TINT)
        em = nt.n('ShaderNodeEmission', Color=(0.02, 0.30, 0.30, 1), Strength=0.25 * L.EMISSIVE_GAIN)
        under = nt.n('ShaderNodeAddShader', i0=transp.outputs[0], i1=em.outputs[0])
        mix = nt.n('ShaderNodeMixShader', i0=fres.outputs[0], i1=under.outputs[0], i2=glossy.outputs[0])
        nt.link(mix.outputs[0], out.inputs['Surface'])
        m.surface_render_method = 'DITHERED'
    return m


@mats.material("water_film")
def _m_water_film(name):
    """Thin overflow film on the outer pool wall: glossy, mostly see-through, faint downward streaks."""
    m, nt, out = new_material(name)
    t = _time(nt)
    geo = nt.n('ShaderNodeNewGeometry')
    P = geo.outputs['Position']
    # streaks along z, scrolling down ~1.5 m/s
    mp = nt.n('ShaderNodeMapping', Vector=nt.vmath('ADD', P, nt.comb_xyz(0.0, 0.0, nt.math('MULTIPLY', t, 1.6))))
    mp.inputs['Scale'].default_value = (7.0, 7.0, 0.7)
    st = nt.n('ShaderNodeTexNoise', Vector=mp.outputs[0], Scale=1.0, Detail=3.0, Roughness=0.55)
    sf = st.outputs['Factor']
    nrm = nt.bump(sf, strength=0.35, distance=0.01)
    glossy = nt.n('ShaderNodeBsdfGlossy', Color=(0.9, 0.97, 0.97, 1), Roughness=0.03, Normal=nrm)
    white = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.75, 0.82, 0.82, 1), Roughness=0.4)
    foam = nt.map_range(sf, 0.62, 0.78, 0.0, 0.5)
    surf = nt.n('ShaderNodeMixShader', i0=foam, i1=glossy.outputs[0], i2=white.outputs[0])
    transp = nt.n('ShaderNodeBsdfTransparent', Color=(0.85, 0.95, 0.95, 1))
    a = nt.map_range(sf, 0.3, 0.75, 0.35, 0.75)
    mix = nt.n('ShaderNodeMixShader', i0=a, i1=transp.outputs[0], i2=surf.outputs[0])
    nt.link(mix.outputs[0], out.inputs['Surface'])
    m.surface_render_method = 'DITHERED'
    return m


FALL_PARAMS = ("thr0", "thr1", "opacity", "streak_u", "streak_t", "fade_in", "edge", "seed", "glassy")


def _fall_material(name, fog=True):
    """ONE shared waterfall material (keeps shader compilation cheap).  Per-object parameters come from
    object custom properties wf_<param> (see FALL_PARAMS), geometry data from point attributes written by
    _build_fall: wf_u (lateral m, top scale), wf_un (lateral -1..1), wf_tau (time of flight s),
    wf_h (fall m), wf_hn (fall fraction 0..1)."""
    m, nt, out = new_material(name)
    t = _time(nt, FALL_STREAK_SPEED)
    pr = {k: nt.attr('wf_' + k, 'OBJECT').outputs['Factor'] for k in FALL_PARAMS}
    U = nt.attr('wf_u').outputs['Factor']
    UN = nt.attr('wf_un').outputs['Factor']
    TAU = nt.attr('wf_tau').outputs['Factor']
    H = nt.attr('wf_h').outputs['Factor']
    HN = nt.attr('wf_hn').outputs['Factor']
    T = nt.math('SUBTRACT', TAU, t)                 # advected coordinate: features fall with the water
    Us = nt.math('MULTIPLY', U, pr['streak_u'])
    Ts = nt.math('MULTIPLY', T, pr['streak_t'])
    # A: broad lanes + slow flow pulses
    n1 = nt.n('ShaderNodeTexNoise', Vector=nt.comb_xyz(nt.math('MULTIPLY', Us, 0.3), nt.math('MULTIPLY', Ts, 0.5),
                                                       pr['seed']), Scale=1.0, Detail=2.0, Roughness=0.5)
    # B: clumps / 'comets' (short in time -> a few metres long at speed), carry the motion
    n2 = nt.n('ShaderNodeTexNoise', Vector=nt.comb_xyz(Us, nt.math('MULTIPLY', Ts, 5.0), nt.math('ADD', pr['seed'], 3.1)),
              Scale=1.0, Detail=3.0, Roughness=0.6, Distortion=0.3)
    # C: fine long streaks
    n3 = nt.n('ShaderNodeTexNoise', Vector=nt.comb_xyz(nt.math('MULTIPLY', Us, 5.0), nt.math('MULTIPLY', Ts, 1.2),
                                                       nt.math('ADD', pr['seed'], 6.3)), Scale=1.0, Detail=2.0)
    c = nt.math('ADD', nt.math('ADD', nt.math('MULTIPLY', n1.outputs['Factor'], 0.40),
                                      nt.math('MULTIPLY', n2.outputs['Factor'], 0.42)),
                nt.math('MULTIPLY', n3.outputs['Factor'], 0.18))
    c = nt.math('ADD', nt.math('MULTIPLY', nt.math('SUBTRACT', c, 0.5), 2.3), 0.5)     # restore contrast
    # coverage threshold rises with the fall -> veil breaks into strands lower down
    thr = nt.mixf(nt.math('POWER', HN, 0.6), pr['thr0'], pr['thr1'])
    soft = nt.map_range(HN, 0.0, 1.0, 0.07, 0.16)
    lo = nt.math('SUBTRACT', thr, soft); hi = nt.math('ADD', thr, soft)
    cov = nt.map_range(c, lo, hi, 0.0, 1.0, interp='SMOOTHSTEP')
    # translucent spray veil between the white clumps (thins out with the fall, modulated by the lanes)
    veil = nt.math('MULTIPLY', nt.map_range(HN, 0.0, 1.0, 0.55, 0.22),
                   nt.map_range(n1.outputs['Factor'], 0.3, 0.65, 0.35, 1.0))
    body = nt.math('MAXIMUM', cov, veil)
    # ragged, feathered edges; the soft margin widens as the fall spreads into spray
    ve = nt.comb_xyz(nt.math('MULTIPLY', U, 0.35), nt.math('MULTIPLY', T, 0.9), nt.math('ADD', pr['seed'], 7.7))
    ne = nt.n('ShaderNodeTexNoise', Vector=ve, Scale=1.0, Detail=2.0)
    e_in = nt.math('ADD', nt.math('SUBTRACT', pr['edge'], nt.math('ADD', 0.25, nt.math('MULTIPLY', HN, 0.35))),
                   nt.math('MULTIPLY', ne.outputs['Factor'], 0.5))
    e_in = nt.math('MINIMUM', e_in, 0.86)
    feather = nt.map_range(nt.math('ABSOLUTE', UN), e_in, 1.0, 1.0, 0.0, interp='SMOOTHSTEP')
    # vertical envelopes
    bottom = nt.map_range(HN, 0.72, 0.995, 1.0, 0.0, interp='SMOOTHSTEP')
    alpha = nt.math('MULTIPLY', nt.math('MULTIPLY', body, feather), bottom)
    # parts of a curved sheet seen edge-on fade out (no stacked opaque edges, no visible card edges)
    lwf = nt.n('ShaderNodeLayerWeight', Blend=0.5)
    facing = nt.math('SUBTRACT', 1.0, lwf.outputs['Facing'])
    alpha = nt.math('MULTIPLY', alpha, nt.map_range(facing, 0.04, 0.42, 0.0, 1.0, interp='SMOOTHSTEP'))
    fin = nt.map_range(nt.math('DIVIDE', H, nt.math('MAXIMUM', pr['fade_in'], 0.01)), 0.0, 1.0, 0.0, 1.0,
                       interp='SMOOTHSTEP')
    alpha = nt.math('MULTIPLY', alpha, fin)
    alpha = nt.math('MULTIPLY', alpha, pr['opacity'], clamp=True)
    # shading: white aerated water = diffuse + translucency; glassy nappe for the first metre
    tone = nt.map_range(nt.math('ADD', nt.math('MULTIPLY', cov, 0.7), nt.math('MULTIPLY', n2.outputs['Factor'], 0.3)),
                        0.1, 0.9, 0.62, 1.0)
    base = nt.mix(1.0, (0.90 * FALL_BRIGHT, 0.92 * FALL_BRIGHT, 0.93 * FALL_BRIGHT, 1), nt.comb_xyz(tone, tone, tone),
                  blend='MULTIPLY')
    nrm = nt.bump(nt.math('ADD', n2.outputs['Factor'], nt.math('MULTIPLY', n1.outputs['Factor'], 2.0)),
                  strength=0.7, distance=0.25)
    diff = nt.n('ShaderNodeBsdfPrincipled', Base_Color=base, Roughness=0.35, Normal=nrm)
    diff.inputs['Specular IOR Level'].default_value = 0.45
    trans = nt.n('ShaderNodeBsdfTranslucent', Color=(0.85, 0.88, 0.90, 1), Normal=nrm)
    white = nt.n('ShaderNodeMixShader', i0=0.4, i1=diff.outputs[0], i2=trans.outputs[0]).outputs[0]
    # glassy (unaerated) water right at the lip
    gl = nt.n('ShaderNodeBsdfGlossy', Color=(0.85, 0.97, 0.95, 1), Roughness=0.04, Normal=nrm)
    tp = nt.n('ShaderNodeBsdfTransparent', Color=(0.55, 0.92, 0.88, 1))
    glass = nt.n('ShaderNodeMixShader', i0=0.45, i1=tp.outputs[0], i2=gl.outputs[0]).outputs[0]
    aer = nt.map_range(H, 0.15, 1.6, 0.0, 1.0, interp='SMOOTHSTEP')
    aer = nt.math('MAXIMUM', aer, nt.math('MULTIPLY', nt.map_range(c, 0.5, 0.7, 0.0, 1.0), 0.6))
    aer = nt.math('MAXIMUM', aer, nt.math('SUBTRACT', 1.0, pr['glassy']))
    white = nt.n('ShaderNodeMixShader', i0=aer, i1=glass, i2=white).outputs[0]
    # the glassy part is solid water: always visible; streak cut only once aerated
    alpha = nt.mixf(aer, nt.math('MULTIPLY', feather, 0.95), alpha)
    if fog:
        white = atmo.fog(nt, white)
    tr = nt.n('ShaderNodeBsdfTransparent', Color=(1, 1, 1, 1))
    mix = nt.n('ShaderNodeMixShader', i0=alpha, i1=tr.outputs[0], i2=white)
    nt.link(mix.outputs[0], out.inputs['Surface'])
    m.surface_render_method = FALL_METHOD
    m.use_backface_culling = False
    return m


def _set_fall_params(ob, **kw):
    d = dict(thr0=0.3, thr1=0.62, opacity=1.0, streak_u=2.2, streak_t=1.4, fade_in=0.0, edge=0.62, seed=0.0,
             glassy=0.0)
    d.update({k: v for k, v in kw.items() if k in d})
    for k, v in d.items():
        ob["wf_" + k] = float(v)


def _mist_material(name, fog=True, warm=0.0):
    """Soft spray / mist puff for (low poly) ellipsoids: opacity falls off toward the silhouette, so the
    puff looks like a soft blob from every direction.  Drifts downward + evolves with time.
    Per-object opacity: custom property wf_density."""
    m, nt, out = new_material(name)
    t = _time(nt)
    lw = nt.n('ShaderNodeLayerWeight', Blend=0.5)
    face = nt.math('SUBTRACT', 1.0, lw.outputs['Facing'])
    core = nt.math('POWER', nt.math('MAXIMUM', face, 0.0), 3.0)
    tc = nt.n('ShaderNodeTexCoord')
    oc = tc.outputs['Object']
    oi = nt.n('ShaderNodeObjectInfo')
    seed = nt.math('MULTIPLY', oi.outputs['Random'], 50.0)
    drift = nt.vmath('ADD', oc, nt.comb_xyz(0.0, 0.0, nt.math('MULTIPLY', t, 0.18)))
    n = nt.n('ShaderNodeTexNoise', noise_dimensions='4D', Vector=drift, W=nt.math('ADD', nt.math('MULTIPLY', t, 0.07), seed),
             Scale=2.2, Detail=3.0, Roughness=0.55)
    wisp = nt.map_range(n.outputs['Factor'], 0.32, 0.72, 0.15, 1.0)
    dens = nt.math('MULTIPLY', nt.attr('wf_density', 'OBJECT').outputs['Factor'], MIST_DENSITY)
    # only the half in front of the fall (local +x = outward) is visible, so a puff that touches the rock
    # behind never shows a hard intersection line
    front = nt.map_range(nt.sep_xyz(oc)[0], -0.15, 0.55, 0.0, 1.0, interp='SMOOTHSTEP')
    alpha = nt.math('MULTIPLY', nt.math('MULTIPLY', nt.math('MULTIPLY', core, wisp), front), dens, clamp=True)
    col = (0.80 + 0.2 * warm, 0.82 + 0.05 * warm, 0.86 - 0.1 * warm, 1)
    # shade as if the puff were a camera-facing card: no sphere terminator
    geo = nt.n('ShaderNodeNewGeometry')
    diff = nt.n('ShaderNodeBsdfDiffuse', Color=col, Normal=geo.outputs['Incoming'])
    trans = nt.n('ShaderNodeBsdfTranslucent', Color=col, Normal=nt.vmath('SCALE', geo.outputs['Incoming'], scale=-1.0))
    sh = nt.n('ShaderNodeMixShader', i0=0.5, i1=diff.outputs[0], i2=trans.outputs[0]).outputs[0]
    if fog:
        sh = atmo.fog(nt, sh)
    tr = nt.n('ShaderNodeBsdfTransparent', Color=(1, 1, 1, 1))
    mix = nt.n('ShaderNodeMixShader', i0=alpha, i1=tr.outputs[0], i2=sh)
    nt.link(mix.outputs[0], out.inputs['Surface'])
    m.surface_render_method = 'BLENDED'
    m.use_backface_culling = True
    try:
        # overlap ON = no depth pre-pass: a puff must never hide the (blended) fall sheets behind it
        m.use_transparency_overlap = True
    except Exception:
        pass
    return m


def _get_mat(name, factory, **kw):
    mm = bpy.data.materials.get(name)
    if mm is None:
        mm = factory(name, **kw)
        mm.name = name
    return mm


# =========================================================================================== rock probing
_SKIP_WORDS = ("mist", "fog", "cloud", "cam", "water", "fall", "spray", "tree", "fol", "leaf", "plant", "grass",
               "haze", "veg", "sky", "bird")


def _rock_dist(scene, dg, base, o, lat, z, reach=60.0):
    """Outward distance (along o from the lip line) of the first solid surface met by a horizontal ray cast
    from `reach` metres in front of the fall back toward the face.  None if nothing is hit."""
    origin = Vector((base[0] + o[0] * reach + lat[0], base[1] + o[1] * reach + lat[1], z))
    d = Vector((-o[0], -o[1], 0.0))
    travelled = 0.0
    for _ in range(6):
        hit, loc, nrm, idx, ob, mtx = scene.ray_cast(dg, origin, d, distance=reach * 2.2 - travelled)
        if not hit:
            return None
        name = ob.name.lower()
        mine = any(c.name == "water" for c in ob.users_collection) and not name.startswith("water_standin")
        if mine or (any(w in name for w in _SKIP_WORDS) and not name.startswith("water_standin")):
            step = (loc - origin).length + 0.05
            origin = loc + d * 0.05; travelled += step
            continue
        return reach - ((loc - origin).length + travelled)
    return None


# =========================================================================================== waterfall core
def _fall_path(h_max, v0, n=150, vt=FALL_VT):
    """Integrate a ballistic path with quadratic drag.  Returns arrays h (fall), d (outward), tau (s), v (m/s)
    sampled at n+1 fall heights (denser at the top)."""
    hs = h_max * (np.linspace(0.0, 1.0, n + 1) ** 1.7)
    k = G / (vt * vt)
    dt = 0.002
    t = 0.0; x = 0.0; z = 0.0; vx = v0; vz = 0.0
    T = [0.0]; X = [0.0]; Z = [0.0]; V = [v0]
    while z < h_max + 1.0:
        sp = math.hypot(vx, vz)
        ax = -k * sp * vx
        az = G - k * sp * vz
        vx += ax * dt; vz += az * dt
        x += vx * dt; z += vz * dt; t += dt
        T.append(t); X.append(x); Z.append(z); V.append(math.hypot(vx, vz))
    Z = np.array(Z); X = np.array(X); T = np.array(T); V = np.array(V)
    d = np.interp(hs, Z, X); tau = np.interp(hs, Z, T); v = np.interp(hs, Z, V)
    return hs, d, tau, v


def _adapt_to_rock(ctx, base, o, lat_dir, width, hs, d, z_top, clear):
    """Push the path outward where the rock behind it would intersect; returns (d_new, gap) where gap = free
    space between sheet and rock (large = free fall, small = grazing)."""
    scene = ctx.scene
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    gap = np.full(len(hs), 99.0)
    need = np.full(len(hs), -1e9)
    for i, h in enumerate(hs):
        if i % 2 and i < len(hs) - 1:
            continue
        z = z_top - h - 0.05
        rmax = None
        for s in (-0.5, -0.25, 0.0, 0.25, 0.5):
            lat = (lat_dir[0] * s * width, lat_dir[1] * s * width)
            r = _rock_dist(scene, dg, base, o, lat, z)
            if r is not None and (rmax is None or r > rmax):
                rmax = r
        if rmax is not None:
            need[i] = rmax + clear
            gap[i] = d[i] - rmax
    # fill the skipped samples
    for i in range(1, len(hs) - 1, 2):
        need[i] = max(need[i - 1], need[i + 1]) if i + 1 < len(hs) else need[i - 1]
        gap[i] = min(gap[i - 1], gap[i + 1]) if i + 1 < len(hs) else gap[i - 1]
    dn = np.maximum(d, need)
    # momentum: once pushed out the water does not come back in
    dn = np.maximum.accumulate(dn)
    # smooth (in fall distance) and re-enforce
    for _ in range(4):
        sm = dn.copy()
        sm[1:-1] = 0.25 * dn[:-2] + 0.5 * dn[1:-1] + 0.25 * dn[2:]
        dn = np.maximum(sm, np.maximum(d, need))
    dn[0] = d[0]
    gap = np.minimum(gap + (dn - d), 99.0)
    return dn, gap


def _build_fall(ctx, col, name, lip, z_bottom, width, facing_deg, layers, v0=FALL_V0, n=150, nu=28,
                spread=FALL_SPREAD, mist=True, mist_scale=1.0, strands=6, rock_clear=FALL_CLEAR, fog=True,
                seed=1):
    """Generic waterfall builder used for the hero fall and add_waterfall().
    lip: (x, y, z) centre of the lip.  layers: list of dicts (offset, width_mul, opacity, thr0, thr1, fade_in,
    streak_u, glassy)."""
    rng = random.Random(seed)
    f = math.radians(facing_deg)
    o = (math.cos(f), math.sin(f))
    lat = (-math.sin(f), math.cos(f))
    z_top = lip[2]
    H = z_top - z_bottom
    hs, d, tau, v = _fall_path(H, v0, n)
    d, gap = _adapt_to_rock(ctx, lip, o, lat, width, hs, d, z_top, rock_clear)
    hn = hs / H
    objs = []
    for li, ly in enumerate(layers):
        un = np.linspace(-1.0, 1.0, nu + 1)
        un = np.sign(un) * np.abs(un) ** 0.85            # a bit denser toward the edges
        W = width * ly.get("width_mul", 1.0) + spread * 2.0 * np.sqrt(np.maximum(hs, 0.0)) * 6.0 * ly.get("spread_mul", 1.0)
        W = np.minimum(W, width * ly.get("width_mul", 1.0) + spread * hs * 2.2)
        half = 0.5 * W
        bulge = ly.get("bulge", 0.5) + 0.012 * hs
        off = ly.get("offset", 0.0) * (0.3 + 0.7 * _smoothstep_np(0.0, 25.0, hs)) + 0.004 * hs * ly.get("offset", 0.0)
        # slow lateral meander so the fall is not a perfect ruled surface
        mean = np.sin(hs * 0.031 + li * 1.7 + seed) * 0.6 * _smoothstep_np(5.0, 80.0, hs)
        # edges lag slightly (water at the edge is slower) -> curled sides
        P = np.zeros((len(hs), len(un), 3))
        HH, UU = np.meshgrid(hs, un, indexing='ij')
        DD = d[:, None] + off[:, None] + bulge[:, None] * (1.0 - UU ** 2) - 0.25 * (np.abs(UU) ** 3) * _smoothstep_np(0, 30, HH)
        LL = UU * half[:, None] + mean[:, None]
        P[..., 0] = lip[0] + o[0] * DD + lat[0] * LL
        P[..., 1] = lip[1] + o[1] * DD + lat[1] * LL
        P[..., 2] = z_top - HH + ly.get("z_off", 0.0)
        attrs = {
            "wf_u": UU * width * 0.5,
            "wf_un": UU,
            "wf_tau": np.repeat(tau[:, None], len(un), axis=1),
            "wf_h": HH,
            "wf_hn": np.repeat(hn[:, None], len(un), axis=1),
        }
        mat = _get_mat("wf_fall" if fog else "wf_fall_nofog", _fall_material, fog=fog)
        ob = _mesh_from_grid("%s_sheet%d" % (name, li), P, attrs, mat, col)
        pp = dict(ly); pp.setdefault("seed", li * 5.3 + seed); pp["glassy"] = 1.0 if ly.get("glassy") else 0.0
        _set_fall_params(ob, **pp)
        objs.append(ob)
    # ---- breakaway strands: narrow ribbons peeling off the edges and the front
    if strands:
        smat = _get_mat("wf_fall" if fog else "wf_fall_nofog", _fall_material, fog=fog)
        for k in range(strands):
            side = -1.0 if k % 2 == 0 else 1.0
            s_lat = side * (0.42 + 0.12 * rng.random()) * width
            if k >= 4:
                s_lat = (rng.random() - 0.5) * 0.6 * width
            w0 = 0.6 + 0.9 * rng.random()
            h0 = 3.0 + rng.random() * 12.0
            sel = hs >= h0
            hh = hs[sel]; dd = d[sel]
            dv = 0.0035 * (hh - h0) ** 1.15 * (1.0 + rng.random())     # peels away outward / sideways
            un = np.linspace(-1.0, 1.0, 5)
            P = np.zeros((len(hh), len(un), 3))
            HH, UU = np.meshgrid(hh, un, indexing='ij')
            ww = w0 * (1.0 + 0.03 * (HH - h0))
            front = 0.6 + (0.9 if k >= 4 else 0.0)
            DD = dd[:, None] + front + dv[:, None] * (0.6 if k < 4 else 1.0)
            LL = s_lat + side * dv[:, None] * (0.7 if k < 4 else 0.1) + UU * ww * 0.5
            P[..., 0] = lip[0] + o[0] * DD + lat[0] * LL
            P[..., 1] = lip[1] + o[1] * DD + lat[1] * LL
            P[..., 2] = z_top - HH
            attrs = {"wf_u": (s_lat + UU * ww * 0.5), "wf_un": UU,
                     "wf_tau": np.repeat(tau[sel][:, None], len(un), axis=1),
                     "wf_h": HH - h0 + 0.5, "wf_hn": np.repeat(hn[sel][:, None], len(un), axis=1)}
            ob = _mesh_from_grid("%s_strand%d" % (name, k), P, attrs, smat, col)
            _set_fall_params(ob, opacity=0.85, seed=11.0 + seed + k * 2.7, thr0=0.40, thr1=0.70, streak_u=3.5,
                             streak_t=1.2, fade_in=4.0, glassy=0.0, edge=0.35)
            objs.append(ob)
    # ---- spray / mist puffs
    if mist:
        objs += _fall_mist(col, name, lip, o, lat, hs, d, gap, width, H, rng, mist_scale, fog)
    return objs, (hs, d, gap)


_PUFF_MESH = {}


def _puff_mesh():
    me = _PUFF_MESH.get("m")
    if me is not None and me.name in bpy.data.meshes:
        return me
    segs, rings = 14, 9
    verts = []
    for j in range(rings + 1):
        th = math.pi * j / rings
        for i in range(segs):
            ph = 2 * math.pi * i / segs
            verts.append((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)))
    faces = []
    for j in range(rings):
        for i in range(segs):
            a = j * segs + i; b = j * segs + (i + 1) % segs
            if j == 0:
                faces.append((a, b + segs, a + segs))          # top cap triangles
            elif j == rings - 1:
                faces.append((a, b, a + segs))                 # bottom cap triangles
            else:
                faces.append((a, b, b + segs, a + segs))
    me = bpy.data.meshes.new("wf_puff")
    me.from_pydata(verts, [], faces)
    import bmesh
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me); bm.free()
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    me.update()
    me.materials.append(None)
    _PUFF_MESH["m"] = me
    return me


def _fall_mist(col, name, lip, o, lat, hs, d, gap, width, H, rng, scale, fog):
    me = _puff_mesh()
    objs = []
    mmat = _get_mat("wf_mist" if fog else "wf_mist_nofog", _mist_material, fog=fog)
    dens_ = (0.45, 0.36, 0.24)

    def puff(h, along, side, r, sq=(1.0, 1.0, 1.0), mi=0):
        h = float(np.clip(h, 0.0, H))
        dd = float(np.interp(h, hs, d))
        g = float(np.interp(h, hs, gap))
        sq = (sq[0] * 0.6, sq[1], sq[2])            # flattened against the wall (local x = outward)
        along = max(along, r * sq[0] - min(g, 30.0) + 0.6)   # never cut into the rock behind the fall
        ob = bpy.data.objects.new("%s_puff" % name, me)
        ob.location = (lip[0] + o[0] * (dd + along) + lat[0] * side,
                       lip[1] + o[1] * (dd + along) + lat[1] * side, lip[2] - h)
        ob.scale = (r * sq[0], r * sq[1], r * sq[2])
        ob.rotation_euler = (0, 0, math.atan2(o[1], o[0]))
        ob.visible_shadow = False
        col.objects.link(ob)
        # material per object (mesh data is shared by all puffs)
        ob.material_slots[0].link = 'OBJECT'
        ob.material_slots[0].material = mmat
        ob["wf_density"] = dens_[mi]
        objs.append(ob)

    s = scale
    # near the lip: light spray in front of the curtain
    for k in range(5):
        puff(4.0 + 3.5 * k + rng.random() * 2, 1.2 + rng.random() * 1.0, (rng.random() - 0.5) * width * 0.9,
             (1.3 + 0.6 * k) * s, (1.2, 0.8, 1.5), mi=2)
    # along the fall, increasing in size; more where the fall grazes the rock
    hh = 30.0
    while hh < H * 0.97:
        g = float(np.interp(hh, hs, gap))
        grazing = g < 3.5
        r = (2.0 + 0.05 * hh) * s * (1.3 if grazing else 1.0)
        for side in ((-0.3, 0.3) if grazing else (rng.choice((-0.25, 0.25)),)):
            puff(hh + rng.random() * 6, 1.0 + r * 0.35 + rng.random(), side * width + (rng.random() - 0.5) * 3,
                 r, (1.3, 1.3, 2.0), mi=1 if grazing else 2)
        hh += 8.0 + 0.08 * hh
    # bottom: the fall dissolves into a mist cloud
    for k in range(7):
        r = (14.0 + 9.0 * rng.random()) * s
        puff(H * (0.86 + 0.13 * rng.random()), 2.0 + rng.random() * 8.0, (rng.random() - 0.5) * (width + 25.0) * s,
             r, (1.5, 1.0, 0.8), mi=0)
    return objs


# =========================================================================================== pool + overflow
def _pool_water(col):
    mb = MB("pool_water")
    z = L.POOL_WATER_Z
    # tiny overlap under the copings so no gap shows at grazing angles
    e = 0.02
    nx, ny = 4, 8
    mb.grid(nx, ny, lambda u, v: (L.POOL_X0 - e + u * (L.POOL_X1 - L.POOL_X0 + 2 * e),
                                  L.POOL_Y0 - e + v * (L.POOL_Y1 - L.POOL_Y0 + 2 * e), z), "water_pool")
    ob = mb.to_object("pool_water", col, smooth_angle=None)
    ob.visible_shadow = False
    return ob


def _film_profile(z_end):
    z0 = L.POOL_WATER_Z
    t = INF_WALL_T
    prof = []          # (outward distance from the inner wall line, z) over the wall top, then down the face
    for i in range(7):  # over the wall top: slightly above the stone, thinning toward the edge
        s = i / 6.0
        prof.append((s * t, z0 + 0.012 - 0.010 * s * s))
    for i in range(1, 6):   # round the outer arris (r ~3 cm)
        a = (math.pi / 2) * i / 5
        prof.append((t + 0.025 * math.sin(a) + 0.006, z0 + 0.002 - 0.03 * (1 - math.cos(a))))
    nz = 10
    drop = (z0 - 0.03) - z_end
    for i in range(1, nz + 1):   # clinging film down the outer face (slowly peeling off the wall)
        s = i / nz
        prof.append((t + 0.031 + 0.012 * s * s, z0 - 0.03 - drop * s))
    return prof


def _overflow(col):
    """Films over the west edge (x = POOL_X0, into the catch channel of the lower west terrace) and the south
    edge (y = POOL_Y0, into the corbelled gutter; not at the outlet), plus the water in channel + gutter."""
    mb = MB("pool_overflow")

    def run(p0, p1, outward, prof):
        L_ = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        nseg = max(2, int(L_ / 1.5))
        verts = []
        for i in range(nseg + 1):
            s = i / nseg
            bx = p0[0] + (p1[0] - p0[0]) * s; by = p0[1] + (p1[1] - p0[1]) * s
            for (dd, zz) in prof:
                verts.append((bx + outward[0] * dd, by + outward[1] * dd, zz))
        m = len(prof); faces = []
        for i in range(nseg):
            for j in range(m - 1):
                a = i * m + j
                faces.append((a, a + m, a + m + 1, a + 1))
        mb.add(verts, faces, "water_film")

    pw = _film_profile(WEST_CHANNEL_WATER_Z - 0.03)
    ps = _film_profile(SOUTH_GUTTER_WATER_Z - 0.03)
    # west edge: from the SW corner (incl. the corner block) to the north end
    run((L.POOL_X0, L.POOL_Y0 - INF_WALL_T), (L.POOL_X0, L.POOL_Y1), (-1.0, 0.0), pw)
    # south edge, west of the outlet and east of the outlet
    run((L.POOL_X0, L.POOL_Y0), (L.SPILL_X0, L.POOL_Y0), (0.0, -1.0), ps)
    run((L.SPILL_X1, L.POOL_Y0), (L.POOL_X1, L.POOL_Y0), (0.0, -1.0), ps)
    ob = mb.to_object("pool_overflow", col, smooth_angle=None, recalc=False)
    ob.visible_shadow = False
    # water surface in the catch channel / gutter (slightly turbulent where the film lands)
    mb = MB("pool_gutter_water")
    xw = L.POOL_X0 - INF_WALL_T
    mb.grid(1, 24, lambda u, v: (xw - WEST_CHANNEL_W + u * WEST_CHANNEL_W, L.SPILL_Y + v * (L.POOL_Y1 + 0.9 - L.SPILL_Y),
                                 WEST_CHANNEL_WATER_Z), "water_channel")
    for (a, b) in ((L.POOL_X0 - INF_WALL_T, L.SPILL_X0), (L.SPILL_X1, L.POOL_X1 + INF_WALL_T)):
        mb.grid(max(1, int((b - a) / 2)), 1, lambda u, v, a=a, b=b: (a + u * (b - a), L.SPILL_Y - 0.75 + v * 0.75,
                                                                      SOUTH_GUTTER_WATER_Z), "water_channel")
    ob = mb.to_object("pool_gutter_water", col, smooth_angle=None)
    ob.visible_shadow = False
    return ob


@mats.material("water_channel")
def _m_water_channel(name):
    """Shallow running water in the gutters: dark glossy mirror with faster ripples."""
    m, nt, out = new_material(name)
    t = _time(nt, 3.0)
    nrm = _ripple_normal(nt, t, strength=0.25)
    bsdf = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.02, 0.05, 0.05, 1), Roughness=0.03, Normal=nrm)
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
    nt.link(bsdf.outputs[0], out.inputs['Surface'])
    return m


def _spill_approach(col, name, x0, x1, y_in, y_lip, z):
    """Water accelerating through the outlet toward the lip (flow sheet slightly above the crest)."""
    mat = _get_mat("wf_fall_nofog", _fall_material, fog=False)
    nv, nu = 6, 20
    P = np.zeros((nv + 1, nu + 1, 3))
    un = np.linspace(-1, 1, nu + 1)
    vv = np.linspace(0, 1, nv + 1)
    xc = 0.5 * (x0 + x1); hw = 0.5 * (x1 - x0)
    for i, s in enumerate(vv):
        P[i, :, 0] = xc + un * hw
        P[i, :, 1] = y_in + (y_lip - y_in) * s
        P[i, :, 2] = z + 0.006 - 0.012 * s * s
    # tau: water accelerates from ~0.6 to ~1.9 m/s over the approach; negative so it continues the fall's
    dist = abs(y_lip - y_in)
    tau = -(1.0 - vv) * dist / 1.2
    attrs = {"wf_u": np.repeat((un * hw)[None, :], nv + 1, 0), "wf_un": np.repeat(un[None, :], nv + 1, 0),
             "wf_tau": np.repeat(tau[:, None], nu + 1, 1), "wf_h": np.zeros((nv + 1, nu + 1)),
             "wf_hn": np.zeros((nv + 1, nu + 1))}
    ob = _mesh_from_grid(name, P, attrs, mat, col)
    _set_fall_params(ob, opacity=1.0, seed=2.0, thr0=0.0, thr1=0.0, glassy=1.0, edge=0.97)
    return ob


# =========================================================================================== stand-ins
def _standins(ctx, col):
    if 'terrace' not in ctx.mods:
        mb = MB("water_standin_terrace")
        zf = L.POOL_WATER_Z - L.POOL_DEPTH
        X0, X1, Y0, Y1 = L.POOL_X0, L.POOL_X1, L.POOL_Y0, L.POOL_Y1
        # basin
        mb.box(X0, X1, Y0, Y1, zf - 0.3, zf, "pool_tile")
        mb.box(X0 - 0.9, X0, Y0 - 0.9, Y1 + 0.9, zf - 0.3, L.POOL_WATER_Z - 0.004, "pool_tile")    # west wall
        mb.box(X1, X1 + 0.9, Y0 - 0.9, Y1 + 0.9, zf - 0.3, 0.0, "pool_tile")                      # east wall
        mb.box(X0, X1, Y1, Y1 + 0.9, zf - 0.3, 0.0, "pool_tile")                                  # north wall
        mb.box(X0, L.SPILL_X0, Y0 - 0.9, Y0, zf - 0.3, L.POOL_WATER_Z - 0.004, "pool_tile")       # south walls
        mb.box(L.SPILL_X1, X1, Y0 - 0.9, Y0, zf - 0.3, L.POOL_WATER_Z - 0.004, "pool_tile")
        mb.box(L.SPILL_X0, L.SPILL_X1, Y0 - 0.9, Y0, zf - 0.3, L.POOL_WATER_Z - 0.03, "pool_tile")  # outlet crest
        # copings east / north at terrace level (stone)
        mb.box(X1 + 0.9, X1 + 1.4, Y0 - 0.9, Y1 + 1.4, -0.3, 0.0, "stone_trim")
        # terrace floor east + north of the pool
        mb.box(X1 + 1.4, 44.0, L.TERRACE_Y0, 20.0, -0.3, 0.0, "marble_wet")
        mb.box(-46.0, X1 + 1.4, Y1 + 0.9, 20.0, -0.3, 0.0, "marble_wet")
        # lower west terrace + catch channel
        xw = X0 - INF_WALL_T
        mb.box(-47.5, xw - WEST_CHANNEL_W, L.SPILL_Y, Y1 + 0.9, L.LOWER_Z - 0.4, L.LOWER_Z, "marble_wet")
        mb.box(xw - WEST_CHANNEL_W, xw, L.SPILL_Y, Y1 + 0.9, L.LOWER_Z - 0.6, WEST_CHANNEL_WATER_Z - 0.25, "stone_dark")
        mb.box(xw - WEST_CHANNEL_W - 0.12, xw - WEST_CHANNEL_W, L.SPILL_Y, Y1 + 0.9, L.LOWER_Z - 0.4, L.LOWER_Z + 0.03,
               "stone_trim")
        # corbelled south gutters
        for (a, b) in ((xw, L.SPILL_X0), (L.SPILL_X1, X1 + 0.9)):
            mb.box(a, b, L.SPILL_Y - 0.9, L.SPILL_Y, SOUTH_GUTTER_WATER_Z - 0.3, SOUTH_GUTTER_WATER_Z - 0.12, "stone_trim")
            mb.box(a, b, L.SPILL_Y - 0.9, L.SPILL_Y - 0.75, SOUTH_GUTTER_WATER_Z - 0.12, SOUTH_GUTTER_WATER_Z + 0.08,
                   "stone_trim")
        # retaining walls under everything
        mb.box(-47.5, 44.0, L.SPILL_Y, L.SPILL_Y + 1.0, -16.0, zf - 0.3, "stone_dark")
        mb.box(xw - 0.2, xw, L.SPILL_Y, Y1 + 0.9, L.LOWER_Z - 0.6, zf, "stone_dark")
        mb.to_object("water_standin_terrace", col)
    if 'castle' not in ctx.mods:
        # crude lit facade so pool reflections can be judged
        mb = MB("water_standin_castle")
        mb.box(-46.0, 62.0, 27.0, 45.0, 0.0, 22.0, "stone")
        mb.box(-9.0, 13.0, 22.0, 46.0, 0.0, 40.0, "stone")
        mb.box(-36.0, -24.0, 24.0, 34.0, 0.0, 36.0, "stone")
        for i in range(14):
            for j in range(3):
                x = -44.0 + i * 7.6
                if -10.0 < x < 14.0:
                    continue
                mb.rnd = ((i * 7 + j * 3) % 10) / 10.0
                z = 4.0 + j * 6.0
                mb.quad((x, 26.98, z), (x + 2.0, 26.98, z), (x + 2.0, 26.98, z + 3.8), (x, 26.98, z + 3.8), "glass_warm")
        mb.rnd = 0.6
        mb.quad((-6.0, 21.98, 3.0), (10.0, 21.98, 3.0), (10.0, 21.98, 30.0), (-6.0, 21.98, 30.0), "glass_warm")
        mb.to_object("water_standin_castle", col)
    if 'landscape' not in ctx.mods:
        # rough rock face under the masonry that bulges out with depth -> the fall grazes it lower down
        mb = MB("water_standin_cliff")
        nx, nz = 40, 60
        def f(u, v):
            x = -110.0 + u * 120.0
            z = -14.0 - v * 260.0
            k = min(1.0, v * 5)
            y = L.SPILL_Y - 0.3 - max(0.0, (-14.0 - z)) * 0.05 + k * (2.2 * math.sin(x * 0.21 + z * 0.05)
                                                                    + 1.3 * math.sin(z * 0.13 + x * 0.05) - 1.5)
            return (x, y, z)
        mb.grid(nx, nz, f, "wf_standin_rock")
        mb.to_object("water_standin_cliff", col)


@mats.material("wf_standin_rock")
def _m_rock(name):
    m, nt, out = new_material(name)
    geo = nt.n('ShaderNodeNewGeometry')
    n = nt.n('ShaderNodeTexNoise', Vector=geo.outputs['Position'], Scale=0.08, Detail=5.0, Roughness=0.6)
    col = nt.ramp(n.outputs['Factor'], [(0.3, (0.07, 0.065, 0.06, 1)), (0.7, (0.20, 0.18, 0.16, 1))])
    b = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=0.9,
             Normal=nt.bump(n.outputs['Factor'], strength=0.8, distance=0.5))
    nt.link(atmo.fog(nt, b.outputs[0]), out.inputs['Surface'])
    return m


# =========================================================================================== public API
_HERO_LAYERS = [
    # core curtain: dense at the top (glassy nappe -> white), breaks up lower down
    dict(offset=0.0, width_mul=1.0, opacity=0.95, thr0=0.30, thr1=0.68, streak_u=2.0, streak_t=1.3, glassy=True,
         bulge=0.35, edge=0.70),
    # front veil: a bit wider, appears after a few metres, more broken
    dict(offset=0.9, width_mul=1.06, opacity=0.6, thr0=0.50, thr1=0.70, streak_u=2.6, streak_t=1.1, fade_in=6.0,
         bulge=0.55, edge=0.55, spread_mul=1.3, z_off=-0.6),
    # back veil, slightly narrower, closer to the wall
    dict(offset=-0.45, width_mul=0.92, opacity=0.7, thr0=0.34, thr1=0.64, streak_u=1.6, streak_t=1.5, fade_in=2.0,
         bulge=0.25, edge=0.6, spread_mul=0.8),
    # outer mist veil: very wide and soft, lower half
    dict(offset=2.2, width_mul=1.5, opacity=0.42, thr0=0.42, thr1=0.52, streak_u=1.0, streak_t=0.6, fade_in=25.0,
         bulge=1.5, edge=0.35, spread_mul=3.5),
]


def add_waterfall(x, y, z_top, z_bottom, width, facing_deg, name, v0=1.6, mist=True, ctx=None, collection=None):
    """Natural waterfall anywhere (simpler than the hero fall: 2 veils + 2 strands + mist puffs).
    (x, y, z_top): centre of the lip; facing_deg: horizontal direction the water leaves toward, degrees CCW
    from +X.  The path is pushed out of any rock found behind it.  Returns the list of created objects."""
    if ctx is None:
        import types
        ctx = types.SimpleNamespace(scene=bpy.context.scene)
    col = collection or get_col("water")
    layers = [
        dict(offset=0.0, width_mul=1.0, opacity=1.0, thr0=0.30, thr1=0.64, streak_u=2.4, streak_t=1.3,
             glassy=False, bulge=0.3, edge=0.6),
        dict(offset=0.8, width_mul=1.15, opacity=0.6, thr0=0.45, thr1=0.70, streak_u=1.4, streak_t=0.9,
             fade_in=5.0, bulge=0.6, edge=0.5, spread_mul=1.5),
    ]
    n = int(np.clip((z_top - z_bottom) / 1.6, 40, 140))
    objs, _ = _build_fall(ctx, col, name, (x, y, z_top), z_bottom, width, facing_deg, layers, v0=v0, n=n, nu=16,
                          strands=2, mist=mist, mist_scale=max(0.5, width / 7.0), seed=abs(hash(name)) % 97)
    return objs


def build(ctx):
    col = get_col("water")
    _standins(ctx, col)
    _pool_water(col)
    _overflow(col)
    lip_z = L.SPILL_TOP_Z + 0.004
    xc = 0.5 * (L.SPILL_X0 + L.SPILL_X1)
    width = (L.SPILL_X1 - L.SPILL_X0) - 0.1
    _spill_approach(col, "fall_hero_approach", L.SPILL_X0 + 0.05, L.SPILL_X1 - 0.05, L.POOL_Y0 + 0.6, L.SPILL_Y, lip_z)
    objs, (hs, d, gap) = _build_fall(ctx, col, "fall_hero", (xc, L.SPILL_Y, lip_z), L.SPILL_BOTTOM_Z, width, -90.0,
                                     _HERO_LAYERS, v0=FALL_V0, n=170, nu=30, strands=6, mist=True, seed=1)
    ntri = 0
    for ob in col.objects:
        if ob.type == 'MESH':
            ob.data.calc_loop_triangles()
            ntri += len(ob.data.loop_triangles)
    print("[water] hero fall: outward offset at -16 m %.1f m, at bottom %.1f m, min gap %.1f m; water tris %d"
          % (float(np.interp(16.0, hs, d)), float(d[-1]), float(gap.min()), ntri))



