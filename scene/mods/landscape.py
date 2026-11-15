"""Landscape module: alpine valley, lake, hero cliff, mist and distant forest around the castle plateau.

Public API (pure numpy, importable without building anything - implemented in landscape_field.py):
    height_at(x, y)   -> z      terrain height (numpy arrays ok)
    forest_mask(x, y) -> 0..1   where forest may grow
    slope_at(x, y)    -> tan(slope)

build(ctx) creates, in the collection "landscape":
    landscape_terrain    polar heightfield mesh out to 3 km, cells ~0.4 m (castle) .. ~12 x 36 m
    landscape_terrain_far  3 .. 46 km, 2560 columns concentrated in the sector the film looks at
                         (az ~75..220 deg: ~14 x 70 m cells at 10 km), buried skirt ring at the seam
    landscape_cliff      hero cliff sheet along layout.CLIFF_TOP (dense, displaced along its normals)
    landscape_water      lake / river surface at layout.VALLEY_Z
    landscape_mist_NN    stack of mist sheets (valley fog banks, low clouds between ridges, waterfall plume)
    landscape_trees      distant conifer cones (850 m .. 8 km, near real size)

Nothing here needs to cast shadows (scene/mods/perf.py switches shadow casting off for the landscape);
all relief shading comes from N.L + bump.  Set the environment variable LS_PLAIN=1 to build flat terrain /
cliff materials (used to measure what the landscape shading costs per frame).

Tunables: constants below, landscape_field.py (terrain shape), landscape_cliff.py (cliff), landscape_mats.py.
"""
import math, time
import numpy as np
import bpy

from .landscape_field import height_at, forest_mask, slope_at   # noqa: F401  (public API)
from . import landscape_field as F
from .. import layout as L, mats
from ..common import get_col

# ---------------------------------------------------------------- terrain grid
N_THETA = 1536            # vertices per ring
R_INNER = 30.0            # first ring (inside: hidden under the terrace)
DR_MIN = 1.5              # radial cell size near the castle
K_RADIAL = 0.012          # radial growth per ring beyond DR_MIN / K_RADIAL metres
R_OUTER = 46000.0
R_GRID = 15000.0          # the helper grid (trees, mist, inner mesh) is evaluated out to here
R_SPLIT = 3000.0          # inner mesh ends / far mesh starts here (far mesh has a buried skirt ring)
N_THETA_FAR = 2560        # far mesh: vertices per ring, concentrated in the sector the film looks at
K_FAR = 0.0072            # far mesh: radial growth per ring
FAR_AZ, FAR_HALF = 147.5, 72.0   # visible sector (deg CCW from +X): centre and half width at full density
DENSE_AZ = 155.0          # centre (deg, CCW from +X) of the angularly dense sector (valley / sun side)
WATER_R = 17000.0


def _log(msg, t0):
    print("[landscape] %-28s %.1fs" % (msg, time.time() - t0))


def new_mesh_object(name, co, loops, loop_start, col, material=None, smooth=True):
    """Fast mesh creation from numpy arrays (co (n,3), flat loop vertex indices, loop starts)."""
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(co))
    me.vertices.foreach_set("co", np.ascontiguousarray(co, dtype=np.float32).ravel())
    me.loops.add(len(loops))
    me.loops.foreach_set("vertex_index", np.ascontiguousarray(loops, dtype=np.int32))
    me.polygons.add(len(loop_start))
    me.polygons.foreach_set("loop_start", np.ascontiguousarray(loop_start, dtype=np.int32))
    me.update(calc_edges=True)
    if material is not None:
        me.materials.append(mats.get(material))
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    return ob


def grid_quads(nr, nc, wrap):
    """Quad loops for a (nr x nc) vertex grid; wrap = columns wrap around.  CCW when rows go outward and
    columns go counter-clockwise."""
    ncq = nc if wrap else nc - 1
    i = np.arange(nr - 1)[:, None]; j = np.arange(ncq)[None, :]
    j1 = (j + 1) % nc
    a = i * nc + j; b = (i + 1) * nc + j; c = (i + 1) * nc + j1; d = i * nc + j1
    return np.stack([a, b, c, d], axis=-1).reshape(-1)


def add_attr(me, name, values, kind='FLOAT', domain='POINT'):
    at = me.attributes.new(name, kind, domain)
    key = {'FLOAT': 'value', 'FLOAT_COLOR': 'color', 'FLOAT_VECTOR': 'vector'}[kind]
    at.data.foreach_set(key, np.ascontiguousarray(values, dtype=np.float32).ravel())
    return at


# =====================================================================================
# terrain
# =====================================================================================
def _polar_layout():
    # angular distribution: denser toward the valley / sun side
    th = np.linspace(0.0, 2.0 * np.pi, 8192, endpoint=False)
    d = np.abs(np.angle(np.exp(1j * (th - math.radians(DENSE_AZ)))))
    w = 0.52 + 0.48 * (1.0 - F.smoothstep(math.radians(100.0), math.radians(150.0), d))
    cdf = np.concatenate([[0.0], np.cumsum(w)])
    u = np.arange(N_THETA) / N_THETA * cdf[-1]
    theta = np.interp(u, cdf, np.concatenate([th, [2.0 * np.pi]]))
    radii = [R_INNER]
    while radii[-1] < R_GRID:
        radii.append(radii[-1] + max(DR_MIN, K_RADIAL * radii[-1]))
    return theta, np.array(radii)


def _grid_masks(X, Y, H, aux, theta, radii):
    dth = (np.roll(theta, -1) - np.roll(theta, 1)) % (2.0 * np.pi)
    dh_t = (np.roll(H, -1, axis=1) - np.roll(H, 1, axis=1)) / (radii[:, None] * dth[None, :])
    dh_r = np.gradient(H, radii, axis=0)
    slope = np.hypot(dh_t, dh_r)
    fm = F.forest_mask(X, Y, _pre=(H, aux, slope))
    shore = 1.0 - F.smoothstep(F.VALLEY_Z + 0.4, F.VALLEY_Z + 3.0, H)
    return slope, fm, shore


def build_terrain_far(col, r0):
    """Far terrain (r0 .. R_OUTER): fine cells in the sector the cameras look at (mountains, ridges)."""
    t0 = time.time()
    th = np.linspace(0.0, 2.0 * np.pi, 16384, endpoint=False)
    d = np.abs(np.angle(np.exp(1j * (th - math.radians(FAR_AZ)))))
    w = 0.2 + 0.8 * (1.0 - F.smoothstep(math.radians(FAR_HALF), math.radians(FAR_HALF + 30.0), d))
    cdf = np.concatenate([[0.0], np.cumsum(w)])
    u = np.arange(N_THETA_FAR) / N_THETA_FAR * cdf[-1]
    theta = np.interp(u, cdf, np.concatenate([th, [2.0 * np.pi]]))
    radii = [r0]
    while radii[-1] < R_OUTER:
        radii.append(radii[-1] * (1.0 + K_FAR))
    radii = np.array(radii)
    nr, nc = len(radii), len(theta)
    X = F.CX + radii[:, None] * np.cos(theta)[None, :]
    Y = F.CY + radii[:, None] * np.sin(theta)[None, :]
    H, aux = F._terrain(X, Y, aux=True)
    slope, fm, shore = _grid_masks(X, Y, H, aux, theta, radii)
    # buried skirt ring just inside r0 hides the T-junctions against the inner mesh
    co = np.empty(((nr + 1) * nc, 3), dtype=np.float32)
    co[:nc, 0] = F.CX + 0.985 * (X[0] - F.CX); co[:nc, 1] = F.CY + 0.985 * (Y[0] - F.CY); co[:nc, 2] = H[0] - 25.0
    co[nc:, 0] = X.ravel(); co[nc:, 1] = Y.ravel(); co[nc:, 2] = H.ravel()
    quads = grid_quads(nr + 1, nc, True)
    ob = new_mesh_object("landscape_terrain_far", co, quads, np.arange(len(quads) // 4) * 4, col, "ls_terrain")
    lm = np.zeros(((nr + 1) * nc, 4), dtype=np.float32)
    lm[:nc, 0] = fm[0]; lm[:nc, 1] = shore[0]
    lm[nc:, 0] = fm.ravel(); lm[nc:, 1] = shore.ravel(); lm[:, 3] = 1.0
    add_attr(ob.data, "lmask", lm, 'FLOAT_COLOR')
    _log("far terrain %dx%d (%dk tris)" % (nc, nr, len(quads) // 2000), t0)
    return ob


def build_terrain(col):
    t0 = time.time()
    theta, radii = _polar_layout()
    nr, nc = len(radii), len(theta)
    X = F.CX + radii[:, None] * np.cos(theta)[None, :]
    Y = F.CY + radii[:, None] * np.sin(theta)[None, :]
    H, aux = F._terrain(X, Y, aux=True)
    _log("terrain heights (%dk verts)" % (H.size // 1000), t0)
    slope, fm, shore = _grid_masks(X, Y, H, aux, theta, radii)
    ep = aux['ep']
    talus = aux['her'] * F.smoothstep(F.FOOT_E - 30.0, F.FOOT_E - 5.0, ep) * \
        (1.0 - F.smoothstep(F.FOOT_E + 70.0, F.FOOT_E + 150.0, ep))
    tg = dict(X=X, Y=Y, H=H, fm=fm, aux=aux, slope=slope, theta=theta, radii=radii)
    # the inner mesh uses the rings up to R_SPLIT; the rest of the grid only serves trees / mist
    nr = int(np.searchsorted(radii, R_SPLIT)) + 1
    tg['r_split'] = float(radii[nr - 1])
    X = X[:nr]; Y = Y[:nr]; H = H[:nr]; fm = fm[:nr]; shore = shore[:nr]; talus = talus[:nr]
    co = np.empty((nr * nc + 1, 3), dtype=np.float32)
    co[:-1, 0] = X.ravel(); co[:-1, 1] = Y.ravel(); co[:-1, 2] = H.ravel()
    co[-1] = (F.CX, F.CY, float(H[0].mean()))
    quads = grid_quads(nr, nc, True)
    j = np.arange(nc)
    fan = np.stack([np.full(nc, nr * nc), j, (j + 1) % nc], axis=1).reshape(-1)
    loops = np.concatenate([quads, fan])
    nq = len(quads) // 4
    loop_start = np.concatenate([np.arange(nq) * 4, nq * 4 + np.arange(nc) * 3])
    ob = new_mesh_object("landscape_terrain", co, loops, loop_start, col, "ls_terrain")
    lm = np.zeros((nr * nc + 1, 4), dtype=np.float32)
    lm[:-1, 0] = fm.ravel(); lm[:-1, 1] = shore.ravel(); lm[:-1, 2] = talus.ravel(); lm[:, 3] = 1.0
    add_attr(ob.data, "lmask", lm, 'FLOAT_COLOR')
    _log("terrain mesh (%dk tris)" % ((nq * 2 + nc) // 1000), t0)
    return ob, tg


# =====================================================================================
# water
# =====================================================================================
def build_water(col):
    n = 128
    radii = np.array([0.0, 60.0, 150.0, 300.0, 600.0, 1000.0, 1600.0, 2500.0, 4000.0, 6000.0, 9000.0, 13000.0,
                      WATER_R])
    th = np.arange(n) / n * 2.0 * np.pi
    cxw, cyw = -1500.0, 500.0
    X = cxw + radii[:, None] * np.cos(th)[None, :]
    Y = cyw + radii[:, None] * np.sin(th)[None, :]
    co = np.stack([X.ravel(), Y.ravel(), np.full(X.size, L.VALLEY_Z)], axis=1)
    quads = grid_quads(len(radii), n, True)
    ob = new_mesh_object("landscape_water", co, quads, np.arange(len(quads) // 4) * 4, col, "ls_water")
    ob.visible_shadow = False
    return ob


# =====================================================================================
def build(ctx):
    t0 = time.time()
    from . import landscape_mats, landscape_mist, landscape_cliff   # noqa: F401  (register the ls_* materials)
    col = get_col("landscape")
    terrain, tg = build_terrain(col)
    build_terrain_far(col, tg['r_split'])
    build_water(col)
    t1 = time.time()
    landscape_cliff.build_cliff(col, new_mesh_object, add_attr, grid_quads)
    t1 = time.time()
    nm = landscape_mist.build_mist(col, tg, new_mesh_object, grid_quads, add_attr)
    _log("mist (%d sheets)" % nm, t1)
    t1 = time.time()
    ob, nt_ = landscape_mist.build_trees(col, tg, new_mesh_object, add_attr)
    _log("far trees (%dk)" % (nt_ // 1000), t1)
    _log("total", t0)
