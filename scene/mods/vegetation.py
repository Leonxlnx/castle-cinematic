"""VEGETATION module: garden plants at the layout anchors + forest / wild vegetation on the natural terrain.

Builds (collection "vegetation"; instancing prototypes live in unlinked collections "veg_proto_*"):
  A. garden (vegetation_garden.py)
     * red Japanese maples at layout.MAPLES   (tube skeleton mesh + leaf-spray instances, frame-driven wind)
     * Italian cypresses at layout.CYPRESSES
     * white-blossom shrubs + trailing greenery + red accents in layout.PLANTERS
  B. forest (vegetation_forest.py): conifers / deciduous / bare trees in 3 LODs, bushes, grass tufts,
     placed with landscape.height_at / landscape.forest_mask (flat-plateau fallback if the landscape module
     is not available yet).

Tunable constants: see the top of vegetation_garden.py and vegetation_forest.py; SEED below.
"""
import math, time, os
import numpy as np
import bpy
from .. import layout as L, mats
from ..common import get_col

SEED = 7
# debug switches (environment variables, so shared files stay untouched):
#   VEG_ONLY=maples,cypresses,shrubs,forest   build only these parts
#   VEG_FLAT=1                                ignore the landscape module (flat plateau fallback)
ONLY = tuple(x for x in os.environ.get("VEG_ONLY", "").split(",") if x)
FORCE_FLAT = bool(os.environ.get("VEG_FLAT", ""))
DEBUG_GROUND = True      # isolated test builds (--mods vegetation) get a stand-in ground + terrace slabs + vc_* cameras


def _plateau_polygon():
    pts = list(L.CLIFF_TOP)
    return np.array([(1600.0, -420.0)] + pts + [(760.0, 940.0), (1600.0, 940.0)])


def _in_poly(x, y, poly):
    x = np.asarray(x, float); y = np.asarray(y, float)
    inside = np.zeros(x.shape, bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]; x1, y1 = poly[(i + 1) % n]
        c = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-30) + x0)
        inside ^= c
    return inside


class Env:
    def __init__(self, ctx):
        self.ctx = ctx
        self.col = ctx.col("vegetation")
        self.L = L
        self.poly = _plateau_polygon()
        self._h = None; self._fm = None
        try:
            if FORCE_FLAT:
                raise ImportError("VEG_FLAT")
            from . import landscape as _ls
            h = getattr(_ls, "height_at", None); fm = getattr(_ls, "forest_mask", None)
            if callable(h) and callable(fm):
                # smoke test the functions before trusting them
                tx = np.array([100.0, 300.0]); ty = np.array([0.0, 50.0])
                a = np.asarray(h(tx, ty), float); b = np.asarray(fm(tx, ty), float)
                if a.shape == tx.shape and b.shape == tx.shape:
                    self._h = h; self._fm = fm
        except Exception as e:           # module missing / broken: flat fallback
            print("[veg] landscape functions unavailable (%s) - using flat plateau fallback" % (type(e).__name__,))
        self.has_landscape = self._h is not None
        print("[veg] ground source:", "landscape.height_at/forest_mask" if self.has_landscape else "FLAT FALLBACK")

    def rng(self, seed):
        return np.random.default_rng(SEED * 10007 + seed)

    def on_plateau(self, x, y):
        return _in_poly(x, y, self.poly)

    def height(self, x, y):
        """Ground z (arrays).  Fallback: flat plateau east of the cliff line, -9999 elsewhere."""
        x = np.asarray(x, float); y = np.asarray(y, float)
        if self._h is not None:
            return np.asarray(self._h(x, y), float)
        return np.where(self.on_plateau(x, y), L.PLATEAU_BASE_Z, -9999.0)

    def height1(self, x, y, default=None):
        h = float(self.height(np.array([x], float), np.array([y], float))[0])
        return default if (h < -5000.0 and default is not None) else h

    def forest_mask(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        if self._fm is not None:
            return np.clip(np.asarray(self._fm(x, y), float), 0.0, 1.0)
        ok = self.on_plateau(x, y)
        for dx, dy in ((5, 0), (-5, 0), (0, 5), (0, -5)):          # keep a margin to the cliff edge
            ok &= self.on_plateau(x + dx, y + dy)
        return np.where(ok, 1.0, 0.0)


def _debug_ground(env):
    from ..common import MB
    from . import vegetation_mats  # noqa
    col = get_col("veg_debug")
    n = 180
    xs = np.linspace(-700, 900, n + 1); ys = np.linspace(-700, 900, n + 1)
    X, Y = np.meshgrid(xs, ys)
    Z = env.height(X.ravel(), Y.ravel()).reshape(X.shape)
    Z = np.where(Z < -5000, -330.0, Z)
    for (x0, x1, y0, y1) in L.BUILT_AREAS:
        inb = (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
        Z = np.where(inb, np.minimum(Z, -3.0), Z)
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    idx = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
    Q = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
    me = bpy.data.meshes.new("veg_dbg_ground")
    me.from_pydata(V.tolist(), [], Q.tolist())
    me.materials.append(mats.get("veg_dbg_ground"))
    ob = bpy.data.objects.new("veg_dbg_ground", me); col.objects.link(ob)
    mb = MB("veg_dbg_paving")
    mb.box(L.TERRACE_X0, L.TERRACE_X1, L.TERRACE_Y0, L.TERRACE_Y1, -3.0, 0.0, "marble_wet")
    mb.box(L.LOUNGE_X0, L.LOUNGE_X1, L.LOUNGE_Y0, L.LOUNGE_Y1, -4.0, L.LOUNGE_Z, "marble_wet")
    mb.box(-60, 62, L.PODIUM_Y0, L.PODIUM_Y1, -3.0, L.PODIUM_Z, "marble_wet")
    for (nm, x0, x1, y0, y1, zt) in L.CASTLE_BLOCKS:
        mb.box(x0, x1, y0, y1, 0.0, zt, "stone")
    for (x0, x1, y0, y1, zb) in L.PLANTERS:
        mb.box(x0, x1, y0, y1, zb, zb + 0.45, "stone_trim")
    mb.to_object("veg_dbg_paving", col)


DEBUG_CAMS = {
    "vc_maple_w": ((24.0, -56.0, 0.6), (44.0, -66.0, 7.5), 24.0),
    "vc_maple_far": ((10.0, -30.0, 6.0), (45.0, -67.0, 5.0), 35.0),
    "vc_backlit": ((45.0, -69.5, 9.6), (37.0, -65.5, 10.6), 50.0),
    "vc_leaf": ((41.6, -70.6, 10.0), (41.0, -68.2, 11.0), 50.0),
    "vc_cyp": ((38.0, 4.0, 3.0), (45.2, 16.0, 5.0), 35.0),
    "vc_cyp_row": ((-34.0, 8.0, 5.0), (-22.0, 25.6, 7.0), 35.0),
    "vc_shrub": ((30.0, -64.4, 0.3), (31.2, -68.6, -0.4), 35.0),
    "vc_shrub2": ((23.0, -57.0, 0.5), (17.6, -60.0, -0.4), 35.0),
    "vc_garden": ((85.0, -85.0, 38.0), (25.0, -25.0, 0.0), 28.0),
    "vc_forest": ((70.0, -30.0, 3.0), (130.0, 20.0, 12.0), 35.0),
    "vc_forest2": ((60.0, -60.0, 12.0), (150.0, -40.0, 8.0), 50.0),
    "vc_forest_up": ((150.0, -150.0, 120.0), (60.0, 60.0, 0.0), 35.0),
}


def _debug_cams(ctx):
    from ..setup import custom_camera
    for name, (p, t, lens) in DEBUG_CAMS.items():
        custom_camera(ctx.scene, p, t, lens, name=name)


def build(ctx):
    t0 = time.time()
    env = Env(ctx)
    from . import vegetation_garden as G
    if DEBUG_GROUND and set(ctx.mods) <= {"vegetation", "cameras"}:
        _debug_ground(env)
    from . import vegetation_shrubs as S
    isolated = set(ctx.mods) <= {"vegetation", "cameras"}
    if isolated:
        _debug_cams(ctx)
    tris = {}
    from . import vegetation_forest as F
    for name, fn in (("maples", G.build_maples), ("cypresses", G.build_cypresses), ("shrubs", S.build_shrubs),
                     ("forest", F.build_forest)):
        if ONLY and name not in ONLY:
            continue
        t = time.time(); tris[name] = fn(env); print("[veg] %s %.1fs" % (name, time.time() - t))
    print("[veg] triangles:", tris, " garden ~%d  forest ~%d" % (
        sum(v for k, v in tris.items() if k != "forest"), tris.get("forest", 0)))
    print("[veg] build %.1fs" % (time.time() - t0))
