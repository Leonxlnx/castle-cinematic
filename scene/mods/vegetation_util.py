"""Low-level helpers for the vegetation module: numpy mesh accumulator, tubes, curves, instancing GN groups."""
import bpy, math
import numpy as np
from mathutils import Matrix, Vector
from ..common import NT, new_gn_group, add_gn_modifier

UP = np.array([0.0, 0.0, 1.0])


def nrm(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def smooth01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# =====================================================================================
# mesh accumulator (triangles only, per-vertex uv / sway / tint / optional custom normal)
# =====================================================================================
class Geo:
    def __init__(self):
        self.V = []; self.F = []; self.UV = []; self.M = []; self.S = []; self.N = []; self.C = []
        self.n = 0

    def add(self, v, f, uv=None, mat=0, sway=None, nrm=None, tint=None):
        v = np.asarray(v, np.float32).reshape(-1, 3)
        f = np.asarray(f, np.int32).reshape(-1, 3)
        k = len(v)
        self.V.append(v); self.F.append(f + self.n)
        self.UV.append(np.zeros((k, 2), np.float32) if uv is None else np.asarray(uv, np.float32).reshape(k, 2))
        self.M.append(np.full(len(f), mat, np.int32))
        self.S.append(np.zeros(k, np.float32) if sway is None
                      else np.broadcast_to(np.asarray(sway, np.float32), (k,)).copy())
        self.N.append(np.zeros((k, 3), np.float32) if nrm is None
                      else np.broadcast_to(np.asarray(nrm, np.float32), (k, 3)).copy())
        self.C.append(np.zeros(k, np.float32) if tint is None
                      else np.broadcast_to(np.asarray(tint, np.float32), (k,)).copy())
        self.n += k
        return self

    def quads(self, v, q, **kw):
        """q: (m,4) -> two triangles each."""
        q = np.asarray(q, np.int32).reshape(-1, 4)
        f = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]], axis=0)
        return self.add(v, f, **kw)

    def arrays(self):
        if not self.V:
            z = np.zeros
            return z((0, 3), np.float32), z((0, 3), np.int32), z((0, 2), np.float32), z(0, np.int32), \
                z(0, np.float32), z((0, 3), np.float32), z(0, np.float32)
        c = np.concatenate
        return c(self.V), c(self.F), c(self.UV), c(self.M), c(self.S), c(self.N), c(self.C)

    def add_geo(self, other, M=None, tint=None, sway=None, mat_map=None):
        """Stamp another Geo with a 4x4 transform (numpy or mathutils)."""
        V, F, UV, Mi, S, N, C = other.arrays()
        if len(V) == 0:
            return self
        if M is not None:
            Mn = np.array(M, dtype=np.float64)
            R = Mn[:3, :3]
            V = (V @ R.T + Mn[:3, 3]).astype(np.float32)
            has = np.abs(N).sum(axis=1) > 1e-9
            if has.any():
                N2 = N @ R.T
                N2[has] = nrm(N2[has])
                N = N2.astype(np.float32)
        if mat_map is not None:
            Mi = np.asarray(mat_map, np.int32)[Mi]
        self.V.append(V); self.F.append(F + self.n); self.UV.append(UV); self.M.append(Mi)
        self.S.append(S if sway is None else np.full(len(V), sway, np.float32))
        self.N.append(N)
        self.C.append(C if tint is None else np.full(len(V), tint, np.float32))
        self.n += len(V)
        return self

    @property
    def ntris(self):
        return sum(len(f) for f in self.F)

    def to_mesh(self, name, materials, custom_normals=False, smooth=True):
        V, F, UV, Mi, S, N, C = self.arrays()
        me = bpy.data.meshes.new(name)
        me.from_pydata(V.tolist(), [], F.tolist())
        for m in materials:
            me.materials.append(m)
        if len(F):
            me.polygons.foreach_set("material_index", Mi)
            me.polygons.foreach_set("use_smooth", np.full(len(F), smooth, dtype=bool))
            uvl = me.uv_layers.new(name="UVMap")
            uvl.data.foreach_set("uv", UV[F.ravel()].ravel())
            a = me.attributes.new("sway", 'FLOAT', 'POINT'); a.data.foreach_set("value", S)
            a = me.attributes.new("tint", 'FLOAT', 'POINT'); a.data.foreach_set("value", C)
            me.update()
            if custom_normals:
                me.normals_split_custom_set_from_vertices(N.tolist())
        return me

    def to_object(self, name, materials, collection=None, **kw):
        me = self.to_mesh(name, materials, **kw)
        ob = bpy.data.objects.new(name, me)
        if collection is not None:
            collection.objects.link(ob)
        return ob


def points_object(name, pos, collection, **attrs):
    """Vertex-only mesh carrying instance attributes (float, int or 3-vector arrays)."""
    pos = np.asarray(pos, np.float32).reshape(-1, 3)
    me = bpy.data.meshes.new(name)
    me.from_pydata(pos.tolist(), [], [])
    for k, v in attrs.items():
        v = np.asarray(v)
        if len(v) != len(pos):
            raise ValueError("attr %s length %d != %d" % (k, len(v), len(pos)))
        if len(pos) == 0:
            continue
        if v.ndim == 2:
            a = me.attributes.new(k, 'FLOAT_VECTOR', 'POINT'); a.data.foreach_set("vector", v.astype(np.float32).ravel())
        elif v.dtype.kind in 'iu':
            a = me.attributes.new(k, 'INT', 'POINT'); a.data.foreach_set("value", v.astype(np.int32))
        else:
            a = me.attributes.new(k, 'FLOAT', 'POINT'); a.data.foreach_set("value", v.astype(np.float32))
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    return ob


# =====================================================================================
# curves / tubes
# =====================================================================================
def hermite(p0, t0, p1, t1, n):
    s = np.linspace(0.0, 1.0, n + 1)[:, None]
    p0, t0, p1, t1 = (np.asarray(a, np.float64) for a in (p0, t0, p1, t1))
    return (2 * s ** 3 - 3 * s ** 2 + 1) * p0 + (s ** 3 - 2 * s ** 2 + s) * t0 + \
        (-2 * s ** 3 + 3 * s ** 2) * p1 + (s ** 3 - s ** 2) * t1


def gnarl(pts, rng, amp, waves=2.2, free_end=False):
    """Add a smooth random wobble to a polyline (zero at the start, zero at the end unless free_end)."""
    n = len(pts)
    s = np.linspace(0.0, 1.0, n)
    out = pts.copy()
    for k in range(2):
        d = nrm(rng.normal(size=3))
        ph = rng.uniform(0, math.tau)
        w = waves * (1 + k * 1.3) * rng.uniform(0.7, 1.3)
        env = np.sin(np.pi * s) if not free_end else np.sin(0.5 * np.pi * s)
        out += d * (np.sin(s * w * math.tau + ph) * env)[:, None] * (amp / (1 + k))
    return out


def arclen(pts):
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def sample_poly(pts, L, s):
    """Point + unit tangent at arclength s on polyline pts with cumulative lengths L."""
    s = min(max(s, 0.0), L[-1] - 1e-9)
    i = int(np.searchsorted(L, s, side='right')) - 1
    i = min(max(i, 0), len(pts) - 2)
    t = (s - L[i]) / max(L[i + 1] - L[i], 1e-9)
    p = pts[i] * (1 - t) + pts[i + 1] * t
    tan = nrm(pts[i + 1] - pts[i])
    return p, tan, i + t


def tube(geo, pts, rad, sides=6, mat=0, sway=None, v0=0.0, cap=False, twist=0.0):
    """Tube along polyline pts (n,3) with radii rad (n,).  UV: u around (metres), v along (metres)."""
    pts = np.asarray(pts, np.float64); n = len(pts)
    rad = np.broadcast_to(np.asarray(rad, np.float64), (n,))
    tan = np.zeros_like(pts)
    tan[1:-1] = pts[2:] - pts[:-2]; tan[0] = pts[1] - pts[0]; tan[-1] = pts[-1] - pts[-2]
    tan = nrm(tan)
    a = np.array([0.0, 0.0, 1.0]) if abs(tan[0][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    nx = nrm(np.cross(tan[0], a))
    N1 = np.zeros_like(pts); N1[0] = nx
    for i in range(1, n):
        v = N1[i - 1] - tan[i] * np.dot(N1[i - 1], tan[i])
        l = np.linalg.norm(v)
        N1[i] = v / l if l > 1e-9 else N1[i - 1]
    N2 = np.cross(tan, N1)
    ang = np.arange(sides + 1) * (math.tau / sides) + twist
    ca = np.cos(ang)[None, :, None]; sa = np.sin(ang)[None, :, None]
    dirs = N1[:, None, :] * ca + N2[:, None, :] * sa                      # (n, sides+1, 3)
    V = pts[:, None, :] + dirs * rad[:, None, None]
    L = arclen(pts) + v0
    rmean = float(rad.mean())
    UV = np.zeros((n, sides + 1, 2))
    UV[:, :, 0] = (np.arange(sides + 1) / sides * math.tau * rmean)[None, :]
    UV[:, :, 1] = L[:, None]
    idx = np.arange(n * (sides + 1)).reshape(n, sides + 1)
    a_ = idx[:-1, :-1].ravel(); b_ = idx[:-1, 1:].ravel(); c_ = idx[1:, 1:].ravel(); d_ = idx[1:, :-1].ravel()
    F = np.concatenate([np.stack([a_, b_, c_], 1), np.stack([a_, c_, d_], 1)])
    sw = None
    if sway is not None:
        sw = np.repeat(np.broadcast_to(np.asarray(sway, np.float64), (n,)), sides + 1)
    geo.add(V.reshape(-1, 3), F, uv=UV.reshape(-1, 2), mat=mat, sway=sw)
    if cap:
        k = geo.n
        ring = V[-1, :sides]
        geo.add(np.concatenate([ring, pts[-1:] + tan[-1:] * rad[-1] * 0.6]),
                [(i, (i + 1) % sides, sides) for i in range(sides)], mat=mat,
                sway=None if sway is None else float(np.broadcast_to(sway, (n,))[-1]),
                uv=np.zeros((sides + 1, 2)))
    return geo


def basis_matrix(pos, X, Y, Z, scale=1.0):
    M = np.eye(4)
    M[:3, 0] = np.asarray(X) * scale; M[:3, 1] = np.asarray(Y) * scale; M[:3, 2] = np.asarray(Z) * scale
    M[:3, 3] = pos
    return M


def frame_from_y(Y, up_hint=UP, roll=0.0):
    """Orthonormal frame with +Y along Y and +Z as close as possible to up_hint (then rolled about Y)."""
    Y = nrm(Y)
    Z = np.asarray(up_hint, np.float64) - Y * np.dot(up_hint, Y)
    if np.linalg.norm(Z) < 1e-6:
        Z = np.array([1.0, 0.0, 0.0]) - Y * Y[0]
    Z = nrm(Z)
    X = np.cross(Y, Z)
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        X, Z = X * c + Z * s, Z * c - X * s
    return X, Y, Z


def euler_from_axes(X, Y, Z):
    """XYZ euler angles (numpy, vectorised) of the rotation whose columns are X, Y, Z.  Arrays (n,3)."""
    X = np.atleast_2d(X); Y = np.atleast_2d(Y); Z = np.atleast_2d(Z)
    r20 = X[:, 2]; r21 = Y[:, 2]; r22 = Z[:, 2]; r10 = X[:, 1]; r00 = X[:, 0]
    b = -np.arcsin(np.clip(r20, -1.0, 1.0))
    a = np.arctan2(r21, r22)
    c = np.arctan2(r10, r00)
    return np.stack([a, b, c], axis=1)


# =====================================================================================
# camera helpers (pruning foliage out of sight lines)
# =====================================================================================
def cam_basis(pos, tgt):
    pos = np.asarray(pos, np.float64); tgt = np.asarray(tgt, np.float64)
    F = nrm(tgt - pos)
    R = nrm(np.cross(F, UP))
    U = np.cross(R, F)
    return pos, F, R, U


def cam_ndc(P, cam, aspect=16.0 / 9.0):
    """cam = (pos, target, lens).  Returns depth f, ndc x (right), ndc y (up); |ndc| <= 1 is inside the frame."""
    pos, F, R, U = cam_basis(cam[0], cam[1])
    d = np.atleast_2d(P) - pos
    f = d @ F
    th = 18.0 / cam[2]
    fs = np.where(np.abs(f) < 1e-6, 1e-6, f)
    return f, (d @ R) / (fs * th), (d @ U) / (fs * th / aspect)


def cam_point(cam, f, nx, ny, aspect=16.0 / 9.0):
    pos, F, R, U = cam_basis(cam[0], cam[1])
    th = 18.0 / cam[2]
    return pos + F * f + R * (nx * th * f) + U * (ny * th / aspect * f)


def _catmull(pts, u):
    """Same spline as scene/mods/cameras.py (uniform Catmull-Rom through the control points)."""
    P = [np.asarray(p, np.float64) for p in pts]
    n = len(P)
    if n == 1:
        return P[0].copy()
    if n == 2:
        return P[0] * (1 - u) + P[1] * u
    x = u * (n - 1)
    i = min(int(x), n - 2)
    t = x - i
    p0 = P[max(i - 1, 0)]; p1 = P[i]; p2 = P[i + 1]; p3 = P[min(i + 2, n - 1)]
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def view_samples(nshot=9):
    """All views the film / look-dev uses: [(name, pos, target, lens)].
    layout.CAMERAS (static) + the shots of scene/mods/cameras.py sampled along their paths (if that module exists)."""
    from .. import layout as L
    out = [(n, np.asarray(p, float), np.asarray(t, float), float(l)) for n, (p, t, l) in L.CAMERAS.items()]
    try:
        from . import cameras as C
        for sh in C.SHOTS:
            lens = sh['lens']
            for k in range(nshot):
                u = k / (nshot - 1)
                p = _catmull(sh['pos'], u); t = _catmull(sh['tgt'], u)
                l = lens[0] + (lens[1] - lens[0]) * u if isinstance(lens, (tuple, list)) else float(lens)
                out.append(("%s@%.2f" % (sh['name'], u), p, t, float(l)))
    except Exception as e:
        print("[veg] cameras.SHOTS not available (%s): using layout.CAMERAS only" % type(e).__name__)
    return out


class ViewSet:
    """Vectorised frustum tests for a list of (name, pos, target, lens) views (16:9)."""

    def __init__(self, views, aspect=16.0 / 9.0):
        self.names = [v[0] for v in views]
        self.pos = np.array([v[1] for v in views], float).reshape(-1, 3)
        k = len(views)
        self.F = np.zeros((k, 3)); self.R = np.zeros((k, 3)); self.U = np.zeros((k, 3))
        self.tx = np.zeros(k); self.ty = np.zeros(k)
        for i, v in enumerate(views):
            _, F, R, U = cam_basis(v[1], v[2])
            self.F[i], self.R[i], self.U[i] = F, R, U
            self.tx[i] = 18.0 / v[3]; self.ty[i] = 18.0 / v[3] / aspect

    def __len__(self):
        return len(self.pos)

    def ndc(self, P):
        """P (n,3) -> depth f, ndc x, ndc y, distance; all (k, n)."""
        d = np.atleast_2d(P)[None, :, :] - self.pos[:, None, :]
        f = np.einsum('knj,kj->kn', d, self.F)
        fs = np.where(np.abs(f) < 1e-6, 1e-6, f)
        x = np.einsum('knj,kj->kn', d, self.R) / (fs * self.tx[:, None])
        y = np.einsum('knj,kj->kn', d, self.U) / (fs * self.ty[:, None])
        return f, x, y, np.linalg.norm(d, axis=2)


# =====================================================================================
# geometry nodes: instancer with optional wind
# =====================================================================================
def wind_offset(nt, sway_sock, amp=0.05, freq=0.22, speed=0.35):
    """Field: position offset = sway * amp * (noise(P*freq, t*speed) - 0.5) (plus a slow shared gust)."""
    pos = nt.n('GeometryNodeInputPosition').outputs[0]
    tm = nt.n('GeometryNodeInputSceneTime').outputs['Seconds']
    noise = nt.n('ShaderNodeTexNoise', noise_dimensions='4D', Vector=nt.vmath('SCALE', pos, scale=freq),
                 W=nt.math('MULTIPLY', tm, speed), Scale=1.0, Detail=1.0, Roughness=0.5)
    v = nt.vmath('SUBTRACT', noise.outputs['Color'], (0.5, 0.5, 0.5))
    v = nt.vmath('MULTIPLY', v, (1.0, 1.0, 0.45))
    return nt.vmath('SCALE', v, scale=nt.math('MULTIPLY', sway_sock, amp * 2.0))


def named_attr(nt, name, typ='FLOAT'):
    return nt.n('GeometryNodeInputNamedAttribute', data_type=typ, Name=name).outputs[0]


def gn_wind_mesh(name, amp=0.05):
    """GN group that sways a mesh by its 'sway' vertex attribute (frame driven, deterministic)."""
    g, nt, gi, go = new_gn_group(name)
    off = wind_offset(nt, named_attr(nt, 'sway'), amp=amp)
    sp = nt.n('GeometryNodeSetPosition', Geometry=gi.outputs[0], Offset=off)
    nt.link(sp.outputs[0], go.inputs[0])
    return g


def gn_instancer(name, proto_col, wind_amp=0.0, flutter=0.0, flutter_speed=1.6, flutter_freq=1.7):
    """Points (mesh vertices with attributes rot (euler vector), scl (vector or float), pid (int), optional
    sway (float)) -> instances of the children of proto_col (picked by pid, alphabetical order)."""
    g, nt, gi, go = new_gn_group(name)
    geo = gi.outputs[0]
    if wind_amp > 0:
        off = wind_offset(nt, named_attr(nt, 'sway'), amp=wind_amp)
        geo = nt.n('GeometryNodeSetPosition', Geometry=geo, Offset=off).outputs[0]
    ci = nt.n('GeometryNodeCollectionInfo', Separate_Children=True, Reset_Children=True)
    ci.inputs['Collection'].default_value = proto_col
    rot = nt.n('FunctionNodeEulerToRotation', Euler=named_attr(nt, 'rot', 'FLOAT_VECTOR')).outputs[0]
    iop = nt.n('GeometryNodeInstanceOnPoints', Points=geo, Instance=ci.outputs[0], Pick_Instance=True,
               Instance_Index=named_attr(nt, 'pid', 'INT'), Rotation=rot,
               Scale=named_attr(nt, 'scl', 'FLOAT_VECTOR'))
    inst = iop.outputs[0]
    if flutter > 0:
        pos = nt.n('GeometryNodeInputPosition').outputs[0]
        tm = nt.n('GeometryNodeInputSceneTime').outputs['Seconds']
        noise = nt.n('ShaderNodeTexNoise', noise_dimensions='4D', Vector=nt.vmath('SCALE', pos, scale=flutter_freq),
                     W=nt.math('MULTIPLY', tm, flutter_speed), Scale=1.0, Detail=1.0, Roughness=0.6)
        v = nt.vmath('SUBTRACT', noise.outputs['Color'], (0.5, 0.5, 0.5))
        v = nt.vmath('SCALE', v, scale=flutter * 2.0)
        r2 = nt.n('FunctionNodeEulerToRotation', Euler=v).outputs[0]
        inst = nt.n('GeometryNodeRotateInstances', Instances=inst, Rotation=r2, Local_Space=True).outputs[0]
    nt.link(inst, go.inputs[0])
    return g


def proto_collection(name):
    """Collection that is NOT linked into the scene: holds instancing prototypes."""
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
    return c
