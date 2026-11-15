"""Shared helpers for all scene modules (Blender 5.2, EEVEE).

Key pieces
  get_col(name, parent)            collection helper
  MB                               fast mesh builder (many primitives -> ONE mesh object, auto UVs in meters)
  NT                               tiny node-tree DSL used by mats.py and module shaders
  look_at(obj, target)             aim cameras / lights
  add_gn_modifier / new_gn_group   geometry-nodes helpers
"""
import bpy, bmesh, math, os, sys, random
import numpy as np
from mathutils import Vector, Matrix, geometry as mgeo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
TAU = math.tau


# =====================================================================================
# collections / objects
# =====================================================================================
def get_col(name, parent=None):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(c)
    return c


def link_obj(obj, collection):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def new_obj(name, data, collection, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
    o = bpy.data.objects.new(name, data)
    collection.objects.link(o)
    o.location = loc; o.rotation_euler = rot; o.scale = scale
    return o


def look_at(obj, target, roll=0.0):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    if roll:
        obj.rotation_euler.rotate_axis('Z', roll)


def rot_z(a):
    return Matrix.Rotation(a, 4, 'Z')


def xf(loc=(0, 0, 0), rz=0.0, scale=1.0, rx=0.0, ry=0.0):
    """Compose a 4x4 transform: scale -> rotate (x, y, z) -> translate."""
    m = Matrix.Translation(Vector(loc))
    if rz: m = m @ Matrix.Rotation(rz, 4, 'Z')
    if ry: m = m @ Matrix.Rotation(ry, 4, 'Y')
    if rx: m = m @ Matrix.Rotation(rx, 4, 'X')
    if scale != 1.0:
        if isinstance(scale, (int, float)):
            m = m @ Matrix.Scale(scale, 4)
        else:
            m = m @ Matrix.Diagonal((scale[0], scale[1], scale[2], 1.0))
    return m


def frame_xf(origin, xaxis_xy):
    """Local frame: local +X along horizontal direction xaxis_xy, local +Z up, local +Y = 'into the wall'
    (90deg CCW of X).  Used to place facade elements: build in local coords with the OUTSIDE at -Y."""
    a = math.atan2(xaxis_xy[1], xaxis_xy[0])
    return Matrix.Translation(Vector(origin)) @ Matrix.Rotation(a, 4, 'Z')


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# =====================================================================================
# mesh builder
# =====================================================================================
class MB:
    """Accumulates geometry for one mesh object.

    * all coordinates in meters, local to the current transform stack (push/pop)
    * each face carries a material slot (by material NAME from mats.py), optional explicit UVs,
      and a per-face random value `rnd` (exposed to shaders as the face attribute "rnd")
    * faces without explicit UVs get box-projected UVs in METERS (u along the wall, v = z; or x,y on floors)
    * to_object() recalculates outward normals, so winding order does not matter for closed shells
    """

    def __init__(self, name="mesh"):
        self.name = name
        self.v = []          # list of (x,y,z)
        self.f = []          # list of index tuples
        self.fm = []         # material index per face
        self.fuv = []        # per face: tuple of (u,v) per corner, or None
        self.frnd = []       # per face float
        self.mats = []       # material names
        self.stack = [Matrix.Identity(4)]
        self.rnd = 0.0       # current per-face random value

    # ---- transform stack
    def push(self, m):
        self.stack.append(self.stack[-1] @ m)
        return self

    def pop(self):
        self.stack.pop()
        return self

    @property
    def M(self):
        return self.stack[-1]

    def mat(self, name):
        if name not in self.mats:
            self.mats.append(name)
        return self.mats.index(name)

    # ---- core add
    def add(self, verts, faces, mat, uvs=None, m=None):
        """verts: iterable of 3-tuples (local), faces: iterable of local index tuples.
        uvs: optional list (one entry per face) of per-corner (u,v) tuples."""
        M = self.M if m is None else self.M @ m
        base = len(self.v)
        if not isinstance(verts, np.ndarray):
            verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        Mn = np.array(M)
        w = verts @ Mn[:3, :3].T + Mn[:3, 3]
        self.v.extend(map(tuple, w.tolist()))
        mi = self.mat(mat)
        n = 0
        for fc in faces:
            self.f.append(tuple(base + i for i in fc)); n += 1
        self.fm.extend([mi] * n)
        self.frnd.extend([self.rnd] * n)
        if uvs is None:
            self.fuv.extend([None] * n)
        else:
            self.fuv.extend(uvs)
        return base

    def stamp(self, other, m=None, rnd=None):
        """Copy all geometry of another MB into this one with transform m (instancing by copy)."""
        if not other.v:
            return
        M = self.M if m is None else self.M @ m
        base = len(self.v)
        Mn = np.array(M)
        w = np.asarray(other.v, dtype=np.float64) @ Mn[:3, :3].T + Mn[:3, 3]
        self.v.extend(map(tuple, w.tolist()))
        remap = [self.mat(n) for n in other.mats]
        self.f.extend([tuple(base + i for i in fc) for fc in other.f])
        self.fm.extend([remap[i] for i in other.fm])
        self.fuv.extend(other.fuv)
        if rnd is None:
            self.frnd.extend(other.frnd)
        else:
            self.frnd.extend([rnd] * len(other.f))

    # ---- primitives
    def quad(self, a, b, c, d, mat, uv=None):
        self.add([a, b, c, d], [(0, 1, 2, 3)], mat, None if uv is None else [uv])

    def tri(self, a, b, c, mat):
        self.add([a, b, c], [(0, 1, 2)], mat)

    def box(self, x0, x1, y0, y1, z0, z1, mat, skip=""):
        """Axis aligned box.  skip: string containing any of '+x -x +y -y +z -z' faces to omit."""
        v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        fs = {"-z": (0, 3, 2, 1), "+z": (4, 5, 6, 7), "-y": (0, 1, 5, 4),
              "+x": (1, 2, 6, 5), "+y": (2, 3, 7, 6), "-x": (3, 0, 4, 7)}
        faces = [f for k, f in fs.items() if k not in skip]
        self.add(v, faces, mat)

    def cbox(self, cx, cy, z0, sx, sy, h, mat, skip=""):
        """Box by centre (cx,cy), base z0, size sx,sy and height h."""
        self.box(cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, z0, z0 + h, mat, skip)

    def prism(self, poly, z0, z1, mat, cap_top=True, cap_bottom=False, mat_top=None):
        """Vertical prism from a plan polygon [(x,y),...] (any winding, may be concave for the sides;
        caps are ear-clipped)."""
        n = len(poly)
        v = [(p[0], p[1], z0) for p in poly] + [(p[0], p[1], z1) for p in poly]
        faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
        self.add(v, faces, mat)
        if cap_top or cap_bottom:
            tris = mgeo.tessellate_polygon([[Vector((p[0], p[1], 0)) for p in poly]])
            if cap_top:
                pts = [(p[0], p[1], z1) for p in poly]
                self.add(pts, self._orient(pts, tris, (0, 0, 1)), mat_top or mat)
            if cap_bottom:
                pts = [(p[0], p[1], z0) for p in poly]
                self.add(pts, self._orient(pts, tris, (0, 0, -1)), mat)

    def frustum(self, poly0, z0, poly1, z1, mat, cap_top=True, mat_top=None):
        """Loft between two plan polygons with the same vertex count (tapered tower, hipped roof...)."""
        n = len(poly0)
        v = [(p[0], p[1], z0) for p in poly0] + [(p[0], p[1], z1) for p in poly1]
        faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
        self.add(v, faces, mat)
        if cap_top:
            tris = mgeo.tessellate_polygon([[Vector((p[0], p[1], 0)) for p in poly1]])
            pts = [(p[0], p[1], z1) for p in poly1]
            self.add(pts, self._orient(pts, tris, (0, 0, 1)), mat_top or mat)

    def lathe(self, profile, segs, mat, center=(0, 0), a0=0.0, a1=TAU, cap_top=False, cap_bottom=False,
              uscale=None):
        """Revolve profile [(r, z), ...] (bottom -> top) around the vertical axis through `center`.
        UVs: u = arc length at the largest radius (meters), v = z."""
        closed = abs((a1 - a0) - TAU) < 1e-6
        cols = segs if closed else segs + 1
        rmax = uscale if uscale else max(p[0] for p in profile)
        verts = []; faces = []; uvs = []
        for i in range(cols):
            a = a0 + (a1 - a0) * i / segs
            ca, sa = math.cos(a), math.sin(a)
            for (r, z) in profile:
                verts.append((center[0] + r * ca, center[1] + r * sa, z))
        m = len(profile)
        for i in range(segs):
            i1 = (i + 1) % cols
            ua = (a1 - a0) * i / segs * rmax; ub = (a1 - a0) * (i + 1) / segs * rmax
            for j in range(m - 1):
                faces.append((i * m + j, i1 * m + j, i1 * m + j + 1, i * m + j + 1))
                za, zb = profile[j][1], profile[j + 1][1]
                if abs(zb - za) < 1e-6:      # horizontal ring: use radius for v so it is not degenerate
                    za, zb = profile[j][0], profile[j + 1][0]
                uvs.append(((ua, za), (ub, za), (ub, zb), (ua, zb)))
        self.add(verts, faces, mat, uvs)
        if closed:
            if cap_top:
                r, z = profile[-1]
                self.add([(center[0] + r * math.cos(TAU * i / segs), center[1] + r * math.sin(TAU * i / segs), z)
                          for i in range(segs)], [tuple(range(segs))], mat)
            if cap_bottom:
                r, z = profile[0]
                self.add([(center[0] + r * math.cos(TAU * i / segs), center[1] + r * math.sin(TAU * i / segs), z)
                          for i in range(segs)], [tuple(range(segs))], mat)

    def cyl(self, cx, cy, z0, r, h, mat, segs=16, cap_top=True, cap_bottom=False, r_top=None):
        self.lathe([(r, z0), (r if r_top is None else r_top, z0 + h)], segs, mat, (cx, cy),
                   cap_top=cap_top, cap_bottom=cap_bottom)

    def sweep(self, profile, path, mat, closed=False, uv_v_scale=1.0):
        """Sweep a 2D profile along a horizontal plan path with mitred corners.
        profile: [(d, z), ...]  d = offset to the RIGHT of the travel direction (i.e. outward when the
                 path runs CCW around a building... use negative d to go left), z = absolute height.
        path:    [(x, y), ...] plan polyline.
        Great for plinths, string courses, cornices, copings, kerbs, pool edges."""
        n = len(path)
        P = [Vector((p[0], p[1])) for p in path]
        offs = []
        for i in range(n):
            if closed:
                a, b, c = P[(i - 1) % n], P[i], P[(i + 1) % n]
            else:
                a = P[i - 1] if i > 0 else None
                b = P[i]
                c = P[i + 1] if i < n - 1 else None
            d1 = (b - a).normalized() if a is not None else None
            d2 = (c - b).normalized() if c is not None else None
            if d1 is None: d1 = d2
            if d2 is None: d2 = d1
            n1 = Vector((d1.y, -d1.x)); n2 = Vector((d2.y, -d2.x))
            mdir = (n1 + n2)
            if mdir.length < 1e-6:
                mdir = n1
            mdir.normalize()
            k = 1.0 / max(0.25, mdir.dot(n1))
            offs.append(mdir * k)
        m = len(profile)
        verts = []
        for i in range(n):
            for (d, z) in profile:
                q = P[i] + offs[i] * d
                verts.append((q.x, q.y, z))
        # cumulative length for UVs
        L = [0.0]
        for i in range(1, n):
            L.append(L[-1] + (P[i] - P[i - 1]).length)
        Lc = L[-1] + (P[0] - P[-1]).length
        # profile arc length
        S = [0.0]
        for j in range(1, m):
            S.append(S[-1] + math.hypot(profile[j][0] - profile[j - 1][0], profile[j][1] - profile[j - 1][1]))
        faces = []; uvs = []
        segs = n if closed else n - 1
        for i in range(segs):
            i1 = (i + 1) % n
            u0 = L[i]; u1 = L[i + 1] if i + 1 < n else Lc
            for j in range(m - 1):
                faces.append((i * m + j, i1 * m + j, i1 * m + j + 1, i * m + j + 1))
                uvs.append(((u0, S[j] * uv_v_scale), (u1, S[j] * uv_v_scale),
                            (u1, S[j + 1] * uv_v_scale), (u0, S[j + 1] * uv_v_scale)))
        self.add(verts, faces, mat, uvs)

    def poly_holes(self, outer, holes, mat, to3d=None, normal=(0.0, -1.0, 0.0)):
        """Planar polygon with holes, triangulated.  outer/holes: lists of 2D points (a,b).
        to3d: function (a,b) -> (x,y,z); default maps (a,b) -> (a, 0, b) i.e. a wall in the local XZ plane.
        normal: desired face normal in LOCAL coords (default -Y = facade outside); triangles are flipped to match."""
        if to3d is None:
            to3d = lambda a, b: (a, 0.0, b)
        loops = [[Vector((p[0], p[1], 0)) for p in outer]] + [[Vector((p[0], p[1], 0)) for p in h] for h in holes]
        tris = mgeo.tessellate_polygon(loops)
        flat = [p for lp in ([outer] + list(holes)) for p in lp]
        pts = [to3d(p[0], p[1]) for p in flat]
        if tris and normal is not None:
            a, b, c = (Vector(pts[i]) for i in tris[0])
            if (b - a).cross(c - a).dot(Vector(normal)) < 0:
                tris = [(t[0], t[2], t[1]) for t in tris]
        self.add(pts, tris, mat)

    def sweep_xz(self, profile, path, mat, closed=True, flip=False):
        """Sweep a profile along a path lying in the local XZ plane (i.e. on a facade whose outside is -Y).
        profile: [(d, y), ...]  d = in-plane offset to the RIGHT of the travel direction (for a CCW opening
                 outline seen from outside, d>0 goes away from the opening into the wall, d<0 into the opening),
                 y = local y (negative = sticking out of the wall, positive = recessed).
        path: [(x, z), ...].  Used for window reveals, hood moulds, frames, tracery bars."""
        n = len(path)
        P = [Vector((p[0], p[1])) for p in path]
        offs = []
        for i in range(n):
            if closed:
                a, b, c = P[(i - 1) % n], P[i], P[(i + 1) % n]
            else:
                a = P[i - 1] if i > 0 else None
                b = P[i]
                c = P[i + 1] if i < n - 1 else None
            d1 = (b - a).normalized() if a is not None and (b - a).length > 1e-9 else None
            d2 = (c - b).normalized() if c is not None and (c - b).length > 1e-9 else None
            if d1 is None: d1 = d2
            if d2 is None: d2 = d1
            n1 = Vector((d1.y, -d1.x)); n2 = Vector((d2.y, -d2.x))
            md = n1 + n2
            if md.length < 1e-6:
                md = n1
            md.normalize()
            offs.append(md * (1.0 / max(0.3, md.dot(n1))))
        m = len(profile)
        verts = []
        for i in range(n):
            for (d, y) in profile:
                q = P[i] + offs[i] * d
                verts.append((q.x, y, q.y))
        faces = []
        segs = n if closed else n - 1
        for i in range(segs):
            i1 = (i + 1) % n
            for j in range(m - 1):
                if flip:
                    faces.append((i * m + j, i1 * m + j, i1 * m + j + 1, i * m + j + 1))
                else:
                    faces.append((i * m + j, i * m + j + 1, i1 * m + j + 1, i1 * m + j))
        self.add(verts, faces, mat)

    @staticmethod
    def _orient(pts, tris, normal):
        if tris:
            a, b, c = (Vector(pts[i]) for i in tris[0])
            if (b - a).cross(c - a).dot(Vector(normal)) < 0:
                return [(t[0], t[2], t[1]) for t in tris]
        return tris

    def extrude_xz(self, outline, y0, y1, mat, cap0=True, cap1=True):
        """Prism along local Y from a CCW outline [(x, z), ...] in the XZ plane (seen from -Y)."""
        n = len(outline)
        v = [(p[0], y0, p[1]) for p in outline] + [(p[0], y1, p[1]) for p in outline]
        faces = [(i, n + i, n + (i + 1) % n, (i + 1) % n) for i in range(n)]
        self.add(v, faces, mat)
        if cap0 or cap1:
            tris = mgeo.tessellate_polygon([[Vector((p[0], p[1], 0)) for p in outline]])
            if cap0:
                pts = [(p[0], y0, p[1]) for p in outline]
                self.add(pts, self._orient(pts, tris, (0, -1 if y0 < y1 else 1, 0)), mat)
            if cap1:
                pts = [(p[0], y1, p[1]) for p in outline]
                self.add(pts, self._orient(pts, tris, (0, 1 if y0 < y1 else -1, 0)), mat)

    def extrude_yz(self, outline, x0, x1, mat, cap0=True, cap1=True):
        """Prism along local X from an outline [(y, z), ...] in the YZ plane (buttresses, gablets, steps)."""
        n = len(outline)
        v = [(x0, p[0], p[1]) for p in outline] + [(x1, p[0], p[1]) for p in outline]
        # make side faces point outward whatever the outline winding is
        area = sum(outline[i][0] * outline[(i + 1) % n][1] - outline[(i + 1) % n][0] * outline[i][1] for i in range(n))
        if (area > 0) == (x1 > x0):
            faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
        else:
            faces = [(i, n + i, n + (i + 1) % n, (i + 1) % n) for i in range(n)]
        self.add(v, faces, mat)
        if cap0 or cap1:
            tris = mgeo.tessellate_polygon([[Vector((p[0], p[1], 0)) for p in outline]])
            if cap0:
                pts = [(x0, p[0], p[1]) for p in outline]
                self.add(pts, self._orient(pts, tris, (-1 if x0 < x1 else 1, 0, 0)), mat)
            if cap1:
                pts = [(x1, p[0], p[1]) for p in outline]
                self.add(pts, self._orient(pts, tris, (1 if x0 < x1 else -1, 0, 0)), mat)

    def grid(self, nx, ny, fn, mat, uv_fn=None):
        """Parametric grid surface: fn(i/nx, j/ny) -> (x,y,z)."""
        verts = [fn(i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)]
        faces = []; uvs = [] if uv_fn else None
        for j in range(ny):
            for i in range(nx):
                a = j * (nx + 1) + i
                faces.append((a, a + 1, a + nx + 2, a + nx + 1))
                if uv_fn:
                    uvs.append((uv_fn(i / nx, j / ny), uv_fn((i + 1) / nx, j / ny),
                                uv_fn((i + 1) / nx, (j + 1) / ny), uv_fn(i / nx, (j + 1) / ny)))
        self.add(verts, faces, mat, uvs)

    # ---- finalize
    def to_object(self, name=None, collection=None, smooth_angle=35.0, recalc=True, shadow=True, weld=False):
        from . import mats as _m  # noqa
        name = name or self.name
        me = bpy.data.meshes.new(name)
        if self.v:
            me.from_pydata(self.v, [], self.f, shade_flat=False)
        obj = bpy.data.objects.new(name, me)
        (collection or bpy.context.scene.collection).objects.link(obj)
        for mn in self.mats:
            me.materials.append(_m.get(mn))
        if not self.v:
            return obj
        npoly = len(me.polygons)
        me.polygons.foreach_set("material_index", np.asarray(self.fm, dtype=np.int32))
        if recalc:
            bm = bmesh.new(); bm.from_mesh(me)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me); bm.free()
        me.update()
        # ---- UVs
        nloop = len(me.loops)
        lv = np.empty(nloop, dtype=np.int32); me.loops.foreach_get("vertex_index", lv)
        co = np.empty(len(me.vertices) * 3, dtype=np.float64); me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        ls = np.empty(npoly, dtype=np.int32); me.polygons.foreach_get("loop_start", ls)
        lt = np.empty(npoly, dtype=np.int32); me.polygons.foreach_get("loop_total", lt)
        pn = np.empty(npoly * 3, dtype=np.float64); me.polygons.foreach_get("normal", pn)
        pn = pn.reshape(-1, 3)
        lp = np.repeat(np.arange(npoly), lt)           # polygon index per loop
        P = co[lv]; N = pn[lp]
        horiz = np.abs(N[:, 2]) >= 0.707
        tl = np.hypot(N[:, 0], N[:, 1]); tl[tl < 1e-9] = 1.0
        tx = -N[:, 1] / tl; ty = N[:, 0] / tl
        u = np.where(horiz, P[:, 0], P[:, 0] * tx + P[:, 1] * ty)
        vv = np.where(horiz, P[:, 1], P[:, 2])
        uv = np.stack([u, vv], axis=1)
        if any(x is not None for x in self.fuv):
            # explicit UVs: recalc may have flipped loop order, so match by vertex index
            polys = me.polygons
            for pi, fu in enumerate(self.fuv):
                if fu is None:
                    continue
                src = self.f[pi]
                s = ls[pi]
                for k in range(lt[pi]):
                    vi = lv[s + k]
                    try:
                        uv[s + k] = fu[src.index(vi)]
                    except ValueError:
                        pass
        layer = me.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", uv.astype(np.float32).ravel())
        # ---- per-face random attribute
        attr = me.attributes.new("rnd", 'FLOAT', 'FACE')
        attr.data.foreach_set("value", np.asarray(self.frnd, dtype=np.float32))
        # ---- shading
        if weld:
            # merge coincident vertices so facetted round surfaces shade smoothly (UVs/attributes survive)
            bm = bmesh.new(); bm.from_mesh(me)
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0008)
            bm.to_mesh(me); bm.free()
            me.update()
            npoly = len(me.polygons)
        me.polygons.foreach_set("use_smooth", np.ones(npoly, dtype=bool))
        if smooth_angle is not None:
            try:
                me.set_sharp_from_angle(angle=math.radians(smooth_angle))
            except Exception:
                pass
        me.update()
        if not shadow:
            obj.visible_shadow = False
        return obj


def circle_pts(r, n, cx=0.0, cy=0.0, a0=0.0):
    return [(cx + r * math.cos(a0 + TAU * i / n), cy + r * math.sin(a0 + TAU * i / n)) for i in range(n)]


def rect_pts(x0, x1, y0, y1):
    """CCW rectangle."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def pointed_arch(w, spring_z, n=10, sharp=1.0, x0=0.0):
    """Outline points of a gothic pointed arch head, from the right springer up over the apex to the left
    springer.  w = opening width, spring_z = height where the arch starts.
    sharp: arc radius as a multiple of w (1.0 = equilateral arch, >1 = lancet, 0.6..0.9 = depressed/drop).
    Returns list of (x, z) going right -> apex -> left (CCW when combined with jambs bottom-left->bottom-right).
    Apex height above spring = sqrt(R^2 - (R - w/2)^2)."""
    R = max(sharp, 0.5001) * w
    half = w / 2.0
    cxr = x0 + half - R      # centre of the arc that forms the RIGHT side
    cxl = x0 - half + R      # centre of the arc that forms the LEFT side
    top = math.sqrt(max(R * R - (R - half) ** 2, 0.0))
    a_end = math.atan2(top, x0 - cxr)     # angle of apex seen from right-arc centre
    pts = []
    for i in range(n + 1):
        a = a_end * i / n
        pts.append((cxr + R * math.cos(a), spring_z + R * math.sin(a)))
    for i in range(1, n + 1):
        a = a_end * (n - i) / n
        pts.append((cxl - R * math.cos(a), spring_z + R * math.sin(a)))
    return pts


def arch_apex(w, sharp=1.0):
    R = max(sharp, 0.5001) * w
    return math.sqrt(max(R * R - (R - w / 2.0) ** 2, 0.0))


# =====================================================================================
# node helpers
# =====================================================================================
class NT:
    """Tiny DSL for building node trees.

        nt = NT(material.node_tree)
        noise = nt.n('ShaderNodeTexNoise', Scale=4.0, Detail=5.0, Vector=some_socket)
        col   = nt.ramp(noise.outputs['Factor'], [(0.0, (0,0,0,1)), (1.0, (1,1,1,1))])
        x     = nt.math('MULTIPLY', a, 2.0)

    kwargs: if the node has a python attribute with that name it is set as an attribute (e.g. operation,
    blend_type, data_type, noise_dimensions), otherwise it is matched against input socket names
    (underscores = spaces, case-insensitive).  Socket values are linked, plain values become defaults.
    Use i0=..., i1=... to address inputs by index.
    """

    def __init__(self, tree, clear=False):
        self.t = tree
        if clear:
            tree.nodes.clear()
        self._x = 0

    def n(self, typ, **kw):
        node = self.t.nodes.new(typ)
        node.location = (self._x, 0); self._x += 180
        # attributes first (they can change which sockets exist)
        rest = {}
        for k, v in kw.items():
            if k in ('label',):
                node.label = v; continue
            # convention: node ATTRIBUTES are lowercase (operation=, blend_type=...), INPUT sockets are
            # Capitalized (Scale=, Base_Color=...) or addressed by index (i0=, i1=...)
            is_index = len(k) > 1 and k[0] == 'i' and k[1:].isdigit()
            if k[0].islower() and not is_index and hasattr(node, k):
                setattr(node, k, v); continue
            rest[k] = v
        for k, v in rest.items():
            sock = None
            if len(k) > 1 and k[0] == 'i' and k[1:].isdigit():
                sock = node.inputs[int(k[1:])]
            else:
                key = k.replace('_', ' ').lower()
                for s in node.inputs:
                    if s.name.lower() == key and s.enabled:
                        sock = s; break
                if sock is None:
                    for s in node.inputs:
                        if s.name.lower() == key:
                            sock = s; break
            if sock is None:
                raise KeyError("node %s has no input/attr %r (inputs: %s)" % (typ, k, [s.name for s in node.inputs]))
            self.set(sock, v)
        return node

    def set(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.t.links.new(v, sock)
        elif isinstance(v, bpy.types.Node):
            self.t.links.new(v.outputs[0], sock)
        else:
            try:
                sock.default_value = v
            except Exception:
                # allow RGB given for RGBA sockets or float for vector
                if isinstance(v, (tuple, list)) and len(v) == 3:
                    sock.default_value = (v[0], v[1], v[2], 1.0)
                else:
                    raise

    def link(self, a, b):
        if isinstance(a, bpy.types.Node):
            a = a.outputs[0]
        self.t.links.new(a, b)

    def math(self, op, a, b=None, c=None, clamp=False):
        node = self.t.nodes.new('ShaderNodeMath'); node.operation = op; node.use_clamp = clamp
        self.set(node.inputs[0], a)
        if b is not None: self.set(node.inputs[1], b)
        if c is not None: self.set(node.inputs[2], c)
        return node.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        node = self.t.nodes.new('ShaderNodeVectorMath'); node.operation = op
        self.set(node.inputs[0], a)
        if b is not None: self.set(node.inputs[1], b)
        if scale is not None: self.set(node.inputs['Scale'], scale)
        return node.outputs['Value'] if op in ('LENGTH', 'DOT_PRODUCT', 'DISTANCE') else node.outputs[0]

    def mix(self, fac, a, b, blend='MIX', clamp=True):
        """Colour mix.  Returns the colour output socket."""
        node = self.t.nodes.new('ShaderNodeMix'); node.data_type = 'RGBA'; node.blend_type = blend
        node.clamp_factor = clamp
        self.set(node.inputs[0], fac); self.set(node.inputs[6], a); self.set(node.inputs[7], b)
        return node.outputs[2]

    def mixf(self, fac, a, b):
        node = self.t.nodes.new('ShaderNodeMix'); node.data_type = 'FLOAT'
        self.set(node.inputs[0], fac); self.set(node.inputs[2], a); self.set(node.inputs[3], b)
        return node.outputs[0]

    def ramp(self, fac, stops, interp='LINEAR'):
        """stops: [(pos, (r,g,b,a)), ...].  Returns colour output socket."""
        node = self.t.nodes.new('ShaderNodeValToRGB')
        cr = node.color_ramp; cr.interpolation = interp
        while len(cr.elements) < len(stops):
            cr.elements.new(0.5)
        for e, (p, c) in zip(cr.elements, stops):
            e.position = p
            e.color = c if len(c) == 4 else (c[0], c[1], c[2], 1.0)
        self.set(node.inputs[0], fac)
        return node.outputs[0]

    def map_range(self, v, fmin, fmax, tmin=0.0, tmax=1.0, clamp=True, interp='LINEAR'):
        node = self.t.nodes.new('ShaderNodeMapRange'); node.clamp = clamp; node.interpolation_type = interp
        self.set(node.inputs[0], v)
        self.set(node.inputs[1], fmin); self.set(node.inputs[2], fmax)
        self.set(node.inputs[3], tmin); self.set(node.inputs[4], tmax)
        return node.outputs[0]

    def bump(self, height, strength=0.3, distance=0.05, normal=None):
        node = self.t.nodes.new('ShaderNodeBump')
        self.set(node.inputs['Strength'], strength)
        self.set(node.inputs['Distance'], distance)
        self.set(node.inputs['Height'], height)
        if normal is not None:
            self.set(node.inputs['Normal'], normal)
        return node.outputs[0]

    def sep_xyz(self, v):
        node = self.t.nodes.new('ShaderNodeSeparateXYZ'); self.set(node.inputs[0], v)
        return node.outputs

    def comb_xyz(self, x=0.0, y=0.0, z=0.0):
        node = self.t.nodes.new('ShaderNodeCombineXYZ')
        self.set(node.inputs[0], x); self.set(node.inputs[1], y); self.set(node.inputs[2], z)
        return node.outputs[0]

    def attr(self, name, typ='GEOMETRY'):
        node = self.t.nodes.new('ShaderNodeAttribute'); node.attribute_name = name; node.attribute_type = typ
        return node

    def value(self, v):
        node = self.t.nodes.new('ShaderNodeValue'); node.outputs[0].default_value = v
        return node.outputs[0]

    def rgb(self, c):
        node = self.t.nodes.new('ShaderNodeRGB')
        node.outputs[0].default_value = c if len(c) == 4 else (c[0], c[1], c[2], 1.0)
        return node.outputs[0]


def new_material(name, clear=True):
    """Returns (material, NT, output_node).  With clear=True the tree only holds a Material Output."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = NT(m.node_tree, clear=clear)
    if clear:
        out = nt.n('ShaderNodeOutputMaterial')
    else:
        out = m.node_tree.nodes.get('Material Output')
    return m, nt, out


def drive_frame(socket_or_prop_owner, data_path, expr="frame/24", index=-1):
    """Attach a driver so a value follows the scene frame (seconds = frame/24).  For animated shaders:
       drive_frame(node.inputs['W'], 'default_value', 'frame/24*0.2')"""
    fc = socket_or_prop_owner.driver_add(data_path, index) if index >= 0 else socket_or_prop_owner.driver_add(data_path)
    fc.driver.expression = expr
    return fc


# =====================================================================================
# geometry nodes helpers
# =====================================================================================
def new_gn_group(name):
    """New geometry node group with Geometry in/out.  Returns (group, NT, input_node, output_node)."""
    g = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    g.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    g.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nt = NT(g)
    gi = nt.n('NodeGroupInput'); go = nt.n('NodeGroupOutput')
    return g, nt, gi, go


def add_gn_modifier(obj, group, name="GN"):
    md = obj.modifiers.new(name, 'NODES')
    md.node_group = group
    return md


def set_smooth(obj, angle=None):
    me = obj.data
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=bool))
    if angle is not None:
        me.set_sharp_from_angle(angle=math.radians(angle))
    me.update()

