"""Pure-numpy terrain field for the landscape module (NO bpy import - safe to import anywhere).

Public, stable API (re-exported by scene/mods/landscape.py):

    height_at(x, y)   -> z      terrain height in metres (numpy arrays or scalars, any shape)
    forest_mask(x, y) -> 0..1   where forest may grow (0 on water, cliffs, snow, built areas, cliff rim)
    slope_at(x, y)    -> tan(slope angle)

All are deterministic and vectorised (evaluated internally in cache-sized chunks).

How the terrain is put together
    e  (x,y)  signed distance to the (extended) cliff line layout.CLIFF_TOP, positive = outside (valley side)
    e' (x,y)  same, but measured to the "closed" line (notches < ~55 m filled) - the hero cliff sheet in
              landscape.py is laid out on iso-lines of e'
    V  (x,y)  valley terrain: lake / river network at VALLEY_Z, islands, forested ridges, alpine ranges
    Hm (x,y)  plateau + eastern massif (z ~ -0.6 at the castle, rising slowly eastward)
    near the castle (q < ~620 m):  plateau -> rim -> [pit hidden behind the hero cliff sheet] -> talus apron -> V
    far away        (q > ~900 m):  plateau/massif -> steep forested flank -> V

Tunables: the constants right below, the lake network tables (_MAIN, _NORTH) and the amplitudes in
_valley() / _massif().
"""
import math
import numpy as np
from .. import layout as L

VALLEY_Z = L.VALLEY_Z
CX, CY = 0.0, 20.0             # "castle centre" used for distance-based blends (q)
R_MID = 55.0                   # closing radius / offset of the mid contour used by the cliff sheet
HERO_Q0, HERO_Q1 = 620.0, 900.0  # hero cliff zone: full inside Q0, gone outside Q1
RIM_Z_BUILT = -13.0            # rock level where the cliff reaches the CLIFF_TOP line under masonry
TALUS_SLOPE = 0.62             # tan(32 deg)
FOOT_E = 75.0                  # nominal offset of the cliff foot from the line
FOOT_Z = -260.0
WATERFALL_C = (-39.5, -50.5)   # plan position of the hero waterfall chute (under layout.SPILL_*)
TREELINE_Z = 430.0
SNOWLINE_Z = 640.0
SUN_AZ = math.radians(L.SUN_AZIMUTH_DEG)
_CHUNK = 32768
F32 = np.float32


# =====================================================================================
# noise (float32)
# =====================================================================================
_rng = np.random.RandomState(90210)
_ang = _rng.rand(65536) * (2.0 * np.pi)
_G = (np.cos(_ang) + 1j * np.sin(_ang)).astype(np.complex64)
_ang = _rng.rand(65536) * (2.0 * np.pi)
_H = (np.cos(_ang) + 1j * np.sin(_ang)).astype(np.complex64)
_ROT_C, _ROT_S = F32(math.cos(0.6435)), F32(math.sin(0.6435))


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lattice(x, y, seed):
    x = x + F32(seed * 17.137)
    y = y - F32(seed * 9.871)
    xf = np.floor(x); yf = np.floor(y)
    fx = x - xf; fy = y - yf
    ix = xf.astype(np.int32) & 255; iy = yf.astype(np.int32) & 255
    ix1 = (ix + 1) & 255; iy1 = ((iy + 1) & 255) << 8
    iy = iy << 8
    u = fx * fx * fx * (fx * (fx * F32(6.0) - F32(15.0)) + F32(10.0))
    v = fy * fy * fy * (fy * (fy * F32(6.0) - F32(15.0)) + F32(10.0))
    return fx, fy, u, v, iy + ix, iy + ix1, iy1 + ix, iy1 + ix1


def _interp(T, fx, fy, u, v, i00, i10, i01, i11):
    g = T[i00]; n00 = g.real * fx + g.imag * fy
    g = T[i10]; n10 = g.real * (fx - F32(1.0)) + g.imag * fy
    g = T[i01]; n01 = g.real * fx + g.imag * (fy - F32(1.0))
    g = T[i11]; n11 = g.real * (fx - F32(1.0)) + g.imag * (fy - F32(1.0))
    a = n00 + u * (n10 - n00)
    b = n01 + u * (n11 - n01)
    return (a + v * (b - a)) * F32(1.5)


def gnoise2(x, y, seed=0):
    """2D gradient noise, roughly -1..1 (float32 in, float32 out)."""
    return _interp(_G, *_lattice(x, y, seed))


def gnoise2p(x, y, seed=0):
    """Two independent noises for the price of ~1.4."""
    lt = _lattice(x, y, seed)
    return _interp(_G, *lt), _interp(_H, *lt)


def fbm(x, y, octaves=4, lac=2.03, gain=0.5, seed=0):
    s = 0.0; a = 1.0; norm = 0.0; lac = F32(lac)
    for o in range(octaves):
        s = s + F32(a) * gnoise2(x, y, seed + o * 3)
        norm += a; a *= gain
        x, y = (x * _ROT_C - y * _ROT_S) * lac, (x * _ROT_S + y * _ROT_C) * lac
    return s * F32(1.0 / norm)


def fbmp(x, y, octaves=3, lac=2.03, gain=0.5, seed=0):
    s1 = 0.0; s2 = 0.0; a = 1.0; norm = 0.0; lac = F32(lac)
    for o in range(octaves):
        n1, n2 = gnoise2p(x, y, seed + o * 3)
        s1 = s1 + F32(a) * n1; s2 = s2 + F32(a) * n2
        norm += a; a *= gain
        x, y = (x * _ROT_C - y * _ROT_S) * lac, (x * _ROT_S + y * _ROT_C) * lac
    return s1 * F32(1.0 / norm), s2 * F32(1.0 / norm)


def ridged(x, y, octaves=6, lac=2.07, gain=0.52, seed=0):
    """Ridged multifractal, 0..1, sharp crests."""
    s = 0.0; a = 1.0; norm = 0.0; w = F32(1.0); lac = F32(lac)
    for o in range(octaves):
        n = F32(1.0) - np.abs(gnoise2(x, y, seed + o * 3))
        n = n * n
        s = s + n * (F32(a) * w)
        norm += a
        w = np.minimum(n * F32(1.5), F32(1.0))
        a *= gain
        x, y = (x * _ROT_C - y * _ROT_S) * lac, (x * _ROT_S + y * _ROT_C) * lac
    return s * F32(1.0 / norm)


# ---- 3D gradient noise (used by the cliff sheet displacement in landscape.py)
_G3 = _rng.normal(size=(4096, 3))
_G3 = (_G3 / np.linalg.norm(_G3, axis=1)[:, None]).astype(np.float32)
_G3x = np.ascontiguousarray(_G3[:, 0]); _G3y = np.ascontiguousarray(_G3[:, 1]); _G3z = np.ascontiguousarray(_G3[:, 2])
_P3 = _rng.permutation(4096).astype(np.int32)


def _gnoise3(x, y, z, seed):
    x = x + F32(seed * 13.31); y = y + F32(seed * 7.77); z = z - F32(seed * 5.19)
    xf = np.floor(x); yf = np.floor(y); zf = np.floor(z)
    fx = x - xf; fy = y - yf; fz = z - zf
    ix = xf.astype(np.int32); iy = yf.astype(np.int32); iz = zf.astype(np.int32)
    u = fx * fx * fx * (fx * (fx * F32(6.0) - F32(15.0)) + F32(10.0))
    v = fy * fy * fy * (fy * (fy * F32(6.0) - F32(15.0)) + F32(10.0))
    w = fz * fz * fz * (fz * (fz * F32(6.0) - F32(15.0)) + F32(10.0))
    px0 = _P3[ix & 4095]; px1 = _P3[(ix + 1) & 4095]
    one = F32(1.0)
    res = []
    for dz in (0, 1):
        fzz = fz - F32(dz)
        row = []
        for dy in (0, 1):
            fyy = fy - F32(dy)
            h0 = _P3[(_P3[(px0 + iy + dy) & 4095] + iz + dz) & 4095]
            h1 = _P3[(_P3[(px1 + iy + dy) & 4095] + iz + dz) & 4095]
            c0 = _G3x[h0] * fx + _G3y[h0] * fyy + _G3z[h0] * fzz
            c1 = _G3x[h1] * (fx - one) + _G3y[h1] * fyy + _G3z[h1] * fzz
            row.append(c0 + u * (c1 - c0))
        res.append(row[0] + v * (row[1] - row[0]))
    return (res[0] + w * (res[1] - res[0])) * F32(1.6)


def gnoise3(x, y, z, seed=0):
    """3D gradient noise ~ -1..1; chunked, float32."""
    x = np.asarray(x, dtype=np.float32); y = np.asarray(y, dtype=np.float32); z = np.asarray(z, dtype=np.float32)
    if x.size <= _CHUNK:
        return _gnoise3(x, y, z, seed)
    out = np.empty(x.shape, dtype=np.float32)
    xr = x.ravel(); yr = y.ravel(); zr = z.ravel(); o = out.ravel()
    for s in range(0, xr.size, _CHUNK):
        o[s:s + _CHUNK] = _gnoise3(xr[s:s + _CHUNK], yr[s:s + _CHUNK], zr[s:s + _CHUNK], seed)
    return out


_W_R = _rng.rand(4096, 4).astype(np.float32)
_W_RX = np.ascontiguousarray(_W_R[:, 0]); _W_RY = np.ascontiguousarray(_W_R[:, 1])
_W_RZ = np.ascontiguousarray(_W_R[:, 2]); _W_RV = np.ascontiguousarray(_W_R[:, 3])


def worley3(x, y, z, seed=0):
    """3D cellular noise: returns (F1, F2, value of the nearest cell 0..1).  Chunked, float32."""
    x = np.asarray(x, dtype=np.float32).ravel(); y = np.asarray(y, dtype=np.float32).ravel()
    z = np.asarray(z, dtype=np.float32).ravel()
    n = x.size
    o1 = np.empty(n, dtype=np.float32); o2 = np.empty(n, dtype=np.float32); ov = np.empty(n, dtype=np.float32)
    for s0 in range(0, n, _CHUNK):
        cx = x[s0:s0 + _CHUNK]; cy = y[s0:s0 + _CHUNK]; cz = z[s0:s0 + _CHUNK]
        xf = np.floor(cx); yf = np.floor(cy); zf = np.floor(cz)
        fx = cx - xf; fy = cy - yf; fz = cz - zf
        ix = xf.astype(np.int32) + seed * 131; iy = yf.astype(np.int32); iz = zf.astype(np.int32)
        f1 = np.full(cx.shape, 1e9, dtype=np.float32); f2 = f1.copy(); val = np.zeros(cx.shape, dtype=np.float32)
        for dx in (-1, 0, 1):
            hx = _P3[(ix + dx) & 4095]
            ax = F32(dx) - fx
            for dy in (-1, 0, 1):
                hy = _P3[(hx + iy + dy) & 4095]
                ay = F32(dy) - fy
                for dz in (-1, 0, 1):
                    h = _P3[(hy + iz + dz) & 4095]
                    ox = ax + _W_RX[h]; oy = ay + _W_RY[h]; oz = (F32(dz) - fz) + _W_RZ[h]
                    d2 = ox * ox + oy * oy + oz * oz
                    closer = d2 < f1
                    f2 = np.where(closer, f1, np.minimum(f2, d2))
                    val = np.where(closer, _W_RV[h], val)
                    f1 = np.where(closer, d2, f1)
        o1[s0:s0 + _CHUNK] = np.sqrt(f1); o2[s0:s0 + _CHUNK] = np.sqrt(f2); ov[s0:s0 + _CHUNK] = val
    return o1, o2, ov


def fbm3(x, y, z, octaves=4, lac=2.03, gain=0.5, seed=0):
    x = np.asarray(x, dtype=np.float32); y = np.asarray(y, dtype=np.float32); z = np.asarray(z, dtype=np.float32)
    out = np.empty(x.shape, dtype=np.float32)
    xr = x.ravel(); yr = y.ravel(); zr = z.ravel(); o = out.ravel()
    lacf = F32(lac)
    for s0 in range(0, xr.size, _CHUNK):
        cx = xr[s0:s0 + _CHUNK]; cy = yr[s0:s0 + _CHUNK]; cz = zr[s0:s0 + _CHUNK]
        s = 0.0; a = 1.0; norm = 0.0
        for k in range(octaves):
            s = s + F32(a) * _gnoise3(cx, cy, cz, seed + k * 3)
            norm += a; a *= gain
            cx, cy, cz = (cx * _ROT_C - cy * _ROT_S) * lacf, (cx * _ROT_S + cy * _ROT_C) * lacf, cz * lacf
        o[s0:s0 + _CHUNK] = s * F32(1.0 / norm)
    return out


# =====================================================================================
# cliff line geometry
# =====================================================================================
_S_EXT = [(45000.0, -24000.0), (9000.0, -4800.0), (4200.0, -2100.0), (2300.0, -950.0), (1250.0, -440.0),
          (720.0, -240.0)]
_N_EXT = [(370.0, 590.0), (680.0, 930.0), (1250.0, 1500.0), (2400.0, 2450.0), (4600.0, 4000.0),
          (9000.0, 6800.0), (45000.0, 27000.0)]
LINE = np.array(_S_EXT + [tuple(map(float, p)) for p in L.CLIFF_TOP] + _N_EXT, dtype=np.float64)
_POLY = np.vstack([LINE, [(70000.0, 27000.0), (70000.0, -24000.0)]])     # closed plateau polygon
_NL = len(LINE)
# simplified line for far points (max deviation < 60 m): the two long arms + a few hero points
_LINE_FAR = np.array(_S_EXT + [(400.0, -150.0), (48.0, -70.0), (-47.5, -50.0), (-72.0, 10.0), (-52.0, 80.0),
                               (-10.0, 170.0), (200.0, 380.0)] + _N_EXT, dtype=np.float64)
_POLY_FAR = np.vstack([_LINE_FAR, [(70000.0, 27000.0), (70000.0, -24000.0)]])


def _seg_dist2(px, py, ax, ay, bx, by):
    dx = bx - ax; dy = by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) * (1.0 / (dx * dx + dy * dy)), 0.0, 1.0)
    qx = px - (ax + t * dx); qy = py - (ay + t * dy)
    return qx * qx + qy * qy, t


_LCUM = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(LINE[:, 0]), np.diff(LINE[:, 1])))])


def _poly_sd(x, y, poly, nline, want_u=False):
    """Signed distance to the open polyline poly[:nline] (sign from the closed polygon `poly`).
    want_u: also return the arclength (along LINE) of the nearest point."""
    d2 = np.full(x.shape, 1e30)
    inside = np.zeros(x.shape, dtype=bool)
    u = np.zeros(x.shape) if want_u else None
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]; bx, by = poly[(i + 1) % n]
        if i < nline - 1:
            dd, t = _seg_dist2(x, y, ax, ay, bx, by)
            if want_u:
                m = dd < d2
                u = np.where(m, _LCUM[i] + t * (_LCUM[i + 1] - _LCUM[i]), u)
            np.minimum(d2, dd, out=d2)
        if ay != by:
            inside ^= ((ay > y) != (by > y)) & (x < (bx - ax) * (y - ay) / (by - ay) + ax)
    d = np.sqrt(d2)
    d = np.where(inside, -d, d)
    return (d, u) if want_u else d


def cliff_sd(x, y):
    """Signed distance to the extended cliff line; positive = outside (valley side).  Exact within 2.5 km of
    the castle, simplified line (error < 60 m) beyond."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    q = np.hypot(x - CX, y - CY)
    near = q < 2500.0
    if near.all():
        return _poly_sd(x, y, _POLY, _NL)
    out = np.empty(x.shape)
    if near.any():
        out[near] = _poly_sd(x[near], y[near], _POLY, _NL)
    far = ~near
    out[far] = _poly_sd(x[far], y[far], _POLY_FAR, len(_LINE_FAR))
    return out


def _build_mid_contour():
    """Outward offset (R_MID) of the hero part of the cliff line, trimmed + smoothed.
    Returns points (n,2) ordered like LINE (far SE -> SW promontory -> far N) at 0.5 m spacing."""
    i0, i1 = 3, _NL - 4                 # out to ~2.5 km on both sides
    P = LINE[i0:i1 + 1]
    pts = []
    for i in range(len(P) - 1):
        a = P[i]; b = P[i + 1]
        d = b - a; ln = float(np.hypot(d[0], d[1])); d = d / ln
        nrm = np.array([-d[1], d[0]])                       # left of travel direction = outward
        m = max(2, int(ln / 0.5))
        t = np.linspace(0.0, 1.0, m, endpoint=False)[:, None]
        pts.append(a + t * (b - a) + nrm * R_MID)
        if i < len(P) - 2:                                  # convex corner -> arc
            d2 = P[i + 2] - b; d2 = d2 / np.hypot(d2[0], d2[1])
            cr = d[0] * d2[1] - d[1] * d2[0]
            if cr < 0:
                a0 = math.atan2(nrm[1], nrm[0])
                n2 = np.array([-d2[1], d2[0]]); a1 = math.atan2(n2[1], n2[0])
                while a1 > a0: a1 -= 2 * math.pi
                k = max(2, int(abs(a1 - a0) * R_MID / 0.5))
                aa = np.linspace(a0, a1, k, endpoint=False)
                pts.append(b + R_MID * np.stack([np.cos(aa), np.sin(aa)], axis=1))
    pts = np.vstack(pts)
    keep = _poly_sd(pts[:, 0], pts[:, 1], _POLY, _NL) >= R_MID - 0.02
    pts = pts[keep]
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    su = np.arange(0.0, s[-1], 0.5)
    px = np.interp(su, s, pts[:, 0]); py = np.interp(su, s, pts[:, 1])
    k = np.exp(-0.5 * (np.arange(-24, 25) / 8.0) ** 2); k /= k.sum()       # sigma 4 m
    pxs = np.convolve(np.pad(px, 24, mode='edge'), k, mode='valid')
    pys = np.convolve(np.pad(py, 24, mode='edge'), k, mode='valid')
    return np.stack([pxs, pys], axis=1)


MID = _build_mid_contour()              # dense (0.5 m)
_MID_C = MID[::8]                       # 4 m polyline for distance queries
_sel = np.nonzero(np.hypot(_MID_C[:, 0] - CX, _MID_C[:, 1] - CY) < 1500.0)[0]
_MA = np.ascontiguousarray(_MID_C[_sel[0]:_sel[-1] + 1])
_MAX = _MA[:, 0].copy(); _MAY = _MA[:, 1].copy()
# arclength (along LINE) of the foot point of every contour vertex: lets us look only at a window of the
# contour around a query point's own foot point
_MU = np.maximum.accumulate(_poly_sd(_MAX, _MAY, _POLY, _NL, want_u=True)[1])
_WIN = np.arange(-40, 41)


def eprime_exact(x, y, u=None):
    """R_MID - distance to the mid contour (valid for points on the plateau side of the mid contour and
    within ~1.1 km of the castle).  u: arclength of the points' foot point on LINE (computed if None)."""
    x = np.asarray(x, dtype=np.float64).ravel(); y = np.asarray(y, dtype=np.float64).ravel()
    if u is None:
        u = _poly_sd(x, y, _POLY, _NL, want_u=True)[1]
    out = np.empty(x.shape)
    nseg = len(_MAX) - 1
    CH = 2048
    for s in range(0, x.size, CH):
        px = x[s:s + CH, None]; py = y[s:s + CH, None]
        i0 = np.searchsorted(_MU, u[s:s + CH])
        idx = np.clip(i0[:, None] + _WIN, 0, nseg - 1)
        ax = _MAX[idx]; ay = _MAY[idx]
        dx = _MAX[idx + 1] - ax; dy = _MAY[idx + 1] - ay
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        qx = px - (ax + t * dx); qy = py - (ay + t * dy)
        out[s:s + CH] = np.sqrt((qx * qx + qy * qy).min(axis=1))
    return R_MID - out


def cliff_coords(x, y):
    """Returns (e, e', q): signed distance to the line, to the closed line, and distance from the castle."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    q = np.hypot(x - CX, y - CY)
    near = q < 1150.0
    if near.any() and not near.all():
        e = np.empty(x.shape); ep = np.empty(x.shape)
        far = ~near
        e[far] = cliff_sd(x[far], y[far]); ep[far] = e[far]
        e[near], ep[near], _ = cliff_coords(x[near], y[near])
        return e, ep, q
    if not near.any():
        e = cliff_sd(x, y)
        return e, e.copy(), q
    e, u = _poly_sd(x, y, _POLY, _NL, want_u=True)
    ep = e.copy()
    band = (e > -65.0) & (e < R_MID - 0.5)
    if band.any():
        ep[band] = np.minimum(eprime_exact(x[band], y[band], u[band]), e[band])
    return e, ep, q


# =====================================================================================
# built areas
# =====================================================================================
def built_sd(x, y):
    """Distance to the union of layout.BUILT_AREAS rectangles (negative inside)."""
    d = np.full(np.shape(x), 1e30)
    for (x0, x1, y0, y1) in L.BUILT_AREAS:
        dx = np.maximum(x0 - x, x - x1); dy = np.maximum(y0 - y, y - y1)
        o = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0)) + np.minimum(np.maximum(dx, dy), 0.0)
        d = np.minimum(d, o)
    return d


# =====================================================================================
# valley: lake / river network, hills, mountains
# =====================================================================================
# (x, y, lake half-width [negative = dry valley floor], valley half-width)
_MAIN = np.array([
    (-25000, 10600, -300, 1300), (-15500, 6600, -200, 1700), (-10800, 4600, 40, 2000),
    (-7600, 3250, 260, 2300), (-5300, 2270, 430, 2500), (-3700, 1750, 560, 2500),
    (-2300, 1150, 640, 2400), (-1250, 520, 580, 2100), (-800, -200, 410, 1700),
    (-560, -1100, 210, 1300), (-250, -2600, 95, 1100), (350, -5200, 70, 1000),
    (-600, -9500, 50, 900), (1500, -15000, -100, 800), (500, -24000, -300, 700)], dtype=np.float64)
_NORTH = np.array([
    (-1150, 650, 300, 2000), (-720, 1700, 260, 1100), (-980, 2900, 190, 900),
    (-520, 4300, 100, 800), (-900, 7000, 45, 700), (100, 12000, -80, 650),
    (-800, 20000, -300, 600)], dtype=np.float64)
_NETS = (_MAIN, _NORTH)


def _water_dist(x, y):
    """dw: distance to the lake outline (negative = in water); dv: distance outside the valley corridor;
    hw: lake half width at the nearest axis point."""
    dw = np.full(x.shape, 1e30); dv = np.full(x.shape, 1e30); hw = np.zeros(x.shape)
    for net in _NETS:
        for i in range(len(net) - 1):
            ax, ay, ha, va = net[i]; bx, by, hb, vb = net[i + 1]
            d2, t = _seg_dist2(x, y, ax, ay, bx, by)
            d = np.sqrt(d2)
            h = ha + t * (hb - ha)
            dd = d - h
            m = dd < dw
            dw = np.where(m, dd, dw); hw = np.where(m, h, hw)
            np.minimum(dv, d - (va + t * (vb - va)), out=dv)
    return dw, dv, hw


def _mountain_noise(x, y):
    """~0..1.2 alpine ridge field (expensive: evaluated only where needed).  x, y float32."""
    w1, w2 = fbmp(x * F32(1 / 5200.0), y * F32(1 / 5200.0), 2, seed=61)
    wx = x + F32(1500.0) * w1; wy = y + F32(1500.0) * w2
    r = ridged(wx * F32(1 / 8200.0), wy * F32(1 / 8200.0), 9, gain=0.575, seed=41) * F32(1.10)
    # couloirs / ribs: a second, finer ridged field carves the flanks
    g = ridged(x * F32(1 / 900.0) + F32(7.7), y * F32(1 / 900.0), 5, gain=0.62, seed=53)
    r = r - F32(0.085) * (F32(1.0) - g) * np.minimum(r * F32(2.2), F32(1.0))
    big = F32(0.5) + F32(0.5) * fbm(x * F32(1 / 21000.0), y * F32(1 / 21000.0), 2, seed=47)
    return np.clip(r * (F32(0.55) + F32(0.75) * big), 0.0, 1.18)


def _soft_cap(part, cap):
    """Compress `part` (>= 0) so that it never exceeds `cap`, keeping its relief."""
    cap = np.maximum(cap, 1.0); h = 0.5 * cap
    return np.where(part <= h, part, cap - h * np.exp(-np.maximum(part - h, 0.0) / h))


def _sun_cap(x, y, q):
    """Max height above the lake (m) so the low sun stays visible from the castle and lights the lake."""
    az = np.arctan2(y - CY, x - CX)
    da = np.abs(np.angle(np.exp(1j * (az - SUN_AZ))))
    return 70.0 + 0.040 * q + smoothstep(math.radians(5.5), math.radians(17.0), da) * 9000.0


def _valley(x, y, q, e, mtn_fn):
    """Valley terrain for a chunk.  Returns (V, dw, dv)."""
    xs = x.astype(F32); ys = y.astype(F32)
    w1, w2 = fbmp(xs * F32(1 / 1500.0), ys * F32(1 / 1500.0), 3, seed=11)
    wx = x + 330.0 * w1; wy = y + 330.0 * w2
    dw, dv, hw = _water_dist(wx, wy)
    # bays / peninsulas (scaled with the local lake width so rivers stay rivers)
    amp = np.clip(hw * 0.62, 18.0, 330.0)
    b1, isl = fbmp(xs * F32(1 / 720.0), ys * F32(1 / 720.0), 4, seed=13)
    dw = dw + amp * b1
    # keep a strip of dry land at the foot of the hero cliff
    dw = np.where(q < 2500.0, np.maximum(dw, 235.0 - e), dw)
    dwp = np.maximum(dw, 0.0)
    rise = 22.0 * (1.0 - np.exp(-dwp * (1.0 / 260.0))) + 0.004 * dwp
    bed = np.maximum(dw * 0.09, -14.0)
    deep = smoothstep(-40.0, -150.0, dw)
    bed = np.maximum(bed, -14.0 + 70.0 * smoothstep(0.10, 0.60, isl) * deep)
    base = np.where(dw > 0.0, rise, bed)
    d1, d2 = fbmp(xs * F32(1 / 150.0), ys * F32(1 / 150.0), 3, seed=29)
    base = base + 4.5 * d1 * smoothstep(-2.0, 25.0, base)
    # layered forested ridges at 1-6 km
    mh = smoothstep(30.0, 900.0, dw) * smoothstep(140.0, 1000.0, e)
    hills = np.zeros(x.shape)
    k = mh > 0.0
    if k.any():
        xk = xs[k]; yk = ys[k]
        r = ridged(xk * F32(1 / 2300.0) + F32(3.3), yk * F32(1 / 2300.0), 5, seed=31)
        f = fbm(xk * F32(1 / 620.0), yk * F32(1 / 620.0), 3, seed=37)
        hills[k] = (330.0 * r * r * 1.25 + 45.0 * f) * mh[k]
    # alpine ranges
    mm = smoothstep(0.0, 3800.0, dv) * smoothstep(3200.0, 7500.0, q)
    mount = np.zeros(x.shape)
    k = mm > 0.0
    if k.any():
        mount[k] = (450.0 + 1750.0 * mtn_fn(k)) * mm[k]
    floor = 0.012 * np.maximum(q - 9000.0, 0.0) * smoothstep(40.0, 300.0, dw)
    relief = hills + mount
    cap = _sun_cap(x, y, q)
    relief = _soft_cap(relief + floor, np.maximum(cap - base, 5.0)) - floor
    return VALLEY_Z + base + relief + floor, dw, dv


def _massif(x, y, q, e, sdb, mtn_fn):
    """Plateau + eastern mountain shoulder (also evaluated outside the line with d_in = 0).
    Returns (Hm incl. built-area depression, natural ground)."""
    xs = x.astype(F32); ys = y.astype(F32)
    din = np.maximum(-e, 0.0)
    n1, n2 = fbmp(xs * F32(1 / 46.0), ys * F32(1 / 46.0), 3, seed=71)
    m1, m2 = fbmp(xs * F32(1 / 270.0), ys * F32(1 / 270.0), 3, seed=73)
    amp = 0.22 + 0.011 * np.clip(sdb, 0.0, 400.0)
    nat = L.PLATEAU_BASE_Z + amp * n1
    nat = nat + 0.028 * np.clip(sdb - 25.0, 0.0, 900.0) * m1
    nat = nat + 50.0 * smoothstep(40.0, 650.0, din)
    part = 500.0 * (1.0 - np.exp(-np.maximum(din - 600.0, 0.0) * (0.13 / 500.0)))
    part = part + 400.0 * (1.0 - np.exp(-np.maximum(q - 900.0, 0.0) * (0.10 / 400.0)))
    mk = smoothstep(1500.0, 9000.0, din)
    k = mk > 0.0
    if k.any():
        part[k] = part[k] + (200.0 + 1500.0 * mtn_fn(k)) * mk[k]
    far = q > 500.0
    if far.any():
        cap = 60.0 + 0.14 * (q[far] - 500.0)
        part[far] = _soft_cap(part[far], cap) + 30.0 * m2[far] * smoothstep(500.0, 1500.0, q[far])
    nat = nat + part
    return nat + (-3.8 - nat) * (1.0 - smoothstep(0.1, 1.3, sdb)), nat


def _terrain_chunk(x, y):
    e, ep, q = cliff_coords(x, y)
    far = q > 1500.0
    if far.any():       # far away the plateau edge wanders (no ruler-straight flank for 40 km)
        wob = 480.0 * smoothstep(1500.0, 5000.0, q[far]) * fbm(x[far].astype(F32) * F32(1 / 3600.0),
                                                              y[far].astype(F32) * F32(1 / 3600.0), 2, seed=101)
        e[far] = e[far] + wob; ep[far] = e[far]
    sdb = built_sd(x, y)
    cache = {}

    def mtn_fn(mask):
        # mountain noise, computed once per chunk for the union of points that ask for it
        if 'm' not in cache:
            far = q > 2800.0
            m = np.zeros(x.shape, dtype=np.float32)
            if far.any():
                m[far] = _mountain_noise(x[far].astype(F32), y[far].astype(F32))
            cache['m'] = m
        return cache['m'][mask]

    W = np.clip(110.0 + 0.3 * (q - 800.0), 110.0, 1800.0)
    need_v = e > -8.0
    need_m = e < 2.0 * W
    V = np.full(x.shape, -1e4); dw = np.full(x.shape, 1e4); dv = np.full(x.shape, 0.0)
    if need_v.any():
        V[need_v], dw[need_v], dv[need_v] = _valley(x[need_v], y[need_v], q[need_v], e[need_v],
                                                    lambda k: mtn_fn(np.nonzero(need_v)[0][k]))
    Hm = np.full(x.shape, 1e4); nat = np.full(x.shape, 1e4)
    if need_m.any():
        Hm[need_m], nat[need_m] = _massif(x[need_m], y[need_m], q[need_m], e[need_m], sdb[need_m],
                                          lambda k: mtn_fn(np.nonzero(need_m)[0][k]))
    # ---- far profile: steep flank from the rim down to the valley
    S = smoothstep(0.0, 1.0, 1.0 - np.maximum(e, 0.0) / (2.0 * W))
    h = np.where(S <= 0.0, V, np.where(S >= 1.0, Hm, V + (Hm - V) * S))
    # ---- hero zone: rim, pit hidden behind the cliff sheet, talus apron
    her = 1.0 - smoothstep(HERO_Q0, HERO_Q1, q)
    nz = (her > 0.0) & (e > -3.5)
    if nz.any():
        xn = x[nz]; yn = y[nz]; en = e[nz]; epn = ep[nz]
        xs = xn.astype(F32); ys = yn.astype(F32)
        rimz = RIM_Z_BUILT + (nat[nz] - 1.5 - RIM_Z_BUILT) * smoothstep(90.0, 180.0, sdb[nz])
        T = Hm[nz] + (rimz - Hm[nz]) * smoothstep(-3.0, -0.8, en)
        dwf = np.hypot(xn - WATERFALL_C[0], yn - WATERFALL_C[1])
        w2 = 1.0 - smoothstep(25.0, 110.0, dwf)
        f1, f2 = fbmp(xs * F32(1 / 210.0), ys * F32(1 / 210.0), 2, seed=83)
        footz = FOOT_Z + 24.0 * f1
        footz = footz + (-250.0 - footz) * w2
        ef = FOOT_E + (5.0 - FOOT_E) * w2
        apron = footz - TALUS_SLOPE * (epn - ef) + 5.0 * fbm(xs * F32(1 / 55.0), ys * F32(1 / 55.0), 3, seed=89) + 9.0 * f2
        low = np.minimum(np.maximum(V[nz], apron), T)
        hn = T + (low - T) * smoothstep(-2.5, -0.8, epn)
        h[nz] = h[nz] + (hn - h[nz]) * her[nz]
    return h, e, ep, q, sdb, dw, dv, her


_AUX = ('e', 'ep', 'q', 'sdb', 'dw', 'dv', 'her')


def _terrain(x, y, aux=False):
    shape = np.shape(x)
    x = np.asarray(x, dtype=np.float64).ravel(); y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    h = np.empty(n)
    A = {k: np.empty(n) for k in _AUX} if aux else None
    for s in range(0, n, _CHUNK):
        r = _terrain_chunk(x[s:s + _CHUNK], y[s:s + _CHUNK])
        h[s:s + _CHUNK] = r[0]
        if aux:
            for k, v in zip(_AUX, r[1:]):
                A[k][s:s + _CHUNK] = v
    if aux:
        return h.reshape(shape), {k: v.reshape(shape) for k, v in A.items()}
    return h.reshape(shape)


def rim_info(x, y):
    """For the cliff sheet builder: (T, rimz) = terrain top surface near the rim and the rock rim level."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    e, ep, q = cliff_coords(x, y)
    sdb = built_sd(x, y)
    Hm, nat = _massif(x, y, q, e, sdb, lambda k: np.zeros(int(np.count_nonzero(k)), dtype=np.float32))
    rimz = RIM_Z_BUILT + (nat - 1.5 - RIM_Z_BUILT) * smoothstep(90.0, 180.0, sdb)
    T = Hm + (rimz - Hm) * smoothstep(-3.0, -0.8, e)
    return T, rimz, e


# =====================================================================================
# public API
# =====================================================================================
def height_at(x, y):
    """Terrain height z (metres) at world position(s) x, y.  Accepts scalars or numpy arrays."""
    return _terrain(x, y)


def slope_at(x, y, step=4.0):
    """Terrain slope as tan(angle)."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    h = _terrain(x, y)
    return np.hypot(_terrain(x + step, y) - h, _terrain(x, y + step) - h) / step


def forest_mask(x, y, _pre=None):
    """0..1 forest density: 0 on water, shores, cliffs / steep rock, the cliff rim, snow and the alpine zone,
    built areas (+ a few metres of clearance); patchy meadows elsewhere."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if _pre is None:
        h, a = _terrain(x, y, aux=True)
        step = 4.0
        slope = np.hypot(_terrain(x + step, y) - h, _terrain(x, y + step) - h) / step
    else:
        h, a, slope = _pre
    xs = x.astype(F32); ys = y.astype(F32)
    m = 1.0 - smoothstep(0.62, 0.95, slope)
    m = m * smoothstep(VALLEY_Z + 1.2, VALLEY_Z + 2.6, h)
    tl, cl = gnoise2p(xs * F32(1 / 800.0), ys * F32(1 / 800.0), 5)
    m = m * (1.0 - smoothstep(TREELINE_Z - 70.0, TREELINE_Z + 60.0, h + 70.0 * tl))
    m = m * smoothstep(5.0, 14.0, a['sdb'])
    # hero cliff: nothing on the rim, the face or the hidden pit; sparse on the upper talus
    ep = a['ep']; her = a['her']
    face = smoothstep(-9.0, -5.0, ep) * (1.0 - smoothstep(FOOT_E - 30.0, FOOT_E + 5.0, ep))
    m = m * (1.0 - face * smoothstep(0.0, 0.3, her))
    # meadows / clearings: mostly on flat ground near the water and on the plateau
    cl2 = fbm(xs * F32(1 / 520.0), ys * F32(1 / 520.0), 3, seed=97)
    flat = 1.0 - smoothstep(0.10, 0.32, slope)
    m = m * (1.0 - 0.9 * flat * smoothstep(0.12, 0.34, cl2))
    return np.clip(m, 0.0, 1.0)
