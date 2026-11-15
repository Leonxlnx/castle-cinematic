"""Hero cliff sheet for the landscape module.

A dense quad sheet laid out along iso-lines of e' (distance to the notch-filled layout.CLIFF_TOP line):
columns follow the smoothed 55 m offset contour (landscape_field.MID) with adaptive spacing, rows run from
a buried lip on the plateau (e' = -5) over the rim and ~290 m down the face into the talus apron.
The nominal surface (lean, buttresses, ledges, vertical waterfall chute) is then displaced along its
normals with layered 3D noise (stepped strata, vertical fractures, columns, lumps).

Rules honoured (see docs/AGENT_BRIEF + coordinator notes):
  * rock crosses the CLIFF_TOP line at z ~ -10.5 (between -14 and -8) wherever built areas are near
  * no rock above z = -6 outside the line near the terrace / pool
  * below the hero waterfall (x -43..-36 on the south face) the face is vertical / slightly receding
    (<= ~0.6 m outside the line) down to z ~ -235

Tunables: COL_SPACING, ROW_*, OFF_*, BUTTRESS_*, the amplitudes in _displacement().
"""
import math, time
import numpy as np
from . import landscape_field as F
from .. import layout as L

COL_SPACING = 0.80        # column spacing (m) in the hero core (measured on the widest level)
COL_SPACING_FAR = 4.5     # at the ends of the sheet
ROW_DT0, ROW_DT1 = 0.62, 1.85   # row spacing at the top / bottom of the face
T_MAX = 292.0             # face height below the rim
E_MAX = 125.0             # max nominal offset (for the column metric)
SHARP_ANGLE = 50.0        # edges sharper than this are shaded hard (crisp block edges)
Q_END = 955.0             # sheet ends where the heightfield flank has taken over
LIP_E = np.array([-5.0, -4.2, -3.5, -3.0, -2.6, -2.2, -1.8, -1.4, -1.0, -0.6, -0.25])
LIP_DZ = np.array([-1.3, -0.8, -0.35, -0.05, 0.15, 0.28, 0.32, 0.32, 0.30, 0.25, 0.15])

LEDGES = None             # filled by build_cliff(): (n,3) points on upward facing ledges (tree spots)


def off_nominal(t):
    """Nominal outward offset (m) of the face at depth t below the rim."""
    return 0.05 + 0.11 * t + 0.0002 * t * t


def off_chute(t):
    """Waterfall chute: vertical / slightly receding wall, flaring out into the plunge basin."""
    return 0.35 - 0.003 * np.minimum(t, 235.0) + 46.0 * F.smoothstep(232.0, 292.0, t) ** 1.4


def _columns():
    """Adaptive columns along the mid contour.  Returns (M (n,2), N (n,2) outward normals, s arclength)."""
    M = F.MID
    tx = np.gradient(M[:, 0]); ty = np.gradient(M[:, 1])
    ln = np.hypot(tx, ty); tx /= ln; ty /= ln
    Nx = -ty; Ny = tx
    top = M - np.stack([Nx, Ny], axis=1) * F.R_MID
    q = np.hypot(top[:, 0] - F.CX, top[:, 1] - F.CY)
    ok = np.nonzero(q < Q_END)[0]
    i0, i1 = ok[0], ok[-1]
    M = M[i0:i1 + 1]; Nx = Nx[i0:i1 + 1]; Ny = Ny[i0:i1 + 1]; top = top[i0:i1 + 1]; q = q[i0:i1 + 1]
    bot = M + np.stack([Nx, Ny], axis=1) * (E_MAX - F.R_MID)
    d_top = np.hypot(np.diff(top[:, 0]), np.diff(top[:, 1]))
    d_mid = np.hypot(np.diff(M[:, 0]), np.diff(M[:, 1]))
    d_bot = np.hypot(np.diff(bot[:, 0]), np.diff(bot[:, 1]))
    target = COL_SPACING + (COL_SPACING_FAR - COL_SPACING) * F.smoothstep(230.0, 900.0, 0.5 * (q[1:] + q[:-1]))
    step = np.maximum(np.maximum(d_top, d_mid), d_bot) / target
    cum = np.concatenate([[0.0], np.cumsum(step)])
    ncol = int(cum[-1]) + 1
    fi = np.interp(np.linspace(0.0, cum[-1], ncol), cum, np.arange(len(cum)))
    ia = np.arange(len(M))
    Mx = np.interp(fi, ia, M[:, 0]); My = np.interp(fi, ia, M[:, 1])
    nx = np.interp(fi, ia, Nx); ny = np.interp(fi, ia, Ny)
    nl = np.hypot(nx, ny); nx /= nl; ny /= nl
    s = fi * 0.5
    return np.stack([Mx, My], axis=1), np.stack([nx, ny], axis=1), s


def _rows():
    t = [0.35]
    while t[-1] < T_MAX:
        t.append(t[-1] + ROW_DT0 + (ROW_DT1 - ROW_DT0) * (t[-1] / T_MAX) ** 0.75)
    return np.array(t)


_BED = np.random.RandomState(4711).rand(4096).astype(np.float32)


def _displacement(P):
    """Displacement along the normal (m) for nominal positions P (n,3): bedding with seams, cellular
    blocks at two scales (flat block faces at different depths, cracks between them), big vertical
    fractures and soft lumps."""
    f32 = np.float32
    x = P[:, 0].astype(f32); y = P[:, 1].astype(f32); z = P[:, 2].astype(f32)
    warp = f32(6.0) * F.fbm3(x / 60.0, y / 60.0, z / 60.0, 2, seed=3)
    zs = z + warp + f32(0.05) * (f32(0.8) * x + f32(0.6) * y)
    zone = F.smoothstep(-0.25, 0.30, F.fbm3(x / 90.0, y / 90.0, z / 50.0, 2, seed=13))   # 0 massive .. 1 bedded
    d = f32(2.3) * F.fbm3(x / 28.0, y / 28.0, z / 28.0, 3, seed=9)                 # lumps
    # bedding: every bed sits a little further out / in, recessed seams between beds
    v = zs / f32(10.5) + f32(0.3)
    i = np.floor(v); f = v - i
    ii = i.astype(np.int64) & 4095
    r0 = _BED[ii]; r1 = _BED[(ii + 1) & 4095]
    d += f32(2.0) * (r0 + (r1 - r0) * F.smoothstep(0.86, 1.0, f) - f32(0.5)) * (f32(0.3) + f32(0.7) * zone)
    d -= f32(0.8) * (f32(1.0) - F.smoothstep(0.0, 0.11, np.minimum(f, f32(1.0) - f))) * zone
    # pillars (tall cells between vertical joints) and blocks: flat-ish faces at different depths that
    # fall into V-shaped cracks at the cell borders (so the surface stays continuous)
    amp = f32(0.35) + f32(0.65) * F.smoothstep(-0.3, 0.2, F.fbm3(x / 70.0, y / 70.0, z / 70.0, 2, seed=17))
    f1, f2, val = F.worley3(x / 7.5, y / 7.5, zs / 14.0, seed=1)
    face = F.smoothstep(0.0, 0.28, f2 - f1)
    d += (face * (f32(1.9) * (val - f32(0.5)) + f32(0.6)) - f32(1.1) * (f32(1.0) - face)) * amp
    f1, f2, val = F.worley3(x / 4.3, y / 4.3, zs / 3.5, seed=2)
    face = F.smoothstep(0.0, 0.34, f2 - f1)
    d += (face * (f32(0.95) * (val - f32(0.5)) + f32(0.25)) - f32(0.5) * (f32(1.0) - face)) * amp
    fr = F.gnoise3(x / 11.0, y / 11.0, z / 80.0, seed=5)
    d -= f32(2.8) * F.smoothstep(0.82, 0.985, f32(1.0) - np.abs(fr))               # major vertical fractures
    d += f32(0.30) * F.fbm3(x / 3.2, y / 3.2, z / 3.2, 3, seed=11)                # fine
    return d


def build_cliff(col, new_mesh_object, add_attr, grid_quads):
    global LEDGES
    t0 = time.time()
    M, N, s = _columns()
    ncol = len(M)
    tr = _rows()
    nlip = len(LIP_E); nrow = nlip + len(tr)
    # ---- per column data
    top = M - N * F.R_MID                                    # point on the closed line (e' = 0)
    Ttop, rimz, _ = F.rim_info(top[:, 0], top[:, 1])
    ztop = rimz + 0.12
    q = np.hypot(top[:, 0] - F.CX, top[:, 1] - F.CY)
    her = 1.0 - F.smoothstep(F.HERO_Q0, F.HERO_Q1, q)
    dwf = np.hypot(top[:, 0] - F.WATERFALL_C[0], top[:, 1] - F.WATERFALL_C[1])
    wch = 1.0 - np.clip((dwf - 6.0) / 24.0, 0.0, 1.0) ** 0.7           # vertical chute weight
    wcore = 1.0 - F.smoothstep(5.0, 11.0, dwf)                         # calm, water-worn core
    sf = s.astype(np.float32)
    # ---- nominal surface
    P = np.empty((nrow, ncol, 3))
    for r in range(nlip):
        xy = M + N * (LIP_E[r] - F.R_MID)
        T, _, _ = F.rim_info(xy[:, 0], xy[:, 1])
        P[r, :, 0] = xy[:, 0]; P[r, :, 1] = xy[:, 1]
        P[r, :, 2] = T + LIP_DZ[r] * her - 1.3 * (1.0 - her)
    tt = tr[:, None]
    S2 = np.broadcast_to(sf[None, :], (len(tr), ncol)); T2 = np.broadcast_to(tt.astype(np.float32), (len(tr), ncol))
    b = F.fbm(S2 / 85.0, T2 / 190.0, 3, seed=201)
    butt = (0.3 + 0.12 * tt) * F.smoothstep(-0.25, 0.55, b)
    led = 0.0
    for k, (tl, wl) in enumerate(((52.0, 8.5), (122.0, 10.0), (192.0, 11.5))):
        l1, p1 = F.gnoise2p(sf / 150.0, sf * 0.0 + 0.37 + k, 203 + k)
        l2, p2 = F.gnoise2p(sf / 55.0, sf * 0.0 + 5.37 + k, 213 + k)
        lvl = tl + 30.0 * l1[None, :] + 7.0 * l2[None, :]
        led = led + wl * F.smoothstep(0.0, 3.5, tt - lvl) * F.smoothstep(-0.22, 0.12, p1[None, :])
    off = off_nominal(tt) + butt + led
    off = off + (off_chute(tt) - off) * wch[None, :]
    P[nlip:, :, 0] = M[None, :, 0] + N[None, :, 0] * (off - F.R_MID)
    P[nlip:, :, 1] = M[None, :, 1] + N[None, :, 1] * (off - F.R_MID)
    P[nlip:, :, 2] = ztop[None, :] - tt
    # ---- displacement direction: horizontally outward on the face (robust at the corners where many
    # columns share one top point; varying it with z still gives ledges and overhangs), up on the lip
    nrm = np.zeros_like(P)
    nrm[nlip:, :, 0] = N[None, :, 0]; nrm[nlip:, :, 1] = N[None, :, 1]
    nrm[:nlip, :, 2] = 1.0
    # ---- displacement
    D = _displacement(P.reshape(-1, 3)).reshape(nrow, ncol).astype(np.float64)
    amp = np.empty((nrow, 1)); amp[:nlip, 0] = 0.10; amp[nlip:, 0] = 0.12 + 0.88 * F.smoothstep(0.3, 5.5, tr)
    chute_calm = 1.0 - 0.72 * wcore[None, :] * np.concatenate([np.ones(nlip), 1.0 - F.smoothstep(225.0, 262.0, tr)])[:, None]
    Dn = D * amp * chute_calm * (0.25 + 0.75 * her[None, :])
    # runnels in the chute (water-worn vertical grooves)
    run = np.abs(F.gnoise2(sf * 1.3, sf * 0.0 + 2.1, 207)) * 0.5
    Dn[nlip:] -= run[None, :] * wcore[None, :] * F.smoothstep(2.0, 12.0, tr)[:, None]
    Pd = P + nrm * Dn[:, :, None]
    # never above the rim level on the face / outside the line
    zcap = ztop[None, :] + 0.3
    Pd[nlip:, :, 2] = np.minimum(Pd[nlip:, :, 2], zcap)
    Pd[:nlip, :, 2] = np.minimum(Pd[:nlip, :, 2], P[:nlip, :, 2] + 0.25)
    co = Pd.reshape(-1, 3)
    quads = grid_quads(nrow, ncol, False).reshape(-1, 4)[:, ::-1].reshape(-1)
    ob = new_mesh_object("landscape_cliff", co, quads, np.arange(len(quads) // 4) * 4, col, "ls_cliff")
    cm = np.zeros((nrow * ncol, 4), dtype=np.float32)
    cm[:, 0] = np.clip(0.5 + (D * amp).ravel() / 9.0, 0.0, 1.0)
    wet = np.broadcast_to((1.0 - F.smoothstep(4.0, 16.0, dwf))[None, :], (nrow, ncol))
    cm[:, 1] = wet.ravel()
    cm[:, 2] = np.concatenate([np.zeros(nlip), tr / T_MAX])[:, None].repeat(ncol, axis=1).ravel()
    cm[:, 3] = 1.0
    add_attr(ob.data, "cmask", cm, 'FLOAT_COLOR')
    try:
        ob.data.set_sharp_from_angle(angle=math.radians(SHARP_ANGLE))
    except Exception as ex:
        print("[landscape] sharp edges:", ex)
    # ---- ledge spots (upward facing parts of the displaced face, away from the chute and the buried ends)
    d2c = np.empty_like(Pd); d2r = np.empty_like(Pd)
    d2c[:, 1:-1] = Pd[:, 2:] - Pd[:, :-2]; d2c[:, 0] = Pd[:, 1] - Pd[:, 0]; d2c[:, -1] = Pd[:, -1] - Pd[:, -2]
    d2r[1:-1] = Pd[2:] - Pd[:-2]; d2r[0] = Pd[1] - Pd[0]; d2r[-1] = Pd[-1] - Pd[-2]
    n2 = np.cross(d2c, d2r); n2 /= np.maximum(np.linalg.norm(n2, axis=2), 1e-9)[:, :, None]
    tfull = np.concatenate([np.zeros(nlip), tr])[:, None]
    okl = (n2[:, :, 2] > 0.82) & (tfull > 22.0) & (tfull < 235.0) & (wch[None, :] < 0.05) & (her[None, :] > 0.9)
    LEDGES = Pd[okl]
    print("[landscape] cliff sheet %d x %d = %dk tris, %d ledge verts  %.1fs" %
          (ncol, nrow, (ncol - 1) * (nrow - 1) * 2 // 1000, len(LEDGES), time.time() - t0))
    return ob
