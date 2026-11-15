"""Framework smoke test: exercises MB primitives + every registered material."""
import math
from ..common import MB, get_col, pointed_arch, circle_pts, rect_pts, xf, frame_xf
from .. import mats


def build(ctx):
    col = get_col("smoke")
    mb = MB("smoke")
    mb.box(-60, 60, -60, 40, -0.3, 0.0, "marble_wet")
    names = mats.names()
    for i, n in enumerate(names):
        x = -40 + i * 5.5
        mb.box(x, x + 4, 10, 11, 0, 6, n)
        mb.lathe([(1.2, 0), (1.2, 4), (0.1, 7)], 24, n, center=(x + 2, 4))
    # arch panel with hole + reveal + hood mould
    mb.push(frame_xf((-20, -10, 0), (1, 0)))
    arch = pointed_arch(3.0, 4.0, n=10)
    outline = [(-1.5, 1.0), (1.5, 1.0)] + arch
    outline = [(p[0] + 4, p[1]) for p in outline]
    mb.poly_holes([(0, 0), (8, 0), (8, 9), (0, 9)], [outline], "stone")
    mb.sweep_xz([(0, 0), (0, 0.15), (-0.12, 0.15), (-0.12, 0.4)], outline, "stone_trim", closed=True)
    mb.rnd = 0.7
    mb.quad((2, 0.4, 0.5), (6, 0.4, 0.5), (6, 0.4, 8), (2, 0.4, 8), "glass_warm")
    mb.sweep_xz([(0.0, 0.0), (0.0, -0.1), (0.16, -0.1), (0.22, 0.0)], [(p[0] + 4, p[1]) for p in arch], "stone_trim", closed=False)
    mb.pop()
    mb.sweep([(0, 0), (0.3, 0), (0.3, 0.4), (0, 0.5)], rect_pts(-30, -22, -30, -24), "stone_dark", closed=True)
    mb.prism(circle_pts(3, 8, 10, -20), 0, 5, "stone", cap_top=True, mat_top="slate")
    mb.to_object("smoke", col)
