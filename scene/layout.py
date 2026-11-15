"""Shared world layout for the cliff-top castle scene.  EVERY module must respect these numbers.

Units: meters.  Z up.  +X = east, +Y = north.
Origin (0,0,0) = centre of the main marble terrace, z=0 = terrace floor level.

Big picture (top view, north up):

                 N (+Y)
        mountains far NW / N / W (8-30 km)
   valley + lake (z=-340)  |  castle (y 20..72)
        to the WEST        |  terrace (y -46..20) with infinity pool on its west edge
   <- sun sets WNW         |  plateau / forested mountain shoulder continues EAST

The hero camera hovers SE of the terrace looking NW: castle fills the right half of frame,
the pool + cliff edge + waterfall sit centre/left, the valley, lake, mountains and the setting
sun are on the left.
"""
import math

# ---------------------------------------------------------------- sun / time of day
SUN_AZIMUTH_DEG = 157.0      # direction TOWARD the sun, measured CCW from +X (east). 157 deg = WNW
SUN_ELEVATION_DEG = 3.0      # just above the horizon, golden/red sunset
def sun_dir():
    """Unit vector pointing from the scene toward the sun."""
    a = math.radians(SUN_AZIMUTH_DEG); e = math.radians(SUN_ELEVATION_DEG)
    return (math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e))

# ---------------------------------------------------------------- heights
TERRACE_Z = 0.0              # main terrace floor
PODIUM_Z = 2.4               # castle ground floor / entrance podium (16 steps up from terrace)
LOUNGE_Z = -1.2              # lower SE lounge terrace (fire pit), 8 steps down
LOWER_Z = -1.3               # lower west terrace between the pool's west wall and the cliff line
POOL_WATER_Z = -0.06         # pool water surface (flush infinity edge)
POOL_DEPTH = 1.5
VALLEY_Z = -340.0            # lake / valley floor level
PLATEAU_BASE_Z = -0.6        # natural ground just outside paved areas

# ---------------------------------------------------------------- terrace (paved marble)
TERRACE_X0, TERRACE_X1 = -46.0, 44.0
TERRACE_Y0, TERRACE_Y1 = -50.0, 20.0     # south edge = cliff line (for x < 14; east of that the lounge starts at y=-46)
# lower lounge terrace with the fire pit (SE corner, camera foreground)
LOUNGE_X0, LOUNGE_X1 = 18.0, 46.0
LOUNGE_Y0, LOUNGE_Y1 = -70.0, -46.0        # south edge = cliff line
FIREPIT_POS = (31.0, -55.0, LOUNGE_Z)      # centre of the round fire pit (radius ~2.2 m)

# infinity pool: long axis N-S on the west cliff edge of the terrace
POOL_X0, POOL_X1 = -44.5, -25.0
POOL_Y0, POOL_Y1 = -49.1, 8.0             # the south wall (0.9 m thick) stands exactly on the cliff line y=-50
# The pool is an infinity pool on its WEST edge (x = POOL_X0) and SOUTH edge (y = POOL_Y0).
# Main waterfall outlet: a 7 m wide notch in the south edge near the SW corner; the water sheet falls
# from here down the retaining wall (whose face is the plane y = SPILL_Y = -50) and the cliff into the mist far below.
# West of the pool there is a LOWER TERRACE at z = LOWER_Z between the pool wall and the cliff line (catch channel,
# paving, lookout pavilion); on the south side a corbelled stone gutter catches the thin overflow film.
SPILL_X0, SPILL_X1 = -43.0, -36.0
SPILL_Y = POOL_Y0 - 0.9       # outer face of the pool wall where the sheet leaves
SPILL_TOP_Z = POOL_WATER_Z
SPILL_BOTTOM_Z = -230.0       # sheet dissolves into mist well above the valley floor

# grand stair from terrace up to castle podium (centred on the great hall, x = 2)
STAIR_X0, STAIR_X1 = -12.0, 16.0
STAIR_Y0, STAIR_Y1 = 13.0, 20.0
# upper podium terrace in front of the wings (z = PODIUM_Z): y 20..27, balustrade along y = 20
PODIUM_Y0, PODIUM_Y1 = 20.0, 27.0
# stair from the main terrace down to the lounge terrace: x 22..30 at y -46..-49 (8 steps, descending south)
LOUNGE_STAIR_X0, LOUNGE_STAIR_X1 = 22.0, 30.0

# ---------------------------------------------------------------- global look constants (tuned at integration)
PRACTICAL_GAIN = 0.8   # multiply the energy of every practical light (lanterns, sconces, fire) by this
EMISSIVE_GAIN = 0.6    # multiply emission strength of glowing materials (windows, flames, pool) by this
# aerial perspective colours (scene-linear RGB).  Far terrain / mist fade toward these (see scene/atmo.py);
# the sky's horizon band must be consistent with them.
HAZE_TOWARD_SUN = (1.00, 0.46, 0.17)    # glowing orange haze around the sun azimuth
HAZE_AWAY = (0.21, 0.20, 0.30)          # cool purple-grey haze away from the sun
HAZE_STRENGTH = 0.55                    # brightness multiplier of haze emission

# ---------------------------------------------------------------- built areas (terrain must stay BELOW z = -3 here)
# (x0, x1, y0, y1): paved terraces, pool, castle.  Natural ground / rock may only show outside these.
BUILT_AREAS = [
    (-47.5, 46.0, -50.0, 28.0),    # main terrace + pool + podium
    (13.0, 49.0, -71.0, -44.0),    # lounge terrace
    (-62.5, -45.0, -22.0, 32.0),   # lower west terrace
    (-72.0, 66.0, 13.0, 78.0),     # castle
    (-74.0, -58.0, -2.0, 14.0),    # lookout bastion
]

# ---------------------------------------------------------------- cliff / plateau outline
# Top edge of the cliff (plateau boundary), CCW polyline seen from above, running from the far SE,
# west along the south face, around the SW promontory and north along the west face.
# Everything WEST / SOUTH of this line drops (near vertical rock, 200-340 m) to the valley.
# Everything EAST of it is the plateau (castle, gardens) rising gently into a forested mountain.
CLIFF_TOP = [
    (400.0, -150.0), (220.0, -110.0), (120.0, -84.0), (64.0, -76.0), (48.0, -70.0),
    (16.0, -70.0), (14.0, -50.0), (-47.5, -50.0),          # south face under lounge + terrace
    (-50.0, -20.0), (-58.0, -6.0), (-70.0, 0.0),            # SW corner -> lookout spur
    (-72.0, 10.0), (-62.0, 16.0), (-60.0, 24.0),
    (-68.0, 30.0), (-70.0, 40.0), (-62.0, 52.0),            # bastion rotunda sits at (-58, 36)
    (-52.0, 80.0), (-40.0, 120.0), (-10.0, 170.0), (60.0, 240.0), (200.0, 380.0),
]
# masonry (terrace retaining walls, built by terrace.py) covers the cliff top down to about this z;
# natural rock should start at/above it so there is never a gap.
MASONRY_BOTTOM_Z = -16.0

# ---------------------------------------------------------------- castle footprint (for avoidance / placement)
# (name, x0, x1, y0, y1, approx top z).  Built by castle.py.
CASTLE_BLOCKS = [
    ("great_hall",   -9.0,  13.0, 22.0, 46.0, 42.0),
    ("east_tower",   17.0,  31.0, 15.0, 31.0, 48.0),
    ("east_wing",    31.0,  62.0, 27.0, 45.0, 26.0),
    ("west_wing",   -46.0,  -9.0, 27.0, 46.0, 20.0),
    ("upper_keep",   -6.0,  32.0, 46.0, 72.0, 70.0),
    ("bastion",     -66.0, -50.0, 28.0, 44.0, 22.0),
    ("lookout",     -72.0, -62.0,  0.0, 12.0,  6.0),
]
CASTLE_BBOX = (-72.0, 66.0, 13.0, 76.0)   # x0,x1,y0,y1 no-go zone for trees / terrain bumps

# ---------------------------------------------------------------- towers (built by castle.py; listed for reference)
# name: (cx, cy, radius, eave_z, roof_height)
TOWERS = {
    "west_round":    (-30.0, 29.0, 4.6, 38.0, 13.0),
    "west_slim":     (-20.5, 40.0, 3.2, 47.0, 12.0),
    "hall_turret_w": (-10.2, 22.6, 2.3, 44.0, 9.5),
    "hall_turret_e": (14.2, 22.6, 2.3, 44.0, 9.5),
    "keep_sw":       (-6.0, 46.0, 3.4, 56.0, 12.0),
    "keep_se":       (32.0, 46.0, 3.4, 56.0, 12.0),
    "keep_n":        (13.0, 72.0, 4.2, 62.0, 16.0),
    "ne_tower":      (52.0, 50.0, 5.0, 40.0, 14.0),
    "bastion":       (-58.0, 36.0, 8.0, 14.0, 0.0),
    "lookout":       (-67.0, 6.0, 4.6, 0.0, 0.0),
}

# ---------------------------------------------------------------- anchors for props / plants
def _stair_lanterns():
    out = []
    n = 16; d = (STAIR_Y1 - STAIR_Y0) / n; h = PODIUM_Z / n
    for k in (2, 6, 10, 14):                       # on the steps, both ends
        z = (k + 1) * h
        y = STAIR_Y0 + (k + 0.5) * d
        out.append((STAIR_X0 + 0.7, y, z, "floor")); out.append((STAIR_X1 - 0.7, y, z, "floor"))
    return out

# (x, y, z_of_floor_under_it, kind)   kind "floor" = ~0.9 m tall candle lantern, "small" = ~0.5 m
LANTERNS = _stair_lanterns() + [
    (STAIR_X0 - 1.4, 12.2, 0.0, "floor"), (STAIR_X1 + 1.4, 12.2, 0.0, "floor"),
    # pool east coping, between the sunbeds
    (-23.6, -37.0, 0.0, "floor"), (-23.6, -25.0, 0.0, "floor"), (-23.6, -11.0, 0.0, "floor"), (-23.6, 3.0, 0.0, "floor"),
    # pool north end deck
    (-44.0, 9.4, 0.0, "floor"), (-34.5, 9.4, 0.0, "floor"), (-25.5, 9.4, 0.0, "floor"),
    # along the south balustrade of the main terrace
    (-20.0, -47.2, 0.0, "small"), (-12.0, -47.2, 0.0, "small"), (-4.0, -47.2, 0.0, "small"), (4.0, -47.2, 0.0, "small"),
    (11.0, -47.2, 0.0, "small"),
    # lounge terrace
    (19.4, -47.2, LOUNGE_Z, "floor"), (44.6, -47.2, LOUNGE_Z, "floor"), (19.6, -67.2, LOUNGE_Z, "floor"),
    (41.4, -67.2, LOUNGE_Z, "floor"), (26.0, -58.6, LOUNGE_Z, "small"), (35.8, -50.2, LOUNGE_Z, "small"),
    # in front of the castle entrance / podium
    (-6.0, 20.8, PODIUM_Z, "floor"), (10.0, 20.8, PODIUM_Z, "floor"),
    (-20.0, 21.0, PODIUM_Z, "floor"), (-34.0, 21.0, PODIUM_Z, "floor"), (36.0, 21.0, PODIUM_Z, "floor"),
]

# sunbeds: (x, y, z, heading)  heading = direction the FEET point to, radians CCW from +X
SUNBEDS = [(-21.2, y, 0.0, math.pi) for y in (-34.0, -31.2, -21.0, -18.2, -7.0, -4.2)] + \
          [(x, 11.6, 0.0, -math.pi / 2) for x in (-41.0, -38.2, -30.8, -28.0)]

# lounge sofas around the fire pit: (x0, x1, y0, y1, facing) seat rectangles; facing = direction people look
SOFAS = [
    (36.0, 38.2, -60.0, -50.0, math.pi),            # east side, facing west (toward fire + sunset)
    (26.0, 36.0, -61.8, -59.6, math.pi / 2),        # south side, facing north (toward fire + castle)
]

# raised flower planters (x0, x1, y0, y1, z_base): white blossom shrubs + greenery
PLANTERS = [
    (-22.0, 11.5, -49.2, -47.9, 0.0),               # along the south balustrade of the main terrace
    (19.0, 42.5, -69.2, -68.0, LOUNGE_Z),           # south edge of the lounge = bottom of the hero frame
    (17.0, 18.2, -66.5, -50.5, LOUNGE_Z),           # west edge of the lounge
    (-22.0, -13.0, 18.4, 19.7, 0.0),                # left of the grand stair
    (32.5, 44.0, 18.4, 19.7, 0.0),                  # right, in front of the east wing podium
]

# slender italian cypresses: (x, y, z_base, height)
CYPRESSES = [(x, 25.6, PODIUM_Z, h) for x, h in ((-43.0, 9.0), (-37.0, 10.5), (-23.5, 9.5), (-17.0, 11.0), (-11.8, 8.5))] + \
            [(x, 25.6, PODIUM_Z, h) for x, h in ((33.5, 10.0), (39.5, 11.5), (45.5, 9.5), (51.5, 11.0), (57.5, 10.0))] + \
            [(-45.2, 12.6, 0.0, 8.0), (-24.2, 12.6, 0.0, 8.5), (-13.6, 12.4, 0.0, 7.5), (17.6, 12.4, 0.0, 7.5),
             (45.0, -45.0, 0.0, 9.0), (45.2, 16.0, 0.0, 10.0)]

# red japanese maples: (x, y, z_base, height, crown_radius).  The first one overhangs the hero camera
# (leaves frame the top-right of the shot, like the references).
MAPLES = [
    (45.6, -67.4, LOUNGE_Z, 10.0, 8.0),             # in a tree pit in the SE corner of the lounge terrace
    (49.5, -44.0, -0.4, 8.0, 6.0),
    (12.6, -47.4, 0.0, 4.0, 3.2),      # in a big pot at the SE corner of the main terrace
    (-23.0, 16.6, 0.0, 4.5, 3.4),
    (60.0, -20.0, -0.5, 9.0, 7.0),
    (58.0, 8.0, -0.5, 7.5, 6.0),
]

# ---------------------------------------------------------------- standard preview cameras
# name: (position, look-at target, focal length mm).  Use these for all look-dev renders.
CAMERAS = {
    "hero":    ((40.0, -74.0, 9.5),   (-14.0, 22.0, 13.0), 28.0),   # like reference image 1
    "wide":    ((120.0, -190.0, 40.0), (-30.0, 20.0, 5.0), 32.0),   # castle on cliff, like reference 3
    "aerial":  ((-330.0, -210.0, -40.0), (-10.0, 30.0, 10.0), 35.0),# from out over the valley looking back E
    "valley":  ((30.0, -60.0, 14.0),  (-900.0, 260.0, -260.0), 35.0),# terrace -> lake, mountains, sun
    "pool":    ((-34.0, -40.0, 1.3),  (2.0, 22.0, 16.0), 24.0),     # low over the pool to the great window
    "facade":  ((2.0, -22.0, 6.0),    (2.0, 22.0, 17.0), 30.0),     # frontal on great hall
    "firepit": ((38.0, -62.0, 0.6),   (10.0, 0.0, 14.0), 26.0),     # lounge foreground, castle behind
    "cliff":   ((-150.0, -130.0, -60.0), (-40.0, -50.0, -40.0), 40.0),# hero waterfall + cliff face
    "top":     ((0.0, -10.0, 420.0),  (0.0, 10.0, 0.0), 35.0),      # plan view
    "sky":     ((30.0, -60.0, 10.0),  (-400.0, 150.0, 120.0), 20.0),# mostly sky toward the sun
}




