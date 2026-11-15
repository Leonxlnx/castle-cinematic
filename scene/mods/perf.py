"""Render-performance pass, applied after all other modules (measured on this laptop, see tools/profile4.py).

The sun shadow was 70% of the frame time because every one of ~7.6M triangles (forest, mountains, cliff) was
rasterised into very fine virtual shadow pages.  Visually it makes almost no difference at sunset whether the
far landscape casts real shadows (it is back-lit and hazy), so only the castle / terraces / props / garden
plants cast shadows.
"""
import bpy

SUN_MAX_SHADOW_RES = 0.05      # metres per shadow texel at the finest level (was 0.004)
SHADOW_RES_SCALE = 0.5
NO_SHADOW_COLLECTIONS = ["landscape", "water"]
NO_SHADOW_NAME_PARTS = ["veg_forest", "landscape_", "wf_", "mist", "flame", "spark", "ember"]


def build(ctx):
    scene = ctx.scene
    ee = scene.eevee
    for l in bpy.data.lights:
        if l.type == 'SUN':
            l.shadow_maximum_resolution = SUN_MAX_SHADOW_RES
            l.use_shadow_jitter = False
            l.specular_factor = 0.0       # no blinding sun glint on the lake / wet marble (owner: lights must not glare)
        elif l.use_shadow:
            l.use_shadow_jitter = False
    ee.shadow_resolution_scale = SHADOW_RES_SCALE
    ee.use_overscan = False
    n = 0
    for cn in NO_SHADOW_COLLECTIONS:
        col = bpy.data.collections.get(cn)
        if col:
            for o in col.all_objects:
                o.visible_shadow = False; n += 1
    for o in bpy.data.objects:
        nm = o.name.lower()
        if any(p in nm for p in NO_SHADOW_NAME_PARTS):
            if o.visible_shadow:
                o.visible_shadow = False; n += 1
    print("[perf] shadow casting disabled on %d objects; sun shadow res %.3f m, scale %.2f" % (n, SUN_MAX_SHADOW_RES, SHADOW_RES_SCALE))


