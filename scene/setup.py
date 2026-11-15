"""Render / scene setup shared by build.py and the final renderer."""
import bpy, math
from mathutils import Vector
from . import layout as L
from .common import get_col, look_at, NT

FPS = 24
DURATION_S = 45
FRAME_END = FPS * DURATION_S   # 1080


def setup_render(scene, res=(1920, 1080), samples=32, quality="preview"):
    r = scene.render
    r.engine = 'BLENDER_EEVEE'
    r.resolution_x, r.resolution_y = res
    r.resolution_percentage = 100
    r.fps = FPS
    scene.frame_start = 1
    scene.frame_end = FRAME_END
    r.image_settings.file_format = 'PNG'
    r.image_settings.color_mode = 'RGB'
    r.image_settings.color_depth = '8'
    r.image_settings.compression = 30
    r.film_transparent = False
    r.use_file_extension = True
    r.threads_mode = 'FIXED'
    r.threads = 12                      # leave 4 logical cores free so the laptop stays responsive

    ee = scene.eevee
    ee.taa_render_samples = samples
    ee.use_shadows = True
    ee.shadow_ray_count = 1
    ee.shadow_step_count = 6
    ee.shadow_resolution_scale = 1.0
    ee.shadow_pool_size = '1024' if hasattr(ee, 'shadow_pool_size') else ee.shadow_pool_size
    ee.use_raytracing = True
    ee.ray_tracing_method = 'SCREEN'
    try:
        o = ee.ray_tracing_options
        o.resolution_scale = '2'
        o.use_denoise = True
        o.trace_max_roughness = 0.5
        o.screen_trace_quality = 0.25
    except Exception as e:
        print("raytrace opts:", e)
    ee.use_fast_gi = True
    ee.fast_gi_method = 'GLOBAL_ILLUMINATION'
    ee.fast_gi_resolution = '2'
    ee.fast_gi_ray_count = 2
    ee.fast_gi_step_count = 8
    ee.fast_gi_distance = 12.0
    # volumetrics (only matter if a module adds volume objects / world volume)
    ee.volumetric_tile_size = '8'
    ee.volumetric_samples = 96
    ee.volumetric_sample_distribution = 0.85
    ee.use_volume_custom_range = True
    ee.volumetric_start = 1.0
    ee.volumetric_end = 600.0
    ee.use_volumetric_shadows = False
    ee.volumetric_ray_depth = 8
    ee.use_overscan = True
    ee.overscan_size = 5.0
    ee.light_threshold = 0.02
    ee.clamp_surface_indirect = 12.0
    scene.render.use_motion_blur = False

    vs = scene.view_settings
    vs.view_transform = 'AgX'
    try:
        vs.look = 'AgX - Medium High Contrast'
    except Exception:
        try:
            vs.look = 'Medium High Contrast'
        except Exception as e:
            print("look:", e)
    vs.exposure = 0.0
    vs.gamma = 1.0
    scene.display_settings.display_device = 'sRGB'


def fallback_world(scene):
    """Neutral dusk sky + warm low sun, used when the sky module is not part of a test build."""
    w = bpy.data.worlds.new("FallbackWorld"); scene.world = w
    w.use_nodes = True
    nt = NT(w.node_tree, clear=True)
    out = nt.n('ShaderNodeOutputWorld')
    tc = nt.n('ShaderNodeTexCoord')
    z = nt.sep_xyz(tc.outputs['Generated'])[2]
    # warm glow toward the sun azimuth
    sd = L.sun_dir()
    d = nt.vmath('DOT_PRODUCT', nt.vmath('NORMALIZE', tc.outputs['Generated']), (sd[0], sd[1], sd[2]))
    sunglow = nt.map_range(d, 0.2, 1.0, 0.0, 1.0)
    base = nt.ramp(z, [(0.0, (0.30, 0.20, 0.20, 1)), (0.12, (0.42, 0.30, 0.30, 1)), (0.5, (0.16, 0.19, 0.32, 1)),
                       (1.0, (0.07, 0.09, 0.20, 1))])
    hz = nt.map_range(z, 0.0, 0.5, 1.0, 0.0)
    warm = nt.mix(nt.math('MULTIPLY', nt.math('POWER', sunglow, 3.0), hz), base, (2.6, 1.0, 0.30, 1), blend='ADD')
    bg = nt.n('ShaderNodeBackground', Color=warm, Strength=1.0)
    nt.link(bg.outputs[0], out.inputs['Surface'])
    sun = bpy.data.lights.new("FallbackSun", 'SUN')
    sun.energy = 2.2; sun.color = (1.0, 0.56, 0.30); sun.angle = math.radians(1.5)
    so = bpy.data.objects.new("FallbackSun", sun)
    get_col("lighting").objects.link(so)
    so.rotation_euler = (Vector(sd)).to_track_quat('Z', 'Y').to_euler()
    return so


def make_cameras(scene):
    col = get_col("cameras")
    cams = {}
    for name, (pos, tgt, lens) in L.CAMERAS.items():
        cd = bpy.data.cameras.new("cam_" + name)
        cd.lens = lens; cd.sensor_width = 36.0
        cd.clip_start = 0.3; cd.clip_end = 80000.0
        o = bpy.data.objects.new("cam_" + name, cd)
        col.objects.link(o)
        o.location = pos
        look_at(o, tgt)
        cams[name] = o
    scene.camera = cams["hero"]
    return cams


def custom_camera(scene, pos, tgt, lens=35.0, name="cam_custom"):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens; cd.clip_start = 0.3; cd.clip_end = 80000.0
    o = bpy.data.objects.new(name, cd)
    get_col("cameras").objects.link(o)
    o.location = pos
    look_at(o, tgt)
    return o
