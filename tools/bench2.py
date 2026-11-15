"""Saner engine benchmark: moderately heavy static scene, one sun with shadows, 60 shadowless point lights,
glossy floor, optional fog.  Prints + flushes a timing line per configuration."""
import bpy, sys, time, math, random
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
mode = argv[0] if argv else "EEVEE"
out = argv[1] if len(argv) > 1 else "//bench.png"

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
random.seed(1)

def P(*a):
    print(*a); sys.stdout.flush()

# ---- one big mesh with ~2.5M triangles (bumpy grid) + 400 boxes
n = 1100
xs = np.linspace(-300, 300, n); ys = np.linspace(-50, 550, n)
X, Y = np.meshgrid(xs, ys)
Z = 6 * np.sin(X * 0.05) * np.cos(Y * 0.04) + 2.0 * np.sin(X * 0.4 + Y * 0.3) + 0.03 * (Y - 100).clip(0) ** 1.3 - 4
verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
idx = np.arange(n * n).reshape(n, n)
quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], axis=-1).reshape(-1, 4)
me = bpy.data.meshes.new("terrain")
me.vertices.add(len(verts)); me.vertices.foreach_set("co", verts.ravel())
me.loops.add(quads.size); me.loops.foreach_set("vertex_index", quads.ravel())
me.polygons.add(len(quads)); me.polygons.foreach_set("loop_start", np.arange(0, quads.size, 4)); me.polygons.foreach_set("loop_total", np.full(len(quads), 4))
me.update()
terr = bpy.data.objects.new("terrain", me); sc.collection.objects.link(terr)

def mat(name, col, rough, noise_scale=None, emit=None):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = col; b.inputs["Roughness"].default_value = rough
    if noise_scale:
        nt = m.node_tree
        nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = noise_scale; nz.inputs["Detail"].default_value = 6
        br = nt.nodes.new("ShaderNodeTexBrick")
        bp = nt.nodes.new("ShaderNodeBump")
        nt.links.new(nz.outputs["Fac"], bp.inputs["Height"]); nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
        nt.links.new(br.outputs["Color"], b.inputs["Base Color"])
    if emit:
        b.inputs["Emission Color"].default_value = emit; b.inputs["Emission Strength"].default_value = 8
    return m

terr.data.materials.append(mat("ground", (0.1, 0.14, 0.06, 1), 0.9, 3.0))
stone = mat("stone", (0.4, 0.36, 0.3, 1), 0.8, 8.0)
glow = mat("glow", (0, 0, 0, 1), 0.1, None, (1, 0.55, 0.2, 1))
bpy.ops.mesh.primitive_plane_add(size=120, location=(0, 20, 0.5))
bpy.context.object.data.materials.append(mat("wet", (0.3, 0.28, 0.25, 1), 0.06))
for i in range(400):
    x = random.uniform(-60, 60); y = random.uniform(40, 140); h = random.uniform(5, 45)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, h / 2)); o = bpy.context.object
    o.scale = (random.uniform(2, 8), random.uniform(2, 8), h)
    o.data.materials.append(glow if i % 9 == 0 else stone)
    bv = o.modifiers.new("b", "BEVEL"); bv.width = 0.1
for i in range(60):
    ld = bpy.data.lights.new("p%d" % i, 'POINT'); ld.energy = 2500; ld.color = (1, 0.6, 0.3); ld.use_shadow = False
    lo = bpy.data.objects.new("p%d" % i, ld); sc.collection.objects.link(lo)
    lo.location = (random.uniform(-55, 55), random.uniform(-20, 100), random.uniform(1.5, 5))
sd = bpy.data.lights.new("sun", 'SUN'); sd.energy = 3; sd.color = (1, 0.6, 0.35)
so = bpy.data.objects.new("sun", sd); sc.collection.objects.link(so); so.rotation_euler = (math.radians(86), 0, math.radians(70))

w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.2, 0.22, 0.4, 1)
w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6

bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 60, 15)); fog = bpy.context.object; fog.scale = (260, 260, 50)
vm = bpy.data.materials.new("fog"); vm.use_nodes = True
vn = vm.node_tree; vn.nodes.clear()
vo = vn.nodes.new("ShaderNodeOutputMaterial"); pv = vn.nodes.new("ShaderNodeVolumePrincipled")
pv.inputs["Density"].default_value = 0.0015
vn.links.new(pv.outputs[0], vo.inputs["Volume"]); fog.data.materials.append(vm)
fog.hide_render = True

cd = bpy.data.cameras.new("c"); cd.lens = 28; cd.clip_end = 5000
co = bpy.data.objects.new("c", cd); sc.collection.objects.link(co); co.location = (30, -45, 9); co.rotation_euler = (math.radians(82), 0, math.radians(20))
sc.camera = co
sc.render.resolution_x = 1920; sc.render.resolution_y = 1080
sc.render.filepath = out; sc.render.image_settings.file_format = 'PNG'
sc.view_settings.view_transform = 'AgX'

def run(tag):
    t = time.time()
    bpy.ops.render.render(write_still=True)
    P("BENCH_RESULT", mode, tag, "seconds=%.1f" % (time.time() - t))

if mode == "EEVEE":
    sc.render.engine = 'BLENDER_EEVEE'
    ee = sc.eevee
    ee.taa_render_samples = 16; ee.use_raytracing = False; ee.use_shadows = True
    run("A_warmup_16s_noRT")
    run("B_16s_noRT")
    ee.use_raytracing = True
    ee.ray_tracing_options.resolution_scale = '2'
    run("C_16s_RT_half")
    ee.taa_render_samples = 32
    run("D_32s_RT_half")
    ee.use_fast_gi = True
    run("E_32s_RT_fastGI")
    fog.hide_render = False
    ee.volumetric_tile_size = '8'; ee.use_volumetric_shadows = False; ee.use_volume_custom_range = True
    ee.volumetric_start = 1; ee.volumetric_end = 500
    run("F_32s_RT_volume_tile8")
    ee.use_volumetric_shadows = True
    run("G_32s_RT_volume_shadows")
    fog.hide_render = True
    ee.taa_render_samples = 64
    run("H_64s_RT")
    sc.render.resolution_x = 2560; sc.render.resolution_y = 1440; ee.taa_render_samples = 32
    run("I_1440p_32s_RT")
else:
    sc.render.engine = 'CYCLES'
    cp = bpy.context.preferences.addons['cycles'].preferences
    if mode == "CYCLES_HIP":
        cp.compute_device_type = 'HIP'; cp.get_devices()
        for d in cp.devices:
            d.use = (d.type == 'HIP')
        sc.cycles.device = 'GPU'
    else:
        sc.cycles.device = 'CPU'
        sc.render.threads_mode = 'FIXED'; sc.render.threads = 12
    sc.cycles.samples = 48
    sc.cycles.use_adaptive_sampling = False
    sc.cycles.use_denoising = True
    run("48spp_1080p")
    run("48spp_1080p_again")
