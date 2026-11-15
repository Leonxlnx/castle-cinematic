"""Compositor finishing: bloom on lights/sun, soft halation, gentle chromatic aberration, vignette,
filmic colour balance and animated film grain.  All in scene-linear before the AgX view transform.

Tunables are the constants below.
"""
import bpy
from ..common import NT, drive_frame

BLOOM_STRENGTH = 0.26
BLOOM_THRESHOLD = 2.8
BLOOM_SIZE = 0.5
HALO_STRENGTH = 0.08          # second, tighter glow for small light sources
DISPERSION = 0.005
VIGNETTE = 0.42               # 0 = none
GRAIN = 0.035
SATURATION = 1.06


def _menu(sock, value):
    try:
        sock.default_value = value
        return True
    except Exception as e:
        print("[comp] menu %s <- %r failed: %s" % (sock.name, value, e))
        return False


def build(ctx):
    scene = ctx.scene
    g = bpy.data.node_groups.new("CastleComp", 'CompositorNodeTree')
    g.interface.new_socket("Image", in_out='OUTPUT', socket_type='NodeSocketColor')
    nt = NT(g)
    rl = nt.n('CompositorNodeRLayers')
    rl.scene = scene
    out = nt.n('NodeGroupOutput')
    img = rl.outputs['Image']

    # ---- bloom (wide) + halo (tight)
    bloom = nt.n('CompositorNodeGlare', Image=img)
    _menu(bloom.inputs['Type'], 'Bloom')
    _menu(bloom.inputs['Quality'], 'High')
    bloom.inputs['Threshold'].default_value = BLOOM_THRESHOLD
    bloom.inputs['Smoothness'].default_value = 0.6
    bloom.inputs['Strength'].default_value = BLOOM_STRENGTH
    bloom.inputs['Saturation'].default_value = 1.0
    bloom.inputs['Size'].default_value = BLOOM_SIZE
    try:
        bloom.inputs['Maximum'].default_value = 14.0
    except Exception:
        pass
    img = bloom.outputs['Image']

    halo = nt.n('CompositorNodeGlare', Image=img)
    _menu(halo.inputs['Type'], 'Fog Glow')
    _menu(halo.inputs['Quality'], 'High')
    halo.inputs['Threshold'].default_value = 2.5
    halo.inputs['Strength'].default_value = HALO_STRENGTH
    halo.inputs['Size'].default_value = 0.35
    img = halo.outputs['Image']

    # ---- gentle chromatic aberration toward the frame edges
    lens = nt.n('CompositorNodeLensdist', Image=img)
    lens.inputs['Distortion'].default_value = 0.0
    lens.inputs['Dispersion'].default_value = DISPERSION
    try:
        lens.inputs['Fit'].default_value = True
    except Exception:
        pass
    img = lens.outputs['Image']

    # ---- colour balance: slightly cool shadows, warm highlights, a touch of contrast
    cb = nt.n('CompositorNodeColorBalance', Image=img)
    _menu(cb.inputs['Type'], 'Lift/Gamma/Gain')
    try:
        cb.inputs[4].default_value = (0.985, 0.995, 1.03, 1.0)     # lift colour
        cb.inputs[6].default_value = (1.0, 0.995, 0.985, 1.0)      # gamma colour
        cb.inputs[8].default_value = (1.035, 1.0, 0.955, 1.0)      # gain colour
    except Exception as e:
        print("[comp] colour balance sockets:", e)
    img = cb.outputs['Image']

    sat = nt.n('CompositorNodeHueSat', Image=img)
    sat.inputs['Saturation'].default_value = SATURATION
    img = sat.outputs['Image']

    # ---- vignette
    if VIGNETTE > 0:
        ell = nt.n('CompositorNodeEllipseMask')
        try:
            ell.inputs['Size'].default_value = (0.86, 0.80)
        except Exception as e:
            print("[comp] ellipse size:", e)
        blur = nt.n('CompositorNodeBlur', Image=ell.outputs['Mask'])
        try:
            blur.inputs['Size'].default_value = (420.0, 420.0)
        except Exception:
            try:
                blur.inputs['Size'].default_value = 420.0
            except Exception as e:
                print("[comp] blur size:", e)
        vmix = nt.map_range(blur.outputs['Image'], 0.0, 1.0, 1.0 - VIGNETTE, 1.0)
        mul = nt.n('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
        mul.inputs[0].default_value = 1.0
        nt.link(img, mul.inputs[6]); nt.link(vmix, mul.inputs[7])
        img = mul.outputs[2]

    # ---- animated film grain (luminance noise, multiplicative so highlights/shadows both get texture)
    if GRAIN > 0:
        co = nt.n('CompositorNodeImageCoordinates', Image=rl.outputs['Image'])
        noise = nt.n('ShaderNodeTexNoise', noise_dimensions='4D', Vector=co.outputs['Pixel'], Scale=0.72, Detail=0.0)
        drive_frame(noise.inputs['W'], 'default_value', 'frame*7.31')
        gr = nt.map_range(noise.outputs['Factor'], 0.25, 0.75, 1.0 - GRAIN, 1.0 + GRAIN)
        mul2 = nt.n('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
        mul2.inputs[0].default_value = 1.0
        nt.link(img, mul2.inputs[6]); nt.link(gr, mul2.inputs[7])
        img = mul2.outputs[2]

    nt.link(img, out.inputs[0])
    scene.compositing_node_group = g
    scene.render.use_compositing = True
    try:
        scene.render.compositor_device = 'GPU'
        scene.render.compositor_precision = 'FULL'
    except Exception as e:
        print("[comp] device:", e)






