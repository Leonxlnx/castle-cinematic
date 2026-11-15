"""Aerial perspective for EEVEE, done inside materials (EEVEE's volumetrics cannot reach km distances well).

    from scene import atmo
    shader = atmo.fog(nt, bsdf.outputs[0])          # returns a shader socket: surface faded into haze
    nt.link(shader, out.inputs['Surface'])

The haze is an exponential height fog (dense in the valley, thin on the peaks) whose colour shifts from a
cool purple-grey to glowing orange when looking toward the sun.  Use it on EVERYTHING that is more than
~150 m from the castle: terrain, lake, forests, mountains, mist cards.  Do NOT use it on the castle/terrace.
"""
import math
from . import layout as L

# fog parameters (shared by all modules so distant layers line up)
DENSITY = 1.0 / 5200.0      # extinction per metre at the reference altitude (valley floor)
SCALE_HEIGHT = 520.0        # metres; fog density halves every ~360 m of altitude
MAX_FOG = 0.93              # never fade completely (keeps mountain silhouettes readable)


def fog_factor(nt, density_mul=1.0):
    """Returns (factor socket 0..1, sun_amount socket 0..1).  factor = how much haze covers the surface."""
    geo = nt.n('ShaderNodeNewGeometry')
    cam = nt.n('ShaderNodeCameraData')
    dist = cam.outputs['View Distance']
    pz = nt.sep_xyz(geo.outputs['Position'])[2]
    # camera altitude = P.z + Incoming.z * dist   (Incoming points from the surface toward the camera)
    iz = nt.sep_xyz(geo.outputs['Incoming'])[2]
    cz = nt.math('ADD', pz, nt.math('MULTIPLY', iz, dist))
    # analytic integral of exp(-(z - z0)/H) along the segment camera -> point
    H = SCALE_HEIGHT
    a = nt.math('EXPONENT', nt.math('DIVIDE', nt.math('SUBTRACT', L.VALLEY_Z, cz), H))     # density at camera
    b = nt.math('EXPONENT', nt.math('DIVIDE', nt.math('SUBTRACT', L.VALLEY_Z, pz), H))     # density at point
    dz = nt.math('SUBTRACT', pz, cz)
    adz = nt.math('MAXIMUM', nt.math('ABSOLUTE', dz), 1.0)
    # mean density = (a - b) * H / dz   (-> a when dz -> 0); use a stable form
    mean = nt.math('DIVIDE', nt.math('MULTIPLY', nt.math('ABSOLUTE', nt.math('SUBTRACT', a, b)), H), adz)
    near = nt.math('MULTIPLY', nt.math('ADD', a, b), 0.5)
    use_near = nt.math('LESS_THAN', nt.math('ABSOLUTE', dz), 2.0)
    mean = nt.mixf(use_near, mean, near)
    od = nt.math('MULTIPLY', nt.math('MULTIPLY', mean, dist), DENSITY * density_mul)
    fac = nt.math('SUBTRACT', 1.0, nt.math('EXPONENT', nt.math('MULTIPLY', od, -1.0)))
    fac = nt.math('MINIMUM', fac, MAX_FOG)
    # how much we look toward the sun (view dir = -Incoming)
    sd = L.sun_dir()
    d = nt.vmath('DOT_PRODUCT', geo.outputs['Incoming'], (-sd[0], -sd[1], -sd[2]))
    sun_amt = nt.math('POWER', nt.map_range(d, -0.2, 1.0, 0.0, 1.0), 2.2)
    return fac, sun_amt


def haze_color(nt, sun_amt):
    c = nt.mix(sun_amt, (L.HAZE_AWAY[0], L.HAZE_AWAY[1], L.HAZE_AWAY[2], 1.0),
               (L.HAZE_TOWARD_SUN[0], L.HAZE_TOWARD_SUN[1], L.HAZE_TOWARD_SUN[2], 1.0))
    return c


def fog(nt, shader_socket, density_mul=1.0):
    """Fade a surface shader into the haze.  Returns the new shader socket."""
    fac, sun_amt = fog_factor(nt, density_mul)
    em = nt.n('ShaderNodeEmission', Color=haze_color(nt, sun_amt), Strength=L.HAZE_STRENGTH)
    mix = nt.n('ShaderNodeMixShader', i0=fac, i1=shader_socket, i2=em.outputs[0])
    return mix.outputs[0]
