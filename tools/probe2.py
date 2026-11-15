import bpy
m = bpy.data.materials.new("x"); m.use_nodes = True
nt = m.node_tree
for t in ["ShaderNodeTexNoise","ShaderNodeTexBrick","ShaderNodeTexVoronoi","ShaderNodeTexWave","ShaderNodeTexGradient","ShaderNodeMixShader","ShaderNodeMix","ShaderNodeAttribute","ShaderNodeFresnel","ShaderNodeLayerWeight","ShaderNodeBsdfPrincipled","ShaderNodeBump","ShaderNodeValToRGB","ShaderNodeMapRange","ShaderNodeSeparateColor","ShaderNodeEmission","ShaderNodeBsdfGlossy","ShaderNodeBsdfTransparent","ShaderNodeVolumePrincipled","ShaderNodeTexSky","ShaderNodeNewGeometry","ShaderNodeTexCoord","ShaderNodeMapping","ShaderNodeLightPath","ShaderNodeObjectInfo","ShaderNodeCameraData","ShaderNodeTexWhiteNoise","ShaderNodeVolumeScatter","ShaderNodeBsdfTranslucent","ShaderNodeHueSaturation","ShaderNodeVectorMath","ShaderNodeDisplacement","ShaderNodeBsdfRefraction","ShaderNodeBackground"]:
    try:
        n = nt.nodes.new(t)
        print("NODE", t, "| IN:", [s.name for s in n.inputs], "| OUT:", [s.name for s in n.outputs])
    except Exception as e:
        print("NODE", t, "ERR", e)
n = nt.nodes.new("ShaderNodeTexNoise"); print("noise attrs", [p.identifier for p in n.bl_rna.properties if p.identifier not in bpy.types.Node.bl_rna.properties.keys()])
n = nt.nodes.new("ShaderNodeTexSky"); print("sky attrs", [p.identifier for p in n.bl_rna.properties if p.identifier not in bpy.types.Node.bl_rna.properties.keys()])
print("mat props", [p.identifier for p in m.bl_rna.properties if not p.is_readonly][:80])
l = bpy.data.lights.new("l", 'POINT'); print("light props", [p.identifier for p in l.bl_rna.properties if not p.is_readonly])
s = bpy.context.scene
print("rt opts", [p.identifier for p in s.eevee.ray_tracing_options.bl_rna.properties])
print("looks", [e.identifier for e in s.view_settings.bl_rna.properties['look'].enum_items][:40])
s.view_settings.view_transform = 'AgX'
print("vt ok", s.view_settings.view_transform)
for lk in ['AgX - Medium High Contrast','Medium High Contrast','AgX - Punchy','Punchy']:
    try:
        s.view_settings.look = lk; print("look ok", lk)
    except Exception as e: print("look fail", lk)
cg = bpy.data.node_groups.new("c", "CompositorNodeTree")
for t in ["CompositorNodeRLayers","CompositorNodeGlare","CompositorNodeLensdist","CompositorNodeEllipseMask","CompositorNodeBlur","CompositorNodeColorBalance","CompositorNodeCurveRGB","CompositorNodeMixRGB","ShaderNodeMix","CompositorNodeSunBeams","CompositorNodeMapValue","ShaderNodeMapRange","NodeGroupOutput","CompositorNodeViewer","CompositorNodeHueSat","CompositorNodeExposure","CompositorNodeTonemap","CompositorNodeBrightContrast","CompositorNodeColorCorrection","CompositorNodeFilter","CompositorNodeDefocus","CompositorNodeVecBlur","CompositorNodeKuwahara","CompositorNodeImageCoordinates","CompositorNodeDenoise","ShaderNodeTexNoise","ShaderNodeMath"]:
    try:
        n = cg.nodes.new(t)
        print("CNODE", t, "| IN:", [s.name for s in n.inputs], "| OUT:", [s.name for s in n.outputs])
    except Exception as e:
        print("CNODE", t, "ERR", str(e)[:60])
