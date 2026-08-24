import math
import array
import numpy as np

try:
    import bpy
    import gpu
    import gpu.state
    import gpu.matrix
    import gpu.shader
    from gpu.types import (
        GPUShaderCreateInfo,
        GPUStageInterfaceInfo,
        GPUVertFormat,
        GPUVertBuf,
        GPUIndexBuf,
        GPUBatch,
    )
    import gpu_extras
    import mathutils
    from bpy_extras import view3d_utils
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False

# Import utilities
from .utils import (
    load_mbs_from_npz,
    save_mbs_to_npz,
    get_armature_object
)

# Global variables for caching and drawing
_cached_mbs = []
_draw_handler = None
_scene_mbs = []
_scene_mbs_dirty = True

_last_active_object_name = ""
_last_active_bone = ""
_last_weights_hash = 0

def get_weights_hash(obj):
    if not obj:
        return 0
    exclude_keys = {"weights", "normal", "family_id", "radius", "alpha", "n", "q", "tau", "R_falloff", "is_virtual", "symmetry_class"}
    weights_data = []
    for key in obj.keys():
        if key in exclude_keys or key.startswith("_"):
            continue
        val = obj[key]
        if isinstance(val, (int, float)):
            weights_data.append((key, float(val)))
    return hash(tuple(sorted(weights_data)))


def is_mwc_metaball(obj, col=None):
    """Robust MWC-metaball check: membership in the MWC_Metaballs collection
    (or the collection-provided object itself) instead of the fragile
    name.startswith('MB_') heuristic that broke whenever a user renamed an
    object. Falls back to the marker custom property for unlinked objects."""
    if not obj:
        return False
    if col is None:
        col = bpy.data.collections.get("MWC_Metaballs")
    if col is not None and obj.name in col.objects:
        return True
    # Objects removed from the collection but still carrying our metadata.
    return obj.type == 'META' and obj.get("family_id") is not None

def get_weight_color(weight):
    """
    Returns (R, G, B) color using Weight Paint rainbow gradient:
    Blue (0.0) -> Cyan (0.25) -> Green (0.5) -> Yellow (0.75) -> Red (1.0)
    """
    if weight <= 0.0:
        return (0.0, 0.0, 1.0)
    elif weight >= 1.0:
        return (1.0, 0.0, 0.0)
        
    if weight < 0.25:
        t = weight / 0.25
        return (0.0, t, 1.0)
    elif weight < 0.5:
        t = (weight - 0.25) / 0.25
        return (0.0, 1.0, 1.0 - t)
    elif weight < 0.75:
        t = (weight - 0.5) / 0.25
        return (t, 1.0, 0.0)
    else:
        t = (weight - 0.75) / 0.25
        return (1.0, 1.0 - t, 0.0)

def _build_sphere_lod(segments, rings):
    """Build a UV-sphere batch pair (solid + wire) at the given tessellation."""
    verts = []
    solid_indices = []
    wire_indices = []

    for r in range(rings + 1):
        theta = r * math.pi / rings
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        for s in range(segments):
            phi = s * 2 * math.pi / segments
            x = sin_theta * math.cos(phi)
            y = sin_theta * math.sin(phi)
            z = cos_theta
            verts.append((x, y, z))

    for r in range(rings):
        for s in range(segments):
            i0 = r * segments + s
            i1 = r * segments + (s + 1) % segments
            i2 = (r + 1) * segments + s
            i3 = (r + 1) * segments + (s + 1) % segments

            solid_indices.append((i0, i3, i1))
            solid_indices.append((i0, i2, i3))

            wire_indices.append((i0, i1))
            wire_indices.append((i0, i2))

    for s in range(segments):
        i0 = rings * segments + s
        i1 = rings * segments + (s + 1) % segments
        wire_indices.append((i0, i1))

    fmt = gpu.types.GPUVertFormat()
    fmt.attr_add(id="pos", comp_type="F32", len=3, fetch_mode="FLOAT")

    vbo = gpu.types.GPUVertBuf(fmt, len=len(verts))
    vbo.attr_fill("pos", verts)

    ibo_solid = gpu.types.GPUIndexBuf(type='TRIS', seq=solid_indices)
    ibo_wire = gpu.types.GPUIndexBuf(type='LINES', seq=wire_indices)

    solid = gpu.types.GPUBatch(type='TRIS', buf=vbo, elem=ibo_solid)
    wire = gpu.types.GPUBatch(type='LINES', buf=vbo, elem=ibo_wire)
    return solid, wire


# LOD tiers: (min projected NDC radius, segments, rings). The first entry is
# always used as fallback for anything larger than the last threshold.
_LOD_TIERS = (
    (0.02, 8, 4),    # tiny on screen
    (0.08, 12, 6),   # small
    (float('inf'), 16, 8),  # normal / large
)
_sphere_lods = None  # list of (solid_batch, wire_batch), built lazily

# ---------------------------------------------------------------------------
# GPU instancing: one draw call for ALL metaball spheres of a given LOD.
#
# The legacy path pushed/popped the fixed-function matrix stack and issued a
# separate batch.draw() per metaball -- with 10k+ cached metaballs that alone
# dominated the frame time. The instanced shader instead takes per-instance
# attributes (world position, radius, RGBA color) plus a shared unit-sphere
# vertex buffer, so N spheres cost exactly one draw call.
# ---------------------------------------------------------------------------
_inst_shader_info = None
_inst_shader_cache = {"shader": None, "tried": False}

_INSTANCE_ATTRS = (
    ("i_pos", "F32", 3),
    ("i_radius", "F32", 1),
    ("i_color", "F32", 4),
)

_SPHERE_ATTRS = (
    ("position", "F32", 3),
    ("normal", "F32", 3),
)

_INST_SHADER_SRC = {
    "vertex_source": """
        void main()
        {
            vec4 world_pos = vec4(i_pos + position * i_radius, 1.0);
            gl_Position = ModelViewProjectionMatrix * world_pos;
            /* Approximate view-space normal: the instance axis-aligned sphere
             * only needs a cheap directional shade, not an exact transform. */
            v_view_normal = normalize(mat3(ModelViewProjectionMatrix) * i_normal);
            v_color = i_color;
        }
    """,
    "fragment_source": """
        void main()
        {
            /* Headlight + rim: bright facing the camera, darker at grazing
             * angles -- keeps overlapping spheres readable without lights. */
            float lambert = clamp(v_view_normal.z, 0.0, 1.0);
            float shade = 0.55 + 0.45 * lambert;
            fragColor = vec4(v_color.rgb * shade, v_color.a);
        }
    """,
}


def _get_instance_shader():
    """Build (once) and return the instanced sphere shader, or None if the
    running Blender does not support GPUShaderCreateInfo-based instancing."""
    if not HAS_BLENDER:
        return None
    if _inst_shader_cache["tried"]:
        return _inst_shader_cache["shader"]
    _inst_shader_cache["tried"] = True
    try:
        info = GPUShaderCreateInfo()
        info.vertex_in(0, "VEC3", "position")
        info.vertex_in(1, "VEC3", "i_normal")
        info.vertex_in(2, "VEC3", "i_pos")
        info.vertex_in(3, "FLOAT", "i_radius")
        info.vertex_in(4, "VEC4", "i_color")
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        # NOTE: this Blender build auto-generates declarations for every
        # vertex_in/push_constant/fragment_out, so the GLSL sources must NOT
        # redeclare them (compile error: 'redeclared'). Varyings are declared
        # through a GPUStageInterfaceInfo (info.vertex_out takes an interface
        # object, not slot/type/name).
        iface = GPUStageInterfaceInfo("mwc_iface")
        iface.smooth("VEC4", "v_color")
        iface.smooth("VEC3", "v_view_normal")
        info.vertex_out(iface)
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_INST_SHADER_SRC["vertex_source"])
        info.fragment_source(_INST_SHADER_SRC["fragment_source"])
        shader = gpu.shader.create_from_info(info)
        _inst_shader_cache["shader"] = shader
    except Exception as e:
        print("MWC instanced preview shader unavailable, using legacy path:", e)
        _inst_shader_cache["shader"] = None
    return _inst_shader_cache["shader"]


def _build_sphere_lod_instanced(segments, rings):
    """Unit-sphere buffers for the instanced path: positions + per-vertex
    normals, indexed triangles. Returns (vbo, ibo, flat_indices) -- a fresh
    GPUBatch is assembled per frame from the expanded vertex stream."""
    verts = []
    indices = []

    for r in range(rings + 1):
        theta = r * math.pi / rings
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        for s in range(segments):
            phi = s * 2 * math.pi / segments
            x = sin_theta * math.cos(phi)
            y = sin_theta * math.sin(phi)
            z = cos_theta
            verts.append((x, y, z))

    for r in range(rings):
        for s in range(segments):
            i0 = r * segments + s
            i1 = r * segments + (s + 1) % segments
            i2 = (r + 1) * segments + s
            i3 = (r + 1) * segments + (s + 1) % segments
            # GPUIndexBuf('TRIS') expects a sequence of index TRIPLETS,
            # not a flat int list ('expected a sequence, got int').
            indices.append((i0, i3, i1))
            indices.append((i0, i2, i3))

    fmt = GPUVertFormat()
    fmt.attr_add(id="position", comp_type="F32", len=3, fetch_mode="FLOAT")
    fmt.attr_add(id="normal", comp_type="F32", len=3, fetch_mode="FLOAT")

    total = len(verts)
    pos_arr = np.array(verts, dtype=np.float32)
    vbo = GPUVertBuf(fmt, len=total)
    vbo.attr_fill(0, pos_arr)
    vbo.attr_fill(1, pos_arr)  # unit sphere: normal == position
    flat_indices = [i for tri in indices for i in tri]
    ibo = GPUIndexBuf(type='TRIS', seq=indices)
    return vbo, ibo, flat_indices


def get_sphere_lods():
    global _sphere_lods
    if not HAS_BLENDER:
        return None
    if _sphere_lods is None:
        _sphere_lods = [_build_sphere_lod(seg, rng) for _, seg, rng in _LOD_TIERS]
    return _sphere_lods


def get_sphere_batches():
    """Backwards-compatible accessor returning the highest-quality LOD."""
    lods = get_sphere_lods()
    if not lods:
        return None, None
    return lods[-1]


# Instanced unit-sphere geometry (one (vbo, ibo, flat_indices) tuple per LOD
# tier), built lazily alongside the legacy batches. A throwaway GPUBatch over
# the expanded vertex stream is assembled at draw time.
_sphere_lods_instanced = None
_sphere_pos_cache = []


def _get_instanced_lods():
    global _sphere_lods_instanced
    if not HAS_BLENDER:
        return None
    if _sphere_lods_instanced is None:
        try:
            _sphere_lods_instanced = [
                _build_sphere_lod_instanced(seg, rng) for _, seg, rng in _LOD_TIERS
            ]
            # Flat float32 copy of each LOD's unit-sphere positions, reused for
            # the expanded per-frame vertex stream (see _draw_instances_gpu).
            global _sphere_pos_cache
            _sphere_pos_cache = []
            for (vbo, _ibo, _flat), (_, seg, rng) in zip(_sphere_lods_instanced, _LOD_TIERS):
                verts = []
                for r in range(rng + 1):
                    theta = r * math.pi / rng
                    sin_theta = math.sin(theta)
                    cos_theta = math.cos(theta)
                    for s in range(seg):
                        phi = s * 2 * math.pi / seg
                        verts.append((sin_theta * math.cos(phi),
                                      sin_theta * math.sin(phi),
                                      cos_theta))
                _sphere_pos_cache.append(np.array(verts, dtype=np.float32))
        except Exception as e:
            print("MWC instanced sphere build failed:", e)
            _sphere_lods_instanced = []
    return _sphere_lods_instanced or None


# Expanded index buffer cache: {(ibo identity, n_instances): GPUIndexBuf}.
# Each base triangle's vertex indices are repeated per instance so they address
# the interleaved (sphere verts x instances) vertex stream. Rebuilt only when
# the instance count of a tier changes.
_ibo_expand_cache = {}


def sphere_ibo_expanded(base_indices, n, sphere_count):
    """Return a GPUIndexBuf addressing the (sphere verts x instances) expanded
    vertex stream: every base vertex index is repeated n times, once per
    instance row. Cached by (base index list identity, instance count)."""
    key = (id(base_indices), n)
    cached = _ibo_expand_cache.get(key)
    if cached is not None:
        return cached
    flat = np.repeat(np.asarray(base_indices, dtype=np.uint32), n)
    triplets = flat.reshape(-1, 3)
    new_ibo = GPUIndexBuf(type='TRIS', seq=triplets)
    _ibo_expand_cache[key] = new_ibo
    return new_ibo


def resolve_mb_transform(mb, mbs, arm_obj, depsgraph):
    """
    Resolve the current world-space (co, radius) for a metaball entry, following
    either a live spawned scene object (mbs is _scene_mbs) or the posed armature
    bone it is cached against (mbs is _cached_mbs). Falls back to the metaball's
    stored static co/radius if neither applies.
    """
    co = mb['co']
    radius = mb['radius']

    if mbs is _scene_mbs:
        obj = bpy.data.objects.get(mb.get('name', ''))
        if obj:
            if depsgraph:
                try:
                    eval_obj = obj.evaluated_get(depsgraph)
                    co_world = eval_obj.matrix_world.to_translation()
                except Exception:
                    co_world = obj.matrix_world.to_translation()
            else:
                co_world = obj.matrix_world.to_translation()
            co = (co_world.x, co_world.y, co_world.z)
            if obj.data and hasattr(obj.data, 'elements') and len(obj.data.elements) > 0:
                radius = obj.data.elements[0].radius * obj.scale.x
    elif mbs is _cached_mbs and arm_obj and arm_obj.type == 'ARMATURE':
        p_bone = mb.get('parent_bone', '')
        co_loc = mb.get('co_local')
        if depsgraph:
            try:
                eval_arm = arm_obj.evaluated_get(depsgraph)
            except Exception:
                eval_arm = arm_obj
        else:
            eval_arm = arm_obj

        if p_bone and co_loc and p_bone in eval_arm.pose.bones:
            pose_bone = eval_arm.pose.bones[p_bone]
            local_vec = mathutils.Vector(co_loc)
            co_world = eval_arm.matrix_world @ (pose_bone.matrix @ local_vec)
            co = (co_world.x, co_world.y, co_world.z)

    return co, radius


def _draw_instances_gpu(shader_inst, inst_lods, instances, wire_instances):
    """One draw call per LOD tier via per-instance attributes."""
    # Group instance attribute rows by LOD tier.
    by_lod = {}
    for co, radius, color, lod_idx in instances:
        by_lod.setdefault(lod_idx, []).append((co, radius, color))

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)
    gpu.state.face_culling_set('BACK')

    shader_inst.bind()
    loc_mvp = shader_inst.uniform_from_name("ModelViewProjectionMatrix")
    if loc_mvp != -1:
        mvp = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()
        flat = array.array('f', [v for row in mvp.transposed() for v in row])
        shader_inst.uniform_vector_float(loc_mvp, flat, 16, 1)

    for lod_idx, items in by_lod.items():
        n = len(items)
        pos = np.empty((n, 3), dtype=np.float32)
        rad = np.empty(n, dtype=np.float32)
        col = np.empty((n, 4), dtype=np.float32)
        for i, (co, radius, color) in enumerate(items):
            pos[i, 0] = co[0]; pos[i, 1] = co[1]; pos[i, 2] = co[2]
            rad[i] = radius
            col[i, 0] = color[0]; col[i, 1] = color[1]
            col[i, 2] = color[2]; col[i, 3] = color[3]

        sphere_vbo, sphere_ibo, base_indices = inst_lods[lod_idx]
        # This Blender build has no GPUBatch.inst(), and vertbuf_add() only
        # appends attributes to buffers of the SAME vertex count -- it cannot
        # carry instance rows. So the per-instance data is expanded into the
        # vertex stream instead: every sphere vertex is repeated once per
        # instance with that instance's i_pos/i_radius/i_color. The GPU cost is
        # still one indexed draw call per LOD tier; only the upload grows.
        sphere_count = _sphere_pos_cache[lod_idx].shape[0]
        total = sphere_count * n

        fmt_all = GPUVertFormat()
        for name, comp, ln in _SPHERE_ATTRS + _INSTANCE_ATTRS:
            fmt_all.attr_add(id=name, comp_type=comp, len=ln, fetch_mode="FLOAT")

        vbo_all = GPUVertBuf(fmt_all, len=total)
        vbo_all.attr_fill(0, _sphere_pos_cache[lod_idx])
        vbo_all.attr_fill(1, _sphere_pos_cache[lod_idx])
        vbo_all.attr_fill(2, np.repeat(pos, sphere_count, axis=0))
        vbo_all.attr_fill(3, np.repeat(rad, sphere_count, axis=0))
        vbo_all.attr_fill(4, np.repeat(col, sphere_count, axis=0))

        expanded_ibo = sphere_ibo_expanded(base_indices, n, sphere_count)
        batch = GPUBatch(type='TRIS', buf=vbo_all, elem=expanded_ibo)
        batch.program_set(shader_inst)
        try:
            batch.draw(shader_inst)
        finally:
            batch.program_set(shader_inst)

    gpu.state.face_culling_set('NONE')

    # Wire overlay for selected metaballs: tiny N (usually 0 or 1), so the
    # legacy POLYLINE path is cheaper than building indexed line batches.
    if wire_instances:
        shader_wire = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(False)
        gpu.state.line_width_set(3.0)
        shader_wire.bind()
        loc_mvp = shader_wire.uniform_from_name("ModelViewProjectionMatrix")
        if loc_mvp != -1:
            mvp = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()
            flat = array.array('f', [v for row in mvp.transposed() for v in row])
            shader_wire.uniform_vector_float(loc_mvp, flat, 16, 1)
        shader_wire.uniform_float("color", (1.0, 0.8, 0.0, 1.0))
        solid_batch, wire_batch = get_sphere_lods()[-1]
        for co, radius in wire_instances:
            gpu.matrix.push()
            gpu.matrix.translate(co)
            gpu.matrix.scale_uniform(radius * 1.02)
            wire_batch.draw(shader_wire)
            gpu.matrix.pop()

    return True


def _draw_legacy(lods, instances, wire_instances):
    """Original per-instance push/draw path (fallback)."""
    shader_solid = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader_wire = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)
    gpu.state.face_culling_set('BACK')

    shader_solid.bind()
    loc_mvp_solid = shader_solid.uniform_from_name("ModelViewProjectionMatrix")
    has_mvp_solid = loc_mvp_solid != -1

    if has_mvp_solid:
        _mvp_frame = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()
        _flat_mvp = array.array('f', [v for row in _mvp_frame.transposed() for v in row])
        shader_solid.uniform_vector_float(loc_mvp_solid, _flat_mvp, 16, 1)

    for co, radius, color, lod_idx in instances:
        shader_solid.uniform_float("color", color)

        gpu.matrix.push()
        gpu.matrix.translate(co)
        gpu.matrix.scale_uniform(radius)

        lods[lod_idx][0].draw(shader_solid)
        gpu.matrix.pop()

    # Wire pass (selected metaballs only)
    gpu.state.face_culling_set('NONE')
    shader_wire.bind()
    loc_mvp_wire = shader_wire.uniform_from_name("ModelViewProjectionMatrix")
    has_mvp_wire = loc_mvp_wire != -1
    gpu.state.line_width_set(3.0)
    shader_wire.uniform_float("color", (1.0, 0.8, 0.0, 1.0))
    if has_mvp_wire:
        _mvp_frame = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()
        _flat_mvp = array.array('f', [v for row in _mvp_frame.transposed() for v in row])
        shader_wire.uniform_vector_float(loc_mvp_wire, _flat_mvp, 16, 1)

    for co, radius in wire_instances:
        gpu.matrix.push()
        gpu.matrix.translate(co)
        gpu.matrix.scale_uniform(radius * 1.02)

        lods[-1][1].draw(shader_wire)
        gpu.matrix.pop()


def draw_callback_px():
    if not HAS_BLENDER:
        return
    context = bpy.context
    scene = context.scene
    if not getattr(scene, "mwc_show_viewport_preview", False):
        return
        
    global _scene_mbs, _scene_mbs_dirty, _last_active_object_name, _last_active_bone, _last_weights_hash
    col = bpy.data.collections.get("MWC_Metaballs")
    
    color_by_bone = getattr(scene, "mwc_color_by_active_bone", False)
    active_bone = ""
    if color_by_bone:
        from .__init__ import get_active_bone_name
        active_bone = get_active_bone_name(context)
        
    if col and col.objects:
        active_obj = context.active_object
        current_active_name = active_obj.name if active_obj else ""
        current_weights_hash = get_weights_hash(active_obj) if (active_obj and active_obj.type == 'META') else 0
        
        # Lazy check if anything has changed
        if (len(col.objects) != len(_scene_mbs) or 
            current_active_name != _last_active_object_name or
            active_bone != _last_active_bone or
            current_weights_hash != _last_weights_hash or
            _scene_mbs_dirty):
            
            _scene_mbs = extract_mbs_from_collection()
            _scene_mbs_dirty = False
            _last_active_object_name = current_active_name
            _last_active_bone = active_bone
            _last_weights_hash = current_weights_hash
            
        mbs = _scene_mbs
    else:
        global _cached_mbs
        mbs = _cached_mbs
        
    if not mbs:
        return
        
    arm_obj = get_armature_object(scene, context)
    
    lods = get_sphere_lods()
    if not lods:
        return
        
    orig_depth_mask = gpu.state.depth_mask_get()
    orig_depth_test = gpu.state.depth_test_get()
    orig_blend = gpu.state.blend_get()
    orig_line_width = gpu.state.line_width_get()
    
    selected_idx = getattr(scene, "mwc_selected_mb_idx", -1)
    active_obj = context.active_object
    active_mb_name = active_obj.name if (active_obj and active_obj.type == 'META' and is_mwc_metaball(active_obj, col)) else ""
    
    try:
        depsgraph = context.evaluated_depsgraph_get()
    except Exception:
        depsgraph = None

    # --- Pre-resolve per-metaball state ONCE for both passes ---
    # (transform, color, selection) used to be recomputed per pass, meaning the
    # whole metaball list was walked twice with matrix math and dict lookups in
    # each. Building one flat list of instance data here halves that work.
    _selected_colors = set()  # identity tracker filled during the instance pass
    
    instances = []      # (co, radius, color) for every visible metaball
    wire_instances = [] # (co, radius) for selected metaballs only
    for idx, mb in enumerate(mbs):
        is_selected = (mb.get('name', '') == active_mb_name) if active_mb_name else (idx == selected_idx)
        
        if color_by_bone and active_bone:
            w = mb['weights'].get(active_bone, 0.0)
            if w <= 0.0 and not is_selected:
                continue
            r, g, b = get_weight_color(w)
            color = (r, g, b, 1.0 if is_selected else 0.45)
        else:
            color = (0.2, 0.6, 1.0, 0.6 if is_selected else 0.35)
            
        co, radius = resolve_mb_transform(mb, mbs, arm_obj, depsgraph)
        
        if is_selected and mbs is _cached_mbs:
            # Allow editing override coords
            try:
                co = (scene.mwc_selected_mb_x, scene.mwc_selected_mb_y, scene.mwc_selected_mb_z)
                radius = scene.mwc_selected_mb_radius
            except AttributeError:
                pass
                
        inst = (co, radius, color)
        instances.append(inst)
        if is_selected:
            wire_instances.append((co, radius))
            _selected_colors.add(color)
        
    if not instances:
        return
    
    # --- View-frustum culling + back-to-front depth sorting ---
    # Project every instance center once with the current view-projection
    # matrix: instances entirely outside the frustum are skipped (they cannot
    # contribute a pixel), and the rest are drawn far-to-near so alpha blending
    # resolves correctly regardless of draw order.
    try:
        proj_m = gpu.matrix.get_projection_matrix()
        view_m = gpu.matrix.get_model_view_matrix()
        vp = proj_m @ view_m
    except Exception:
        vp = None

    if vp is not None:
        vp_rows = [list(row) for row in vp]
        m00, m01, m02, m03 = vp_rows[0]
        m10, m11, m12, m13 = vp_rows[1]
        m20, m21, m22, m23 = vp_rows[2]
        m30, m31, m32, m33 = vp_rows[3]

        culled_instances = []
        tier_bounds = [t[0] for t in _LOD_TIERS]
        for co, radius, color in instances:
            x, y, z = co[0], co[1], co[2]
            cw = m30 * x + m31 * y + m32 * z + m33
            if cw <= 1e-9:
                continue  # behind the camera
            cx = m00 * x + m01 * y + m02 * z + m03
            cy = m10 * x + m11 * y + m12 * z + m13
            cz = m20 * x + m21 * y + m22 * z + m23
            # NDC extent of the sphere bounding square; skip if fully outside.
            ndc_x, ndc_y, ndc_z = cx / cw, cy / cw, cz / cw
            r_ndc = radius * 1.5 / cw  # conservative bound (>= projected radius)
            if (ndc_x + r_ndc < -1.0 or ndc_x - r_ndc > 1.0 or
                    ndc_y + r_ndc < -1.0 or ndc_y - r_ndc > 1.0 or
                    ndc_z - r_ndc > 1.0):
                continue
            # Screen-size based LOD: pick the coarsest sphere tessellation
            # whose threshold still covers this projected radius.
            lod_idx = len(tier_bounds) - 1
            for li, bound in enumerate(tier_bounds):
                if r_ndc < bound:
                    lod_idx = li
                    break
            culled_instances.append((co, radius, color, cz, lod_idx))

        if not culled_instances:
            return
        # Painter's algorithm: far first, near last.
        culled_instances.sort(key=lambda item: item[3], reverse=True)
        # Track which surviving entries are selected (by identity of the tuple
        # objects, which is stable through the sort).
        selected_ids = {id(inst) for inst in culled_instances if inst[2] in _selected_colors}
        instances = [(inst[0], inst[1], inst[2], inst[4]) for inst in culled_instances]
        wire_instances = [(inst[0], inst[1]) for inst in culled_instances if id(inst) in selected_ids]

    # --- GPU instancing: pack all surviving instances into per-LOD attribute
    # buffers and issue ONE draw call per LOD tier (3 total), instead of one
    # matrix push/draw/pop pair per metaball. Falls back to the legacy
    # per-instance path automatically when the create_from_info shader API is
    # unavailable in this Blender build.
    shader_inst = _get_instance_shader()
    inst_lods = _get_instanced_lods() if shader_inst is not None else None

    if shader_inst is not None and inst_lods:
        try:
            _draw_instances_gpu(shader_inst, inst_lods, instances, wire_instances)
        except Exception as e:
            print("MWC instanced draw failed, using legacy path:", e)
            _draw_legacy(lods, instances, wire_instances)
    else:
        _draw_legacy(lods, instances, wire_instances)
        
    gpu.state.blend_set(orig_blend)
    gpu.state.depth_test_set(orig_depth_test)
    gpu.state.depth_mask_set(orig_depth_mask)
    gpu.state.line_width_set(orig_line_width)
    gpu.state.face_culling_set('NONE')


def register_draw_handler():
    global _draw_handler
    if not HAS_BLENDER:
        return
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback_px, (), 'WINDOW', 'POST_VIEW'
        )


def unregister_draw_handler():
    global _draw_handler
    if not HAS_BLENDER:
        return
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        _draw_handler = None

def load_cached_mbs_from_file():
    global _cached_mbs
    res = load_mbs_from_npz()
    if res is not None:
        _cached_mbs, _ = res
    else:
        _cached_mbs = []

def tag_redraw_all_views(self, context):
    if not HAS_BLENDER:
        return
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

def update_viewport_preview(self, context):
    if self.mwc_show_viewport_preview:
        load_cached_mbs_from_file()
    tag_redraw_all_views(self, context)

def create_blender_metaballs(original_mbs, virtual_mbs, source_obj_name):
    """
    Creates the hidden collection, cleans up old metaballs, and populates
    the collection with new Blender metaball objects.
    """
    col_name = "MWC_Metaballs"
    col = bpy.data.collections.get(col_name)
    if col:
        # Delete old objects and their metaball data blocks
        for obj in list(col.objects):
            data = obj.data
            bpy.data.objects.remove(obj)
            if data and isinstance(data, bpy.types.MetaBall):
                bpy.data.metaballs.remove(data)
        bpy.data.collections.remove(col)
        
    # Garbage collect orphaned metaball blocks
    for mb in list(bpy.data.metaballs):
        if mb.users == 0:
            bpy.data.metaballs.remove(mb)
            
    # Create new collection and link to active scene
    col = bpy.data.collections.new(col_name)
    bpy.context.scene.collection.children.link(col)
    col.hide_viewport = False  # Visible by default
    
    # Ensure it is not excluded from the view layer
    lc = bpy.context.view_layer.layer_collection.children.get(col_name)
    if lc:
        lc.exclude = False

    all_mbs = original_mbs + virtual_mbs
    family_counters = {}
    
    for mb_data in all_mbs:
        fam_id = mb_data['family_id']
        co = mb_data['co']
        r = mb_data['radius']
        weights = mb_data['weights']
        normal = mb_data['normal']
        
        # Unique name per family
        base_name = f"MB_{source_obj_name}_F{fam_id}"
        if fam_id not in family_counters:
            name = base_name
            family_counters[fam_id] = 1
        else:
            count = family_counters[fam_id]
            name = f"{base_name}.{count:03d}"
            family_counters[fam_id] += 1
            
        # Create metaball data block
        mb = bpy.data.metaballs.new(name)
        element = mb.elements.new()
        element.co = (0.0, 0.0, 0.0)
        element.radius = r
        
        # Create object
        obj = bpy.data.objects.new(name, mb)
        col.objects.link(obj)
        
        # Parent to armature bone if active armature and dominant bone exist
        arm_obj = get_armature_object(bpy.context.scene)
        parented = False
        if arm_obj and arm_obj.type == 'ARMATURE' and weights:
            dominant_bone = max(weights.items(), key=lambda item: item[1])[0]
            if dominant_bone in arm_obj.pose.bones:
                obj.parent = arm_obj
                obj.parent_type = 'BONE'
                obj.parent_bone = dominant_bone
                obj.matrix_world = mathutils.Matrix.Translation(co)
                parented = True
                
        if not parented:
            obj.location = co
        
        # Write data to custom properties
        obj["family_id"] = fam_id
        obj["normal"] = normal
        obj["radius"] = r
        obj["is_virtual"] = mb_data.get('is_virtual', False)
        obj["symmetry_class"] = mb_data.get('symmetry_class', 'L')
        
        # Store weights in custom properties
        for g_name, w_val in weights.items():
            obj[g_name] = w_val

def extract_mbs_from_collection():
    if not HAS_BLENDER:
        return []
    col = bpy.data.collections.get("MWC_Metaballs")
    if not col:
        return []
        
    extracted_mbs = []
    for obj in col.objects:
        if obj.type == 'META' and len(obj.data.elements) > 0:
            element = obj.data.elements[0]
            radius = element.radius * obj.scale.x
            co = obj.matrix_world.to_translation()
            
            fam_id = obj.get("family_id", 0)
            normal = obj.get("normal", [0.0, 0.0, 1.0])
            symmetry_class = obj.get("symmetry_class", "L")
            
            # Read weights from custom properties
            weights = {}
            exclude_keys = {"weights", "normal", "family_id", "radius", "alpha", "n", "q", "tau", "R_falloff", "is_virtual", "symmetry_class"}
            for key in obj.keys():
                if key in exclude_keys:
                    continue
                if key.startswith("_"):
                    continue
                val = obj[key]
                if isinstance(val, (int, float)):
                    weights[key] = float(val)
                    
            extracted_mbs.append({
                'name': obj.name,
                'co': [co.x, co.y, co.z],
                'radius': radius,
                'normal': list(normal) if hasattr(normal, "__iter__") else [0.0, 0.0, 1.0],
                'weights': weights,
                'family_id': int(fam_id),
                'symmetry_class': str(symmetry_class),
            })
    return extracted_mbs

def save_collection_to_cache(scene):
    mbs = extract_mbs_from_collection()
    if not mbs:
        return False
        
    col = bpy.data.collections.get("MWC_Metaballs")
    alpha = col.get("alpha", scene.mwc_alpha)
    n = col.get("n", 2)
    q = col.get("q", 1.5)
    tau = col.get("tau", 0.001)
    r_falloff_coeff = col.get("r_falloff_coeff", 2.5)
    
    save_mbs_to_npz(mbs, alpha, n, q, tau, r_falloff_coeff)
    return True
