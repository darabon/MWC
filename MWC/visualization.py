import math
import numpy as np

try:
    import bpy
    import gpu
    import gpu.state
    import gpu.matrix
    import gpu.shader
    import mathutils
    from bpy_extras import view3d_utils
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False

# Import utilities
from .translation import t
from .utils import (
    load_mbs_from_npz,
    save_mbs_to_npz
)

# Global variables for caching and drawing
_cached_mbs = []
_sphere_solid_batch = None
_sphere_wire_batch = None
_draw_handler = None
_scene_mbs = []
_scene_mbs_dirty = True


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

def get_sphere_batches():
    global _sphere_solid_batch, _sphere_wire_batch
    if not HAS_BLENDER:
        return None, None
    if _sphere_solid_batch is not None:
        return _sphere_solid_batch, _sphere_wire_batch
        
    segments = 16
    rings = 8
    
    verts = []
    solid_indices = []
    wire_indices = []
    
    # Vertices
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
            
    # Indices
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
    
    _sphere_solid_batch = gpu.types.GPUBatch(type='TRIS', buf=vbo, elem=ibo_solid)
    _sphere_wire_batch = gpu.types.GPUBatch(type='LINES', buf=vbo, elem=ibo_wire)
    
    return _sphere_solid_batch, _sphere_wire_batch


def draw_callback_px():
    if not HAS_BLENDER:
        return
    context = bpy.context
    scene = context.scene
    if not getattr(scene, "mwc_show_viewport_preview", False):
        return
        
    global _scene_mbs, _scene_mbs_dirty
    col = bpy.data.collections.get("MWC_Metaballs")
    if col and col.objects:
        if _scene_mbs_dirty:
            _scene_mbs = extract_mbs_from_collection()
            _scene_mbs_dirty = False
        mbs = _scene_mbs
    else:
        global _cached_mbs
        mbs = _cached_mbs
        
    if not mbs:
        return
        
    color_by_bone = getattr(scene, "mwc_color_by_active_bone", False)
    active_bone = ""
    if color_by_bone:
        from .__init__ import get_active_bone_name
        active_bone = get_active_bone_name(context)
    arm_obj = getattr(scene, "mwc_armature", None)
        
    solid_batch, wire_batch = get_sphere_batches()
    if not solid_batch or not wire_batch:
        return
        
    shader_solid = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader_wire = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    
    orig_depth_mask = gpu.state.depth_mask_get()
    orig_depth_test = gpu.state.depth_test_get()
    orig_blend = gpu.state.blend_get()
    orig_line_width = gpu.state.line_width_get()
    
    selected_idx = getattr(scene, "mwc_selected_mb_idx", -1)
    active_obj = context.active_object
    active_mb_name = active_obj.name if (active_obj and active_obj.type == 'META' and active_obj.name.startswith("MB_")) else ""
    
    import array
    
    # Solid pass
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)
    gpu.state.face_culling_set('BACK')
    
    shader_solid.bind()
    loc_mvp_solid = shader_solid.uniform_from_name("ModelViewProjectionMatrix")
    
    for idx, mb in enumerate(mbs):
        is_selected = False
        if active_mb_name:
            if mb.get('name') == active_mb_name:
                is_selected = True
        elif idx == selected_idx:
            is_selected = True
            
        if color_by_bone and active_bone:
            w = mb['weights'].get(active_bone, 0.0)
            if w <= 0.0 and not is_selected:
                continue
            r, g, b = get_weight_color(w)
            color = (r, g, b, 0.45)
        else:
            color = (0.2, 0.6, 1.0, 0.35)
            
        shader_solid.uniform_float("color", color)
        
        co = mb['co']
        radius = mb['radius']
        
        # Posed space dynamic update
        if mbs is _scene_mbs:
            obj = bpy.data.objects.get(mb.get('name', ''))
            if obj:
                co_world = obj.matrix_world.to_translation()
                co = (co_world.x, co_world.y, co_world.z)
                if obj.data and hasattr(obj.data, 'elements') and len(obj.data.elements) > 0:
                    radius = obj.data.elements[0].radius * obj.scale.x
        elif mbs is _cached_mbs and arm_obj and arm_obj.type == 'ARMATURE':
            p_bone = mb.get('parent_bone', '')
            co_loc = mb.get('co_local')
            if p_bone and co_loc and p_bone in arm_obj.pose.bones:
                pose_bone = arm_obj.pose.bones[p_bone]
                local_vec = mathutils.Vector(co_loc)
                co_world = arm_obj.matrix_world @ (pose_bone.matrix @ local_vec)
                co = (co_world.x, co_world.y, co_world.z)
        
        if is_selected:
            # Allow editing override coords
            try:
                co = (scene.mwc_selected_mb_x, scene.mwc_selected_mb_y, scene.mwc_selected_mb_z)
                radius = scene.mwc_selected_mb_radius
            except:
                pass
                
        gpu.matrix.push()
        gpu.matrix.translate(co)
        gpu.matrix.scale_uniform(radius)
        
        if loc_mvp_solid != -1:
            projection = gpu.matrix.get_projection_matrix()
            modelview = gpu.matrix.get_model_view_matrix()
            mvp = projection @ modelview
            flat_mvp = [val for row in mvp.transposed() for val in row]
            buf = array.array('f', flat_mvp)
            shader_solid.uniform_vector_float(loc_mvp_solid, buf, 16, 1)
            
        solid_batch.draw(shader_solid)
        gpu.matrix.pop()
        
    # Wire pass
    gpu.state.face_culling_set('NONE')
    shader_wire.bind()
    loc_mvp_wire = shader_wire.uniform_from_name("ModelViewProjectionMatrix")
    
    for idx, mb in enumerate(mbs):
        is_selected = False
        if active_mb_name:
            if mb.get('name') == active_mb_name:
                is_selected = True
        elif idx == selected_idx:
            is_selected = True
            
        if not is_selected:
            continue
            
        co = mb['co']
        radius = mb['radius']
        
        # Posed space dynamic update
        if mbs is _scene_mbs:
            obj = bpy.data.objects.get(mb.get('name', ''))
            if obj:
                co_world = obj.matrix_world.to_translation()
                co = (co_world.x, co_world.y, co_world.z)
                if obj.data and hasattr(obj.data, 'elements') and len(obj.data.elements) > 0:
                    radius = obj.data.elements[0].radius * obj.scale.x
        elif mbs is _cached_mbs and arm_obj and arm_obj.type == 'ARMATURE':
            p_bone = mb.get('parent_bone', '')
            co_loc = mb.get('co_local')
            if p_bone and co_loc and p_bone in arm_obj.pose.bones:
                pose_bone = arm_obj.pose.bones[p_bone]
                local_vec = mathutils.Vector(co_loc)
                co_world = arm_obj.matrix_world @ (pose_bone.matrix @ local_vec)
                co = (co_world.x, co_world.y, co_world.z)
        
        # Allow editing override coords
        try:
            co = (scene.mwc_selected_mb_x, scene.mwc_selected_mb_y, scene.mwc_selected_mb_z)
            radius = scene.mwc_selected_mb_radius
        except:
            pass
            
        gpu.state.line_width_set(3.0)
        color = (1.0, 0.8, 0.0, 1.0)
        scale_factor = radius * 1.02
            
        shader_wire.uniform_float("color", color)
        
        gpu.matrix.push()
        gpu.matrix.translate(co)
        gpu.matrix.scale_uniform(scale_factor)
        
        if loc_mvp_wire != -1:
            projection = gpu.matrix.get_projection_matrix()
            modelview = gpu.matrix.get_model_view_matrix()
            mvp = projection @ modelview
            flat_mvp = [val for row in mvp.transposed() for val in row]
            buf = array.array('f', flat_mvp)
            shader_wire.uniform_vector_float(loc_mvp_wire, buf, 16, 1)
            
        wire_batch.draw(shader_wire)
        gpu.matrix.pop()
        
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
        arm_obj = bpy.context.scene.mwc_armature
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
        weights_str_parts = []
        for g_name, w_val in weights.items():
            weights_str_parts.append(f"{g_name}: {w_val:.3f}")
            obj[g_name] = w_val
        obj["weights"] = ", ".join(weights_str_parts)

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
