bl_info = {
    "name": "Metaball Weight Container (MWC) 1.1",
    "author": "ARTEREKET",
    "version": (1, 1),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > MWC 1.1 Tab",
    "description": "Experimental: Generate metaballs saved in .npz archive and apply them to target mesh.",
    "category": "Weight",
}

import math
import json
import os
import sys
import tempfile

import numpy as np

# Reload submodules if already imported (support Blender reload scripts)
if "bpy" in locals():
    import importlib
    if "translation" in locals():
        importlib.reload(translation)
    if "utils" in locals():
        importlib.reload(utils)
    if "generation" in locals():
        importlib.reload(generation)
    if "visualization" in locals():
        importlib.reload(visualization)
    if "transfer" in locals():
        importlib.reload(transfer)

from . import generation, transfer, translation, utils, visualization

# Try importing blender modules to support multiprocessing where bpy is not available
try:
    import bmesh
    import bpy
    import gpu
    import gpu.matrix
    import gpu.shader
    import gpu.state
    import mathutils
    from bpy_extras import view3d_utils
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False

# Relative imports from current package modules
from . import visualization
from . import transfer as transfer_module
from .generation import calculate_mwc_metaballs
from .transfer import apply_mwc_weights
from .translation import t
from .utils import (
    HAS_BLENDER,
    clean_curve_mapping_node,
    get_armature_object,
    get_cache_filepath,
    get_curve_mapping_node,
    load_mbs_from_npz,
    reset_cache_location,
    save_mbs_to_npz,
    triangles_intersect,
)
from .visualization import (
    create_blender_metaballs,
    extract_mbs_from_collection,
    load_cached_mbs_from_file,
    register_draw_handler,
    save_collection_to_cache,
    tag_redraw_all_views,
    unregister_draw_handler,
    update_viewport_preview,
)

# Dummy classes to prevent class registration errors in subprocesses
if HAS_BLENDER:
    OperatorBase = bpy.types.Operator
    PanelBase = bpy.types.Panel
    PropertyGroupBase = bpy.types.PropertyGroup
else:
    class OperatorBase: pass
    class PanelBase: pass
    class PropertyGroupBase: pass


def get_active_bone_name(context):
    if not HAS_BLENDER:
        return ""
    scene = context.scene
    # 1. Check active object first
    obj = context.active_object
    if obj:
        if obj.type == 'ARMATURE':
            if obj.mode == 'POSE' and obj.pose and obj.pose.active_bone:
                return obj.pose.active_bone.name
            elif obj.mode == 'EDIT' and hasattr(obj.data, "edit_bones") and obj.data.edit_bones.active:
                return obj.data.edit_bones.active.name
        elif obj.type == 'MESH':
            vg = obj.vertex_groups.active
            if vg:
                return vg.name

    # 2. Fallback to Armature specified in scene or auto-detected
    arm_obj = get_armature_object(scene, context)
    if arm_obj and arm_obj.type == 'ARMATURE':
        if arm_obj.mode == 'POSE' and arm_obj.pose and arm_obj.pose.active_bone:
            return arm_obj.pose.active_bone.name
        elif arm_obj.mode == 'EDIT' and hasattr(arm_obj.data, "edit_bones") and arm_obj.data.edit_bones.active:
            return arm_obj.data.edit_bones.active.name

    # 3. Fallback to Target mesh specified in scene
    target_obj = getattr(scene, "mwc_target_obj", None)
    if target_obj and target_obj.type == 'MESH':
        vg = target_obj.vertex_groups.active
        if vg:
            return vg.name

    # 4. Fallback to Source mesh specified in scene
    source_obj = getattr(scene, "mwc_source_obj", None)
    if source_obj and source_obj.type == 'MESH':
        vg = source_obj.vertex_groups.active
        if vg:
            return vg.name

    return ""


def read_cache_status():
    """Read-only cache status probe safe to call from UI draw() code.

    Returns (exists, count, alpha). Never writes to ID properties -- writing
    to ID classes during draw is forbidden in Blender and raises
    'Writing to ID classes in this context is not allowed'.
    """
    filepath = get_cache_filepath()
    if not os.path.exists(filepath):
        return False, 0, 0.0
    try:
        with np.load(filepath, allow_pickle=True) as data:
            if 'co' in data:
                count = int(data['co'].shape[0])
                alpha = float(data['alpha']) if 'alpha' in data else 0.70
                return True, count, alpha
    except Exception as e:
        print("Error reading MWC cache status:", e)
    return False, 0, 0.0


def sync_cache_properties(scene):
    """Write cache status into Scene ID properties.

    Only callable from operator execute()/modal() contexts -- never from draw().
    """
    exists, count, alpha = read_cache_status()
    scene.mwc_cache_exists = exists
    scene.mwc_cache_count = count
    scene.mwc_cache_alpha = alpha


class MWC17_OT_ClearCache(OperatorBase):
    bl_idname = "mwc17.clear_cache"
    bl_label = "Clear Cache"
    bl_description = "Delete the saved metaballs cache file"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        filepath = get_cache_filepath()
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                self.report({'INFO'}, t("Cache successfully cleared."))
            except Exception as e:
                self.report({'ERROR'}, f"Failed to delete cache: {e}")
        else:
            self.report({'INFO'}, "Cache is already empty.")

        # Delete MWC_Metaballs collection from scene
        col_name = "MWC_Metaballs"
        col = bpy.data.collections.get(col_name)
        if col:
            for obj in list(col.objects):
                data = obj.data
                bpy.data.objects.remove(obj)
                if data and isinstance(data, bpy.types.MetaBall):
                    bpy.data.metaballs.remove(data)
            bpy.data.collections.remove(col)

        visualization._cached_mbs = []
        scene = context.scene
        scene.mwc_cache_exists = False
        scene.mwc_cache_count = 0
        scene.mwc_cache_alpha = 0.0
        scene.mwc_selected_mb_idx = -1
        scene.mwc_selected_mb_weights.clear()

        # Mark viewport preview cache dirty
        visualization._scene_mbs = []
        visualization._scene_mbs_dirty = True

        return {'FINISHED'}


class MWC17_OT_SpawnViewportMetaballs(OperatorBase):
    bl_idname = "mwc17.spawn_viewport_metaballs"
    bl_label = "Spawn Viewport Metaballs"
    bl_description = "Spawn metaballs from cache as actual scene objects for viewport editing"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if context.active_object and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        scene = context.scene
        loaded = load_mbs_from_npz()
        if not loaded:
            self.report({'ERROR'}, t("Metaball collection MWC_Metaballs is empty or not found!"))
            return {'CANCELLED'}

        mbs, meta = loaded
        source_name = scene.mwc_source_obj.name if scene.mwc_source_obj else "Cache"

        # Build Blender metaballs in the collection
        create_blender_metaballs(mbs, [], source_name)

        # Store settings on the collection for later use
        col = bpy.data.collections.get("MWC_Metaballs")
        if col:
            col["alpha"] = meta.get('alpha', scene.mwc_alpha)
            col["n"] = meta.get('n', 2)
            col["q"] = meta.get('q', 1.5)
            col["tau"] = meta.get('tau', 0.001)
            col["r_falloff_coeff"] = meta.get('r_falloff_coeff', 2.5)

        visualization._scene_mbs_dirty = True

        tag_redraw_all_views(self, context)
        self.report({'INFO'}, t("Successfully spawned {} metaball objects in the viewport.", len(mbs)))
        return {'FINISHED'}


class MWC17_OT_ClearViewportMetaballs(OperatorBase):
    bl_idname = "mwc17.clear_viewport_metaballs"
    bl_label = "Clear Viewport Metaballs"
    bl_description = "Remove metaball objects from the scene viewport without deleting the cache file"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        col_name = "MWC_Metaballs"
        col = bpy.data.collections.get(col_name)
        if col:
            for obj in list(col.objects):
                data = obj.data
                bpy.data.objects.remove(obj)
                if data and isinstance(data, bpy.types.MetaBall):
                    bpy.data.metaballs.remove(data)
            bpy.data.collections.remove(col)

        visualization._scene_mbs = []
        visualization._scene_mbs_dirty = True

        scene = context.scene
        scene.mwc_selected_mb_idx = -1
        scene.mwc_selected_mb_weights.clear()

        tag_redraw_all_views(self, context)
        self.report({'INFO'}, t("Viewport metaballs cleared."))
        return {'FINISHED'}


class MWC17_OT_CreateMetaballs(OperatorBase):
    bl_idname = "mwc17.create_metaballs"
    bl_label = "Create Metaballs"
    bl_description = "Generate metaballs from vertex weights of the source mesh and save to cache"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if context.active_object and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        scene = context.scene
        src_obj = scene.mwc_source_obj

        if not src_obj:
            self.report({'ERROR'}, t("Please select the source object!"))
            return {'CANCELLED'}

        if src_obj.type != 'MESH':
            self.report({'ERROR'}, t("Source object must be a mesh!"))
            return {'CANCELLED'}

        alpha = scene.mwc_alpha
        k_coeff = scene.mwc_subdivision_k
        merge_close = scene.mwc_merge_close
        merge_factor = scene.mwc_merge_factor

        use_joint_scaling = scene.mwc_use_joint_scaling
        armature_obj = get_armature_object(scene, context)
        joint_scale = scene.mwc_joint_scale
        middle_scale = scene.mwc_middle_scale

        use_thickness_scaling = scene.mwc_use_thickness_scaling
        thickness_factor = scene.mwc_thickness_factor

        creation_type = scene.mwc_creation_type
        use_symmetry = scene.mwc_symmetry

        # Calculate original and virtual metaballs
        import time as _time
        t_start = _time.perf_counter()
        wm = context.window_manager
        wm.progress_begin(0, 100)
        try:
            wm.progress_update(5)
            original_mbs, virtual_mbs = calculate_mwc_metaballs(
                src_obj, alpha, creation_type,
                k_coeff=k_coeff,
                merge_close=merge_close,
                merge_factor=merge_factor,
                use_symmetry=use_symmetry,
                use_joint_scaling=use_joint_scaling,
                armature_obj=armature_obj,
                joint_scale=joint_scale,
                middle_scale=middle_scale,
                use_thickness_scaling=use_thickness_scaling,
                thickness_factor=thickness_factor
            )
            wm.progress_update(85)

            # Save metaballs to NPZ cache directly
            all_mbs = original_mbs + virtual_mbs
            save_mbs_to_npz(
                all_mbs, alpha,
                scene.mwc_n,
                scene.mwc_q,
                scene.mwc_tau,
                scene.mwc_r_falloff_coeff
            )
            wm.progress_update(95)
        finally:
            wm.progress_end()

        # Sync cache properties in scene
        scene.mwc_cache_exists = True
        scene.mwc_cache_count = len(all_mbs)
        scene.mwc_cache_alpha = alpha

        load_cached_mbs_from_file()
        tag_redraw_all_views(self, context)

        elapsed = _time.perf_counter() - t_start
        self.report({'INFO'}, t("Successfully created {} metaballs.", len(all_mbs)) +
                    " (%.2fs, %d src / %d virtual)" % (elapsed, len(original_mbs), len(virtual_mbs)))
        return {'FINISHED'}


class MWC17_OT_ApplyWeights(OperatorBase):
    bl_idname = "mwc17.apply_weights"
    bl_label = "Apply Weights"
    bl_description = "Bake weights from cached metaballs onto the target mesh"
    bl_options = {'REGISTER', 'UNDO'}

    confirmed: bpy.props.BoolProperty(default=False, options={'HIDDEN'})

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text=t("Warning: Object contains self-intersecting geometry!"), icon="ERROR")
        col.label(text=t("This can lead to weight distribution instability."))
        col.separator()
        col.label(text=t("Do you really want to continue?"))

    def invoke(self, context, event):
        if self.confirmed:
            return self.execute(context)

        scene = context.scene
        target_obj = scene.mwc_target_obj

        if not target_obj:
            self.report({'ERROR'}, t("Please select the target object!"))
            return {'CANCELLED'}

        if target_obj.type != 'MESH':
            self.report({'ERROR'}, t("Target object must be a mesh!"))
            return {'CANCELLED'}

        # Check self-intersection
        has_self_intersection = False
        try:
            from mathutils.bvhtree import BVHTree
            depsgraph = context.evaluated_depsgraph_get()
            bvh = BVHTree.FromObject(target_obj, depsgraph)
            overlaps = bvh.overlap(bvh)

            mesh = target_obj.data
            for f1_idx, f2_idx in overlaps:
                if f1_idx >= f2_idx:
                    continue
                v1 = mesh.polygons[f1_idx].vertices
                v2 = mesh.polygons[f2_idx].vertices
                shared = False
                for idx1 in v1:
                    if idx1 in v2:
                        shared = True
                        break
                if not shared:
                    # Perform precise geometric intersection check
                    p_a1 = mesh.vertices[v1[0]].co
                    p_a2 = mesh.vertices[v1[1]].co
                    p_a3 = mesh.vertices[v1[2]].co

                    p_b1 = mesh.vertices[v2[0]].co
                    p_b2 = mesh.vertices[v2[1]].co
                    p_b3 = mesh.vertices[v2[2]].co

                    if len(v1) == 3 and len(v2) == 3:
                        if triangles_intersect((p_a1, p_a2, p_a3), (p_b1, p_b2, p_b3)):
                            has_self_intersection = True
                            break
                    else:
                        tris_a = []
                        for i in range(1, len(v1) - 1):
                            tris_a.append((mesh.vertices[v1[0]].co, mesh.vertices[v1[i]].co, mesh.vertices[v1[i+1]].co))
                        tris_b = []
                        for j in range(1, len(v2) - 1):
                            tris_b.append((mesh.vertices[v2[0]].co, mesh.vertices[v2[j]].co, mesh.vertices[v2[j+1]].co))

                        intersect = False
                        for ta in tris_a:
                            for tb in tris_b:
                                if triangles_intersect(ta, tb):
                                    intersect = True
                                    break
                            if intersect:
                                break
                        if intersect:
                            has_self_intersection = True
                            break
        except Exception as e:
            print("Error checking self-intersection:", e)

        if has_self_intersection:
            self.confirmed = True
            return context.window_manager.invoke_props_dialog(self, width=400)

        return self.execute(context)

    def execute(self, context):
        if context.active_object and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        scene = context.scene
        target_obj = scene.mwc_target_obj

        if not target_obj:
            self.report({'ERROR'}, t("Please select the target object!"))
            return {'CANCELLED'}

        if target_obj.type != 'MESH':
            self.report({'ERROR'}, t("Target object must be a mesh!"))
            return {'CANCELLED'}

        # Try extracting metaballs from scene collection first
        mbs = extract_mbs_from_collection()
        meta = None
        if mbs:
            # Sync cache on disk
            save_collection_to_cache(scene)
            col = bpy.data.collections.get("MWC_Metaballs")
            meta = {
                'n': col.get('n', 2),
                'q': col.get('q', 1.5),
                'tau': col.get('tau', 0.001),
                'r_falloff_coeff': col.get('r_falloff_coeff', 2.5)
            }
        else:
            # Fallback to cache NPZ
            loaded = load_mbs_from_npz()
            if not loaded:
                self.report({'ERROR'}, t("Metaball collection MWC_Metaballs is empty or not found!"))
                return {'CANCELLED'}
            mbs, meta = loaded

        n = scene.mwc_n
        q = scene.mwc_q
        tau = scene.mwc_tau
        r_falloff_coeff = scene.mwc_r_falloff_coeff

        use_normal_filter = scene.mwc_use_normal_filter
        normal_p = scene.mwc_normal_filter_p

        use_smoothing = scene.mwc_use_smoothing
        smoothing_strength = scene.mwc_smoothing_strength
        smoothing_iterations = scene.mwc_smoothing_iterations

        use_geodesic = scene.mwc_use_geodesic
        geodesic_mode = scene.mwc_geodesic_mode
        use_custom_curve = scene.mwc_use_custom_curve
        curve_node = get_curve_mapping_node() if use_custom_curve else None

        # Apply the weights with a live progress bar in the status bar
        import time as _time
        t_start = _time.perf_counter()
        wm = context.window_manager
        last_stage = [-1]

        def _progress(frac, msg=""):
            stage = int(frac * 100)
            if stage != last_stage[0]:
                last_stage[0] = stage
                wm.progress_update(stage)

        wm.progress_begin(0, 100)
        try:
            apply_mwc_weights(
                target_obj, mbs, n, q, tau, r_falloff_coeff,
                use_normal_filter=use_normal_filter, normal_p=normal_p,
                use_smoothing=use_smoothing,
                smoothing_strength=smoothing_strength,
                smoothing_iterations=smoothing_iterations,
                use_geodesic=use_geodesic,
                use_custom_curve=use_custom_curve,
                curve_node=curve_node,
                geodesic_mode=geodesic_mode,
                progress_cb=_progress
            )
        finally:
            wm.progress_end()

        elapsed = _time.perf_counter() - t_start
        engine = "SciPy" if getattr(transfer_module, 'HAS_SCIPY', False) else "NumPy"
        self.report({'INFO'},
                    t("Weights successfully applied to target mesh.") +
                    " (%.2fs, %s engine)" % (elapsed, engine))
        return {'FINISHED'}


class MWC17_PT_MainPanel(PanelBase):
    bl_label = "Metaball Weight Container (MWC) 1.1"
    bl_idname = "MWC17_PT_MainPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MWC 1.1'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # General Cache Info status shown at the very top (always visible).
        # Read-only: writing to Scene properties inside draw() is forbidden.
        box_cache = layout.box()
        cache_exists, cache_count, cache_alpha = read_cache_status()
        if cache_exists:
            box_cache.label(text=t("Loaded {} metaballs (Alpha: {})", cache_count, f"{cache_alpha:.2f}"), icon="FILE_TICK")
            box_cache.operator("mwc17.clear_cache", text=t("Clear Cache"), icon="TRASH")
        else:
            box_cache.label(text=t("Metaballs Cache Status:") + " " + t("Empty (Run 'Create')"), icon="FILE_BLANK")


class MWC17_PT_Creation(PanelBase):
    bl_label = "1. Generation & Cache Creation"
    bl_idname = "MWC17_PT_Creation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MWC 1.1'
    bl_parent_id = "MWC17_PT_MainPanel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # 1. Source selection
        layout.label(text="Source Data (Creation):")
        layout.prop(scene, "mwc_source_obj", text="Source Mesh")

        # 2. Parameters block
        box = layout.box()
        box.label(text="Creation Parameters:")
        box.prop(scene, "mwc_alpha", text="Alpha")
        box.prop(scene, "mwc_subdivision_k", text="Subdivision K")
        box.prop(scene, "mwc_merge_close", text="Merge Close")
        sub_merge = box.column()
        sub_merge.active = scene.mwc_merge_close
        sub_merge.prop(scene, "mwc_merge_factor", text="Merge Factor")

        # Joint-Aware Scaling controls
        box.prop(scene, "mwc_use_joint_scaling", text="Use Joint Scaling")
        sub_joint = box.column()
        sub_joint.active = scene.mwc_use_joint_scaling
        sub_joint.prop(scene, "mwc_armature", text="Armature")
        sub_joint.prop(scene, "mwc_joint_scale", text="Joint Scale Factor")
        sub_joint.prop(scene, "mwc_middle_scale", text="Middle Scale Factor")

        # Thickness-Aware Scaling controls
        box.prop(scene, "mwc_use_thickness_scaling", text="Use Thickness Scaling")
        sub_thick = box.column()
        sub_thick.active = scene.mwc_use_thickness_scaling
        sub_thick.prop(scene, "mwc_thickness_factor", text="Thickness Factor")

        # 4. Enum list creation type
        layout.prop(scene, "mwc_creation_type", text="Grouping")
        layout.prop(scene, "mwc_symmetry", text="Symmetry")

        # 5. Create button -- drawn BEFORE the preset row so that any failure in
        # the (less critical) preset UI can never hide the primary action.
        layout.operator("mwc17.create_metaballs", text="Create", icon="META_BALL")

        # Parameter presets
        try:
            row_preset = layout.row(align=True)
            row_preset.operator("mwc17.load_preset", text="Load Preset", icon="FILE_REFRESH")
            row_preset.operator("mwc17.save_preset", text="", icon="SAVE_AS")
        except Exception as e:
            print("MWC preset row draw failed:", e)


class MWC17_PT_Visualization(PanelBase):
    bl_label = "2. Viewport & Editing"
    bl_idname = "MWC17_PT_Visualization"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MWC 1.1'
    bl_parent_id = "MWC17_PT_MainPanel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Spawn/Clear viewport metaballs
        row_spawn = layout.row(align=True)
        row_spawn.operator("mwc17.spawn_viewport_metaballs", text="Spawn Viewport Metaballs", icon="META_BALL")
        row_spawn.operator("mwc17.clear_viewport_metaballs", text="Clear Viewport Metaballs", icon="X")

        # Viewport Preview controls
        box_view = layout.box()
        box_view.label(text="Viewport Preview", icon="VIEW3D")
        box_view.prop(scene, "mwc_show_viewport_preview", text="Show Preview")
        box_view.operator("mwc17.select_nearest_mb", text="Select Nearest", icon="CURSOR")

        sub_preview = box_view.column()
        sub_preview.active = scene.mwc_show_viewport_preview
        sub_preview.prop(scene, "mwc_color_by_active_bone", text="Color by Active Bone")

        # Active Metaball Editor
        active_obj = context.active_object
        col_name = "MWC_Metaballs"
        col = bpy.data.collections.get(col_name)
        is_mwc_mb = False
        if active_obj and active_obj.type == 'META' and col and active_obj.name in col.objects:
            is_mwc_mb = True

        box_edit = layout.box()
        box_edit.label(text="Active Metaball Editor", icon="EDITMODE_HLT")

        # Add a button to spawn a new metaball
        box_edit.operator("mwc17.add_active_mb", text="Add Metaball", icon="ADD")

        if is_mwc_mb:
            box_edit.label(text=f"Object: {active_obj.name}")

            # Snap to cursor button
            box_edit.operator("mwc17.snap_active_mb_to_cursor", text="Snap to Cursor", icon="CURSOR")

            # Native Radius slider of the metaball element
            if active_obj.data.elements:
                element = active_obj.data.elements[0]
                box_edit.prop(element, "radius", text="Alpha")

            # Bone weights list
            box_weights = box_edit.box()
            box_weights.label(text="Bone Weights", icon="BONE_DATA")

            # Read custom properties representing bone weights
            exclude_keys = {"weights", "normal", "family_id", "radius", "alpha", "n", "q", "tau", "R_falloff", "is_virtual", "symmetry_class", "_RNA_UI"}
            for key in list(active_obj.keys()):
                if key in exclude_keys:
                    continue
                if key.startswith("_"):
                    continue

                row = box_weights.row(align=True)
                row.prop(active_obj, f'["{key}"]', text=key, slider=True)
                op_rem = row.operator("mwc17.remove_active_mb_weight", text="", icon="X")
                op_rem.bone_name = key

            # Add new bone weight
            row_add = box_weights.row(align=True)
            row_add.prop(scene, "mwc_new_bone_name", text="")
            op_add = row_add.operator("mwc17.add_active_mb_weight", text="", icon="ADD")
            op_add.bone_name = scene.mwc_new_bone_name

            # Save/sync cache button
            box_edit.operator("mwc17.save_collection_to_cache", text="Save to Cache", icon="FILE_TICK")
        else:
            box_edit.label(text="Select a metaball in the viewport to edit it", icon="INFO")


class MWC17_PT_Transfer(PanelBase):
    bl_label = "3. Weight Transfer & Baking"
    bl_idname = "MWC17_PT_Transfer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MWC 1.1'
    bl_parent_id = "MWC17_PT_MainPanel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Performance engine status: shows whether the C-accelerated SciPy
        # path (vectorized KD queries + Dijkstra) is active.
        engine_box = layout.box()
        if getattr(transfer_module, 'HAS_SCIPY', False):
            engine_box.label(text="Engine: SciPy (C accelerated)", icon="MODIFIER_ON")
        else:
            engine_box.label(text="Engine: NumPy fallback", icon="MODIFIER")

        # Target selection
        layout.label(text="Weight Application:")
        layout.prop(scene, "mwc_target_obj", text="Target Mesh")

        # Normal filter and transfer settings block
        box_apply = layout.box()
        box_apply.label(text="Transfer Parameters:")

        box_transfer = box_apply.box()
        sub_transfer = box_transfer.column()
        sub_transfer.prop(scene, "mwc_use_geodesic", text="Geodesic Distance")
        if scene.mwc_use_geodesic:
            sub_transfer.prop(scene, "mwc_geodesic_mode", text="Geodesic Performance Mode")
        sub_transfer.prop(scene, "mwc_use_custom_curve", text="Custom Falloff Curve")

        if scene.mwc_use_custom_curve:
            node = get_curve_mapping_node(create=False)
            if node:
                sub_transfer.template_curve_mapping(node, "mapping")
            else:
                sub_transfer.operator("mwc17.init_curve", text="Initialize Curve", icon="NODETREE")
        else:
            sub_transfer.prop(scene, "mwc_n", text="Wyvill Exponent (n)")

        sub_transfer.prop(scene, "mwc_q", text="Mixing Exponent (q)")
        sub_transfer.prop(scene, "mwc_tau", text="Threshold (tau)")
        sub_transfer.prop(scene, "mwc_r_falloff_coeff", text="R Falloff Coeff")

        box_apply.prop(scene, "mwc_use_normal_filter", text="Normal Filter")
        sub_col = box_apply.column()
        sub_col.active = scene.mwc_use_normal_filter
        sub_col.prop(scene, "mwc_normal_filter_p", text="Strictness (p)")

        box_apply.prop(scene, "mwc_use_smoothing", text="Smoothing")
        sub_smooth = box_apply.column()
        sub_smooth.active = scene.mwc_use_smoothing
        sub_smooth.prop(scene, "mwc_smoothing_strength", text="Smoothing Strength")
        sub_smooth.prop(scene, "mwc_smoothing_iterations", text="Iterations")

        # 7. Apply button
        layout.operator("mwc17.apply_weights", text="Apply", icon="MOD_SKIN")


def get_creation_type_items(self, context):
    return [
        ('SINGLE', "Single Object", "All metaballs belong to one group/family"),
        ('MULTIPLY', "Multiple Objects", "Separate metaballs into families based on geometry islands")
    ]


def _preset_items(_self=None, _context=None):
    """Enum items for the preset loader, discovered from the presets dir."""
    if not HAS_BLENDER:
        return [("", "", "")]
    directory = _mwc_presets_dir()
    items = []
    try:
        for fname in sorted(os.listdir(directory)):
            if fname.endswith(".json"):
                stem = fname[:-5]
                items.append((stem, stem, f"Apply preset {stem}"))
    except OSError:
        pass
    if not items:
        items = [("", "(no presets yet)", "Save a preset first")]
    return items


def update_use_custom_curve(self, context):
    if self.mwc_use_custom_curve:
        get_curve_mapping_node(create=True)


class MWC17_OT_InitCurve(OperatorBase):
    bl_idname = "mwc17.init_curve"
    bl_label = "Initialize Custom Curve"
    bl_description = "Initialize the custom curve node group safely in write-allowed context"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        get_curve_mapping_node(create=True)
        return {'FINISHED'}


class MWC17_MBWeightItem(PropertyGroupBase):
    pass


if HAS_BLENDER:
    MWC17_MBWeightItem.__annotations__ = {
        "bone_name": bpy.props.StringProperty(name="Bone"),
        "weight": bpy.props.FloatProperty(name="Weight", min=0.0, max=1.0, default=0.0)
    }


class MWC17_OT_SelectNearestMB(OperatorBase):
    bl_idname = "mwc17.select_nearest_mb"
    bl_label = "Select Nearest Metaball"
    bl_description = "Select the metaball closest to the 3D Cursor"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        if not visualization._cached_mbs:
            load_cached_mbs_from_file()
            if not visualization._cached_mbs:
                self.report({'WARNING'}, "Metaballs cache is empty. Please Create metaballs first.")
                return {'CANCELLED'}

        cursor_loc = scene.cursor.location
        min_dist = float('inf')
        nearest_idx = -1

        for idx, mb in enumerate(visualization._cached_mbs):
            co = mathutils.Vector(mb['co'])
            dist = (co - cursor_loc).length
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx

        if nearest_idx != -1:
            scene.mwc_selected_mb_idx = nearest_idx
            mb = visualization._cached_mbs[nearest_idx]
            scene.mwc_selected_mb_x = mb['co'][0]
            scene.mwc_selected_mb_y = mb['co'][1]
            scene.mwc_selected_mb_z = mb['co'][2]
            scene.mwc_selected_mb_radius = mb['radius']

            scene.mwc_selected_mb_weights.clear()
            for bone_name, weight in mb['weights'].items():
                item = scene.mwc_selected_mb_weights.add()
                item.bone_name = bone_name
                item.weight = weight

            tag_redraw_all_views(self, context)
            self.report({'INFO'}, f"Selected metaball {nearest_idx} (Distance: {min_dist:.4f})")
            return {'FINISHED'}

        return {'CANCELLED'}


class MWC17_OT_AddActiveMB(OperatorBase):
    bl_idname = "mwc17.add_active_mb"
    bl_label = "Add Metaball"
    bl_description = "Create a new metaball at the 3D Cursor location"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not HAS_BLENDER:
            return {'CANCELLED'}
        scene = context.scene
        col_name = "MWC_Metaballs"
        col = bpy.data.collections.get(col_name)
        if not col:
            col = bpy.data.collections.new(col_name)
            context.scene.collection.children.link(col)
            col.hide_viewport = False

        cursor_loc = context.scene.cursor.location
        active_bone = get_active_bone_name(context)

        src_name = scene.mwc_source_obj.name if scene.mwc_source_obj else "Obj"
        base_name = f"MB_{src_name}_F0"
        unique_name = bpy.path.clean_name(base_name)

        mb = bpy.data.metaballs.new(unique_name)
        element = mb.elements.new()
        element.co = (0.0, 0.0, 0.0)
        element.radius = scene.mwc_cache_alpha if scene.mwc_cache_alpha > 0 else 0.2

        obj = bpy.data.objects.new(unique_name, mb)
        obj.location = cursor_loc.copy()
        col.objects.link(obj)

        # Select and make active
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj

        obj["family_id"] = 0
        obj["normal"] = [0.0, 0.0, 1.0]
        obj["radius"] = element.radius
        obj["is_virtual"] = False
        obj["symmetry_class"] = "C"

        if active_bone:
            obj[active_bone] = 1.0

        tag_redraw_all_views(self, context)
        self.report({'INFO'}, "Added new metaball at 3D Cursor.")
        return {'FINISHED'}


class MWC17_OT_AddActiveMBWeight(OperatorBase):
    bl_idname = "mwc17.add_active_mb_weight"
    bl_label = "Add Bone Weight"
    bl_description = "Add a new bone weight custom property to the active metaball"
    bl_options = {'REGISTER', 'UNDO'}

    bone_name: bpy.props.StringProperty(name="Bone Name", default="")

    def execute(self, context):
        if not HAS_BLENDER:
            return {'CANCELLED'}
        active_obj = context.active_object
        if not visualization.is_mwc_metaball(active_obj):
            self.report({'WARNING'}, "No active metaball selected.")
            return {'CANCELLED'}

        bone = self.bone_name.strip()
        if not bone:
            self.report({'WARNING'}, "Bone name cannot be empty.")
            return {'CANCELLED'}

        active_obj[bone] = 0.0
        tag_redraw_all_views(self, context)
        return {'FINISHED'}


class MWC17_OT_RemoveActiveMBWeight(OperatorBase):
    bl_idname = "mwc17.remove_active_mb_weight"
    bl_label = "Remove Bone Weight"
    bl_description = "Remove this bone weight custom property"
    bl_options = {'REGISTER', 'UNDO'}

    bone_name: bpy.props.StringProperty(name="Bone Name", default="")

    def execute(self, context):
        if not HAS_BLENDER:
            return {'CANCELLED'}
        active_obj = context.active_object
        if not visualization.is_mwc_metaball(active_obj):
            return {'CANCELLED'}

        if self.bone_name in active_obj:
            del active_obj[self.bone_name]
            tag_redraw_all_views(self, context)
            return {'FINISHED'}
        return {'CANCELLED'}


class MWC17_OT_SnapActiveMBToCursor(OperatorBase):
    bl_idname = "mwc17.snap_active_mb_to_cursor"
    bl_label = "Snap Active to Cursor"
    bl_description = "Move active metaball to the 3D Cursor location"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not HAS_BLENDER:
            return {'CANCELLED'}
        active_obj = context.active_object
        if not visualization.is_mwc_metaball(active_obj):
            self.report({'WARNING'}, "No active metaball selected.")
            return {'CANCELLED'}

        active_obj.location = context.scene.cursor.location.copy()
        tag_redraw_all_views(self, context)
        self.report({'INFO'}, "Snapped active metaball to 3D Cursor.")
        return {'FINISHED'}


class MWC17_OT_SaveCollectionToCache(OperatorBase):
    bl_idname = "mwc17.save_collection_to_cache"
    bl_label = "Save Metaballs to Cache"
    bl_description = "Save current scene metaballs back to the .npz cache file"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if save_collection_to_cache(context.scene):
            self.report({'INFO'}, "Saved scene metaballs to cache.")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "Metaballs collection is empty or not found.")
            return {'CANCELLED'}


# ---------------------------------------------------------------------------
# Preset system -- save/load every generation & transfer parameter as a named
# JSON preset under presets/operator/mwc17.preset/. Character pipelines reuse
# the same settings across dozens of garments; typing them by hand each time
# is error-prone, and Blender's own preset framework would only cover props
# drawn directly in a panel column.
# ---------------------------------------------------------------------------
_MWC_PRESET_PROPS = (
    # Generation
    "mwc_alpha", "mwc_subdivision_k", "mwc_merge_close", "mwc_merge_factor",
    "mwc_use_joint_scaling", "mwc_joint_scale", "mwc_middle_scale",
    "mwc_use_thickness_scaling", "mwc_thickness_factor",
    "mwc_creation_type", "mwc_symmetry",
    # Transfer
    "mwc_n", "mwc_q", "mwc_tau", "mwc_r_falloff_coeff",
    "mwc_use_normal_filter", "mwc_normal_filter_p",
    "mwc_use_smoothing", "mwc_smoothing_strength", "mwc_smoothing_iterations",
    "mwc_use_geodesic", "mwc_geodesic_mode", "mwc_use_custom_curve",
)


def _mwc_presets_dir():
    return os.path.join(bpy.utils.resource_path('USER'), "scripts", "presets", "operator", "mwc17.preset")


class MWC17_OT_SavePreset(OperatorBase):
    bl_idname = "mwc17.save_preset"
    bl_label = "Save Preset"
    bl_description = "Save all MWC generation and transfer parameters as a reusable preset"
    bl_options = {'REGISTER', 'UNDO'}

    preset_name: bpy.props.StringProperty(name="Name", default="MyPreset")

    def execute(self, context):
        name = self.preset_name.strip()
        if not name:
            self.report({'WARNING'}, t("Preset name cannot be empty."))
            return {'CANCELLED'}
        safe = bpy.path.clean_name(name)
        scene = context.scene
        data = {prop: getattr(scene, prop) for prop in _MWC_PRESET_PROPS}
        data["_version"] = (1, 2)
        directory = _mwc_presets_dir()
        try:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, safe + ".json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to save preset: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, t("Preset '{}' saved.", safe))
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=300)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "preset_name", text=t("Name"))


class MWC17_OT_LoadPreset(OperatorBase):
    bl_idname = "mwc17.load_preset"
    bl_label = "Load Preset"
    bl_description = "Apply a saved MWC parameter preset to the current scene"
    bl_options = {'REGISTER', 'UNDO'}

    preset_name: bpy.props.EnumProperty(
        name="Preset",
        description="Saved parameter preset to apply",
        items=_preset_items,
    )

    def execute(self, context):
        if not self.preset_name:
            self.report({'WARNING'}, t("No presets found."))
            return {'CANCELLED'}
        path = os.path.join(_mwc_presets_dir(), self.preset_name + ".json")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to read preset: {e}")
            return {'CANCELLED'}

        scene = context.scene
        applied = 0
        for prop, value in data.items():
            if prop.startswith("_") or prop not in _MWC_PRESET_PROPS:
                continue
            try:
                setattr(scene, prop, value)
                applied += 1
            except Exception as e:
                print(f"MWC preset: could not set {prop}={value!r}: {e}")
        self.report({'INFO'}, t("Applied {} parameters from preset.", applied))
        return {'FINISHED'}


classes = (
    MWC17_MBWeightItem,
    MWC17_OT_ClearCache,
    MWC17_OT_CreateMetaballs,
    MWC17_OT_ApplyWeights,
    MWC17_OT_InitCurve,
    MWC17_OT_SaveCollectionToCache,
    MWC17_OT_AddActiveMB,
    MWC17_OT_AddActiveMBWeight,
    MWC17_OT_RemoveActiveMBWeight,
    MWC17_OT_SnapActiveMBToCursor,
    MWC17_OT_SpawnViewportMetaballs,
    MWC17_OT_ClearViewportMetaballs,
    MWC17_OT_SavePreset,
    MWC17_OT_LoadPreset,
    MWC17_PT_MainPanel,
    MWC17_PT_Creation,
    MWC17_PT_Visualization,
    MWC17_PT_Transfer,
)


def mwc17_depsgraph_update(scene, depsgraph):
    col = bpy.data.collections.get("MWC_Metaballs")
    if col:
        if len(col.objects) != len(visualization._scene_mbs):
            visualization._scene_mbs_dirty = True

    active_obj = bpy.context.active_object
    if active_obj and active_obj.type == 'META' and visualization.is_mwc_metaball(active_obj, col):
        visualization._scene_mbs_dirty = True
        return

    for update in depsgraph.updates:
        if update.id.name.startswith("MB_") or update.id.name == "MWC_Metaballs":
            visualization._scene_mbs_dirty = True
            break


def _on_load_post(_scene, _depsgraph):
    # The per-blend cache path is derived from bpy.data.filepath; after a
    # file load it must be re-derived or the addon would keep reading the
    # previous blend's cache namespace.
    reset_cache_location()
    load_cached_mbs_from_file()


def register():
    # Register translations first so they are available for properties registered below
    try:
        from .translation import translations_dict
        bpy.app.translations.register(__name__, translations_dict)
    except Exception as e:
        print("MWC translation registration failed:", e)

    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.mwc_source_obj = bpy.props.PointerProperty(
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH',
        description="Mesh based on which metaballs with weights will be generated"
    )

    bpy.types.Scene.mwc_alpha = bpy.props.FloatProperty(
        name="Alpha",
        default=0.70,
        min=0.1,
        max=2.0,
        description="Metaball radius coefficient"
    )

    bpy.types.Scene.mwc_n = bpy.props.IntProperty(
        name="n",
        default=2,
        min=1,
        max=10,
        description="Exponent n in Wyvill formula"
    )

    bpy.types.Scene.mwc_q = bpy.props.FloatProperty(
        name="q",
        default=1.5,
        min=1.0,
        max=10.0,
        description="Transition mixing stiffness parameter"
    )

    bpy.types.Scene.mwc_tau = bpy.props.FloatProperty(
        name="tau",
        default=0.001,
        min=0.0001,
        max=0.1,
        precision=4,
        description="Threshold for cutting off small weights"
    )

    bpy.types.Scene.mwc_r_falloff_coeff = bpy.props.FloatProperty(
        name="R Falloff Coeff",
        default=2.5,
        min=1.0,
        max=10.0,
        description="R_falloff radius coefficient relative to average radius"
    )

    bpy.types.Scene.mwc_subdivision_k = bpy.props.FloatProperty(
        name="Subdivision K",
        default=2.0,
        min=1.0,
        max=10.0,
        description="Coefficient for local subdivision of long edges (L > K * max(R1, R2))"
    )

    bpy.types.Scene.mwc_merge_close = bpy.props.BoolProperty(
        name="Merge Close",
        default=True,
        description="Merge close metaballs with same dominant bone"
    )

    bpy.types.Scene.mwc_merge_factor = bpy.props.FloatProperty(
        name="Merge Factor",
        default=0.5,
        min=0.1,
        max=2.0,
        description="Merge distance coefficient (factor * (R1 + R2))"
    )

    bpy.types.Scene.mwc_creation_type = bpy.props.EnumProperty(
        items=get_creation_type_items,
        name="Creation Type",
        description="Whether to create a single group of metaballs or group by geometric islands"
    )

    bpy.types.Scene.mwc_symmetry = bpy.props.BoolProperty(
        name="Symmetry",
        default=False,
        description="Generate only left half of metaballs and automatically mirror to the right"
    )

    bpy.types.Scene.mwc_target_obj = bpy.props.PointerProperty(
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH',
        description="Target mesh onto which weights from metaballs will be projected"
    )

    bpy.types.Scene.mwc_use_normal_filter = bpy.props.BoolProperty(
        name="Use Normal Filter",
        default=False,
        description="Enable weight filtering by vertex normals to prevent leaks. Do not enable if the mesh has thickness (solidify)."
    )

    bpy.types.Scene.mwc_normal_filter_p = bpy.props.FloatProperty(
        name="Normal Filter Exponent (p)",
        default=1.0,
        min=0.1,
        max=10.0,
        description="Strictness exponent of the normal filter (higher p = stricter filtering)"
    )

    bpy.types.Scene.mwc_use_smoothing = bpy.props.BoolProperty(
        name="Use Smoothing",
        default=True,
        description="Smooth transferred weights across adjacent vertices"
    )

    bpy.types.Scene.mwc_smoothing_strength = bpy.props.FloatProperty(
        name="Smoothing Strength",
        default=0.5,
        min=0.0,
        max=1.0,
        description="Weight smoothing strength per iteration"
    )

    bpy.types.Scene.mwc_smoothing_iterations = bpy.props.IntProperty(
        name="Smoothing Iterations",
        default=3,
        min=1,
        max=20,
        description="Number of passes (iterations) of smoothing"
    )

    bpy.types.Scene.mwc_use_geodesic = bpy.props.BoolProperty(
        name="Geodesic Distance",
        default=False,
        description="Use geodesic distance along mesh edges (Dijkstra) instead of Euclidean to prevent weight leakage"
    )

    bpy.types.Scene.mwc_geodesic_mode = bpy.props.EnumProperty(
        name="Geodesic Performance Mode",
        description="Choose multi-threading/multi-processing method for Dijkstra calculation",
        items=[
            ('SEQ', "Sequential", "Sequential single-threaded calculation (safe)"),
            ('THREAD', "Thread Pool", "Parallel multi-threaded calculation (recommended, works on all OS)"),
            ('PROCESS', "Process Pool", "Parallel multi-process calculation (experimental, fastest on multi-core CPUs for heavy 100k+ vertex meshes)")
        ],
        default='THREAD'
    )

    bpy.types.Scene.mwc_use_custom_curve = bpy.props.BoolProperty(
        name="Use Custom Curve",
        default=False,
        description="Use custom Bezier curve editor for weight falloff instead of Wyvill formula",
        update=update_use_custom_curve
    )

    bpy.types.Scene.mwc_cache_exists = bpy.props.BoolProperty(
        name="Cache Exists",
        default=False
    )

    bpy.types.Scene.mwc_cache_count = bpy.props.IntProperty(
        name="Cache Count",
        default=0
    )

    bpy.types.Scene.mwc_cache_alpha = bpy.props.FloatProperty(
        name="Cache Alpha",
        default=0.0
    )

    bpy.types.Scene.mwc_use_joint_scaling = bpy.props.BoolProperty(
        name="Use Joint Scaling",
        default=False,
        description="Adapt metaball radius in joints and center of bones using armature joint positions"
    )

    bpy.types.Scene.mwc_armature = bpy.props.PointerProperty(
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'ARMATURE',
        name="Armature",
        description="Skeletal Armature object to retrieve joint positions from"
    )

    bpy.types.Scene.mwc_joint_scale = bpy.props.FloatProperty(
        name="Joint Scale Factor",
        default=0.5,
        min=0.1,
        max=1.0,
        description="Radius multiplier near joint connections"
    )

    bpy.types.Scene.mwc_middle_scale = bpy.props.FloatProperty(
        name="Middle Scale Factor",
        default=1.2,
        min=1.0,
        max=2.0,
        description="Radius multiplier in the middle of long bones"
    )

    bpy.types.Scene.mwc_use_thickness_scaling = bpy.props.BoolProperty(
        name="Use Thickness Scaling",
        default=False,
        description="Limit metaball radius to local mesh thickness via raycast to prevent leaks"
    )

    bpy.types.Scene.mwc_thickness_factor = bpy.props.FloatProperty(
        name="Thickness Factor",
        default=0.5,
        min=0.1,
        max=2.0,
        description="Clamping factor for thickness (radius = min(radius, thickness * factor))"
    )

    bpy.types.Scene.mwc_selected_mb_idx = bpy.props.IntProperty(
        name="Selected Index",
        default=-1,
        update=tag_redraw_all_views
    )
    bpy.types.Scene.mwc_selected_mb_x = bpy.props.FloatProperty(
        name="X",
        precision=4,
        update=tag_redraw_all_views
    )
    bpy.types.Scene.mwc_selected_mb_y = bpy.props.FloatProperty(
        name="Y",
        precision=4,
        update=tag_redraw_all_views
    )
    bpy.types.Scene.mwc_selected_mb_z = bpy.props.FloatProperty(
        name="Z",
        precision=4,
        update=tag_redraw_all_views
    )
    bpy.types.Scene.mwc_selected_mb_radius = bpy.props.FloatProperty(
        name="Radius",
        min=0.001,
        max=10.0,
        default=0.1,
        precision=4,
        update=tag_redraw_all_views
    )
    bpy.types.Scene.mwc_selected_mb_weights = bpy.props.CollectionProperty(
        type=MWC17_MBWeightItem
    )
    bpy.types.Scene.mwc_new_bone_name = bpy.props.StringProperty(
        name="New Bone Name",
        default=""
    )
    bpy.types.Scene.mwc_show_viewport_preview = bpy.props.BoolProperty(
        name="Show Preview",
        default=False,
        update=update_viewport_preview
    )
    bpy.types.Scene.mwc_color_by_active_bone = bpy.props.BoolProperty(
        name="Color by Active Bone",
        default=False,
        update=tag_redraw_all_views
    )

    register_draw_handler()
    load_cached_mbs_from_file()
    if HAS_BLENDER:
        bpy.app.handlers.depsgraph_update_post.append(mwc17_depsgraph_update)
        if _on_load_post not in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.append(_on_load_post)


def unregister():
    # Unregister translations safely
    try:
        bpy.app.translations.unregister(__name__)
    except Exception as e:
        print("MWC translation unregistration failed:", e)

    unregister_draw_handler()
    if HAS_BLENDER:
        if mwc17_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.remove(mwc17_depsgraph_update)
        if _on_load_post in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.remove(_on_load_post)

    visualization._cached_mbs = []
    visualization._scene_mbs = []
    visualization._scene_mbs_dirty = True

    # Remove properties safely
    for prop in [
        "mwc_source_obj",
        "mwc_alpha", "mwc_n", "mwc_q", "mwc_tau", "mwc_r_falloff_coeff",
        "mwc_subdivision_k", "mwc_merge_close", "mwc_merge_factor",
        "mwc_creation_type", "mwc_symmetry", "mwc_target_obj",
        "mwc_use_normal_filter", "mwc_normal_filter_p",
        "mwc_use_smoothing", "mwc_smoothing_strength", "mwc_smoothing_iterations",
        "mwc_use_geodesic", "mwc_geodesic_mode", "mwc_use_custom_curve",
        "mwc_cache_exists", "mwc_cache_count", "mwc_cache_alpha",
        "mwc_use_joint_scaling", "mwc_armature", "mwc_joint_scale", "mwc_middle_scale",
        "mwc_use_thickness_scaling", "mwc_thickness_factor",
        "mwc_selected_mb_idx", "mwc_selected_mb_x", "mwc_selected_mb_y", "mwc_selected_mb_z",
        "mwc_selected_mb_radius", "mwc_selected_mb_weights", "mwc_new_bone_name",
        "mwc_show_viewport_preview", "mwc_color_by_active_bone"
    ]:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    # Clean hidden curve mapping node group
    clean_curve_mapping_node()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
