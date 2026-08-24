import os
import hashlib
import tempfile
import numpy as np

try:
    import bpy
    import mathutils
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False


# Cache filename policy: per-blend isolation with a shared fallback.
#
# The v1.0 cache was a single global file in the temp dir -- opening two
# characters in two Blender instances silently overwrote each other's cache.
# v1.2 derives a suffix from the absolute .blend path (stable across sessions,
# filesystem moves of the temp dir, and Blender restarts) so every blend gets
# its own cache namespace, while unsaved/unknown files keep sharing the
# generic "default" cache.
_CACHE_DEFAULT_NAME = "mwc_metaballs_cache"
_blend_cache_suffix = None


def _compute_cache_suffix():
    if not HAS_BLENDER:
        return None
    try:
        filepath = bpy.data.filepath
    except Exception:
        return None
    if not filepath:
        return None
    norm = os.path.normcase(os.path.abspath(filepath))
    digest = hashlib.md5(norm.encode("utf-8", "surrogateescape")).hexdigest()[:12]
    base = os.path.splitext(os.path.basename(filepath))[0][:48] or "untitled"
    safe_base = "".join(c if c.isalnum() or c in "-_" else "_" for c in base)
    return f"{safe_base}_{digest}"


def get_cache_filepath():
    global _blend_cache_suffix
    if _blend_cache_suffix is None:
        _blend_cache_suffix = _compute_cache_suffix()
    name = (_CACHE_DEFAULT_NAME + "@" + _blend_cache_suffix) if _blend_cache_suffix else _CACHE_DEFAULT_NAME
    return os.path.join(tempfile.gettempdir(), name + ".npz")


def reset_cache_location():
    """Re-derive the per-blend cache path (call after save-as / load)."""
    global _blend_cache_suffix
    _blend_cache_suffix = _compute_cache_suffix()

def get_vertex_weights(obj, v_idx):
    """
    Get vertex weights for a specific vertex in the source mesh.
    """
    weights = {}
    for vg in obj.vertex_groups:
        try:
            w_val = vg.weight(v_idx)
            if w_val >= 0.001:
                weights[vg.name] = float(w_val)
        except RuntimeError:
            # Vertex is not in this vertex group
            pass
    return weights

def build_vertex_weights_map(obj):
    """
    Build {vertex_index: {bone_name: weight}} for the whole mesh in a single
    pass over its vertices. Orders of magnitude faster than calling
    get_vertex_weights() per vertex (which iterates every vertex group and
    raises/catches RuntimeError for each empty one).
    """
    weights_map = {}
    vgs = list(obj.vertex_groups)
    if not vgs:
        return weights_map
    vg_names = [vg.name for vg in vgs]
    n_vg = len(vgs)
    for v in obj.data.vertices:
        entries = {}
        for ge in v.groups:
            g = ge.group
            if g < n_vg:
                w_val = ge.weight
                if w_val >= 0.001:
                    entries[vg_names[g]] = float(w_val)
        if entries:
            weights_map[v.index] = entries
    return weights_map

def save_mbs_to_npz(mbs, alpha, n, q, tau, r_falloff_coeff):
    filepath = get_cache_filepath()
    bone_names_set = set()
    for mb in mbs:
        bone_names_set.update(mb['weights'].keys())
    bone_names = sorted(list(bone_names_set))
    G = len(bone_names)
    M = len(mbs)
    
    co = np.array([mb['co'] for mb in mbs], dtype=np.float32)
    radius = np.array([mb['radius'] for mb in mbs], dtype=np.float32)
    normal = np.array([mb['normal'] for mb in mbs], dtype=np.float32)
    family_id = np.array([mb['family_id'] for mb in mbs], dtype=np.int32)
    
    sym_map = {'L': 1, 'R': 2, 'C': 3}
    symmetry_class = np.array([sym_map.get(mb.get('symmetry_class', 'L'), 1) for mb in mbs], dtype=np.int8)
    
    weights = np.zeros((M, G), dtype=np.float32)
    bone_to_idx = {name: idx for idx, name in enumerate(bone_names)}
    for j, mb in enumerate(mbs):
        for b_name, w_val in mb['weights'].items():
            weights[j, bone_to_idx[b_name]] = w_val
            
    # Calculate local coordinates relative to dominant bones if armature exists
    co_local_list = []
    parent_bones_list = []
    
    arm_obj = None
    if HAS_BLENDER:
        arm_obj = get_armature_object()
        
    for mb in mbs:
        co_world = mathutils.Vector(mb['co'])
        parent_bone = ""
        co_local = co_world
        
        if arm_obj and arm_obj.type == 'ARMATURE' and mb['weights']:
            dominant_bone = max(mb['weights'].items(), key=lambda item: item[1])[0]
            if dominant_bone in arm_obj.pose.bones:
                pose_bone = arm_obj.pose.bones[dominant_bone]
                co_arm = arm_obj.matrix_world.inverted() @ co_world
                co_local = pose_bone.matrix.inverted() @ co_arm
                parent_bone = dominant_bone
                
        co_local_list.append(list(co_local))
        parent_bones_list.append(parent_bone)
        
    # Atomic write: save to a temp file first, then os.replace() it into
    # place. A crash mid-save can never corrupt the existing cache.
    tmp_path = filepath + ".tmp%d" % os.getpid()
    try:
        with open(tmp_path, 'wb') as fh:
            np.savez(fh,
                     co=co,
                     radius=radius,
                     normal=normal,
                     family_id=family_id,
                     symmetry_class=symmetry_class,
                     bone_names=np.array(bone_names),
                     weights=weights,
                     alpha=alpha,
                     n=n,
                     q=q,
                     tau=tau,
                     r_falloff_coeff=r_falloff_coeff,
                     co_local=np.array(co_local_list, dtype=np.float32),
                     parent_bone=np.array(parent_bones_list))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, filepath)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def load_mbs_from_npz():
    filepath = get_cache_filepath()
    if not os.path.exists(filepath):
        return None
    try:
        with np.load(filepath, allow_pickle=True) as data:
            co = data['co']
            radius = data['radius']
            normal = data['normal']
            family_id = data['family_id']
            symmetry_class_int = data['symmetry_class']
            bone_names = [b.decode('utf-8') if isinstance(b, bytes) else str(b) for b in data['bone_names'].tolist()]
            weights = data['weights']

            co_local = data['co_local'] if 'co_local' in data.files else None
            parent_bone = data['parent_bone'].tolist() if 'parent_bone' in data.files else None
            if parent_bone is not None:
                parent_bone = [b.decode('utf-8') if isinstance(b, bytes) else str(b) for b in parent_bone]

            metadata = {
                'alpha': float(data['alpha']) if 'alpha' in data else 0.70,
                'n': int(data['n']) if 'n' in data else 2,
                'q': float(data['q']) if 'q' in data else 1.5,
                'tau': float(data['tau']) if 'tau' in data else 0.001,
                'r_falloff_coeff': float(data['r_falloff_coeff']) if 'r_falloff_coeff' in data else 2.5
            }
        
        sym_map_inv = {1: 'L', 2: 'R', 3: 'C'}
        
        arm_obj = None
        if HAS_BLENDER:
            arm_obj = get_armature_object()
            
        mbs = []
        M = len(co)
        for j in range(M):
            mb_weights = {}
            for g_idx, b_name in enumerate(bone_names):
                val = float(weights[j, g_idx])
                if val >= 0.001:
                    mb_weights[b_name] = val
                    
            mb_co = co[j].tolist()
            if co_local is not None and parent_bone is not None:
                p_bone_name = parent_bone[j]
                if p_bone_name:
                    if arm_obj and arm_obj.type == 'ARMATURE' and p_bone_name in arm_obj.pose.bones:
                        pose_bone = arm_obj.pose.bones[p_bone_name]
                        local_vec = mathutils.Vector(co_local[j])
                        co_world = arm_obj.matrix_world @ (pose_bone.matrix @ local_vec)
                        mb_co = list(co_world)
                        
            mbs.append({
                'co': mb_co,
                'radius': float(radius[j]),
                'normal': normal[j].tolist(),
                'weights': mb_weights,
                'family_id': int(family_id[j]),
                'symmetry_class': sym_map_inv.get(symmetry_class_int[j], 'L'),
                'co_local': co_local[j].tolist() if co_local is not None else None,
                'parent_bone': parent_bone[j] if parent_bone is not None else ""
            })

        return mbs, metadata
    except Exception as e:
        print("Error loading MWC cache NPZ:", e)
        # A corrupt/unreadable cache should not keep poisoning the session:
        # remove it so the UI falls back to the clean "Empty" state.
        try:
            os.remove(filepath)
            print("Removed corrupt MWC cache file.")
        except OSError:
            pass
        return None

JOINT_INFLUENCE_RANGE = 0.1

def precompute_bone_joints(armature_obj):
    """
    Collect all bone head/tail joint positions in world space ONCE.
    Returns an (2*B, 3) float64 array, or None if unusable.
    """
    if not armature_obj or getattr(armature_obj, 'type', None) != 'ARMATURE':
        return None
    arm_matrix = armature_obj.matrix_world
    joints = []
    for bone in armature_obj.data.bones:
        joints.append(tuple(arm_matrix @ bone.head))
        joints.append(tuple(arm_matrix @ bone.tail))
    if not joints:
        return None
    return np.array(joints, dtype=np.float64)

def joint_aware_multipliers(points, joints, joint_scale=0.5, middle_scale=1.2):
    """
    Vectorized per-point radius multipliers.

    points: (N, 3) float array in world space.
    joints: (J, 3) float array of bone head/tail positions (see
            precompute_bone_joints).

    Returns an (N,) float64 array. Mathematically identical to calling
    get_joint_aware_multiplier() per point, but computes the minimum
    distance-to-joint for all points in bulk instead of looping over every
    bone for every point in Python.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if joints is None or len(joints) == 0 or n == 0:
        return np.ones(n, dtype=np.float64)

    min_dist = np.full(n, np.inf, dtype=np.float64)
    for j in range(len(joints)):
        d = np.sqrt(((pts - joints[j]) ** 2).sum(axis=1))
        np.minimum(min_dist, d, out=min_dist)

    factor = np.minimum(min_dist / JOINT_INFLUENCE_RANGE, 1.0)
    s_factor = factor * factor * (3.0 - 2.0 * factor)
    return joint_scale + s_factor * (middle_scale - joint_scale)

def get_joint_aware_multiplier(P, armature_obj, joint_scale=0.5, middle_scale=1.2):
    """Scalar convenience wrapper around joint_aware_multipliers()."""
    if not armature_obj:
        return 1.0
    joints = precompute_bone_joints(armature_obj)
    if joints is None:
        return 1.0
    return float(joint_aware_multipliers(
        np.array([tuple(P)], dtype=np.float64), joints, joint_scale, middle_scale
    )[0])

def is_bone_central(name):
    # Detect typical central bone naming conventions
    lower = name.lower()
    for center_word in ["spine", "chest", "neck", "head", "hips", "pelvis", "root"]:
        if center_word in lower:
            # Verify it's not marked with side suffixes
            if not (lower.endswith(".l") or lower.endswith(".r") or lower.endswith("_l") or lower.endswith("_r")):
                return True
    return False

def swap_bone_side(name):
    # Case-insensitive suffix detection so rigs named with lowercase
    # suffixes (.l / _r) are mirrored correctly too; the original case of
    # the flipped letter is preserved.
    if not name:
        return name
    lower = name.lower()
    for sep in ('.', '_'):
        if lower.endswith(sep + 'l'):
            return name[:-1] + ('R' if name[-1].isupper() else 'r')
        if lower.endswith(sep + 'r'):
            return name[:-1] + ('L' if name[-1].isupper() else 'l')
    for sep in ('.', '_'):
        token_l = sep + 'l' + sep
        idx = lower.find(token_l)
        if idx != -1:
            ch = name[idx + 1]
            return name[:idx + 1] + ('R' if ch.isupper() else 'r') + name[idx + 2:]
        token_r = sep + 'r' + sep
        idx = lower.find(token_r)
        if idx != -1:
            ch = name[idx + 1]
            return name[:idx + 1] + ('L' if ch.isupper() else 'l') + name[idx + 2:]
    return name

def segment_intersects_tri(p1, p2, v1, v2, v3):
    """
    Moller-Trumbore ray-triangle intersection algorithm adapted for segment.
    """
    edge1 = v2 - v1
    edge2 = v3 - v1
    pvec = (p2 - p1).cross(edge2)
    det = edge1.dot(pvec)
    if abs(det) < 1e-8:
        return False
    inv_det = 1.0 / det
    tvec = p1 - v1
    u = tvec.dot(pvec) * inv_det
    if u < 0.0 or u > 1.0:
        return False
    qvec = tvec.cross(edge1)
    v = (p2 - p1).dot(qvec) * inv_det
    if v < 0.0 or u + v > 1.0:
        return False
    t = edge2.dot(qvec) * inv_det
    if 0.0 <= t <= 1.0:
        return True
    return False

def triangles_intersect(ta, tb):
    # Returns True if triangle ta intersects triangle tb (checking segment intersections)
    # ta, tb are tuples of 3 mathutils.Vector points
    for i in range(3):
        p1, p2 = ta[i], ta[(i+1)%3]
        if segment_intersects_tri(p1, p2, tb[0], tb[1], tb[2]):
            return True
    for j in range(3):
        p1, p2 = tb[j], tb[(j+1)%3]
        if segment_intersects_tri(p1, p2, ta[0], ta[1], ta[2]):
            return True
    return False

def get_curve_mapping_node(create=True):
    if not HAS_BLENDER:
        return None
    tree_name = ".hidden_mwc_curve_tree"
    if tree_name not in bpy.data.node_groups:
        if not create:
            return None
        ng = bpy.data.node_groups.new(tree_name, 'ShaderNodeTree')
        ng.use_fake_user = True
        node = ng.nodes.new('ShaderNodeRGBCurve')
        node.name = "CurveNode"
        curve = node.mapping.curves[3]
        if len(curve.points) >= 2:
            curve.points[0].location = (0.0, 1.0)
            curve.points[1].location = (1.0, 0.0)
        node.mapping.initialize()
    else:
        ng = bpy.data.node_groups[tree_name]
        if "CurveNode" not in ng.nodes:
            if not create:
                return None
            node = ng.nodes.new('ShaderNodeRGBCurve')
            node.name = "CurveNode"
            curve = node.mapping.curves[3]
            if len(curve.points) >= 2:
                curve.points[0].location = (0.0, 1.0)
                curve.points[1].location = (1.0, 0.0)
            node.mapping.initialize()
        else:
            node = ng.nodes["CurveNode"]
    return node

def clean_curve_mapping_node():
    if not HAS_BLENDER:
        return
    tree_name = ".hidden_mwc_curve_tree"
    if tree_name in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[tree_name])


def get_armature_object(scene=None, context=None):
    if not HAS_BLENDER:
        return None
        
    # Resolve scene if not provided or restricted
    if scene is None:
        try:
            scene = bpy.context.scene
        except AttributeError:
            pass
            
    if scene is None:
        try:
            if len(bpy.data.scenes) > 0:
                scene = bpy.data.scenes[0]
        except Exception:
            pass
            
    if scene is None:
        # Fallback to bpy.data.objects if scene is completely unavailable
        try:
            for obj in bpy.data.objects:
                if obj.type == 'ARMATURE':
                    return obj
        except Exception:
            pass
        return None

    # 1. Try specified scene armature
    try:
        arm_obj = getattr(scene, "mwc_armature", None)
        if arm_obj and arm_obj.type == 'ARMATURE':
            return arm_obj
    except Exception:
        pass
        
    # 2. Try source object's armature modifier
    try:
        src_obj = getattr(scene, "mwc_source_obj", None)
        if src_obj and src_obj.type == 'MESH':
            for mod in src_obj.modifiers:
                if mod.type == 'ARMATURE' and mod.object:
                    return mod.object
    except Exception:
        pass
                
    # 3. Try active object's armature modifier
    if context:
        try:
            act_obj = context.active_object
            if act_obj and act_obj.type == 'MESH':
                for mod in act_obj.modifiers:
                    if mod.type == 'ARMATURE' and mod.object:
                        return mod.object
        except Exception:
            pass
                    
    # 4. Fallback to any armature in the scene
    try:
        for obj in scene.objects:
            if obj.type == 'ARMATURE':
                return obj
    except Exception:
        pass
            
    # 5. Last fallback to bpy.data.objects
    try:
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE':
                return obj
    except Exception:
        pass
            
    return None
