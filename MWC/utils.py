import os
import tempfile
import numpy as np

try:
    import bpy
    import mathutils
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False

# Import translation system helper
from .translation import t

def get_cache_filepath():
    return os.path.join(tempfile.gettempdir(), "mwc_metaballs_cache.npz")

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
        try:
            arm_obj = bpy.context.scene.mwc_armature
        except Exception:
            try:
                arm_obj = bpy.data.scenes[0].mwc_armature
            except Exception:
                pass
        
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
        
    np.savez(filepath,
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

def load_mbs_from_npz():
    filepath = get_cache_filepath()
    if not os.path.exists(filepath):
        return None
    try:
        data = np.load(filepath, allow_pickle=True)
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
        
        sym_map_inv = {1: 'L', 2: 'R', 3: 'C'}
        
        arm_obj = None
        if HAS_BLENDER:
            try:
                arm_obj = bpy.context.scene.mwc_armature
            except Exception:
                try:
                    arm_obj = bpy.data.scenes[0].mwc_armature
                except Exception:
                    pass
            
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
            
        metadata = {
            'alpha': float(data['alpha']) if 'alpha' in data else 0.70,
            'n': int(data['n']) if 'n' in data else 2,
            'q': float(data['q']) if 'q' in data else 1.5,
            'tau': float(data['tau']) if 'tau' in data else 0.001,
            'r_falloff_coeff': float(data['r_falloff_coeff']) if 'r_falloff_coeff' in data else 2.5
        }
        return mbs, metadata
    except Exception as e:
        print("Error loading MWC cache NPZ:", e)
        return None

def project_point_on_segment(P, A, B):
    """
    Project point P onto segment AB. Returns projection point and factor t in [0, 1].
    """
    AB = B - A
    ab_len_sq = AB.length_squared
    if ab_len_sq < 1e-8:
        return A, 0.0
    t = (P - A).dot(AB) / ab_len_sq
    t = max(0.0, min(1.0, t))
    proj = A + t * AB
    return proj, t

def get_joint_aware_multiplier(P, armature_obj, joint_scale=0.5, middle_scale=1.2):
    if not armature_obj:
        return 1.0
        
    min_dist_to_joint = float('inf')
    found_any_bone = False
    
    arm_matrix = armature_obj.matrix_world
    for bone in armature_obj.data.bones:
        # Get head and tail in armature space, then convert to world
        head_w = arm_matrix @ bone.head
        tail_w = arm_matrix @ bone.tail
        
        # Project metaball point P onto bone segment
        proj, t_val = project_point_on_segment(P, head_w, tail_w)
        
        dist_to_head = (P - head_w).length
        dist_to_tail = (P - tail_w).length
        
        min_dist_to_joint = min(min_dist_to_joint, dist_to_head, dist_to_tail)
        found_any_bone = True
        
    if not found_any_bone:
        return 1.0
        
    # Standard Bone Joint Radius Interpolator:
    # Scale decreases to joint_scale near joints and increases to middle_scale in the middle
    # Based on a characteristic joint influence range (default: 0.1m)
    joint_influence_range = 0.1
    if min_dist_to_joint < joint_influence_range:
        factor = min_dist_to_joint / joint_influence_range
        # Smooth interpolation
        s_factor = 3 * (factor ** 2) - 2 * (factor ** 3)
        return joint_scale + s_factor * (middle_scale - joint_scale)
        
    return middle_scale

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
    if name.endswith(".L"):
        return name[:-2] + ".R"
    elif name.endswith(".R"):
        return name[:-2] + ".L"
    elif name.endswith("_L"):
        return name[:-2] + "_R"
    elif name.endswith("_R"):
        return name[:-2] + "_L"
    elif ".L." in name:
        return name.replace(".L.", ".R.")
    elif ".R." in name:
        return name.replace(".R.", ".L.")
    elif "_L_" in name:
        return name.replace("_L_", "_R_")
    elif "_R_" in name:
        return name.replace("_R_", "_L_")
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
