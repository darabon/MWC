import numpy as np
import math


try:
    import bpy
    import bmesh
    import mathutils
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False

# Import utilities
from .utils import (
    get_vertex_weights, 
    get_joint_aware_multiplier, 
    is_bone_central, 
    swap_bone_side
)

def count_mesh_islands(obj):
    """
    Count disconnected topological components (mesh islands) using DFS.
    """
    if not obj or obj.type != 'MESH' or not HAS_BLENDER:
        return 0
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    
    visited = set()
    islands_count = 0
    
    for v in bm.verts:
        if v not in visited:
            islands_count += 1
            stack = [v]
            visited.add(v)
            while stack:
                curr = stack.pop()
                for edge in curr.link_edges:
                    other = edge.other_vert(curr)
                    if other not in visited:
                        visited.add(other)
                        stack.append(other)
    bm.free()
    return islands_count

def merge_close_metaballs(mbs, merge_factor):
    """
    Groups and merges metaballs with same family_id and dominant weight bone
    if they are within merge_factor * (R1 + R2) distance.
    """
    if len(mbs) < 2:
        return mbs
        
    # Group metaballs by (family_id, dominant_bone)
    groups = {}
    for mb in mbs:
        fam = mb['family_id']
        w = mb['weights']
        if not w:
            dom_bone = None
        else:
            dom_bone = max(w, key=w.get)
            
        key = (fam, dom_bone)
        if key not in groups:
            groups[key] = []
        groups[key].append(mb)
        
    merged_mbs = []
    
    for key, group_mbs in groups.items():
        fam, dom_bone = key
        # If no dominant bone or group has only 1 metaball, keep as is
        if dom_bone is None or len(group_mbs) < 2:
            merged_mbs.extend(group_mbs)
            continue
            
        # Greedy clustering
        unvisited = list(group_mbs)
        while unvisited:
            seed = unvisited.pop(0)
            cluster = [seed]
            
            i = 0
            while i < len(cluster):
                curr = cluster[i]
                curr_co = np.array(curr['co'])
                curr_r = curr['radius']
                
                j = 0
                while j < len(unvisited):
                    other = unvisited[j]
                    other_co = np.array(other['co'])
                    other_r = other['radius']
                    
                    dist = np.linalg.norm(curr_co - other_co)
                    if dist < merge_factor * (curr_r + other_r):
                        cluster.append(other)
                        unvisited.pop(j)
                    else:
                        j += 1
                i += 1
                
            if len(cluster) == 1:
                merged_mbs.append(cluster[0])
            else:
                cos = np.array([mb['co'] for mb in cluster])
                radii = np.array([mb['radius'] for mb in cluster])
                
                sum_r = np.sum(radii)
                new_co = np.sum(cos * radii[:, np.newaxis], axis=0) / (sum_r if sum_r > 0 else 1.0)
                
                nos = np.array([mb['normal'] for mb in cluster])
                new_no = np.sum(nos, axis=0)
                no_len = np.linalg.norm(new_no)
                if no_len > 1e-6:
                    new_no /= no_len
                else:
                    new_no = np.array([0.0, 0.0, 1.0])
                    
                dists_to_new = np.linalg.norm(cos - new_co, axis=1)
                new_r = float(np.max(dists_to_new + radii))
                
                new_weights = {}
                all_bones = set()
                for mb in cluster:
                    all_bones.update(mb['weights'].keys())
                for b in all_bones:
                    val = sum(mb['radius'] * mb['weights'].get(b, 0.0) for mb in cluster) / (sum_r if sum_r > 0 else 1.0)
                    if val >= 0.001:
                        new_weights[b] = val
                        
                sum_w = sum(new_weights.values())
                if sum_w > 0:
                    new_weights = {k: v / sum_w for k, v in new_weights.items()}
                else:
                    new_weights = {}
                    
                merged_mbs.append({
                    'co': new_co.tolist(),
                    'radius': new_r,
                    'weights': new_weights,
                    'normal': new_no.tolist(),
                    'family_id': fam,
                    'is_virtual': False,
                    'vertex_index': -1,
                    'symmetry_class': cluster[0].get('symmetry_class', 'L')
                })
                
    return merged_mbs

def calculate_mwc_metaballs(obj, alpha, creation_type, k_coeff=2.0, merge_close=True, merge_factor=0.5, use_symmetry=False,
                            use_joint_scaling=False, armature_obj=None, joint_scale=0.5, middle_scale=1.2,
                            use_thickness_scaling=False, thickness_factor=0.5):
    """
    Performs initial calculations, topological island grouping, virtual edge subdivision,
    and returns lists of original and virtual metaball data structures.
    """
    matrix_world = obj.matrix_world
    
    # Get evaluated mesh (deforms applied)
    eval_obj = obj
    has_eval_mesh = False
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        has_eval_mesh = True
    except Exception as e:
        print("Failed to get evaluated mesh, falling back to base mesh:", e)
        mesh = obj.data
        
    bm = bmesh.new()
    bm.from_mesh(mesh)
    
    if has_eval_mesh:
        eval_obj.to_mesh_clear()
        
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    
    # 1. Group vertices by topological islands (connected components)
    visited = set()
    islands = []
    
    for v in bm.verts:
        if v not in visited:
            island = []
            stack = [v]
            visited.add(v)
            while stack:
                curr = stack.pop()
                island.append(curr.index)
                for edge in curr.link_edges:
                    other = edge.other_vert(curr)
                    if other not in visited:
                        visited.add(other)
                        stack.append(other)
            islands.append(island)
            
    # Map vertex index to family ID (island index)
    vert_family = {}
    if creation_type == 'SINGLE':
        for v in bm.verts:
            vert_family[v.index] = 0
    else:
        for island_idx, island in enumerate(islands):
            for v_idx in island:
                vert_family[v_idx] = island_idx
                
    # 2. Compute initial global edge lengths
    edge_lengths_global = {}
    for e in bm.edges:
        p1 = matrix_world @ e.verts[0].co
        p2 = matrix_world @ e.verts[1].co
        edge_lengths_global[e.index] = (p1 - p2).length
        
    # Map vertex index to its incident edges and their lengths
    vert_edge_lengths = {v.index: [] for v in bm.verts}
    for e in bm.edges:
        l = edge_lengths_global[e.index]
        vert_edge_lengths[e.verts[0].index].append((e.index, l))
        
    # 3. Create virtual metaballs along long subdivided edges
    virtual_metaballs = []
    subdivided_edge_lengths = {}
    
    for e in bm.edges:
        L = edge_lengths_global[e.index]
        
        v1 = e.verts[0]
        v2 = e.verts[1]
        
        r1_base = alpha * (sum(edge_lengths_global[ed.index] for ed in v1.link_edges) / max(len(v1.link_edges), 1))
        r2_base = alpha * (sum(edge_lengths_global[ed.index] for ed in v2.link_edges) / max(len(v2.link_edges), 1))
        
        # Adjust with joint-scaling if enabled
        if use_joint_scaling and armature_obj:
            r1_base *= get_joint_aware_multiplier(matrix_world @ v1.co, armature_obj, joint_scale, middle_scale)
            r2_base *= get_joint_aware_multiplier(matrix_world @ v2.co, armature_obj, joint_scale, middle_scale)
            
        r_max = max(r1_base, r2_base)
        
        # Local edge subdivision threshold
        if L > k_coeff * r_max:
            # Determine division count
            divs = int(math.ceil(L / (k_coeff * r_max)))
            if divs > 1:
                L_sub = L / divs
                subdivided_edge_lengths[(v1.index, e.index)] = L_sub
                subdivided_edge_lengths[(v2.index, e.index)] = L_sub
                
                # Fetch global space endpoints
                p1 = np.array(matrix_world @ v1.co)
                p2 = np.array(matrix_world @ v2.co)
                
                n1 = np.array(v1.normal)
                n2 = np.array(v2.normal)
                
                w1 = get_vertex_weights(obj, v1.index)
                w2 = get_vertex_weights(obj, v2.index)
                
                fam_id = vert_family[v1.index]
                
                for i in range(1, divs):
                    t = i / divs
                    co_new = (1 - t) * p1 + t * p2
                    no_new = (1 - t) * n1 + t * n2
                    no_new_len = np.linalg.norm(no_new)
                    if no_new_len > 1e-6:
                        no_new /= no_new_len
                        
                    # Linear Weight Blending
                    groups = set(w1.keys()).union(w2.keys())
                    w_new = {}
                    for g in groups:
                        val1 = w1.get(g, 0.0)
                        val2 = w2.get(g, 0.0)
                        val = (1 - t) * val1 + t * val2
                        if val >= 0.001:
                            w_new[g] = float(val)
                            
                    r_new = alpha * L_sub
                    
                    # 1. Joint-aware scaling
                    if use_joint_scaling and armature_obj:
                        r_new *= get_joint_aware_multiplier(mathutils.Vector(co_new), armature_obj, joint_scale, middle_scale)
                        
                    # 2. Thickness-aware scaling
                    if use_thickness_scaling:
                        inv_mw = matrix_world.inverted()
                        co_local = inv_mw @ mathutils.Vector(co_new)
                        no_local = inv_mw.to_3x3() @ mathutils.Vector(no_new)
                        no_local.normalize()
                        
                        result, hit_loc, _, _ = eval_obj.ray_cast(co_local - 1e-4 * no_local, -no_local)
                        if result:
                            thickness_world = (co_local - hit_loc).length * matrix_world.to_scale().x
                            r_new = min(r_new, thickness_world * thickness_factor)
                            
                    virtual_metaballs.append({
                        'co': co_new.tolist(),
                        'radius': r_new,
                        'weights': w_new,
                        'normal': no_new.tolist(),
                        'family_id': fam_id,
                        'is_virtual': True
                    })
                    
    # 4. Recalculate original radii using updated edge lengths
    new_radii = {}
    for v in bm.verts:
        lengths = []
        for e_idx, original_len in vert_edge_lengths[v.index]:
            if (v.index, e_idx) in subdivided_edge_lengths:
                lengths.append(subdivided_edge_lengths[(v.index, e_idx)])
            else:
                lengths.append(original_len)
        if len(lengths) > 0:
            new_radii[v.index] = alpha * (sum(lengths) / len(lengths))
        else:
            new_radii[v.index] = 0.0
            
    # 5. Compile original metaballs list
    original_metaballs = []
    normal_matrix = matrix_world.to_3x3()
    for v in bm.verts:
        co = matrix_world @ v.co
        no = (normal_matrix @ v.normal).normalized()
        w = get_vertex_weights(obj, v.index)
        
        r_final = new_radii[v.index]
        
        # 1. Joint-aware scaling
        if use_joint_scaling and armature_obj:
            r_final *= get_joint_aware_multiplier(co, armature_obj, joint_scale, middle_scale)
            
        # 2. Thickness-aware scaling
        if use_thickness_scaling:
            result, hit_loc, _, _ = eval_obj.ray_cast(v.co - 1e-4 * v.normal, -v.normal)
            if result:
                thickness_world = (v.co - hit_loc).length * matrix_world.to_scale().x
                r_final = min(r_final, thickness_world * thickness_factor)
                
        original_metaballs.append({
            'co': list(co),
            'radius': r_final,
            'weights': w,
            'normal': list(no),
            'family_id': vert_family[v.index],
            'is_virtual': False,
            'vertex_index': v.index
        })
        
    bm.free()
    
    if use_symmetry:
        left_mbs = []
        inv_mw = matrix_world.inverted()
        inv_mw_3x3 = inv_mw.to_3x3()
        mw_3x3 = matrix_world.to_3x3()
        
        all_raw = original_metaballs + virtual_metaballs
        
        for mb in all_raw:
            w = mb.get('weights', {})
            dom_bone = max(w, key=w.get) if w else None
            
            if is_bone_central(dom_bone):
                mb['symmetry_class'] = 'C'
                left_mbs.append(mb)
            else:
                co_local = inv_mw @ mathutils.Vector(mb['co'])
                # Keep Left side (local X >= -1e-4)
                if co_local.x >= -1e-4:
                    mb['symmetry_class'] = 'L'
                    left_mbs.append(mb)
                    
                    # Mirror to Right if not central (local X > 1e-3)
                    if co_local.x > 1e-3:
                        co_loc_mir = mathutils.Vector((-co_local.x, co_local.y, co_local.z))
                        co_glob_mir = matrix_world @ co_loc_mir
                        
                        no_local = (inv_mw_3x3 @ mathutils.Vector(mb['normal'])).normalized()
                        no_loc_mir = mathutils.Vector((-no_local.x, no_local.y, no_local.z)).normalized()
                        no_glob_mir = (mw_3x3 @ no_loc_mir).normalized()
                        
                        # Mirror weights (swap bone side suffix)
                        mirrored_weights = {}
                        for bone_name, weight in mb['weights'].items():
                            mirrored_weights[swap_bone_side(bone_name)] = weight
                            
                        mirrored_mb = {
                            'co': list(co_glob_mir),
                            'radius': mb['radius'],
                            'weights': mirrored_weights,
                            'normal': list(no_glob_mir),
                            'family_id': mb['family_id'],
                            'is_virtual': mb['is_virtual'],
                            'symmetry_class': 'R'
                        }
                        left_mbs.append(mirrored_mb)
        original_metaballs = left_mbs
        virtual_metaballs = []
    else:
        # Standard run: assign symmetry class based on dominant bone and coordinates
        inv_mw = matrix_world.inverted()
        for mb in original_metaballs:
            w = mb.get('weights', {})
            dom_bone = max(w, key=w.get) if w else None
            if is_bone_central(dom_bone):
                mb['symmetry_class'] = 'C'
            else:
                co_local = inv_mw @ mathutils.Vector(mb['co'])
                mb['symmetry_class'] = 'L' if co_local.x >= 0.0 else 'R'
        for mb in virtual_metaballs:
            w = mb.get('weights', {})
            dom_bone = max(w, key=w.get) if w else None
            if is_bone_central(dom_bone):
                mb['symmetry_class'] = 'C'
            else:
                co_local = inv_mw @ mathutils.Vector(mb['co'])
                mb['symmetry_class'] = 'L' if co_local.x >= 0.0 else 'R'
            
    if merge_close:
        all_mbs = original_metaballs + virtual_metaballs
        merged_mbs = merge_close_metaballs(all_mbs, merge_factor)
        return merged_mbs, []
    else:
        return original_metaballs, virtual_metaballs
