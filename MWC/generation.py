import numpy as np
import math


try:
    import bpy
    import bmesh
    import mathutils
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False

# Optional SciPy acceleration for the connected-components pass.
try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _scipy_cc
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# Import utilities
from .utils import (
    build_vertex_weights_map,
    precompute_bone_joints,
    joint_aware_multipliers,
    is_bone_central,
    swap_bone_side
)


def _topological_islands(V_count, edge_vert_idx):
    """
    Connected components of the vertex graph in O(V + E).

    Uses scipy's C implementation when available; otherwise an iterative
    numpy/union-find-free BFS over a CSR-style adjacency built with bincount +
    argsort (no per-vertex Python sets). Returns an int array of shape (V,) --
    the component label of every vertex.
    """
    if V_count == 0:
        return np.zeros(0, dtype=np.int64)
    if len(edge_vert_idx) == 0:
        return np.arange(V_count, dtype=np.int64)

    if HAS_SCIPY:
        rows = np.concatenate([edge_vert_idx[:, 0], edge_vert_idx[:, 1]])
        cols = np.concatenate([edge_vert_idx[:, 1], edge_vert_idx[:, 0]])
        graph = coo_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(V_count, V_count), dtype=np.int8)
        _n, labels = _scipy_cc(graph, directed=False)
        return labels.astype(np.int64)

    # Fallback: build CSR adjacency and run an iterative stack-based flood fill.
    src = edge_vert_idx[:, 0].astype(np.int64)
    dst = edge_vert_idx[:, 1].astype(np.int64)
    both_src = np.concatenate([src, dst])
    both_dst = np.concatenate([dst, src])
    order = np.argsort(both_src, kind='stable')
    nbrs_sorted = both_dst[order]
    starts = np.searchsorted(both_src[order], np.arange(V_count), side='left')
    ends = np.searchsorted(both_src[order], np.arange(V_count), side='right')

    labels = np.full(V_count, -1, dtype=np.int64)
    stack = []
    label = 0
    for seed in range(V_count):
        if labels[seed] != -1:
            continue
        labels[seed] = label
        stack.append(seed)
        while stack:
            u = stack.pop()
            for v in nbrs_sorted[starts[u]:ends[u]]:
                if labels[v] == -1:
                    labels[v] = label
                    stack.append(v)
        label += 1
    return labels

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
            
        # Greedy clustering -- all pairwise distances are computed once in a
        # single vectorized pass instead of np.array allocations per pair.
        n_g = len(group_mbs)
        cos_arr = np.array([mb['co'] for mb in group_mbs], dtype=np.float64)
        rad_arr = np.array([mb['radius'] for mb in group_mbs], dtype=np.float64)

        # Threshold matrix: dist < merge_factor * (r_i + r_j)
        diff = cos_arr[:, np.newaxis, :] - cos_arr[np.newaxis, :, :]
        dist_m = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
        thresh_m = merge_factor * (rad_arr[:, np.newaxis] + rad_arr[np.newaxis, :])
        adj = dist_m < thresh_m

        # Connected components via BFS over the precomputed adjacency (same
        # greedy transitive-closure semantics as before).
        visited_local = [False] * n_g
        for s in range(n_g):
            if visited_local[s]:
                continue
            comp = [s]
            visited_local[s] = True
            qi = 0
            while qi < len(comp):
                curr = comp[qi]
                qi += 1
                neighbors = np.flatnonzero(adj[curr])
                for nb in neighbors.tolist():
                    if not visited_local[nb]:
                        visited_local[nb] = True
                        comp.append(nb)

            if len(comp) == 1:
                merged_mbs.append(group_mbs[comp[0]])
                continue

            cluster = [group_mbs[c] for c in comp]
            cos = cos_arr[comp]
            radii = rad_arr[comp]

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

    # Snapshot the raw arrays we need with C-level foreach_get BEFORE the
    # evaluated mesh is freed. NOTE: BMVertSeq/BMEdgeSeq have no foreach_get --
    # only bpy.types.Mesh does, so this must happen while `mesh` is alive.
    V_count = len(mesh.vertices)
    E_count = len(mesh.edges)
    cos_local = np.empty(V_count * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", cos_local)
    edge_vert_idx = np.empty(E_count * 2, dtype=np.int32)
    if E_count > 0:
        mesh.edges.foreach_get("vertices", edge_vert_idx)
        edge_vert_idx = edge_vert_idx.reshape(E_count, 2).astype(np.int64)

    if has_eval_mesh:
        eval_obj.to_mesh_clear()
        
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    # Precompute per-vertex weights and bone joints ONCE (O(V) instead of O(V*G)
    # with exception handling per lookup).
    weights_map = build_vertex_weights_map(obj)
    joints = precompute_bone_joints(armature_obj) if use_joint_scaling else None

    # 1. Group vertices by topological islands (connected components) --
    # O(V + E) via scipy C graph pass (or CSR flood-fill fallback).
    if creation_type == 'SINGLE':
        vert_family_arr = np.zeros(max(V_count, 1), dtype=np.int64)
    else:
        vert_family_arr = _topological_islands(V_count, edge_vert_idx)

    # Map vertex index to family ID (island index)
    vert_family = {}
    fam_list = vert_family_arr.tolist()
    for v_idx in range(V_count):
        vert_family[v_idx] = fam_list[v_idx]
                
    # 2. Compute initial global edge lengths -- fully vectorized via numpy
    # (single foreach_get pass + BLAS norm instead of per-edge Python loop).
    V_count = len(bm.verts)
    E_count = len(bm.edges)
    verts_list = list(bm.verts)

    # Per-vertex joint multipliers computed ONCE for all vertices (needed both
    # for base radii below and for final original radii). World-space vertex
    # coordinates are resolved with BLAS instead of per-vertex mathutils
    # matrix products.
    mw_np64 = np.asarray(matrix_world, dtype=np.float64)
    cos_world_all = np.zeros((max(V_count, 1), 3), dtype=np.float64)
    if V_count > 0:
        cos_world_all[:V_count] = cos_local.reshape(V_count, 3) @ mw_np64[:3, :3].T + mw_np64[:3, 3]

    vert_mults = np.ones(max(V_count, 1), dtype=np.float64)
    if joints is not None:
        vert_mults = joint_aware_multipliers(cos_world_all[:V_count], joints, joint_scale, middle_scale)

    edge_lengths_global = {}
    vert_edge_lengths = {v.index: [] for v in bm.verts}
    base_radii = np.zeros(max(V_count, 1), dtype=np.float64)
    edge_lens = np.zeros(max(E_count, 1), dtype=np.float64)
    if E_count > 0:
        cos_world = cos_world_all[:V_count]

        edge_vec = cos_world[edge_vert_idx[:, 0]] - cos_world[edge_vert_idx[:, 1]]
        edge_lens = np.sqrt(np.einsum('ij,ij->i', edge_vec, edge_vec))

        eidx_arr = edge_vert_idx[:, 0].tolist()
        lens_list = edge_lens.tolist()
        for ei in range(E_count):
            edge_lengths_global[ei] = lens_list[ei]
            vert_edge_lengths[eidx_arr[ei]].append((ei, lens_list[ei]))

        # Per-vertex average incident edge length in O(E) with bincount
        # (replaces per-vertex Python sums over link_edges).
        # Every edge contributes to BOTH endpoints: flatten (E,2) indices and
        # duplicate the lengths so weights match the index list.
        flat_ends = edge_vert_idx.ravel()
        lens_both = np.concatenate([edge_lens, edge_lens])
        deg_full = np.bincount(flat_ends, minlength=V_count)
        len_sum_full = np.bincount(flat_ends, weights=lens_both, minlength=V_count)
        avg_len = np.divide(len_sum_full, np.maximum(deg_full, 1))
        base_radii = alpha * avg_len
        if joints is not None:
            base_radii *= vert_mults[:V_count]
        
    # 3. Create virtual metaballs along long subdivided edges -- fully
    # batch-vectorized: all long edges are resolved in one numpy pass (points,
    # normals, joint multipliers, blended weights), then only the per-edge
    # weight-dict assembly remains in Python.
    virtual_metaballs = []
    subdivided_edge_lengths = {}

    # Thickness-aware scaling: build a BVH tree ONCE (C-level raycasts) instead
    # of calling eval_obj.ray_cast per point (which re-walks the whole mesh).
    # Built from the live bmesh -- the evaluated `mesh` may already be freed.
    bvh_tree = None
    if use_thickness_scaling:
        try:
            from mathutils.bvhtree import BVHTree
            bvh_tree = BVHTree.FromBMesh(bm)
        except Exception as e:
            print("BVH build failed, falling back to object ray_cast:", e)
            bvh_tree = None

    inv_mw_thickness = matrix_world.inverted()
    inv_mw_3x3_thick = inv_mw_thickness.to_3x3()
    mw_scale_x = matrix_world.to_scale().x

    def _thickness_at(co_world, no_world, r_current):
        """Return min(r_current, local_thickness * factor) via one BVH raycast."""
        co_local = inv_mw_thickness @ mathutils.Vector((co_world[0], co_world[1], co_world[2]))
        no_local = inv_mw_3x3_thick @ mathutils.Vector((no_world[0], no_world[1], no_world[2]))
        if no_local.length > 1e-9:
            no_local.normalize()
        if bvh_tree is not None:
            result, hit_loc, _, _ = bvh_tree.ray_cast(co_local - 1e-4 * no_local, -no_local)
        else:
            result, hit_loc, _, _ = eval_obj.ray_cast(co_local - 1e-4 * no_local, -no_local)
        if result:
            thickness_world = (co_local - hit_loc).length * mw_scale_x
            return min(r_current, thickness_world * thickness_factor)
        return r_current

    if E_count > 0:
        r1_arr = base_radii[edge_vert_idx[:, 0]]
        r2_arr = base_radii[edge_vert_idx[:, 1]]
        r_max_arr = np.maximum(r1_arr, r2_arr)
        long_mask = edge_lens > k_coeff * r_max_arr
        long_eidx = np.flatnonzero(long_mask)

        for ei in long_eidx.tolist():
            v1i = int(edge_vert_idx[ei, 0])
            v2i = int(edge_vert_idx[ei, 1])
            L = float(edge_lens[ei])
            r_max = float(r_max_arr[ei])

            divs = int(math.ceil(L / (k_coeff * r_max)))
            if divs <= 1:
                continue
            L_sub = L / divs
            subdivided_edge_lengths[(v1i, ei)] = L_sub
            subdivided_edge_lengths[(v2i, ei)] = L_sub

            # Endpoints/normals straight from the precomputed world arrays --
            # no per-vertex mathutils products inside this loop anymore.
            p1 = cos_world_all[v1i]
            p2 = cos_world_all[v2i]
            n1 = verts_list[v1i].normal
            n2 = verts_list[v2i].normal

            w1 = weights_map.get(v1i, {})
            w2 = weights_map.get(v2i, {})
            fam_id = vert_family[v1i]

            # Joint multipliers for all interpolated points of this edge at once.
            ts = np.arange(1, divs, dtype=np.float64) / divs
            pts_new = (1.0 - ts[:, np.newaxis]) * p1[np.newaxis, :] + ts[:, np.newaxis] * p2[np.newaxis, :]
            nos_new = (1.0 - ts[:, np.newaxis]) * np.asarray(n1, dtype=np.float64)[np.newaxis, :] + \
                ts[:, np.newaxis] * np.asarray(n2, dtype=np.float64)[np.newaxis, :]
            no_lens = np.linalg.norm(nos_new, axis=1)
            nz_mask = no_lens > 1e-6
            nos_new[nz_mask] /= no_lens[nz_mask, np.newaxis]

            mults = (joint_aware_multipliers(pts_new, joints, joint_scale, middle_scale)
                     if joints is not None else None)

            w_keys_union = set(w1.keys()).union(w2.keys())
            w1_vals = {g: w1.get(g, 0.0) for g in w_keys_union}
            w2_vals = {g: w2.get(g, 0.0) for g in w_keys_union}

            # Blend weights for ALL interpolated points of this edge in one
            # vectorized pass (outer (divs-1) x inner G matrix), then split
            # back to per-point dicts only where above the 0.001 threshold.
            if w_keys_union:
                key_list = list(w_keys_union)
                v1_arr = np.fromiter((w1_vals[g] for g in key_list), dtype=np.float64, count=len(key_list))
                v2_arr = np.fromiter((w2_vals[g] for g in key_list), dtype=np.float64, count=len(key_list))
                blended = np.outer(1.0 - ts, v1_arr) + np.outer(ts, v2_arr)
                keep = blended >= 0.001
                per_point_keys = [
                    [key_list[gi] for gi in np.flatnonzero(keep[pi])]
                    for pi in range(blended.shape[0])
                ]
                per_point_vals = [
                    blended[pi][keep[pi]].tolist()
                    for pi in range(blended.shape[0])
                ]
            else:
                per_point_keys = [[] for _ in range(len(ts))]
                per_point_vals = [[] for _ in range(len(ts))]

            pts_list = pts_new.tolist()
            nos_list = nos_new.tolist()
            mults_list = mults.tolist() if mults is not None else None

            for i in range(1, divs):
                pi = i - 1
                r_new = alpha * L_sub

                # 1. Joint-aware scaling
                if mults_list is not None:
                    r_new *= mults_list[pi]

                # 2. Thickness-aware scaling (single C-level BVH raycast)
                if use_thickness_scaling:
                    r_new = _thickness_at(pts_new[pi], nos_new[pi], r_new)

                virtual_metaballs.append({
                    'co': pts_list[pi],
                    'radius': float(r_new),
                    'weights': dict(zip(per_point_keys[pi], per_point_vals[pi])),
                    'normal': nos_list[pi],
                    'family_id': fam_id,
                    'is_virtual': True
                })
                    
    # 4. Recalculate original radii using updated edge lengths -- vectorized:
    # build per-(vertex,edge) arrays once, substitute subdivided lengths, then
    # average with bincount instead of nested Python loops.
    new_radii = {}
    if E_count > 0:
        pair_v = edge_vert_idx[:, 0]
        orig_lens = edge_lens
        sub_mask = np.zeros(E_count, dtype=bool)
        sub_vals = np.zeros(E_count, dtype=np.float64)
        for (v_idx, e_idx), l_sub in subdivided_edge_lengths.items():
            if v_idx == pair_v[e_idx]:
                sub_mask[e_idx] = True
                sub_vals[e_idx] = l_sub
        final_lens = np.where(sub_mask, sub_vals, orig_lens)
        deg_v = np.bincount(pair_v, minlength=V_count)
        sum_v = np.bincount(pair_v, weights=final_lens, minlength=V_count)
        avg_r = np.divide(sum_v, np.maximum(deg_v, 1)) * alpha
        for vi in range(V_count):
            new_radii[vi] = float(avg_r[vi]) if deg_v[vi] > 0 else 0.0
    else:
        for v in bm.verts:
            new_radii[v.index] = 0.0
            
    # 5. Compile original metaballs list -- world coordinates/normals come
    # from the precomputed BLAS arrays; only the per-vertex weight dicts are
    # assembled in Python.
    original_metaballs = []
    cos_w_list = cos_world_all[:V_count].tolist()
    mults_list_all = vert_mults[:V_count].tolist()
    for vi, v in enumerate(verts_list):
        no = (matrix_world.to_3x3() @ v.normal).normalized()
        w = weights_map.get(v.index, {})

        r_final = new_radii[vi]

        # 1. Joint-aware scaling
        if joints is not None:
            r_final *= mults_list_all[vi]
            
        # 2. Thickness-aware scaling (single C-level BVH raycast)
        if use_thickness_scaling:
            r_final = _thickness_at(cos_w_list[vi], no, r_final)
                
        original_metaballs.append({
            'co': cos_w_list[vi],
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
        
        # Batch-resolve dominant bones and local X coordinates once, then run
        # the classification/mirroring branch per metaball (the classification
        # itself is vectorized; only actual mirroring stays per-metaball).
        dom_bones = [max(mb['weights'], key=mb['weights'].get) if mb['weights'] else None for mb in all_raw]
        central_flags = [is_bone_central(db) if db else False for db in dom_bones]

        non_central_idx = [i for i, mb in enumerate(all_raw)
                           if not (mb['weights'] and central_flags[i])]
        if non_central_idx:
            co_np = np.array([all_raw[i]['co'] for i in non_central_idx], dtype=np.float64)
            inv_r = np.asarray(inv_mw.to_3x3(), dtype=np.float64)
            inv_t = np.asarray(inv_mw.translation, dtype=np.float64)
            x_locals = (co_np @ inv_r.T + inv_t)[:, 0]
        else:
            x_locals = np.zeros(0, dtype=np.float64)
        x_local_by_pos = dict(zip(non_central_idx, x_locals.tolist()))

        for idx, mb in enumerate(all_raw):
            w = mb.get('weights', {})
            dom_bone = dom_bones[idx]
            
            if w and central_flags[idx]:
                mb['symmetry_class'] = 'C'
                left_mbs.append(mb)
            else:
                co_local_x = x_local_by_pos.get(idx, 0.0)
                # Keep Left side (local X >= -1e-4)
                if co_local_x >= -1e-4:
                    mb['symmetry_class'] = 'L'
                    left_mbs.append(mb)
                    
                    # Mirror to Right if not central (local X > 1e-3)
                    if co_local_x > 1e-3:
                        co_loc_full = inv_mw @ mathutils.Vector(mb['co'])
                        co_loc_mir = mathutils.Vector((-co_local_x, co_loc_full.y, co_loc_full.z))
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
                            'vertex_index': -1,
                            'symmetry_class': 'R'
                        }
                        left_mbs.append(mirrored_mb)
        original_metaballs = left_mbs
        virtual_metaballs = []
    else:
        # Standard run: assign symmetry class based on dominant bone and
        # coordinates -- batch-classified with one numpy pass over local X.
        inv_mw = matrix_world.inverted()
        inv_t = np.asarray(inv_mw.translation, dtype=np.float64)
        inv_r = np.asarray(inv_mw.to_3x3(), dtype=np.float64)
        all_std = original_metaballs + virtual_metaballs
        if all_std:
            cos_np = np.array([mb['co'] for mb in all_std], dtype=np.float64)
            x_loc = cos_np @ inv_r.T + inv_t
            x_col = x_loc[:, 0]
        else:
            x_col = np.zeros(0, dtype=np.float64)
        for k, mb in enumerate(all_std):
            w = mb.get('weights', {})
            dom_bone = max(w, key=w.get) if w else None
            if dom_bone and is_bone_central(dom_bone):
                mb['symmetry_class'] = 'C'
            else:
                mb['symmetry_class'] = 'L' if x_col[k] >= 0.0 else 'R'
            
    if merge_close:
        all_mbs = original_metaballs + virtual_metaballs
        merged_mbs = merge_close_metaballs(all_mbs, merge_factor)
        return merged_mbs, []
    else:
        return original_metaballs, virtual_metaballs
