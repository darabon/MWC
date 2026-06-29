import heapq
import numpy as np

try:
    import bpy
    import mathutils
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False

# Import utilities
from .translation import t
from .utils import get_curve_mapping_node

_worker_adj = None

def _init_process_worker(shared_adj):
    global _worker_adj
    _worker_adj = shared_adj

def dijkstra_pruned(adj, start_node, start_dist, max_dist):
    distances = {start_node: start_dist}
    pq = [(start_dist, start_node)]
    while pq:
        dist, u = heapq.heappop(pq)
        if dist > distances[u]:
            continue
        for v, weight in adj[u]:
            new_dist = dist + weight
            if new_dist <= max_dist:
                if new_dist < distances.get(v, float('inf')):
                    distances[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
    return distances

def dijkstra_thread_chunk_worker(tasks_chunk, adj):
    results = {}
    for j, v_start, d_start, rj in tasks_chunk:
        results[j] = dijkstra_pruned(adj, v_start, d_start, rj)
    return results

def dijkstra_process_chunk_worker(tasks_chunk):
    # Uses global _worker_adj initialized in child process
    results = {}
    for j, v_start, d_start, rj in tasks_chunk:
        results[j] = dijkstra_pruned(_worker_adj, v_start, d_start, rj)
    return results

def apply_mwc_weights(target_obj, mbs, n, q, tau, r_falloff_multiplier,
                      use_normal_filter=True, normal_p=1.0, symmetry_beta=False,
                      use_smoothing=False, smoothing_strength=0.5, smoothing_iterations=3,
                      use_geodesic=False, use_custom_curve=False, curve_node=None,
                      geodesic_mode='THREAD'):
    """
    Performs high-performance NumPy-vectorized MWC blending (v1.1 optimized with KD-Tree and Dijkstra).
    """
    matrix_world = target_obj.matrix_world
    
    # Get evaluated mesh (deforms applied)
    eval_obj = target_obj
    has_eval_mesh = False
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = target_obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        has_eval_mesh = True
    except Exception as e:
        print("Failed to get evaluated target mesh, falling back to base mesh:", e)
        mesh = target_obj.data
        
    try:
        # 1. Gather target coordinates and normals in global space
        V = len(mesh.vertices)
        target_cos = np.zeros(V * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", target_cos)
        target_cos = target_cos.reshape((V, 3))
        
        # Multiply by world matrix
        mw_3x3 = np.array(matrix_world.to_3x3(), dtype=np.float32)
        mw_trans = np.array(matrix_world.translation, dtype=np.float32)
        target_cos = np.dot(target_cos, mw_3x3.T) + mw_trans
        
        target_nos = np.zeros(V * 3, dtype=np.float32)
        mesh.vertices.foreach_get("normal", target_nos)
        target_nos = target_nos.reshape((V, 3))
        target_nos = np.dot(target_nos, mw_3x3.T)
        
        # Normalize normals row-wise
        norms = np.linalg.norm(target_nos, axis=1, keepdims=True)
        norms = np.where(norms > 1e-6, norms, 1e-6)
        target_nos /= norms
        
        # 2. Setup metaball arrays
        M = len(mbs)
        C = np.array([mb['co'] for mb in mbs], dtype=np.float32)
        R = np.array([mb['radius'] for mb in mbs], dtype=np.float32)
        N = np.array([mb['normal'] for mb in mbs], dtype=np.float32)
        F = np.array([mb['family_id'] for mb in mbs], dtype=np.int32)
        
        # Bone mapping
        bone_names_set = set()
        for mb in mbs:
            bone_names_set.update(mb['weights'].keys())
        bone_names = sorted(list(bone_names_set))
        G = len(bone_names)
        
        if G == 0:
            return
            
        bone_to_idx = {name: idx for idx, name in enumerate(bone_names)}
        
        # Metaball weight matrix
        W = np.zeros((M, G), dtype=np.float32)
        for j, mb in enumerate(mbs):
            for b_name, w_val in mb['weights'].items():
                W[j, bone_to_idx[b_name]] = w_val
                
        # Compute R_falloff
        valid_radii = R[R > 0.0]
        r_avg = np.mean(valid_radii) if len(valid_radii) > 0 else 1.0
        R_falloff = r_falloff_multiplier * r_avg
        
        MB_Sym = np.zeros(M, dtype=np.int8)
        for j, mb in enumerate(mbs):
            sym_str = mb.get('symmetry_class', 'L')
            if sym_str == 'L':
                MB_Sym[j] = 1
            elif sym_str == 'R':
                MB_Sym[j] = 2
            else: # 'C' (Central)
                MB_Sym[j] = 3
        
        # Default symmetry based on global coordinates to account for unapplied object transforms
        target_labels_sym = np.where(target_cos[:, 0] >= 0.0, 1, 2).astype(np.int8)
        
        # 3. Create KD-Tree for fast nearest-neighbor search
        # We build two trees: one for vertices (for range queries) and one for metaballs (for family assignment)
        target_kd = mathutils.kdtree.KDTree(V)
        for i, co in enumerate(target_cos):
            target_kd.insert(mathutils.Vector(co), i)
        target_kd.balance()
        
        kd_mbs = mathutils.kdtree.KDTree(M)
        for j, co in enumerate(C):
            kd_mbs.insert(mathutils.Vector(co), j)
        kd_mbs.balance()
        
        # Find closest metaball for each target vertex to inherit family ID
        closest_indices = np.zeros(V, dtype=np.int32)
        for i in range(V):
            pt = mathutils.Vector(target_cos[i])
            target_sym = target_labels_sym[i]
            found = False
            # Look for the nearest metaball of matching symmetry class OR central class (3)
            for co_mb, mb_idx, dist in kd_mbs.find_n(pt, min(10, M)):
                if MB_Sym[mb_idx] == target_sym or MB_Sym[mb_idx] == 3:
                    closest_indices[i] = mb_idx
                    found = True
                    break
            if not found:
                _, idx, _ = kd_mbs.find(pt)
                closest_indices[i] = idx
            
        vertex_families = F[closest_indices]
        
        # Main weights blending variables
        final_weights = np.zeros((V, G), dtype=np.float32)
        den_accum = np.zeros(V, dtype=np.float32)
        
        # Build adjacency list if using Geodesic
        visited_dists_dict = {}
        if use_geodesic:
            adj = {i: [] for i in range(V)}
            for edge in mesh.edges:
                v1, v2 = edge.vertices
                p1 = target_cos[v1]
                p2 = target_cos[v2]
                dist = float(np.linalg.norm(p1 - p2))
                adj[v1].append((v2, dist))
                adj[v2].append((v1, dist))
                
            # Collect tasks for parallel dijkstra
            tasks = []
            for j in range(M):
                rj = max(R[j], 1e-6)
                _, v_start, d_start = target_kd.find(mathutils.Vector(C[j]))
                if d_start < rj:
                    tasks.append((j, v_start, d_start, rj))
                    
            if geodesic_mode == 'THREAD' and len(tasks) > 1:
                import concurrent.futures
                import os
                
                num_workers = os.cpu_count() or 4
                chunks = [[] for _ in range(num_workers)]
                for idx, task in enumerate(tasks):
                    chunks[idx % num_workers].append(task)
                chunks = [c for c in chunks if c]
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
                    futures = [executor.submit(dijkstra_thread_chunk_worker, chunk, adj) for chunk in chunks]
                    for fut in concurrent.futures.as_completed(futures):
                        visited_dists_dict.update(fut.result())
                        
            elif geodesic_mode == 'PROCESS' and len(tasks) > 1:
                import concurrent.futures
                import os
                
                num_workers = os.cpu_count() or 4
                chunks = [[] for _ in range(num_workers)]
                for idx, task in enumerate(tasks):
                    chunks[idx % num_workers].append(task)
                chunks = [c for c in chunks if c]
                
                try:
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=len(chunks),
                        initializer=_init_process_worker,
                        initargs=(adj,)
                    ) as executor:
                        futures = [executor.submit(dijkstra_process_chunk_worker, chunk) for chunk in chunks]
                        for fut in concurrent.futures.as_completed(futures):
                            visited_dists_dict.update(fut.result())
                except Exception as e:
                    # Fallback to ThreadPoolExecutor if ProcessPoolExecutor fails
                    print(f"[MWC] ProcessPoolExecutor failed: {e}. Falling back to ThreadPoolExecutor.")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
                        futures = [executor.submit(dijkstra_thread_chunk_worker, chunk, adj) for chunk in chunks]
                        for fut in concurrent.futures.as_completed(futures):
                            visited_dists_dict.update(fut.result())
            else:
                # Sequential fallback (or if only 1 task)
                for j, v_start, d_start, rj in tasks:
                    visited_dists_dict[j] = dijkstra_pruned(adj, v_start, d_start, rj)
    
        # Precompute Custom Curve LUT if active
        if use_custom_curve and curve_node:
            mapping = curve_node.mapping
            combined_curve = mapping.curves[3]
            lut_x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
            lut_y = np.array([mapping.evaluate(combined_curve, x) for x in lut_x], dtype=np.float32)
        else:
            lut_x = None
            lut_y = None
            
        # 4. Range queries and Wyvill field calculations
        for j in range(M):
            rj = max(R[j], 1e-6)
            
            if use_geodesic:
                if j not in visited_dists_dict:
                    continue
                visited_dists = visited_dists_dict[j]
                indices = np.array(list(visited_dists.keys()), dtype=np.int32)
                dists = np.array(list(visited_dists.values()), dtype=np.float32)
            else:
                # Query KD-Tree for vertices within range rj
                candidates = target_kd.find_range(mathutils.Vector(C[j]), rj)
                if not candidates:
                    continue
                indices = np.array([idx for _, idx, _ in candidates], dtype=np.int32)
                dists = np.array([d for _, _, d in candidates], dtype=np.float32)
                
            # Filter by family ID and symmetry class
            if MB_Sym[j] == 3:
                mask = (vertex_families[indices] == F[j])
            else:
                mask = (vertex_families[indices] == F[j]) & (target_labels_sym[indices] == MB_Sym[j])
                
            if not np.any(mask):
                continue
                
            active_indices = indices[mask]
            d = dists[mask]
            
            # Influence calculation
            if use_custom_curve and lut_x is not None:
                f = np.interp(d / rj, lut_x, lut_y)
            else:
                f = (1.0 - (d / rj)**2)**n
                
            if use_normal_filter:
                nos = target_nos[active_indices]
                dot = np.sum(nos * N[j], axis=1)
                f *= np.maximum(0.0, dot)**normal_p
                
            fq = f**q
            
            # Accumulate blending values
            den_accum[active_indices] += fq
            final_weights[active_indices] += fq[:, np.newaxis] * W[j]
            
        # Divide by denominator
        mask_active = den_accum > 0.0
        final_weights[mask_active] /= den_accum[mask_active][:, np.newaxis]
        
        # 5. RBF Extrapolation for orphans (vertices with no active metaballs in range)
        orphans_mask = den_accum == 0.0
        if np.any(orphans_mask):
            orphan_indices = np.where(orphans_mask)[0]
            unique_fam_ids = np.unique(vertex_families[orphan_indices])
            
            for f_id in unique_fam_ids:
                # Group orphans by symmetry class
                for s_val in [1, 2]:
                    mb_fam_indices = np.where((F == f_id) & ((MB_Sym == s_val) | (MB_Sym == 3)))[0]
                    if len(mb_fam_indices) == 0:
                        continue
                        
                    orphans_fam_mask = (vertex_families[orphan_indices] == f_id) & (target_labels_sym[orphan_indices] == s_val)
                    orphans_fam_global = orphan_indices[orphans_fam_mask]
                    
                    if len(orphans_fam_global) == 0:
                        continue
                        
                    v_cos = target_cos[orphans_fam_global]
                    mb_cos = C[mb_fam_indices]
                    
                    # Pairwise distance matrix of shape (V_o, M_f)
                    d_matrix = np.linalg.norm(v_cos[:, np.newaxis, :] - mb_cos[np.newaxis, :, :], axis=2)
                    phi = np.exp(-(d_matrix**2) / (R_falloff**2))
                    
                    if use_normal_filter:
                        v_nos = target_nos[orphans_fam_global]
                        mb_nos = N[mb_fam_indices]
                        dot_matrix = np.dot(v_nos, mb_nos.T)
                        normal_factor = np.maximum(0.0, dot_matrix) ** normal_p
                        phi *= normal_factor
                        
                    sum_phi = np.sum(phi, axis=1)
                    
                    valid_rbf_mask = sum_phi > tau
                    if np.any(valid_rbf_mask):
                        valid_global_indices = orphans_fam_global[valid_rbf_mask]
                        num = np.dot(phi[valid_rbf_mask], W[mb_fam_indices])
                        scale_den = sum_phi[valid_rbf_mask][:, np.newaxis]
                        final_weights[valid_global_indices] = num / scale_den
                    
        # 6. Fallback for completely unassigned orphans using Normal-Filtered IDW (Inverse Distance Weighting)
        sums = np.sum(final_weights, axis=1)
        unassigned_mask = sums <= 1e-6
        if np.any(unassigned_mask):
            unassigned_indices = np.where(unassigned_mask)[0]
            
            for idx in unassigned_indices:
                pt = mathutils.Vector(target_cos[idx])
                target_sym = target_labels_sym[idx]
                
                # Find nearest metaballs of matching symmetry class (or central class 3)
                # To be efficient, we search the KD-Tree for the 8 nearest neighbors,
                # then filter for matching symmetry and take up to 4.
                nearest = []
                for co_mb, mb_idx, dist in kd_mbs.find_n(pt, min(8, M)):
                    if MB_Sym[mb_idx] == target_sym or MB_Sym[mb_idx] == 3:
                        nearest.append((mb_idx, max(dist, 1e-6)))
                        if len(nearest) == 4:
                            break
                            
                if not nearest:
                    # Failsafe: if none match symmetry, just take any nearest
                    for co_mb, mb_idx, dist in kd_mbs.find_n(pt, 4):
                        nearest.append((mb_idx, max(dist, 1e-6)))
                
                # Calculate IDW weights blended with normal filter
                w_sum = 0.0
                blended_w = np.zeros(G, dtype=np.float32)
                
                # 1st pass: try distance + normal filter
                for mb_idx, dist in nearest:
                    dist_factor = 1.0 / (dist ** 2)
                    
                    # Apply normal filter if enabled
                    if use_normal_filter:
                        dot = float(target_nos[idx].dot(N[mb_idx]))
                        normal_factor = max(0.0, dot) ** normal_p
                    else:
                        normal_factor = 1.0
                        
                    w_val = dist_factor * normal_factor
                    blended_w += w_val * W[mb_idx]
                    w_sum += w_val
                    
                # If w_sum is zero (due to strict normal filtering), fallback to pure distance-based IDW
                if w_sum <= 1e-6:
                    w_sum = 0.0
                    blended_w = np.zeros(G, dtype=np.float32)
                    for mb_idx, dist in nearest:
                        w_val = 1.0 / (dist ** 2)
                        blended_w += w_val * W[mb_idx]
                        w_sum += w_val
                        
                if w_sum > 1e-6:
                    final_weights[idx] = blended_w / w_sum
                else:
                    # Absolute fallback to the single closest metaball
                    final_weights[idx] = W[closest_indices[idx]]
            
        # 6.5. Apply Laplacian Smoothing if enabled
        if use_smoothing and smoothing_iterations > 0 and smoothing_strength > 0.0:
            edges_src = []
            edges_dst = []
            for edge in mesh.edges:
                v1, v2 = edge.vertices
                edges_src.append(v1)
                edges_dst.append(v2)
                edges_src.append(v2)
                edges_dst.append(v1)
                
            edges_src = np.array(edges_src, dtype=np.int32)
            edges_dst = np.array(edges_dst, dtype=np.int32)
            
            degrees = np.zeros(V, dtype=np.int32)
            np.add.at(degrees, edges_src, 1)
            degrees_mask = degrees > 0
            
            for _ in range(smoothing_iterations):
                sum_neigh_weights = np.zeros((V, G), dtype=np.float32)
                np.add.at(sum_neigh_weights, edges_src, final_weights[edges_dst])
                
                avg_weights = np.zeros_like(final_weights)
                avg_weights[degrees_mask] = sum_neigh_weights[degrees_mask] / degrees[degrees_mask][:, np.newaxis]
                
                final_weights = (1.0 - smoothing_strength) * final_weights + smoothing_strength * avg_weights
            
        # 7. Normalize all final weights to sum up to 1.0
        sums = np.sum(final_weights, axis=1, keepdims=True)
        mask_nonzero = (sums > 1e-6).flatten()
        final_weights[mask_nonzero] /= sums[mask_nonzero]
        
        # 8. Batch write weights to target object's vertex groups
        target_obj.vertex_groups.clear()
        
        vgs = {}
        for g_name in bone_names:
            vgs[g_name] = target_obj.vertex_groups.new(name=g_name)
            
        for g_idx, g_name in enumerate(bone_names):
            vg = vgs[g_name]
            weights = final_weights[:, g_idx]
            
            active_verts = np.where(weights >= 0.001)[0]
            if len(active_verts) == 0:
                continue
                
            active_weights = weights[active_verts]
            
            # Round weights to group vertices and optimize vg.add calls
            rounded_weights = np.round(active_weights, 4)
            unique_weights = np.unique(rounded_weights)
            
            for w in unique_weights:
                indices = active_verts[rounded_weights == w]
                vg.add(indices.tolist(), float(w), 'REPLACE')
    finally:
        if has_eval_mesh:
            eval_obj.to_mesh_clear()
