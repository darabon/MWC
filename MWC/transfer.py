import heapq
import numpy as np

try:
    import bpy
    import mathutils
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False

# Optional SciPy acceleration (C-implemented KD-Tree and Dijkstra).
# Everything gracefully falls back to the pure-Python/mathutils paths below.
try:
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as _scipy_dijkstra
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# Import utilities
from .utils import get_curve_mapping_node



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

def dijkstra_chunk_worker(tasks_chunk, adj):
    """
    Runs pruned Dijkstra for a chunk of metaball tasks. Pure-Python/stdlib core
    (numpy only for packing), so it is safe to execute in a worker thread OR a
    separate worker process (ProcessPoolExecutor pickles this function + its
    arguments to the child process).

    Returns {task_index: (indices_int32_array, dists_float32_array)}.
    """
    results = {}
    for j, v_start, d_start, rj in tasks_chunk:
        dd = dijkstra_pruned(adj, v_start, d_start, rj)
        if dd:
            idx = np.fromiter(dd.keys(), dtype=np.int32, count=len(dd))
            ds = np.fromiter(dd.values(), dtype=np.float32, count=len(dd))
            results[j] = (idx, ds)
    return results


def _geodesic_scipy_batch(edges_arr, edge_lens, V, tasks):
    """
    Batched pruned Dijkstra over the mesh edge graph using scipy's C
    implementation. Mathematically identical to running dijkstra_pruned per
    metaball: for each source we truncate at its own threshold
    (rj - d_start) after computing with a shared limit, and shift the
    resulting shortest-path distances by the metaball's start offset.

    Returns {task_index: (indices_int32_array, dists_float32_array)}.
    """
    row = np.concatenate([edges_arr[:, 0], edges_arr[:, 1]]).astype(np.int64)
    col = np.concatenate([edges_arr[:, 1], edges_arr[:, 0]]).astype(np.int64)
    # csr_matrix drops explicit zeros, which would silently delete collapsed
    # (zero-length) edges that the heapq reference implementation traverses;
    # substitute a negligible epsilon to keep connectivity identical.
    data = np.concatenate([edge_lens, edge_lens]).astype(np.float64)
    data[data <= 0.0] = 1e-12
    graph = csr_matrix((data, (row, col)), shape=(V, V))

    starts = np.array([t[1] for t in tasks], dtype=np.int32)
    offsets = np.array([t[2] for t in tasks], dtype=np.float64)
    thresholds = np.array([t[3] for t in tasks], dtype=np.float64) - offsets
    limit = float(thresholds.max())

    results = {}
    # Chunk sources so the (sources x V) distance matrix stays bounded.
    bytes_per_source = max(V * 8, 8)
    chunk = int(max(1, min(256, 2 * 10**8 // bytes_per_source)))
    for s0 in range(0, len(starts), chunk):
        sl = slice(s0, min(s0 + chunk, len(starts)))
        D = _scipy_dijkstra(graph, directed=True, indices=starts[sl],
                            limit=limit, return_predecessors=False)
        D = np.atleast_2d(D)
        for k, ti in enumerate(range(sl.start, sl.stop)):
            j, _vs, d_start, rj = tasks[ti]
            dk = D[k]
            valid = np.isfinite(dk) & (dk <= thresholds[ti])
            vis = np.where(valid)[0]
            if len(vis):
                results[j] = (vis.astype(np.int32),
                              (dk[vis] + d_start).astype(np.float32))
    return results


def normal_filter_power(dot_values, p):
    """
    Shared normal-filter formula: clamp negative dot products to zero and raise
    to power p. Works on scalars, 1D arrays, and 2D matrices alike (NumPy
    broadcasting), so the same function backs the main blending loop, the RBF
    orphan extrapolation, and the IDW fallback instead of three separate copies.
    """
    return np.maximum(0.0, dot_values) ** p


class _PointTree:
    """
    Nearest-neighbor / range-query structure over a fixed point cloud.

    Uses scipy.spatial.cKDTree (vectorized C queries) when available and falls
    back to a mathutils.kdtree wrapper with identical query semantics.
    """
    __slots__ = ("pts", "_scipy", "_mu")

    def __init__(self, pts):
        self.pts = np.asarray(pts, dtype=np.float64)
        self._scipy = None
        self._mu = None
        if HAS_SCIPY and len(self.pts) > 0:
            try:
                self._scipy = cKDTree(self.pts)
            except Exception:
                self._scipy = None
        if self._scipy is None and HAS_BLENDER and len(self.pts) > 0:
            self._mu = mathutils.kdtree.KDTree(len(self.pts))
            for i in range(len(self.pts)):
                self._mu.insert(mathutils.Vector(self.pts[i]), i)
            self._mu.balance()

    def query_nearest(self, Q, k=1):
        """
        Returns (dists (N,k) float32, idxs (N,k) int64). Missing neighbors
        (when k > cloud size) come back with dist=inf.
        """
        Qd = np.asarray(Q, dtype=np.float64)
        N = len(Qd)
        if self._scipy is not None:
            d, i = self._scipy.query(Qd, k=k)
            if k == 1:
                d = np.asarray(d).reshape(N, 1)
                i = np.asarray(i).reshape(N, 1)
            return d.astype(np.float32), i.astype(np.int64)
        d = np.full((N, k), np.inf, dtype=np.float32)
        ii = np.zeros((N, k), dtype=np.int64)
        for r in range(N):
            found = self._mu.find_n(mathutils.Vector(Qd[r]), k)
            for c, (_co, idx, dist) in enumerate(found):
                d[r, c] = dist
                ii[r, c] = idx
        return d, ii

    def query_range_bulk(self, centers, radii):
        """
        centers (N,3), radii (N,) -> list of length N with either
        (indices_int32, dists_float32) per center or None when empty.
        """
        Cd = np.asarray(centers, dtype=np.float64)
        Rd = np.asarray(radii, dtype=np.float64)
        out = []
        if self._scipy is not None:
            try:
                lists = self._scipy.query_ball_point(Cd, Rd)
            except Exception:
                # Older scipy versions only accept scalar radii.
                lists = [self._scipy.query_ball_point(Cd[i], float(Rd[i]))
                         for i in range(len(Cd))]
            for i, lst in enumerate(lists):
                if lst:
                    idx = np.asarray(lst, dtype=np.int32)
                    diff = self.pts[idx] - Cd[i]
                    dist = np.sqrt((diff * diff).sum(axis=1)).astype(np.float32)
                    out.append((idx, dist))
                else:
                    out.append(None)
            return out
        for i in range(len(Cd)):
            cand = self._mu.find_range(mathutils.Vector(Cd[i]), float(Rd[i]))
            if cand:
                idx = np.array([c[1] for c in cand], dtype=np.int32)
                dist = np.array([c[2] for c in cand], dtype=np.float32)
                out.append((idx, dist))
            else:
                out.append(None)
        return out



def apply_mwc_weights(target_obj, mbs, n, q, tau, r_falloff_multiplier,
                      use_normal_filter=True, normal_p=1.0,
                      use_smoothing=False, smoothing_strength=0.5, smoothing_iterations=3,
                      use_geodesic=False, use_custom_curve=False, curve_node=None,
                      geodesic_mode='THREAD', progress_cb=None):
    """
    Performs high-performance NumPy-vectorized MWC blending.
    v1.1: optional SciPy acceleration (vectorized KD queries + C Dijkstra),
    reduceat-based Laplacian smoothing, BLAS-friendly RBF extrapolation and a
    fully vectorized IDW fallback. Falls back to the previous implementations
    whenever SciPy is unavailable.
    """

    def _report(frac, msg=""):
        if progress_cb is None:
            return
        try:
            progress_cb(min(max(float(frac), 0.0), 1.0), msg)
        except Exception:
            pass

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
        
        if G == 0 or M == 0 or V == 0:
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
        
        # 3. Point-cloud search structures.
        # One tree over target vertices (range + nearest queries) and trees
        # over metaballs (global + per symmetry side) for family assignment
        # and the IDW fallback. Each side tree holds L+Central (or R+Central)
        # so every vertex resolves its nearest matching-symmetry metaball in
        # a single query.
        _report(0.08, "Building search structures")
        target_tree = _PointTree(target_cos)

        mb_indices_by_sym = {}
        side_trees = {}
        for sym_key, allowed_syms in ((1, (1, 3)), (2, (2, 3))):
            local_indices = np.where(np.isin(MB_Sym, allowed_syms))[0]
            mb_indices_by_sym[sym_key] = local_indices
            side_trees[sym_key] = (_PointTree(C[local_indices])
                                   if len(local_indices) > 0 else None)
        kd_mbs_all = _PointTree(C)

        # Find closest metaball for each target vertex to inherit family ID
        # (fully vectorized per symmetry side).
        closest_indices = np.zeros(V, dtype=np.int64)
        for sym_key in (1, 2):
            vids = np.where(target_labels_sym == sym_key)[0]
            if len(vids) == 0:
                continue
            tree = side_trees[sym_key]
            if tree is not None:
                _d, li = tree.query_nearest(target_cos[vids], k=1)
                closest_indices[vids] = mb_indices_by_sym[sym_key][li[:, 0]]
            else:
                # No metaball of matching/central symmetry exists at all; fall
                # back to the globally nearest metaball regardless of symmetry.
                _d, gi = kd_mbs_all.query_nearest(target_cos[vids], k=1)
                closest_indices[vids] = gi[:, 0]

        vertex_families = F[closest_indices]
        
        # Main weights blending variables
        final_weights = np.zeros((V, G), dtype=np.float32)
        den_accum = np.zeros(V, dtype=np.float32)
        
        # Shared edge arrays (geodesic adjacency + Laplacian smoothing).
        visited_dists_dict = {}
        edges_arr = None
        edge_lens = None
        need_edges = use_geodesic or (use_smoothing and smoothing_iterations > 0 and smoothing_strength > 0.0)
        if need_edges:
            E = len(mesh.edges)
            edges_arr = np.empty(E * 2, dtype=np.int32)
            mesh.edges.foreach_get("vertices", edges_arr)
            edges_arr = edges_arr.reshape((E, 2))
            e_cos = target_cos[edges_arr]  # (E, 2, 3)
            edge_lens = np.linalg.norm(e_cos[:, 0, :] - e_cos[:, 1, :], axis=1)

        if use_geodesic:
            _report(0.15, "Geodesic distances")
            tasks = []
            if M > 0:
                sd, si = target_tree.query_nearest(C, k=1)
                rjs_all = np.maximum(R, 1e-6)
                for j in range(M):
                    if sd[j] < rjs_all[j]:
                        tasks.append((j, int(si[j, 0]), float(sd[j]), float(rjs_all[j])))

            scipy_done = False
            if HAS_SCIPY and len(tasks) > 0:
                try:
                    visited_dists_dict = _geodesic_scipy_batch(edges_arr, edge_lens, V, tasks)
                    scipy_done = True
                except Exception as e:
                    print("MWC scipy geodesic failed, falling back to heapq engine:", e)
                    visited_dists_dict = {}

            if not scipy_done:
                adj = {i: [] for i in range(V)}
                for e_i in range(edges_arr.shape[0]):
                    v1 = int(edges_arr[e_i, 0])
                    v2 = int(edges_arr[e_i, 1])
                    dist = float(edge_lens[e_i])
                    adj[v1].append((v2, dist))
                    adj[v2].append((v1, dist))

                if geodesic_mode in ('THREAD', 'PROCESS') and len(tasks) > 1:
                    import concurrent.futures
                    import os

                    num_workers = os.cpu_count() or 4
                    chunks = [[] for _ in range(num_workers)]
                    for idx, task in enumerate(tasks):
                        chunks[idx % num_workers].append(task)
                    chunks = [c for c in chunks if c]

                    executor_cls = (concurrent.futures.ProcessPoolExecutor
                                     if geodesic_mode == 'PROCESS'
                                     else concurrent.futures.ThreadPoolExecutor)

                    try:
                        with executor_cls(max_workers=len(chunks)) as executor:
                            futures = [executor.submit(dijkstra_chunk_worker, chunk, adj) for chunk in chunks]
                            for fut in concurrent.futures.as_completed(futures):
                                visited_dists_dict.update(fut.result())
                    except Exception as e:
                        # ProcessPoolExecutor can fail to spawn in some embedded/Blender
                        # environments (pickling issues, missing __main__ guard, etc).
                        # Fall back to sequential rather than losing the whole Apply.
                        print("MWC geodesic parallel execution failed, falling back to sequential:", e)
                        visited_dists_dict = {}
                        for j, v_start, d_start, rj in tasks:
                            dd = dijkstra_pruned(adj, v_start, d_start, rj)
                            if dd:
                                idx_a = np.fromiter(dd.keys(), dtype=np.int32, count=len(dd))
                                ds_a = np.fromiter(dd.values(), dtype=np.float32, count=len(dd))
                                visited_dists_dict[j] = (idx_a, ds_a)

                else:
                    # Sequential fallback (or if only 1 task)
                    for j, v_start, d_start, rj in tasks:
                        dd = dijkstra_pruned(adj, v_start, d_start, rj)
                        if dd:
                            idx_a = np.fromiter(dd.keys(), dtype=np.int32, count=len(dd))
                            ds_a = np.fromiter(dd.values(), dtype=np.float32, count=len(dd))
                            visited_dists_dict[j] = (idx_a, ds_a)

        # Precompute Custom Curve LUT if active
        if use_custom_curve and curve_node:
            mapping = curve_node.mapping
            combined_curve = mapping.curves[3]
            lut_x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
            lut_y = np.array([mapping.evaluate(combined_curve, x) for x in lut_x], dtype=np.float32)
        else:
            lut_x = None
            lut_y = None
            
        # 4. Influence calculations.
        # Euclidean mode: resolve ALL metaball-to-vertex ranges in one bulk
        # query pass (C-level with scipy) instead of querying per metaball
        # inside the blending loop.
        _report(0.35, "Range queries")
        range_results = None
        if not use_geodesic:
            rjs = np.maximum(R, 1e-6)
            range_results = target_tree.query_range_bulk(C, rjs)

        _report(0.45, "Blending")
        for j in range(M):
            rj = max(R[j], 1e-6)
            
            if use_geodesic:
                res = visited_dists_dict.get(j)
                if res is None:
                    continue
                indices, dists = res
                if len(indices) == 0:
                    continue
            else:
                res = range_results[j]
                if res is None:
                    continue
                indices, dists = res
                
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
                f *= normal_filter_power(dot, normal_p)
                
            fq = f**q
            
            # Accumulate blending values
            den_accum[active_indices] += fq
            final_weights[active_indices] += fq[:, np.newaxis] * W[j]
            
        # Divide by denominator
        mask_active = den_accum > 0.0
        final_weights[mask_active] /= den_accum[mask_active][:, np.newaxis]
        
        # 5. RBF Extrapolation for orphans (vertices with no active metaballs in range).
        # Pairwise distances via the |a-b|^2 = |a|^2 + |b|^2 - 2ab Gram identity
        # (BLAS matmul, 3x less memory than broadcasting the coordinate delta),
        # computed around a local origin to avoid catastrophic cancellation.
        _report(0.62, "RBF extrapolation")
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
                        
                    v_cos = target_cos[orphans_fam_global].astype(np.float64)
                    mb_cos = C[mb_fam_indices].astype(np.float64)
                    origin = mb_cos.mean(axis=0)
                    v_cos -= origin
                    mb_cos -= origin

                    # Squared distance matrix of shape (V_o, M_f)
                    d2 = (v_cos * v_cos).sum(axis=1)[:, np.newaxis] \
                        + (mb_cos * mb_cos).sum(axis=1)[np.newaxis, :] \
                        - 2.0 * (v_cos @ mb_cos.T)
                    np.maximum(d2, 0.0, out=d2)
                    phi = np.exp(-d2 / (float(R_falloff) ** 2))
                    
                    if use_normal_filter:
                        v_nos = target_nos[orphans_fam_global]
                        mb_nos = N[mb_fam_indices]
                        dot_matrix = np.dot(v_nos, mb_nos.T)
                        phi *= normal_filter_power(dot_matrix, normal_p)
                        
                    sum_phi = np.sum(phi, axis=1)
                    
                    valid_rbf_mask = sum_phi > tau
                    if np.any(valid_rbf_mask):
                        valid_global_indices = orphans_fam_global[valid_rbf_mask]
                        num = np.dot(phi[valid_rbf_mask], W[mb_fam_indices])
                        scale_den = sum_phi[valid_rbf_mask][:, np.newaxis]
                        final_weights[valid_global_indices] = num / scale_den
                    
        # 6. Fallback for completely unassigned orphans using Normal-Filtered
        # IDW (Inverse Distance Weighting) -- fully vectorized per symmetry side.
        _report(0.75, "IDW fallback")
        sums = np.sum(final_weights, axis=1)
        unassigned_mask = sums <= 1e-6
        if np.any(unassigned_mask):
            ua = np.where(unassigned_mask)[0]
            ua_sym = target_labels_sym[ua]

            for sym_key in (1, 2):
                sel = np.where(ua_sym == sym_key)[0]
                if len(sel) == 0:
                    continue
                vids = ua[sel]
                tree = side_trees[sym_key]
                if tree is not None:
                    kk = min(4, len(mb_indices_by_sym[sym_key]))
                    d, li = tree.query_nearest(target_cos[vids], k=kk)
                    gi = mb_indices_by_sym[sym_key][li]
                else:
                    kk = min(4, M)
                    d, gi = kd_mbs_all.query_nearest(target_cos[vids], k=kk)

                d = np.maximum(d, 1e-6)
                valid = np.isfinite(d)
                w = np.where(valid, 1.0 / (d * d), 0.0)

                if use_normal_filter:
                    dot = np.einsum('vi,vki->vk', target_nos[vids], N[gi])
                    wn = w * normal_filter_power(dot, normal_p)
                else:
                    wn = w

                w_sum = wn.sum(axis=1)
                strict = w_sum <= 1e-6
                if np.any(strict):
                    # Fallback to pure distance-based IDW (no normal filter).
                    wn[strict] = w[strict]
                    w_sum = wn.sum(axis=1)

                ok = w_sum > 1e-6
                if np.any(ok):
                    num = np.einsum('vk,vkg->vg', wn[ok], W[gi[ok]])
                    final_weights[vids[ok]] = num / w_sum[ok][:, np.newaxis]
                if not np.all(ok):
                    # Absolute fallback to the single closest metaball.
                    bad = vids[~ok]
                    final_weights[bad] = W[closest_indices[bad]]
            
        # 6.5. Apply Laplacian Smoothing if enabled.
        # Segment-summed via np.add.reduceat over a CSR-style ordering of the
        # directed edge list -- orders of magnitude faster than np.add.at and
        # free of per-iteration (V x G) scatter allocations.
        _report(0.85, "Smoothing")
        if use_smoothing and smoothing_iterations > 0 and smoothing_strength > 0.0:
            if edges_arr is None:
                E = len(mesh.edges)
                edges_arr = np.empty(E * 2, dtype=np.int32)
                mesh.edges.foreach_get("vertices", edges_arr)
                edges_arr = edges_arr.reshape((E, 2))

            src_dir = np.concatenate([edges_arr[:, 0], edges_arr[:, 1]]).astype(np.int64)
            dst_dir = np.concatenate([edges_arr[:, 1], edges_arr[:, 0]]).astype(np.int64)
            order = np.argsort(src_dir, kind='stable')
            src_sorted = src_dir[order]
            dst_sorted = dst_dir[order]
            uniq_src, starts = np.unique(src_sorted, return_index=True)
            deg = np.diff(np.append(starts, len(src_sorted))).astype(np.float32)

            for _ in range(smoothing_iterations):
                neigh = final_weights[dst_sorted]
                seg_sums = np.add.reduceat(neigh, starts, axis=0)
                avg_weights = np.zeros_like(final_weights)
                avg_weights[uniq_src] = seg_sums / deg[:, np.newaxis]

                final_weights = (1.0 - smoothing_strength) * final_weights + smoothing_strength * avg_weights
            
        # 7. Normalize all final weights to sum up to 1.0
        _report(0.95, "Writing vertex groups")
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

        _report(1.0, "Done")
    finally:
        if has_eval_mesh:
            eval_obj.to_mesh_clear()
