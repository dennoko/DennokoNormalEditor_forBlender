"""Core mathematical algorithms for Dennoko Normal Editor.

Ports Unity's FalloffUtil, SmoothNormalCalculator, NormalDeltaApplier, and StrengthMaskUtil
to Python + NumPy for real-time performance on high-poly avatar meshes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np

DEFAULT_SEAM_WELD_TOLERANCE = 0.001
DEFAULT_GEO_SMOOTH_ITERATIONS = 2
DEFAULT_SMOOTH_ITERATIONS = 2


# ---------------------------------------------------------------------------
# Falloff (FalloffUtil.cs)
# ---------------------------------------------------------------------------

def falloff_weight(t: float | np.ndarray, falloff_type: str = "SMOOTH") -> float | np.ndarray:
    """Compute falloff weight for normalized distance t in [0, 1] (0 = center, 1 = edge)."""
    if isinstance(t, np.ndarray):
        s = np.clip(1.0 - t, 0.0, 1.0)
        mask = (t < 1.0).astype(np.float32)
        if falloff_type == "LINEAR":
            w = s
        elif falloff_type == "SHARP":
            w = s * s
        elif falloff_type == "CONSTANT":
            w = np.ones_like(s)
        else:  # SMOOTH
            w = s * s * (3.0 - 2.0 * s)
        return (w * mask).astype(np.float32)

    if t >= 1.0:
        return 0.0
    if t <= 0.0:
        return 1.0
    s = 1.0 - t
    if falloff_type == "LINEAR":
        return s
    if falloff_type == "SHARP":
        return s * s
    if falloff_type == "CONSTANT":
        return 1.0
    return s * s * (3.0 - 2.0 * s)


# ---------------------------------------------------------------------------
# Vector Math: SafeNormalize, RotateVector (Rodrigues), SlerpNormal
# ---------------------------------------------------------------------------

def safe_normalize_vec(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Normalize a 1D (3,) or 2D (N, 3) vector array safely."""
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        sq = float(np.dot(v, v))
        if sq > 1e-12:
            return v * (1.0 / math.sqrt(sq))
        if fallback is not None:
            fb = np.asarray(fallback, dtype=np.float32)
            fb_sq = float(np.dot(fb, fb))
            if fb_sq > 1e-12:
                return fb * (1.0 / math.sqrt(fb_sq))
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    sq = np.einsum("ij,ij->i", v, v)
    valid = sq > 1e-12
    inv_len = np.where(valid, 1.0 / np.sqrt(np.where(valid, sq, 1.0)), 0.0).astype(np.float32)
    out = v * inv_len[:, None]
    if not np.all(valid):
        if fallback is not None:
            fb = safe_normalize_vec(fallback)
            if fb.ndim == 1:
                out[~valid] = fb
            else:
                out[~valid] = fb[~valid]
        else:
            out[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return out


def rotate_vector(from_vec: np.ndarray, to_vec: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply the minimal rotation mapping unit vector 'from_vec' to 'to_vec' onto 'v' (Rodrigues).

    Supports both a single (3,) vector 'v' and a batch (N, 3) array 'v'.
    """
    f = safe_normalize_vec(np.asarray(from_vec, dtype=np.float32))
    t = safe_normalize_vec(np.asarray(to_vec, dtype=np.float32), f)
    v_arr = np.asarray(v, dtype=np.float32)

    cos_theta = float(np.dot(f, t))
    axis = np.cross(f, t)
    sin_theta = float(np.linalg.norm(axis))

    if sin_theta <= 1e-7:
        if cos_theta >= 0.0:
            return v_arr.copy()
        # 180-degree reflection around an orthogonal axis
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(float(f[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        perp = safe_normalize_vec(np.cross(f, ref))
        if v_arr.ndim == 1:
            return 2.0 * float(np.dot(perp, v_arr)) * perp - v_arr
        dots = np.einsum("ij,j->i", v_arr, perp)[:, None]
        return 2.0 * dots * perp[None, :] - v_arr

    k = (axis / sin_theta).astype(np.float32)
    if v_arr.ndim == 1:
        return v_arr * cos_theta + np.cross(k, v_arr) * sin_theta + k * (float(np.dot(k, v_arr)) * (1.0 - cos_theta))

    cross_kv = np.cross(k[None, :], v_arr)
    dot_kv = np.einsum("ij,j->i", v_arr, k)[:, None]
    return (v_arr * cos_theta + cross_kv * sin_theta + k[None, :] * (dot_kv * (1.0 - cos_theta))).astype(np.float32)


def slerp_normal(from_vec: np.ndarray, to_vec: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Spherical linear interpolation between unit vectors 'from_vec' and 'to_vec' by weight 't' in [0, 1].

    Supports both single (3,) vectors and batch (N, 3) arrays with (N,) or scalar 't'.
    """
    f = safe_normalize_vec(from_vec)
    to = safe_normalize_vec(to_vec, f)

    if f.ndim == 1 and to.ndim == 1 and not isinstance(t, np.ndarray):
        t_val = max(0.0, min(1.0, float(t)))
        if t_val <= 0.0:
            return f
        if t_val >= 1.0:
            return to
        dot = max(-1.0, min(1.0, float(np.dot(f, to))))
        if dot > 0.9995:
            return safe_normalize_vec(f + (to - f) * t_val, to)
        if dot < -0.9995:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(float(f[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
            ortho = safe_normalize_vec(np.cross(f, ref))
            angle = math.pi * t_val
            return (f * math.cos(angle) + ortho * math.sin(angle)).astype(np.float32)
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        if sin_theta <= 1e-6:
            return safe_normalize_vec(f + (to - f) * t_val, to)
        w0 = math.sin((1.0 - t_val) * theta) / sin_theta
        w1 = math.sin(t_val * theta) / sin_theta
        return safe_normalize_vec(f * w0 + to * w1, to)

    # Vectorized (N, 3) batch path
    if f.ndim == 1:
        f = np.broadcast_to(f, to.shape).copy()
    if to.ndim == 1:
        to = np.broadcast_to(to, f.shape).copy()

    t_arr = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    if t_arr.ndim == 0:
        t_arr = np.full(f.shape[0], float(t_arr), dtype=np.float32)

    dot = np.clip(np.einsum("ij,ij->i", f, to), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    use_slerp = (dot <= 0.9995) & (dot >= -0.9995) & (sin_theta > 1e-6)
    safe_sin = np.where(use_slerp, sin_theta, 1.0)
    w0 = np.where(use_slerp, np.sin((1.0 - t_arr) * theta) / safe_sin, 1.0 - t_arr)[:, None]
    w1 = np.where(use_slerp, np.sin(t_arr * theta) / safe_sin, t_arr)[:, None]
    res = safe_normalize_vec(f * w0 + to * w1, to)

    # Handle anti-parallel (dot < -0.9995)
    anti = dot < -0.9995
    if np.any(anti):
        for idx in np.nonzero(anti)[0]:
            res[idx] = slerp_normal(f[idx], to[idx], float(t_arr[idx]))

    return res


def evaluate_normal(base_normal: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Compute normalize(base_normal + delta), falling back to base_normal if degenerate."""
    return safe_normalize_vec(np.asarray(base_normal, dtype=np.float32) + np.asarray(delta, dtype=np.float32), base_normal)


# ---------------------------------------------------------------------------
# Cross-Mesh Seam Weld & Smooth Geometric Normals (SmoothNormalCalculator.cs)
# ---------------------------------------------------------------------------

@dataclass
class MeshTopologyInput:
    world_vertices: np.ndarray  # (V, 3) float32
    triangles: np.ndarray       # (T, 3) int32


@dataclass
class CrossMeshTopology:
    group_count: int
    vertex_to_group: list[np.ndarray]       # per-mesh (V,) int32 -> group_id
    group_positions: np.ndarray             # (G, 3) float32
    smooth_geometric_normals: np.ndarray    # (G, 3) float32
    neighbor_offsets: np.ndarray            # (G + 1,) int32
    neighbor_indices: np.ndarray            # (E,) int32
    smoothed_buffer_a: np.ndarray           # (G, 3) float32
    smoothed_buffer_b: np.ndarray           # (G, 3) float32


def build_cross_mesh_weld_groups(
    meshes: Sequence[MeshTopologyInput],
    weld_tolerance: float = DEFAULT_SEAM_WELD_TOLERANCE,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Group vertices across all registered meshes whose world distance is <= weld_tolerance."""
    tol = max(0.0, float(weld_tolerance))
    tol_sq = tol * tol
    cell_size = max(tol * 2.0, 0.002)
    inv_cell_size = 1.0 / cell_size

    vertex_to_group: list[np.ndarray] = []
    representatives: list[np.ndarray] = []
    grid: dict[tuple[int, int, int], list[int]] = {}

    for m_input in meshes:
        verts = m_input.world_vertices
        if verts is None or len(verts) == 0:
            vertex_to_group.append(np.empty(0, dtype=np.int32))
            continue

        n_v = len(verts)
        v_map = np.empty(n_v, dtype=np.int32)
        cell_coords = np.floor(verts * inv_cell_size).astype(np.int32)

        for i in range(n_v):
            pos = verts[i]
            cx, cy, cz = int(cell_coords[i, 0]), int(cell_coords[i, 1]), int(cell_coords[i, 2])
            found_group = -1
            best_dist_sq = tol_sq

            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        bucket = grid.get((cx + dx, cy + dy, cz + dz))
                        if not bucket:
                            continue
                        for cand in bucket:
                            diff = representatives[cand] - pos
                            dist_sq = float(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
                            if dist_sq <= best_dist_sq:
                                best_dist_sq = dist_sq
                                found_group = cand

            if found_group < 0:
                found_group = len(representatives)
                representatives.append(pos.copy())
                own_key = (cx, cy, cz)
                bucket = grid.get(own_key)
                if bucket is None:
                    grid[own_key] = [found_group]
                else:
                    bucket.append(found_group)

            v_map[i] = found_group

        vertex_to_group.append(v_map)

    if representatives:
        group_positions = np.array(representatives, dtype=np.float32)
    else:
        group_positions = np.zeros((0, 3), dtype=np.float32)

    return vertex_to_group, group_positions


def build_adjacency(
    meshes: Sequence[MeshTopologyInput],
    vertex_to_group: Sequence[np.ndarray],
    group_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build CSR adjacency graph across welded vertex groups from all mesh triangles."""
    if group_count <= 0:
        return np.zeros(1, dtype=np.int32), np.zeros(0, dtype=np.int32)

    edge_chunks: list[np.ndarray] = []
    for m_idx, m_input in enumerate(meshes):
        tris = m_input.triangles
        v_map = vertex_to_group[m_idx]
        if tris is None or len(tris) == 0 or len(v_map) == 0:
            continue
        g_tris = v_map[tris]  # (T, 3)
        g0, g1, g2 = g_tris[:, 0], g_tris[:, 1], g_tris[:, 2]
        e01 = np.stack([g0, g1], axis=1)
        e10 = np.stack([g1, g0], axis=1)
        e12 = np.stack([g1, g2], axis=1)
        e21 = np.stack([g2, g1], axis=1)
        e20 = np.stack([g2, g0], axis=1)
        e02 = np.stack([g0, g2], axis=1)
        all_e = np.concatenate([e01, e10, e12, e21, e20, e02], axis=0)
        valid = all_e[:, 0] != all_e[:, 1]
        if np.any(valid):
            edge_chunks.append(all_e[valid])

    if not edge_chunks:
        return np.zeros(group_count + 1, dtype=np.int32), np.zeros(0, dtype=np.int32)

    edges = np.concatenate(edge_chunks, axis=0)
    # Deduplicate directed edges (a, b) via 64-bit key a * group_count + b
    keys = edges[:, 0].astype(np.int64) * int(group_count) + edges[:, 1].astype(np.int64)
    unique_keys = np.unique(keys)
    src = (unique_keys // int(group_count)).astype(np.int32)
    dst = (unique_keys % int(group_count)).astype(np.int32)

    counts = np.bincount(src, minlength=group_count).astype(np.int32)
    offsets = np.empty(group_count + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets, dst


def laplacian_smooth_groups(
    group_count: int,
    neighbor_offsets: np.ndarray,
    neighbor_indices: np.ndarray,
    iterations: int,
    buffer: np.ndarray,
    scratch: np.ndarray | None = None,
) -> np.ndarray:
    """Laplacian smooth group normals across CSR neighbors."""
    if iterations <= 0 or group_count <= 0 or len(neighbor_indices) == 0:
        return buffer

    counts = neighbor_offsets[1:] - neighbor_offsets[:-1]
    has_neighbors = counts > 0
    safe_counts = np.where(has_neighbors, counts, 1).astype(np.float32)[:, None]

    # Precompute source group index per CSR entry for fast np.add.at
    row_ids = np.repeat(np.arange(group_count, dtype=np.int32), counts)

    for _ in range(iterations):
        neighbor_sum = np.zeros((group_count, 3), dtype=np.float32)
        np.add.at(neighbor_sum, row_ids, buffer[neighbor_indices])
        neighbor_avg = neighbor_sum / safe_counts
        blended = np.where(has_neighbors[:, None], buffer + neighbor_avg, buffer)
        buffer[:] = safe_normalize_vec(blended, buffer)

    return buffer


def compute_smooth_geometric_normals(
    meshes: Sequence[MeshTopologyInput],
    vertex_to_group: Sequence[np.ndarray],
    group_count: int,
    neighbor_offsets: np.ndarray,
    neighbor_indices: np.ndarray,
    smooth_iterations: int = DEFAULT_GEO_SMOOTH_ITERATIONS,
) -> np.ndarray:
    """Compute area-weighted smooth geometric normals across welded groups + Laplacian smoothing."""
    result = np.zeros((group_count, 3), dtype=np.float32)
    if group_count == 0:
        return result

    for m_idx, m_input in enumerate(meshes):
        verts = m_input.world_vertices
        tris = m_input.triangles
        v_map = vertex_to_group[m_idx]
        if verts is None or tris is None or len(tris) == 0 or len(v_map) == 0:
            continue

        p0 = verts[tris[:, 0]]
        p1 = verts[tris[:, 1]]
        p2 = verts[tris[:, 2]]
        face_normals = np.cross(p1 - p0, p2 - p0).astype(np.float32)

        np.add.at(result, v_map[tris[:, 0]], face_normals)
        np.add.at(result, v_map[tris[:, 1]], face_normals)
        np.add.at(result, v_map[tris[:, 2]], face_normals)

    result = safe_normalize_vec(result)
    laplacian_smooth_groups(group_count, neighbor_offsets, neighbor_indices, smooth_iterations, result)
    return result


def build_topology(
    meshes: Sequence[MeshTopologyInput],
    weld_tolerance: float = DEFAULT_SEAM_WELD_TOLERANCE,
    geo_smooth_iterations: int = DEFAULT_GEO_SMOOTH_ITERATIONS,
) -> CrossMeshTopology:
    """Build complete CrossMeshTopology for all registered target meshes."""
    if not meshes:
        return CrossMeshTopology(
            group_count=0,
            vertex_to_group=[],
            group_positions=np.zeros((0, 3), dtype=np.float32),
            smooth_geometric_normals=np.zeros((0, 3), dtype=np.float32),
            neighbor_offsets=np.zeros(1, dtype=np.int32),
            neighbor_indices=np.zeros(0, dtype=np.int32),
            smoothed_buffer_a=np.zeros((0, 3), dtype=np.float32),
            smoothed_buffer_b=np.zeros((0, 3), dtype=np.float32),
        )

    v_to_g, g_pos = build_cross_mesh_weld_groups(meshes, weld_tolerance)
    g_count = len(g_pos)
    offsets, indices = build_adjacency(meshes, v_to_g, g_count)
    smooth_geo = compute_smooth_geometric_normals(
        meshes, v_to_g, g_count, offsets, indices, geo_smooth_iterations
    )
    return CrossMeshTopology(
        group_count=g_count,
        vertex_to_group=v_to_g,
        group_positions=g_pos,
        smooth_geometric_normals=smooth_geo,
        neighbor_offsets=offsets,
        neighbor_indices=indices,
        smoothed_buffer_a=np.zeros((g_count, 3), dtype=np.float32),
        smoothed_buffer_b=np.zeros((g_count, 3), dtype=np.float32),
    )


def compute_smoothed_group_normals(
    topology: CrossMeshTopology,
    current_world_normals: Sequence[np.ndarray],
    iterations: int = DEFAULT_SMOOTH_ITERATIONS,
) -> np.ndarray:
    """Average current world normals within each weld group and Laplacian smooth across neighbors."""
    g_count = topology.group_count
    if g_count == 0:
        return np.zeros((0, 3), dtype=np.float32)

    out = topology.smoothed_buffer_a
    out.fill(0.0)

    for m_idx, normals in enumerate(current_world_normals):
        if m_idx >= len(topology.vertex_to_group):
            break
        v_map = topology.vertex_to_group[m_idx]
        if normals is None or len(normals) == 0 or len(v_map) == 0:
            continue
        count = min(len(normals), len(v_map))
        np.add.at(out, v_map[:count], normals[:count])

    out[:] = safe_normalize_vec(out, topology.smooth_geometric_normals)
    laplacian_smooth_groups(
        g_count,
        topology.neighbor_offsets,
        topology.neighbor_indices,
        iterations,
        out,
        topology.smoothed_buffer_b,
    )
    return out


# ---------------------------------------------------------------------------
# UV0 Strength Mask (StrengthMaskUtil.cs)
# ---------------------------------------------------------------------------

def sample_mask_bilinear(luminance_grid: np.ndarray, uvs: np.ndarray, clamp: bool = True) -> np.ndarray:
    """Bilinear sample (H, W) luminance array at (N, 2) UV coordinates in [0, 1]."""
    h, w = luminance_grid.shape
    if h == 0 or w == 0 or len(uvs) == 0:
        return np.ones(len(uvs), dtype=np.float32)

    x = uvs[:, 0] * w - 0.5
    y = uvs[:, 1] * h - 0.5
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)

    if clamp:
        xa = np.clip(x0, 0, w - 1)
        xb = np.clip(x0 + 1, 0, w - 1)
        ya = np.clip(y0, 0, h - 1)
        yb = np.clip(y0 + 1, 0, h - 1)
    else:
        xa = np.mod(x0, w)
        xb = np.mod(x0 + 1, w)
        ya = np.mod(y0, h)
        yb = np.mod(y0 + 1, h)

    c00 = luminance_grid[ya, xa]
    c10 = luminance_grid[ya, xb]
    c01 = luminance_grid[yb, xa]
    c11 = luminance_grid[yb, xb]

    bottom = c00 + (c10 - c00) * fx
    top = c01 + (c11 - c01) * fx
    return np.clip(bottom + (top - bottom) * fy, 0.0, 1.0).astype(np.float32)


def compute_vertex_mask_from_blender_image(mesh, image) -> np.ndarray | None:
    """Compute per-vertex mask values in [0, 1] from a Blender Image using UV0.

    Matches StrengthMaskUtil: luminance = 0.299*R + 0.587*G + 0.114*B.
    If a vertex has multiple loops with different UVs, takes the minimum across its loops.
    Returns None if mesh has no UV layer or image is empty.
    """
    if mesh is None or image is None:
        return None
    if not mesh.uv_layers or len(mesh.uv_layers) == 0:
        return None
    n_verts = len(mesh.vertices)
    n_loops = len(mesh.loops)
    if n_verts == 0 or n_loops == 0:
        return None

    w, h = image.size[0], image.size[1]
    if w <= 0 or h <= 0:
        return None

    channels = image.channels
    if channels < 3:
        return None

    px = np.empty(w * h * channels, dtype=np.float32)
    image.pixels.foreach_get(px)
    px = px.reshape((h, w, channels))
    lum = (0.299 * px[:, :, 0] + 0.587 * px[:, :, 1] + 0.114 * px[:, :, 2]).astype(np.float32)

    uv_layer = mesh.uv_layers[0]
    uvs = np.empty(n_loops * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", uvs)
    uvs = uvs.reshape((n_loops, 2))

    loop_samples = sample_mask_bilinear(lum, uvs, clamp=True)

    loop_verts = np.empty(n_loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_verts)

    vert_mask = np.ones(n_verts, dtype=np.float32)
    np.minimum.at(vert_mask, loop_verts, loop_samples)
    return vert_mask


def combine_masks_by_group(
    vertex_masks: Sequence[np.ndarray | None],
    vertex_to_group: Sequence[np.ndarray],
    group_count: int,
) -> list[np.ndarray | None]:
    """Propagate the minimum mask value across each weld group (matching StrengthMaskUtil.CombineByGroup)."""
    count = len(vertex_masks)
    if not vertex_to_group or group_count <= 0:
        return list(vertex_masks)

    group_min = np.ones(group_count, dtype=np.float32)
    any_mask = False

    for t in range(count):
        own = vertex_masks[t]
        if own is None or t >= len(vertex_to_group):
            continue
        v_map = vertex_to_group[t]
        if v_map is None or len(own) != len(v_map):
            continue
        np.minimum.at(group_min, v_map, own)
        any_mask = True

    if not any_mask:
        return [None] * count

    result: list[np.ndarray | None] = []
    for t in range(count):
        own = vertex_masks[t]
        if t >= len(vertex_to_group) or vertex_to_group[t] is None:
            result.append(own)
            continue
        v_map = vertex_to_group[t]
        combined = group_min[v_map].copy()
        if own is not None and len(own) == len(combined):
            np.minimum(combined, own, out=combined)
        if np.all(combined >= 1.0 - 1e-6):
            result.append(None)
        else:
            result.append(combined)

    return result
