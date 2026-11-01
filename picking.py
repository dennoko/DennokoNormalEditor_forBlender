"""Target mesh state, skinning matrix evaluation, and BVHTree occlusion-culled vertex picking."""

from __future__ import annotations

from dataclasses import dataclass, field
import bpy
from bpy_extras import view3d_utils
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
import numpy as np

from . import modifier_utils
from . import core_math


@dataclass(eq=False)
class TargetState:
    obj: bpy.types.Object
    mask_image: bpy.types.Image | None = None

    # Geometry & normals
    base_local_normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    working_deltas: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    snapshot_deltas: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))

    world_vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    world_normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    snapshot_world_normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    triangles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int32))

    # Per-vertex (N, 3, 3) or object-level (3, 3) NORMAL transforms between local and world space.
    # These are inverse-transpose matrices of the position transforms, so they stay correct for
    # non-uniformly scaled objects / bones.
    local_to_world_rot: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    world_to_local_rot: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    has_per_vertex_skin: bool = False

    # Mask
    raw_vertex_mask: np.ndarray | None = None
    mask: np.ndarray | None = None

    # Occlusion raycasting BVHTree in world space
    bvh: BVHTree | None = None

    touched: bool = False

    def local_to_world_normals(self, local_n: np.ndarray) -> np.ndarray:
        """Transform (N, 3) or (M, 3) local normals to unit world normals."""
        if not self.has_per_vertex_skin:
            world_n = local_n @ self.local_to_world_rot.T
        else:
            world_n = np.einsum("nij,nj->ni", self.local_to_world_rot, local_n)
        return core_math.safe_normalize_vec(world_n)

    def local_to_world_subset(self, indices: np.ndarray, local_n: np.ndarray) -> np.ndarray:
        """Transform a subset of local normals at 'indices' to unit world normals."""
        if not self.has_per_vertex_skin:
            world_n = local_n @ self.local_to_world_rot.T
        else:
            world_n = np.einsum("nij,nj->ni", self.local_to_world_rot[indices], local_n)
        return core_math.safe_normalize_vec(world_n)

    def world_to_local_subset(self, indices: np.ndarray, world_n: np.ndarray) -> np.ndarray:
        """Transform a subset of world normals at 'indices' back to unit local rest-pose normals."""
        if not self.has_per_vertex_skin:
            local_n = world_n @ self.world_to_local_rot.T
        else:
            local_n = np.einsum("nij,nj->ni", self.world_to_local_rot[indices], world_n)
        return core_math.safe_normalize_vec(local_n, self.base_local_normals[indices])

    def update_world_normals_from_working(self) -> None:
        """Recompute world_normals from base_local_normals + working_deltas."""
        if len(self.base_local_normals) == 0:
            return
        local_n = core_math.evaluate_normal(self.base_local_normals, self.working_deltas)
        self.world_normals = self.local_to_world_normals(local_n)


def _normal_transforms(pos_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (local->world, world->local) normal transforms for (3, 3) or (N, 3, 3) position matrices.

    Normals transform with the inverse-transpose of the position matrix: n_w ~ (M^-1)^T n_l and
    therefore n_l ~ M^T n_w. Singular matrices fall back to identity.
    """
    eye = np.eye(3, dtype=np.float32)
    if pos_mat.ndim == 2:
        if abs(float(np.linalg.det(pos_mat))) <= 1e-12:
            return eye.copy(), eye.copy()
        l2w = np.linalg.inv(pos_mat).T.astype(np.float32)
        w2l = pos_mat.T.astype(np.float32)
        return l2w, w2l

    dets = np.linalg.det(pos_mat)
    valid = np.abs(dets) > 1e-12
    safe = np.where(valid[:, None, None], pos_mat, eye[None, :, :])
    l2w = np.transpose(np.linalg.inv(safe), (0, 2, 1)).astype(np.float32)
    w2l = np.transpose(safe, (0, 2, 1)).astype(np.float32)
    return l2w, w2l


def _compute_skin_rotations(obj: bpy.types.Object, mesh: bpy.types.Mesh) -> tuple[np.ndarray, np.ndarray, bool]:
    """Compute local<->world 3x3 normal transforms, accounting for posed Armature if active."""
    mw = np.array(obj.matrix_world.to_3x3(), dtype=np.float32)
    obj_l2w, obj_w2l = _normal_transforms(mw)

    # Check if an active posed Armature modifier exists
    arm_obj = None
    for m in obj.modifiers:
        if m.type == "ARMATURE" and m.show_viewport and m.object is not None:
            if getattr(m.object.data, "pose_position", "POSE") != "REST":
                arm_obj = m.object
                break

    if arm_obj is None or not obj.vertex_groups:
        return obj_l2w, obj_w2l, False

    # Check if any bone is actually posed away from rest
    pose_bones = arm_obj.pose.bones
    rest_bones = arm_obj.data.bones
    amt_mw = arm_obj.matrix_world
    amt_mw_inv = amt_mw.inverted_safe()

    vg_idx_to_mat: dict[int, np.ndarray] = {}
    any_posed = False
    for vg in obj.vertex_groups:
        pb = pose_bones.get(vg.name)
        rb = rest_bones.get(vg.name)
        if pb is None or rb is None or not rb.use_deform:
            continue
        delta_amt = pb.matrix @ rb.matrix_local.inverted_safe()
        # Check if close to identity
        if not any_posed:
            diff = sum(abs(delta_amt[r][c] - (1.0 if r == c else 0.0)) for r in range(4) for c in range(4))
            if diff > 1e-4:
                any_posed = True
        world_deform = (amt_mw @ delta_amt @ amt_mw_inv @ obj.matrix_world).to_3x3()
        vg_idx_to_mat[vg.index] = np.array(world_deform, dtype=np.float32)

    if not any_posed or not vg_idx_to_mat:
        return obj_l2w, obj_w2l, False

    n_verts = len(mesh.vertices)
    l2w = np.empty((n_verts, 3, 3), dtype=np.float32)
    for i, v in enumerate(mesh.vertices):
        acc = np.zeros((3, 3), dtype=np.float32)
        total_w = 0.0
        for g in v.groups:
            w = g.weight
            if w > 0.0:
                b_mat = vg_idx_to_mat.get(g.group)
                if b_mat is not None:
                    acc += b_mat * w
                    total_w += w
        if total_w <= 1e-6:
            l2w[i] = mw
        else:
            l2w[i] = acc * (1.0 / total_w)

    # l2w currently holds per-vertex POSITION deform matrices; convert to normal transforms
    n_l2w, n_w2l = _normal_transforms(l2w)
    return n_l2w, n_w2l, True


def build_target_state(context: bpy.types.Context, obj: bpy.types.Object, mask_image: bpy.types.Image | None = None) -> TargetState | None:
    """Initialize a TargetState from a mesh Object."""
    if obj is None or obj.type != "MESH" or obj.data is None:
        return None

    modifier_utils.ensure_dne_modifier(obj)
    mesh = obj.data
    n_verts = len(mesh.vertices)
    if n_verts == 0:
        return None

    base_local_n = modifier_utils.read_base_local_normals(mesh)
    deltas = modifier_utils.read_deltas_numpy(mesh)
    l2w, w2l, has_skin = _compute_skin_rotations(obj, mesh)

    # Read evaluated world vertices (or rest vertices transformed by matrix_world if vertex count differs)
    depsgraph = context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()

    try:
        mw_4x4 = np.array(obj.matrix_world, dtype=np.float32)
        rot_3x3 = mw_4x4[:3, :3]
        trans_3 = mw_4x4[:3, 3]

        if eval_mesh is not None and len(eval_mesh.vertices) == n_verts:
            local_verts = np.empty(n_verts * 3, dtype=np.float32)
            eval_mesh.vertices.foreach_get("co", local_verts)
            local_verts = local_verts.reshape((n_verts, 3))
        else:
            local_verts = np.empty(n_verts * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", local_verts)
            local_verts = local_verts.reshape((n_verts, 3))

        world_verts = (local_verts @ rot_3x3.T) + trans_3

        # Extract triangles from base mesh loop_triangles
        mesh.calc_loop_triangles()
        n_tris = len(mesh.loop_triangles)
        if n_tris > 0:
            tris = np.empty(n_tris * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", tris)
            tris = tris.reshape((n_tris, 3))
        else:
            tris = np.zeros((0, 3), dtype=np.int32)
    finally:
        if eval_mesh is not None:
            eval_obj.to_mesh_clear()

    # Build BVHTree in world space for fast occlusion culling
    bvh = None
    if len(tris) > 0:
        try:
            bvh = BVHTree.FromPolygons(world_verts.tolist(), tris.tolist(), all_triangles=True)
        except Exception:
            bvh = None

    raw_mask = core_math.compute_vertex_mask_from_blender_image(mesh, mask_image) if mask_image else None

    state = TargetState(
        obj=obj,
        mask_image=mask_image,
        base_local_normals=base_local_n,
        working_deltas=deltas.copy(),
        snapshot_deltas=deltas.copy(),
        world_vertices=world_verts,
        triangles=tris,
        local_to_world_rot=l2w,
        world_to_local_rot=w2l,
        has_per_vertex_skin=has_skin,
        raw_vertex_mask=raw_mask,
        mask=raw_mask,
        bvh=bvh,
        touched=False,
    )
    state.update_world_normals_from_working()
    state.snapshot_world_normals = state.world_normals.copy()
    return state


def _is_visible(obj: bpy.types.Object) -> bool:
    try:
        return bool(obj.visible_get())
    except Exception:
        return True


def is_vertex_occluded(
    vertex_world: np.ndarray,
    targets: list[TargetState],
    ray_origin: Vector,
    ray_dir_fallback: Vector,
    is_ortho: bool,
) -> bool:
    """Check if vertex_world is occluded by any registered target's triangles (matching EditSession.Occlusion.cs)."""
    v_pos = Vector((float(vertex_world[0]), float(vertex_world[1]), float(vertex_world[2])))
    if is_ortho:
        # In orthographic view, cast ray backwards from vertex towards the camera plane
        ray_dir = -ray_dir_fallback.normalized()
        max_dist = 1000.0
        start_pos = v_pos + ray_dir * 1e-3
        for t in targets:
            if t.bvh is None:
                continue
            loc, _norm, _idx, dist = t.bvh.ray_cast(start_pos, ray_dir, max_dist)
            if loc is not None and dist is not None and dist > 5e-4:
                return True
        return False

    to_vert = v_pos - ray_origin
    dist_to_vert = to_vert.length
    if dist_to_vert <= 1e-5:
        return False
    ray_dir = to_vert / dist_to_vert
    # Allow a tiny depth tolerance (matching Unity DepthEpsilonRatio) so the vertex's own incident faces don't self-occlude
    max_hit_dist = dist_to_vert * 0.996 - 5e-4
    if max_hit_dist <= 0.0:
        return False

    for t in targets:
        if t.bvh is None:
            continue
        loc, _norm, _idx, hit_dist = t.bvh.ray_cast(ray_origin, ray_dir, max_hit_dist)
        if loc is not None:
            return True
    return False


def try_pick_vertex(
    region: bpy.types.Region,
    rv3d: bpy.types.RegionView3D,
    mouse_xy: tuple[float, float],
    targets: list[TargetState],
    pick_threshold_px: float = 24.0,
    max_candidates: int = 8,
) -> tuple[TargetState | None, int]:
    """Pick the closest visible (unoccluded) vertex to mouse_xy within pick_threshold_px."""
    if region is None or rv3d is None or not targets:
        return None, -1

    # Hidden targets can neither be picked nor occlude other targets
    targets = [t for t in targets if _is_visible(t.obj)]
    if not targets:
        return None, -1

    persp = np.array(rv3d.perspective_matrix, dtype=np.float32)
    half_w = region.width * 0.5
    half_h = region.height * 0.5
    mx, my = float(mouse_xy[0]), float(mouse_xy[1])
    thresh_sq = pick_threshold_px * pick_threshold_px

    candidates: list[tuple[float, TargetState, int]] = []

    for target in targets:
        verts = target.world_vertices
        if verts is None or len(verts) == 0:
            continue
        # Homogeneous projection: clip = verts @ persp[:3, :3].T + persp[:, 3]
        clip_x = verts[:, 0] * persp[0, 0] + verts[:, 1] * persp[0, 1] + verts[:, 2] * persp[0, 2] + persp[0, 3]
        clip_y = verts[:, 0] * persp[1, 0] + verts[:, 1] * persp[1, 1] + verts[:, 2] * persp[1, 2] + persp[1, 3]
        clip_w = verts[:, 0] * persp[3, 0] + verts[:, 1] * persp[3, 1] + verts[:, 2] * persp[3, 2] + persp[3, 3]

        in_front = clip_w > 1e-5
        if not np.any(in_front):
            continue

        inv_w = np.where(in_front, 1.0 / np.where(in_front, clip_w, 1.0), 0.0)
        sx = (clip_x * inv_w + 1.0) * half_w
        sy = (clip_y * inv_w + 1.0) * half_h

        dx = sx - mx
        dy = sy - my
        dist_sq = dx * dx + dy * dy
        valid_mask = in_front & (dist_sq < thresh_sq)
        if not np.any(valid_mask):
            continue

        valid_indices = np.nonzero(valid_mask)[0]
        valid_dists = dist_sq[valid_indices]
        order = np.argsort(valid_dists)[:max_candidates]
        for k in order:
            v_idx = int(valid_indices[k])
            d_sq = float(valid_dists[k])
            candidates.append((d_sq, target, v_idx))

    if not candidates:
        return None, -1

    candidates.sort(key=lambda item: item[0])
    candidates = candidates[:max_candidates]

    is_ortho = not rv3d.is_perspective
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_xy)
    ray_dir_fallback = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_xy)

    for _d_sq, target, v_idx in candidates:
        if not is_vertex_occluded(target.world_vertices[v_idx], targets, ray_origin, ray_dir_fallback, is_ortho):
            return target, v_idx

    return None, -1

