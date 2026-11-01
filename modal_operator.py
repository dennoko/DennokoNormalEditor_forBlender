"""Modal editing session and core manipulation logic for Dennoko Normal Editor.

Faithfully ports Unity's EditSession, EditSession.Manipulation, and EditSession.Overlay
behavior into a Blender 4.5 / 5.0+ modal operator.
"""

from __future__ import annotations

from dataclasses import dataclass
import bpy
from bpy_extras import view3d_utils
from mathutils import Vector
import numpy as np

from . import core_math
from . import modifier_utils
from . import picking
from . import overlay_gpu


@dataclass(eq=False)
class TargetInfluence:
    target: picking.TargetState
    t_idx: int
    indices: np.ndarray         # (M,) int32 vertex indices
    group_ids: np.ndarray       # (M,) int32 weld group IDs
    weights: np.ndarray         # (M,) float32 primary brush weights
    mirror_weights: np.ndarray  # (M,) float32 mirror brush weights


@dataclass
class UndoEntry:
    label: str
    deltas_by_obj: dict[str, np.ndarray]
    brush_radius: float
    selected_obj_name: str | None
    selected_index: int


class EditSessionState:
    """Runtime state for an active Dennoko Normal Editor modal session."""

    min_radius: float = 0.001
    max_radius: float = 0.500
    radius_scroll_step: float = 1.05

    def __init__(self, context: bpy.types.Context) -> None:
        # True while the session itself writes scene settings (suppresses update-callback re-entry)
        self.suppress_notify: bool = False

        self.targets: list[picking.TargetState] = []
        self.topology: core_math.CrossMeshTopology | None = None
        self.influences: list[TargetInfluence] = []

        # Selection state
        self.has_selection: bool = False
        self.selected_target: picking.TargetState | None = None
        self.selected_index: int = -1
        self.selected_obj_name: str | None = None
        self.center_world: np.ndarray = np.zeros(3, dtype=np.float32)
        self.mirror_center_world: np.ndarray = np.zeros(3, dtype=np.float32)
        self.mirror_active: bool = False
        self.selected_snapshot_normal: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # 3D Handle state
        self.handle_position: np.ndarray = np.zeros(3, dtype=np.float32)
        self.handle_rest_position: np.ndarray = np.zeros(3, dtype=np.float32)
        self.handle_rest_length: float = 0.05
        self.hover_handle_part: str | None = None
        self.dragging_handle: bool = False
        self.dragging_axis: str | None = None
        self.drag_start_mouse: tuple[float, float] = (0.0, 0.0)
        self.drag_start_handle_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self.handle_moved: bool = False
        self.radius_changed_during_grab: bool = False
        self.grab_start_radius: float | None = None

        # Hover state
        self.hover_target: picking.TargetState | None = None
        self.hover_index: int = -1

        # Overlay UI state
        self.has_custom_overlay_pos: bool = False
        self.overlay_pos: tuple[float, float] = (0.0, 0.0)
        self.dragging_overlay: bool = False
        self.overlay_drag_offset: tuple[float, float] = (0.0, 0.0)
        self.dragging_radius_slider: bool = False

        self.active_effect: str = "NONE"
        self.slider_smooth: float = 0.0
        self.slider_restore_shape: float = 0.0
        self.slider_flatten: float = 0.0
        self.slider_erase: float = 0.0

        # Undo / Redo stacks
        self.undo_stack: list[UndoEntry] = []
        self.redo_stack: list[UndoEntry] = []

        self.draw_handler_3d = None
        self.draw_handler_2d = None

        self.refresh_targets(context)

    # -----------------------------------------------------------------------
    # Scene property accessors
    # -----------------------------------------------------------------------

    @property
    def props(self):
        return bpy.context.scene.dennoko_normal_editor

    def _set_prop(self, name: str, value) -> None:
        """Write a scene setting without re-entering the session through its update callback."""
        self.suppress_notify = True
        try:
            setattr(self.props, name, value)
        finally:
            self.suppress_notify = False

    @property
    def brush_radius(self) -> float:
        return float(self.props.brush_radius)

    @brush_radius.setter
    def brush_radius(self, val: float) -> None:
        self._set_prop("brush_radius", max(self.min_radius, min(self.max_radius, float(val))))

    @property
    def falloff(self) -> str:
        return str(self.props.falloff)

    @falloff.setter
    def falloff(self, val: str) -> None:
        self._set_prop("falloff", val)

    @property
    def mirror(self) -> bool:
        return bool(self.props.mirror)

    @mirror.setter
    def mirror(self, val: bool) -> None:
        self._set_prop("mirror", bool(val))

    @property
    def mirror_axis(self) -> str:
        return str(self.props.mirror_axis)

    @mirror_axis.setter
    def mirror_axis(self, val: str) -> None:
        self._set_prop("mirror_axis", val)

    # -----------------------------------------------------------------------
    # Target & Topology Initialization
    # -----------------------------------------------------------------------

    def refresh_targets(self, context: bpy.types.Context) -> None:
        """Load all registered target meshes, build cross-mesh weld topology, and combine UV masks."""
        scene_props = context.scene.dennoko_normal_editor
        new_targets: list[picking.TargetState] = []
        seen_names: set[str] = set()

        for item in scene_props.targets:
            obj = item.obj
            if obj is None or obj.type != "MESH" or obj.data is None:
                continue
            if obj.name in seen_names:
                continue
            seen_names.add(obj.name)
            st = picking.build_target_state(context, obj, item.strength_mask)
            if st is not None:
                new_targets.append(st)

        self.targets = new_targets

        # Build Cross-Mesh Weld Topology & SmoothGeometricNormals
        topo_inputs = [
            core_math.MeshTopologyInput(world_vertices=t.world_vertices, triangles=t.triangles)
            for t in self.targets
        ]
        self.topology = core_math.build_topology(topo_inputs)

        # Combine UV0 strength masks across welded vertex groups
        raw_masks = [t.raw_vertex_mask for t in self.targets]
        combined_masks = core_math.combine_masks_by_group(
            raw_masks,
            self.topology.vertex_to_group if self.topology else [],
            self.topology.group_count if self.topology else 0,
        )
        for t, cm in zip(self.targets, combined_masks):
            t.mask = cm

        self.hover_target = None
        self.hover_index = -1

        if self.has_selection:
            # Re-validate selection: TargetState objects were rebuilt, so remap by object name
            remapped = next((t for t in self.targets if t.obj.name == self.selected_obj_name), None)
            if remapped is None or not (0 <= self.selected_index < len(remapped.world_vertices)):
                self.clear_selection()
            else:
                self.selected_target = remapped
                self.center_world = remapped.world_vertices[self.selected_index].copy()
                self.recompute_mirror_center()
                self.build_influences()
                self.commit_snapshot()
                self.sync_handle_to_selection(None, None)
        else:
            self.commit_snapshot()

    def rebuild_masks_only(self) -> None:
        """Recompute UV0 strength masks when mask image settings change."""
        for item in self.props.targets:
            if item.obj is None:
                continue
            for t in self.targets:
                if t.obj == item.obj:
                    t.mask_image = item.strength_mask
                    t.raw_vertex_mask = (
                        core_math.compute_vertex_mask_from_blender_image(t.obj.data, t.mask_image)
                        if t.mask_image
                        else None
                    )
        raw_masks = [t.raw_vertex_mask for t in self.targets]
        combined_masks = core_math.combine_masks_by_group(
            raw_masks,
            self.topology.vertex_to_group if self.topology else [],
            self.topology.group_count if self.topology else 0,
        )
        for t, cm in zip(self.targets, combined_masks):
            t.mask = cm
        if self.has_selection:
            self.build_influences()

    # -----------------------------------------------------------------------
    # Mirror Space Math
    # -----------------------------------------------------------------------

    def _get_mirror_root_matrix(self):
        """Find common Armature root or active target matrix_world for mirror plane."""
        for t in self.targets:
            if t.obj.parent and t.obj.parent.type == "ARMATURE":
                return t.obj.parent.matrix_world
            for m in t.obj.modifiers:
                if m.type == "ARMATURE" and m.object is not None:
                    return m.object.matrix_world
        if self.selected_target is not None:
            return self.selected_target.obj.matrix_world
        if self.targets:
            return self.targets[0].obj.matrix_world
        from mathutils import Matrix
        return Matrix.Identity(4)

    def _reflect_vec3(self, v: Vector) -> Vector:
        ax = self.mirror_axis
        if ax == "Y":
            return Vector((v.x, -v.y, v.z))
        if ax == "Z":
            return Vector((v.x, v.y, -v.z))
        return Vector((-v.x, v.y, v.z))

    def _axis_component(self, v: Vector) -> float:
        ax = self.mirror_axis
        if ax == "Y":
            return v.y
        if ax == "Z":
            return v.z
        return v.x

    def mirror_direction(self, world_dir: np.ndarray) -> np.ndarray:
        root_mw = self._get_mirror_root_matrix()
        root_inv = root_mw.inverted_safe()
        d_vec = Vector((float(world_dir[0]), float(world_dir[1]), float(world_dir[2])))
        in_root = root_inv.to_3x3() @ d_vec
        reflected = self._reflect_vec3(in_root)
        out_world = root_mw.to_3x3() @ reflected
        return core_math.safe_normalize_vec(np.array([out_world.x, out_world.y, out_world.z], dtype=np.float32), world_dir)

    def recompute_mirror_center(self) -> None:
        self.mirror_active = False
        if not self.mirror or not self.has_selection:
            return

        root_mw = self._get_mirror_root_matrix()
        root_inv = root_mw.inverted_safe()
        c_vec = Vector((float(self.center_world[0]), float(self.center_world[1]), float(self.center_world[2])))
        center_in_root = root_inv @ c_vec

        eps = max(1e-4, self.brush_radius * 0.02)
        if abs(self._axis_component(center_in_root)) < eps:
            return

        self.mirror_active = True
        mir_in_root = self._reflect_vec3(center_in_root)
        mir_world = root_mw @ mir_in_root
        self.mirror_center_world = np.array([mir_world.x, mir_world.y, mir_world.z], dtype=np.float32)

    # -----------------------------------------------------------------------
    # Influence Calculation & Snapshot
    # -----------------------------------------------------------------------

    def build_influences(self) -> None:
        """Build influence vertex lists and falloff weights across all registered targets."""
        self.influences.clear()
        if not self.has_selection:
            return

        radius = max(1e-5, self.brush_radius)
        radius_sq = radius * radius
        inv_radius = 1.0 / radius
        falloff = self.falloff
        has_topo = self.topology is not None and self.topology.group_count > 0

        for t_idx, target in enumerate(self.targets):
            verts = target.world_vertices
            if verts is None or len(verts) == 0:
                continue

            if has_topo and t_idx < len(self.topology.vertex_to_group):
                group_map = self.topology.vertex_to_group[t_idx]
                eval_positions = self.topology.group_positions[group_map]
            else:
                group_map = np.zeros(len(verts), dtype=np.int32)
                eval_positions = verts

            diff = eval_positions - self.center_world[None, :]
            dist_sq = np.einsum("ij,ij->i", diff, diff)
            in_primary = dist_sq < radius_sq
            weights = np.zeros(len(verts), dtype=np.float32)
            if np.any(in_primary):
                t_norm = np.sqrt(dist_sq[in_primary]) * inv_radius
                weights[in_primary] = core_math.falloff_weight(t_norm, falloff)

            mirror_weights = np.zeros(len(verts), dtype=np.float32)
            if self.mirror_active:
                m_diff = eval_positions - self.mirror_center_world[None, :]
                m_dist_sq = np.einsum("ij,ij->i", m_diff, m_diff)
                in_mirror = m_dist_sq < radius_sq
                if np.any(in_mirror):
                    mt_norm = np.sqrt(m_dist_sq[in_mirror]) * inv_radius
                    mirror_weights[in_mirror] = core_math.falloff_weight(mt_norm, falloff)

            if target.mask is not None and len(target.mask) == len(verts):
                weights *= target.mask
                mirror_weights *= target.mask

            active_mask = (weights > 0.0) | (mirror_weights > 0.0)
            if not np.any(active_mask):
                continue

            indices = np.nonzero(active_mask)[0].astype(np.int32)
            self.influences.append(
                TargetInfluence(
                    target=target,
                    t_idx=t_idx,
                    indices=indices,
                    group_ids=group_map[indices],
                    weights=weights[indices],
                    mirror_weights=mirror_weights[indices],
                )
            )

    def commit_snapshot(self) -> None:
        """Freeze current working deltas & world normals as the baseline for the next drag."""
        snapshot_normals_list: list[np.ndarray] = []
        for target in self.targets:
            target.snapshot_deltas = target.working_deltas.copy()
            target.touched = False
            target.update_world_normals_from_working()
            target.snapshot_world_normals = target.world_normals.copy()
            snapshot_normals_list.append(target.snapshot_world_normals)

        if (
            self.has_selection
            and self.selected_target is not None
            and 0 <= self.selected_index < len(self.selected_target.snapshot_world_normals)
        ):
            self.selected_snapshot_normal = self.selected_target.snapshot_world_normals[self.selected_index].copy()
        else:
            self.selected_snapshot_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        if self.topology is not None and self.topology.group_count > 0:
            core_math.compute_smoothed_group_normals(
                self.topology,
                snapshot_normals_list,
                iterations=core_math.DEFAULT_SMOOTH_ITERATIONS,
            )

    def sync_handle_to_selection(self, region: bpy.types.Region | None, rv3d: bpy.types.RegionView3D | None) -> None:
        if not self.has_selection or self.selected_target is None:
            return
        if not (0 <= self.selected_index < len(self.selected_target.world_vertices)):
            return

        self.center_world = self.selected_target.world_vertices[self.selected_index].copy()
        wn = self.selected_target.world_normals[self.selected_index]
        if region is not None and rv3d is not None:
            self.handle_rest_length = overlay_gpu.get_handle_rest_length(region, rv3d, self.center_world)
        self.handle_rest_position = self.center_world + wn * self.handle_rest_length
        self.handle_position = self.handle_rest_position.copy()

    def begin_selection(
        self,
        target: picking.TargetState,
        index: int,
        region: bpy.types.Region | None,
        rv3d: bpy.types.RegionView3D | None,
    ) -> None:
        if self.is_interacting():
            self.cancel_interaction()
        self.has_selection = True
        self.selected_target = target
        self.selected_index = index
        self.selected_obj_name = target.obj.name
        self.center_world = target.world_vertices[index].copy()
        self.reset_effect_sliders()

        self.recompute_mirror_center()
        self.build_influences()
        self.commit_snapshot()
        self.sync_handle_to_selection(region, rv3d)

    def clear_selection(self) -> None:
        if self.is_interacting():
            self.cancel_interaction()
        self.dragging_handle = False
        self.dragging_axis = None
        self.reset_effect_sliders()
        self.has_selection = False
        self.mirror_active = False
        self.selected_target = None
        self.selected_index = -1
        self.selected_obj_name = None
        self.influences.clear()
        self.hover_target = None
        self.hover_index = -1

    def is_interacting(self) -> bool:
        """True while a drag (handle, effect slider, radius slider, or overlay header) is in progress."""
        return (
            self.dragging_handle
            or self.dragging_overlay
            or self.dragging_radius_slider
            or self.active_effect != "NONE"
        )

    def abort_interaction_state(self) -> None:
        """Reset all drag flags without touching mesh data."""
        self.dragging_handle = False
        self.dragging_axis = None
        self.dragging_overlay = False
        self.dragging_radius_slider = False
        self.handle_moved = False
        self.radius_changed_during_grab = False
        self.reset_effect_sliders()

    def cancel_interaction(self) -> None:
        """Abort an in-progress drag and revert its uncommitted live preview to the last snapshot."""
        restore_radius = (
            self.grab_start_radius
            if self.dragging_handle and self.radius_changed_during_grab and self.grab_start_radius is not None
            else None
        )
        self.abort_interaction_state()
        if restore_radius is not None:
            self.brush_radius = restore_radius
        self._reset_touched_to_snapshot()
        self._publish_touched_targets()
        for target in self.targets:
            target.touched = False
        if self.has_selection:
            self.recompute_mirror_center()
            self.build_influences()

    def has_uncommitted_changes(self) -> bool:
        return any(t.touched for t in self.targets)

    def reset_effect_sliders(self) -> None:
        self.active_effect = "NONE"
        self.slider_smooth = 0.0
        self.slider_restore_shape = 0.0
        self.slider_flatten = 0.0
        self.slider_erase = 0.0

    # -----------------------------------------------------------------------
    # Live Normal Manipulation: Tilt & 4 Effects
    # -----------------------------------------------------------------------

    def _reset_touched_to_snapshot(self) -> None:
        for target in self.targets:
            if not target.touched:
                continue
            target.working_deltas[:] = target.snapshot_deltas
            target.world_normals[:] = target.snapshot_world_normals

    def _write_world_normals_subset(self, inf: TargetInfluence, new_world_normals: np.ndarray) -> None:
        target = inf.target
        if not target.touched:
            target.working_deltas[:] = target.snapshot_deltas
            target.world_normals[:] = target.snapshot_world_normals
            target.touched = True

        idx = inf.indices
        target.world_normals[idx] = new_world_normals
        new_local = target.world_to_local_subset(idx, new_world_normals)
        target.working_deltas[idx] = new_local - target.base_local_normals[idx]

    def _publish_touched_targets(self) -> None:
        for target in self.targets:
            if not target.touched:
                continue
            modifier_utils.write_deltas_numpy(target.obj.data, target.working_deltas)

    def apply_tilt(self) -> None:
        """Tilt normals within influence radius toward the dragged 3D handle direction."""
        dragged_vec = self.handle_position - self.center_world
        target_normal = core_math.safe_normalize_vec(dragged_vec, self.selected_snapshot_normal)

        mirror_from = self.mirror_direction(self.selected_snapshot_normal) if self.mirror_active else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        mirror_to = self.mirror_direction(target_normal) if self.mirror_active else np.array([0.0, 0.0, 1.0], dtype=np.float32)

        self._reset_touched_to_snapshot()

        for inf in self.influences:
            idx = inf.indices
            current_world = inf.target.snapshot_world_normals[idx].copy()

            has_w = inf.weights > 0.0
            if np.any(has_w):
                tilted = core_math.rotate_vector(self.selected_snapshot_normal, target_normal, current_world[has_w])
                current_world[has_w] = core_math.slerp_normal(current_world[has_w], tilted, inf.weights[has_w])

            if self.mirror_active:
                has_mw = inf.mirror_weights > 0.0
                if np.any(has_mw):
                    m_tilted = core_math.rotate_vector(mirror_from, mirror_to, current_world[has_mw])
                    current_world[has_mw] = core_math.slerp_normal(current_world[has_mw], m_tilted, inf.mirror_weights[has_mw])

            self._write_world_normals_subset(inf, current_world)

        self._publish_touched_targets()

    def apply_effect(self, mode: str, strength: float) -> None:
        """Apply Smooth, RestoreShape, Flatten, or Erase at 'strength' in [0, 1]."""
        strength = max(0.0, min(1.0, float(strength)))
        self._reset_touched_to_snapshot()

        if mode == "NONE" or strength <= 0.0 or not self.influences:
            self._publish_touched_targets()
            return

        has_topo = self.topology is not None and self.topology.group_count > 0
        mirror_flatten_dir = self.mirror_direction(self.selected_snapshot_normal) if self.mirror_active else self.selected_snapshot_normal

        for inf in self.influences:
            target = inf.target
            idx = inf.indices
            combined_w = np.clip(inf.weights + inf.mirror_weights, 0.0, 1.0) * strength

            if mode == "ERASE":
                if not target.touched:
                    target.working_deltas[:] = target.snapshot_deltas
                    target.world_normals[:] = target.snapshot_world_normals
                    target.touched = True
                rem = target.snapshot_deltas[idx] * (1.0 - combined_w)[:, None]
                target.working_deltas[idx] = rem
                local_n = core_math.evaluate_normal(target.base_local_normals[idx], rem)
                target.world_normals[idx] = target.local_to_world_subset(idx, local_n)
                continue

            snap_n = target.snapshot_world_normals[idx]
            new_world_n = snap_n.copy()

            if mode == "SMOOTH" and has_topo:
                goals = self.topology.smoothed_buffer_a[inf.group_ids]
                new_world_n = core_math.slerp_normal(snap_n, goals, combined_w)
            elif mode == "RESTORE_SHAPE" and has_topo:
                goals = self.topology.smooth_geometric_normals[inf.group_ids]
                new_world_n = core_math.slerp_normal(snap_n, goals, combined_w)
            elif mode == "FLATTEN":
                has_w = inf.weights > 0.0
                if np.any(has_w):
                    new_world_n[has_w] = core_math.slerp_normal(
                        new_world_n[has_w], self.selected_snapshot_normal, inf.weights[has_w] * strength
                    )
                if self.mirror_active:
                    has_mw = inf.mirror_weights > 0.0
                    if np.any(has_mw):
                        new_world_n[has_mw] = core_math.slerp_normal(
                            new_world_n[has_mw], mirror_flatten_dir, inf.mirror_weights[has_mw] * strength
                        )

            self._write_world_normals_subset(inf, new_world_n)

        self._publish_touched_targets()

    # -----------------------------------------------------------------------
    # Undo / Redo Stack
    # -----------------------------------------------------------------------

    def push_undo_snapshot_before_commit(self, label: str, prev_radius: float | None = None) -> None:
        """Record the pre-operation snapshot deltas so Ctrl+Z restores in 1 step."""
        deltas_map = {t.obj.name: t.snapshot_deltas.copy() for t in self.targets}
        sel_name = self.selected_target.obj.name if (self.has_selection and self.selected_target) else None
        self.undo_stack.append(
            UndoEntry(
                label=label,
                deltas_by_obj=deltas_map,
                brush_radius=prev_radius if prev_radius is not None else self.brush_radius,
                selected_obj_name=sel_name,
                selected_index=self.selected_index,
            )
        )
        if len(self.undo_stack) > 64:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _capture_current_entry(self, label: str) -> UndoEntry:
        deltas_map = {t.obj.name: t.working_deltas.copy() for t in self.targets}
        sel_name = self.selected_target.obj.name if (self.has_selection and self.selected_target) else None
        return UndoEntry(
            label=label,
            deltas_by_obj=deltas_map,
            brush_radius=self.brush_radius,
            selected_obj_name=sel_name,
            selected_index=self.selected_index,
        )

    def _apply_undo_entry(self, entry: UndoEntry, region: bpy.types.Region | None, rv3d: bpy.types.RegionView3D | None) -> None:
        self.brush_radius = entry.brush_radius
        for t in self.targets:
            saved = entry.deltas_by_obj.get(t.obj.name)
            if saved is not None and len(saved) == len(t.working_deltas):
                t.working_deltas[:] = saved
                modifier_utils.write_deltas_numpy(t.obj.data, t.working_deltas)
                t.update_world_normals_from_working()

        if entry.selected_obj_name is not None and entry.selected_index >= 0:
            for t in self.targets:
                if t.obj.name == entry.selected_obj_name and entry.selected_index < len(t.world_vertices):
                    self.has_selection = True
                    self.selected_target = t
                    self.selected_index = entry.selected_index
                    self.selected_obj_name = t.obj.name
                    self.center_world = t.world_vertices[entry.selected_index].copy()
                    break

        if self.has_selection:
            self.recompute_mirror_center()
            self.build_influences()
            self.commit_snapshot()
            self.sync_handle_to_selection(region, rv3d)
        else:
            self.commit_snapshot()

    def undo(self, region: bpy.types.Region | None, rv3d: bpy.types.RegionView3D | None) -> bool:
        if not self.undo_stack:
            return False
        entry = self.undo_stack.pop()
        self.redo_stack.append(self._capture_current_entry(entry.label))
        self._apply_undo_entry(entry, region, rv3d)
        return True

    def redo(self, region: bpy.types.Region | None, rv3d: bpy.types.RegionView3D | None) -> bool:
        if not self.redo_stack:
            return False
        entry = self.redo_stack.pop()
        self.undo_stack.append(self._capture_current_entry(entry.label))
        self._apply_undo_entry(entry, region, rv3d)
        return True


# Global active session reference
_ACTIVE_SESSION: EditSessionState | None = None


def get_active_session() -> EditSessionState | None:
    return _ACTIVE_SESSION


def tag_redraw_all_view3d() -> None:
    """Request a redraw of every 3D Viewport in every window."""
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def end_active_session(push_undo: bool = False) -> bool:
    """Terminate the active session (if any): revert uncommitted previews, remove draw handlers.

    The running modal operator notices the missing session on its next event and finishes.
    Returns True if a session was ended.
    """
    global _ACTIVE_SESSION
    session = _ACTIVE_SESSION
    if session is None:
        return False
    _ACTIVE_SESSION = None

    if session.is_interacting():
        try:
            session.cancel_interaction()
        except Exception:
            session.abort_interaction_state()

    for attr in ("draw_handler_3d", "draw_handler_2d"):
        handler = getattr(session, attr)
        if handler is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(handler, "WINDOW")
            except Exception:
                pass
            setattr(session, attr, None)

    tag_redraw_all_view3d()

    if push_undo:
        try:
            bpy.ops.ed.undo_push(message="Dennoko Normal Editor Session")
        except Exception:
            pass
    return True


def notify_brush_settings_changed() -> None:
    """Scene brush settings (radius / falloff / mirror) changed from the UI."""
    session = _ACTIVE_SESSION
    if session is None or session.suppress_notify:
        return
    if session.has_selection:
        session.recompute_mirror_center()
        session.build_influences()
    tag_redraw_all_view3d()


def notify_mask_settings_changed() -> None:
    """A target's strength mask image changed from the UI."""
    session = _ACTIVE_SESSION
    if session is None:
        return
    session.rebuild_masks_only()
    tag_redraw_all_view3d()


def on_global_undo() -> None:
    """Blender's global undo/redo replaced mesh data: rebuild the session state from the scene."""
    session = _ACTIVE_SESSION
    if session is None:
        return
    # Mesh data has already been replaced, so only reset flags (no write-back of stale previews)
    session.abort_interaction_state()
    try:
        session.refresh_targets(bpy.context)
    except Exception:
        end_active_session(push_undo=False)
        return
    tag_redraw_all_view3d()


# ---------------------------------------------------------------------------
# Viewport region lookup
# ---------------------------------------------------------------------------

def _point_in_region(region: bpy.types.Region, x: int, y: int) -> bool:
    return region.x <= x < region.x + region.width and region.y <= y < region.y + region.height


def locate_view3d_region(
    context: bpy.types.Context, event: bpy.types.Event
) -> tuple[bpy.types.Area, bpy.types.Region, bpy.types.RegionView3D] | None:
    """Return (area, WINDOW region, RegionView3D) of the 3D Viewport under the mouse.

    Returns None when the mouse is over another editor, or over a header / toolbar / sidebar /
    HUD region that overlaps the viewport, so those UI elements keep working during a session.
    """
    window = context.window
    screen = window.screen if window is not None else context.screen
    if screen is None:
        return None
    ex, ey = event.mouse_x, event.mouse_y
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        if not (area.x <= ex < area.x + area.width and area.y <= ey < area.y + area.height):
            continue
        for region in area.regions:
            # Hidden regions report a 1px size
            if region.type != "WINDOW" and region.width > 1 and region.height > 1 and _point_in_region(region, ex, ey):
                return None
        for region in area.regions:
            if region.type == "WINDOW" and _point_in_region(region, ex, ey):
                rv3d = region.data
                if rv3d is None:
                    return None
                return area, region, rv3d
        return None
    return None


# ---------------------------------------------------------------------------
# Modal Operator
# ---------------------------------------------------------------------------

class DNE_OT_modal_edit_session(bpy.types.Operator):
    """Interactive proportional normal editing session with 3D handle and viewport overlay."""

    bl_idname = "dne.modal_edit_session"
    bl_label = "Dennoko Normal Editor Session"
    bl_options = {"REGISTER"}

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        global _ACTIVE_SESSION

        if context.area is None or context.area.type != "VIEW_3D":
            self.report({"WARNING"}, "3D Viewport is required.")
            return {"CANCELLED"}

        # If already active, toggle off
        if _ACTIVE_SESSION is not None:
            end_active_session(push_undo=True)
            return {"CANCELLED"}

        # Ensure Object mode (edits written in Edit Mode would be overwritten when leaving it)
        if context.mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
            if context.mode != "OBJECT":
                self.report({"WARNING"}, "オブジェクトモードに切り替えてから実行してください。")
                return {"CANCELLED"}

        # Auto-register selected mesh objects if target list is empty
        scene_props = context.scene.dennoko_normal_editor
        valid_targets = [it.obj for it in scene_props.targets if it.obj is not None and it.obj.type == "MESH"]
        if not valid_targets:
            selected_meshes = [o for o in context.selected_objects if o.type == "MESH"]
            if not selected_meshes and context.active_object and context.active_object.type == "MESH":
                selected_meshes = [context.active_object]
            for m_obj in selected_meshes:
                item = scene_props.targets.add()
                item.obj = m_obj
                modifier_utils.ensure_dne_modifier(m_obj)

        session = EditSessionState(context)
        if not session.targets:
            self.report({"WARNING"}, "編集対象のメッシュがありません。メッシュを選択するか対象リストに追加してください。")
            return {"CANCELLED"}

        _ACTIVE_SESSION = session
        self._session = session
        self._drag_view = None
        session.draw_handler_3d = bpy.types.SpaceView3D.draw_handler_add(
            overlay_gpu.draw_3d_gizmos, (session,), "WINDOW", "POST_VIEW"
        )
        session.draw_handler_2d = bpy.types.SpaceView3D.draw_handler_add(
            overlay_gpu.draw_2d_overlay, (session,), "WINDOW", "POST_PIXEL"
        )

        context.window_manager.modal_handler_add(self)
        tag_redraw_all_view3d()
        return {"RUNNING_MODAL"}

    def cancel(self, context: bpy.types.Context) -> None:
        # Called when Blender force-cancels the modal (file load, window close, ...)
        if _ACTIVE_SESSION is not None and _ACTIVE_SESSION is getattr(self, "_session", None):
            end_active_session(push_undo=False)

    def _finish(self) -> set[str]:
        end_active_session(push_undo=True)
        return {"FINISHED"}

    def modal(self, context: bpy.types.Context, event: bpy.types.Event):
        session = _ACTIVE_SESSION
        if session is None or session is not getattr(self, "_session", None):
            # Ended elsewhere (N-panel button, Apply, file load) or superseded by a new session
            return {"FINISHED"}

        # Entering Edit/Sculpt/... mode replaces mesh data underneath the session
        if context.mode != "OBJECT":
            return self._finish()

        interacting = session.is_interacting()
        if not interacting:
            self._drag_view = None

        # While dragging, keep using the viewport the drag started in (the mouse may leave it)
        if interacting and self._drag_view is not None:
            area, region, rv3d = self._drag_view
        else:
            view = locate_view3d_region(context, event)
            area, region, rv3d = view if view is not None else (None, None, None)

        # 1. Esc: cancel drag > deselect vertex > exit session (matching Unity)
        if event.type == "ESC" and event.value == "PRESS":
            if interacting:
                session.cancel_interaction()
                session.sync_handle_to_selection(region, rv3d)
                self._drag_view = None
            elif session.has_selection:
                session.clear_selection()
            else:
                return self._finish()
            tag_redraw_all_view3d()
            return {"RUNNING_MODAL"}

        # 2. Undo / Redo inside modal session (Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y).
        #    Always captured so Blender's global undo cannot swap mesh data under the session.
        if event.type in {"Z", "Y"} and event.value == "PRESS" and event.ctrl and not event.alt:
            if not interacting:
                if event.type == "Y" or event.shift:
                    session.redo(region, rv3d)
                else:
                    session.undo(region, rv3d)
                tag_redraw_all_view3d()
            return {"RUNNING_MODAL"}

        # Mouse is outside every 3D Viewport, or over a header / toolbar / sidebar: let Blender handle it
        if region is None or rv3d is None:
            if event.type == "MOUSEMOVE" and (session.hover_target is not None or session.hover_handle_part is not None):
                session.hover_target = None
                session.hover_index = -1
                session.hover_handle_part = None
                tag_redraw_all_view3d()
            return {"PASS_THROUGH"}

        mx = float(event.mouse_x - region.x)
        my = float(event.mouse_y - region.y)
        mouse_xy = (mx, my)

        # Right click cancels an in-progress drag (Blender convention)
        if interacting and event.type == "RIGHTMOUSE" and event.value == "PRESS":
            session.cancel_interaction()
            session.sync_handle_to_selection(region, rv3d)
            self._drag_view = None
            tag_redraw_all_view3d()
            return {"RUNNING_MODAL"}

        # 3. Mouse Wheel while grabbing 3D handle -> resize brush radius (matching Unity HandleRadiusScroll)
        if session.has_selection and session.dragging_handle and event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            # Blender convention matching user_manual_ja.md: Wheel Up = shrink (-1), Wheel Down = grow (+1)
            notches = -1.0 if event.type == "WHEELUPMOUSE" else 1.0
            session.brush_radius = session.brush_radius * (session.radius_scroll_step ** notches)
            session.radius_changed_during_grab = True
            session.recompute_mirror_center()
            session.build_influences()
            if session.handle_moved:
                session.apply_tilt()
            tag_redraw_all_view3d()
            return {"RUNNING_MODAL"}

        # 4. Active Dragging States (Overlay Header / Radius Slider / Effect Slider / 3D Handle)
        panel = overlay_gpu.compute_overlay_rect(region, session, area)
        widgets = overlay_gpu.build_overlay_widgets(panel)

        if session.dragging_overlay:
            if event.type == "MOUSEMOVE":
                session.overlay_pos = (
                    mx - session.overlay_drag_offset[0],
                    my - session.overlay_drag_offset[1],
                )
                tag_redraw_all_view3d()
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                session.dragging_overlay = False
                tag_redraw_all_view3d()
                return {"RUNNING_MODAL"}

        if session.dragging_radius_slider:
            r_rect = widgets["radius_slider"]
            if event.type == "MOUSEMOVE":
                ratio = max(0.0, min(1.0, (mx - r_rect.x) / max(1.0, r_rect.w)))
                session.brush_radius = session.min_radius + ratio * (session.max_radius - session.min_radius)
                if session.has_selection:
                    session.recompute_mirror_center()
                    session.build_influences()
                tag_redraw_all_view3d()
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                session.dragging_radius_slider = False
                tag_redraw_all_view3d()
                return {"RUNNING_MODAL"}

        if session.active_effect != "NONE":
            eff = session.active_effect
            s_rect = widgets[f"eff_slider_{eff}"]
            if event.type == "MOUSEMOVE":
                val = max(0.0, min(1.0, (mx - s_rect.x) / max(1.0, s_rect.w)))
                self._set_effect_slider_value(session, eff, val)
                session.apply_effect(eff, val)
                tag_redraw_all_view3d()
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                if session.has_uncommitted_changes():
                    session.push_undo_snapshot_before_commit(f"Effect {eff}")
                session.reset_effect_sliders()
                session.commit_snapshot()
                session.sync_handle_to_selection(region, rv3d)
                tag_redraw_all_view3d()
                return {"RUNNING_MODAL"}

        if session.dragging_handle:
            if event.type == "MOUSEMOVE":
                self._update_3d_handle_drag(session, region, rv3d, mouse_xy)
                session.handle_moved = True
                session.apply_tilt()
                tag_redraw_all_view3d()
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                session.dragging_handle = False
                session.dragging_axis = None
                if session.handle_moved or session.radius_changed_during_grab:
                    session.push_undo_snapshot_before_commit("Tilt", prev_radius=session.grab_start_radius)
                    session.commit_snapshot()
                    session.sync_handle_to_selection(region, rv3d)
                session.handle_moved = False
                session.radius_changed_during_grab = False
                tag_redraw_all_view3d()
                return {"RUNNING_MODAL"}

        # 5. Overlay Panel Mouse Interaction
        if panel.contains(mx, my):
            session.hover_target = None
            session.hover_index = -1
            session.hover_handle_part = None
            if event.type == "LEFTMOUSE" and event.value == "PRESS":
                if self._handle_overlay_click(session, region, rv3d, widgets, mx, my):
                    return self._finish()  # Closed via [×]
                if session.is_interacting():
                    self._drag_view = (area, region, rv3d)
                tag_redraw_all_view3d()
                return {"RUNNING_MODAL"}
            if event.type in {"MOUSEMOVE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "RIGHTMOUSE"}:
                return {"RUNNING_MODAL"}

        # 6. Pass through viewport navigation (MMB, Alt+LMB, Wheel when not grabbing, Numpad)
        if event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "TRACKPADPAN", "TRACKPADZOOM"}:
            if not session.dragging_handle and session.has_selection:
                session.sync_handle_to_selection(region, rv3d)
            return {"PASS_THROUGH"}
        if event.alt and event.type in {"LEFTMOUSE", "MIDDLEMOUSE", "RIGHTMOUSE"}:
            return {"PASS_THROUGH"}

        # 7. Hover updates in 3D Viewport
        if event.type == "MOUSEMOVE":
            if session.has_selection:
                session.sync_handle_to_selection(region, rv3d)
                session.hover_handle_part = overlay_gpu.hit_test_3d_handle(region, rv3d, mouse_xy, session.handle_position)
            else:
                session.hover_handle_part = None
                session.hover_target, session.hover_index = picking.try_pick_vertex(
                    region, rv3d, mouse_xy, session.targets
                )
            tag_redraw_all_view3d()
            return {"RUNNING_MODAL"}

        # 8. Left Click in 3D Viewport: Grab 3D Handle or Pick/Select Vertex
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            # Keep handle position synced with current camera view before hit-testing
            if session.has_selection:
                session.sync_handle_to_selection(region, rv3d)
                part = overlay_gpu.hit_test_3d_handle(region, rv3d, mouse_xy, session.handle_position)
                if part is not None:
                    session.dragging_handle = True
                    session.dragging_axis = part
                    session.drag_start_mouse = mouse_xy
                    session.drag_start_handle_pos = session.handle_position.copy()
                    session.grab_start_radius = session.brush_radius
                    session.handle_moved = False
                    session.radius_changed_during_grab = False
                    self._drag_view = (area, region, rv3d)
                    tag_redraw_all_view3d()
                    return {"RUNNING_MODAL"}

            picked_target, picked_idx = picking.try_pick_vertex(region, rv3d, mouse_xy, session.targets)
            if picked_target is not None and picked_idx >= 0:
                session.begin_selection(picked_target, picked_idx, region, rv3d)
            else:
                session.clear_selection()
            tag_redraw_all_view3d()
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}

    def _set_effect_slider_value(self, session: EditSessionState, eff: str, val: float) -> None:
        session.slider_smooth = val if eff == "SMOOTH" else 0.0
        session.slider_restore_shape = val if eff == "RESTORE_SHAPE" else 0.0
        session.slider_flatten = val if eff == "FLATTEN" else 0.0
        session.slider_erase = val if eff == "ERASE" else 0.0

    def _handle_overlay_click(
        self,
        session: EditSessionState,
        region: bpy.types.Region,
        rv3d: bpy.types.RegionView3D,
        widgets: dict[str, overlay_gpu.Rect2D],
        mx: float,
        my: float,
    ) -> bool:
        """Handle mouse press inside the 2D overlay panel. Returns True if session should exit."""
        if widgets["close"].contains(mx, my):
            return True

        if widgets["header"].contains(mx, my):
            session.dragging_overlay = True
            session.has_custom_overlay_pos = True
            session.overlay_drag_offset = (mx - session.overlay_pos[0], my - session.overlay_pos[1])
            return False

        r_rect = widgets["radius_slider"]
        if r_rect.contains(mx, my):
            session.dragging_radius_slider = True
            ratio = max(0.0, min(1.0, (mx - r_rect.x) / max(1.0, r_rect.w)))
            session.brush_radius = session.min_radius + ratio * (session.max_radius - session.min_radius)
            if session.has_selection:
                session.recompute_mirror_center()
                session.build_influences()
            return False

        for f_id in ("SMOOTH", "LINEAR", "SHARP", "CONSTANT"):
            if widgets[f"falloff_{f_id}"].contains(mx, my):
                session.falloff = f_id
                if session.has_selection:
                    session.build_influences()
                return False

        if widgets["mirror_toggle"].contains(mx, my):
            session.mirror = not session.mirror
            if session.has_selection:
                session.recompute_mirror_center()
                session.build_influences()
            return False

        for ax in ("X", "Y", "Z"):
            if widgets[f"axis_{ax}"].contains(mx, my):
                session.mirror_axis = ax
                if session.has_selection:
                    session.recompute_mirror_center()
                    session.build_influences()
                return False

        if session.has_selection:
            for eff in ("SMOOTH", "RESTORE_SHAPE", "FLATTEN", "ERASE"):
                # 1-Click 100% apply button
                if widgets[f"eff_btn_{eff}"].contains(mx, my):
                    session.apply_effect(eff, 1.0)
                    session.push_undo_snapshot_before_commit(f"Effect {eff}")
                    session.commit_snapshot()
                    session.sync_handle_to_selection(region, rv3d)
                    return False

                # Live preview slider
                s_rect = widgets[f"eff_slider_{eff}"]
                if s_rect.contains(mx, my):
                    session.active_effect = eff
                    val = max(0.0, min(1.0, (mx - s_rect.x) / max(1.0, s_rect.w)))
                    self._set_effect_slider_value(session, eff, val)
                    session.apply_effect(eff, val)
                    return False

        return False

    def _update_3d_handle_drag(
        self,
        session: EditSessionState,
        region: bpy.types.Region,
        rv3d: bpy.types.RegionView3D,
        mouse_xy: tuple[float, float],
    ) -> None:
        """Update session.handle_position from mouse movement in FREE or X/Y/Z axis-constrained mode."""
        start_hp = Vector((
            float(session.drag_start_handle_pos[0]),
            float(session.drag_start_handle_pos[1]),
            float(session.drag_start_handle_pos[2]),
        ))
        p_start = view3d_utils.region_2d_to_location_3d(region, rv3d, session.drag_start_mouse, start_hp)
        p_curr = view3d_utils.region_2d_to_location_3d(region, rv3d, mouse_xy, start_hp)
        delta_3d = p_curr - p_start

        axis = session.dragging_axis
        if axis == "X":
            delta_3d = Vector((delta_3d.x, 0.0, 0.0))
        elif axis == "Y":
            delta_3d = Vector((0.0, delta_3d.y, 0.0))
        elif axis == "Z":
            delta_3d = Vector((0.0, 0.0, delta_3d.z))

        new_hp = start_hp + delta_3d
        session.handle_position = np.array([new_hp.x, new_hp.y, new_hp.z], dtype=np.float32)
