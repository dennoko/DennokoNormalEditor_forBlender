"""Automated headless test suite for Dennoko Normal Editor in Blender 5.0."""

from __future__ import annotations

import math
import os
import sys
import unittest
import bpy
from mathutils import Vector
import numpy as np

# Add parent directory so we can import the addon package as DennokoNormalEditor
ADDON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(ADDON_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import DennokoNormalEditor as dne
from DennokoNormalEditor import core_math, modifier_utils, modal_operator, picking


class TestDennokoNormalEditor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            dne.register()
        except Exception:
            pass

    def setUp(self) -> None:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        try:
            dne.register()
        except Exception:
            pass

    def test_01_falloff_curves(self) -> None:
        """Verify Smooth, Linear, Sharp, and Constant falloff curves match Unity FalloffUtil."""
        self.assertAlmostEqual(float(core_math.falloff_weight(0.0, "SMOOTH")), 1.0, places=6)
        self.assertAlmostEqual(float(core_math.falloff_weight(1.0, "SMOOTH")), 0.0, places=6)
        self.assertAlmostEqual(float(core_math.falloff_weight(0.5, "SMOOTH")), 0.5, places=6)
        self.assertAlmostEqual(float(core_math.falloff_weight(0.5, "LINEAR")), 0.5, places=6)
        self.assertAlmostEqual(float(core_math.falloff_weight(0.5, "SHARP")), 0.25, places=6)
        self.assertAlmostEqual(float(core_math.falloff_weight(0.5, "CONSTANT")), 1.0, places=6)

        arr = np.array([0.0, 0.25, 0.5, 1.0, 1.5], dtype=np.float32)
        w_arr = core_math.falloff_weight(arr, "SHARP")
        np.testing.assert_allclose(w_arr, [1.0, 0.5625, 0.25, 0.0, 0.0], atol=1e-6)

    def test_02_rotate_and_slerp_vector(self) -> None:
        """Verify Rodrigues rotation and SlerpNormal on single and batch vectors."""
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        rot = core_math.rotate_vector(z_axis, x_axis, z_axis)
        np.testing.assert_allclose(rot, x_axis, atol=1e-6)

        mid = core_math.slerp_normal(z_axis, x_axis, 0.5)
        expected_mid = np.array([math.sqrt(0.5), 0.0, math.sqrt(0.5)], dtype=np.float32)
        np.testing.assert_allclose(mid, expected_mid, atol=1e-5)

        # Anti-parallel (180 degrees) slerp stays unit length
        anti = core_math.slerp_normal(z_axis, -z_axis, 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(anti)), 1.0, places=5)

    def test_03_modifier_and_preserve_unmodified_sharp_normals(self) -> None:
        """Verify Geometry Nodes modifier updates edited vertices while preserving unedited split normals."""
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        obj = bpy.context.active_object
        mesh = obj.data

        mod = modifier_utils.ensure_dne_modifier(obj)
        self.assertIsNotNone(mod)
        self.assertEqual(mod.type, "NODES")

        base_vn = modifier_utils.read_base_local_normals(mesh)
        target_n0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        deltas = np.zeros_like(base_vn)
        deltas[0] = target_n0 - base_vn[0]
        modifier_utils.write_deltas_numpy(mesh, deltas)

        self.assertEqual(modifier_utils.count_edited_vertices(mesh), 1)

        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_mesh = obj.evaluated_get(depsgraph).to_mesh()
        try:
            # Corner 0 belongs to vertex 0 -> should become (0, 0, 1)
            c0 = np.array(eval_mesh.corner_normals[0].vector, dtype=np.float32)
            np.testing.assert_allclose(c0, target_n0, atol=1e-4)

            # Corner 1 belongs to vertex 1 (unedited flat cube face) -> should remain (-1, 0, 0)
            c1 = np.array(eval_mesh.corner_normals[1].vector, dtype=np.float32)
            np.testing.assert_allclose(c1, [-1.0, 0.0, 0.0], atol=1e-4)

            # Temporary attribute '_dne_delta' should be stripped from evaluated output mesh
            self.assertIsNone(eval_mesh.attributes.get(modifier_utils.DELTA_ATTR_NAME))
        finally:
            obj.evaluated_get(depsgraph).to_mesh_clear()

    def test_04_cross_mesh_seam_weld_and_effects(self) -> None:
        """Verify two separate meshes (Head + Body seam) weld across boundary and smooth/flatten/erase."""
        # Create two 20cm grid planes sharing a boundary along Y = 0
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=2, y_subdivisions=2, size=0.2, location=(0.0, -0.1, 0.0))
        body_obj = bpy.context.active_object
        body_obj.name = "Body"

        bpy.ops.mesh.primitive_grid_add(x_subdivisions=2, y_subdivisions=2, size=0.2, location=(0.0, 0.1, 0.0))
        head_obj = bpy.context.active_object
        head_obj.name = "Head"

        # Perturb Head normals so there is a seam discontinuity at Y = 0
        modifier_utils.ensure_dne_modifier(body_obj)
        modifier_utils.ensure_dne_modifier(head_obj)
        head_base = modifier_utils.read_base_local_normals(head_obj.data)
        tilted_head = core_math.safe_normalize_vec(head_base + np.array([0.6, 0.5, 0.0], dtype=np.float32))
        modifier_utils.write_deltas_numpy(head_obj.data, tilted_head - head_base)

        scene_props = bpy.context.scene.dennoko_normal_editor
        scene_props.targets.clear()
        for o in (body_obj, head_obj):
            it = scene_props.targets.add()
            it.obj = o
        scene_props.brush_radius = 0.08
        scene_props.falloff = "CONSTANT"

        session = modal_operator.EditSessionState(bpy.context)
        self.assertEqual(len(session.targets), 2)

        # 9 verts + 9 verts = 18 total, but 3 boundary vertices at Y=0 are shared -> 15 weld groups!
        self.assertEqual(session.topology.group_count, 15)

        # Find a boundary vertex on Body at (0, 0, 0)
        body_state = session.targets[0]
        head_state = session.targets[1]
        b_idx = int(np.argmin(np.linalg.norm(body_state.world_vertices, axis=1)))
        h_idx = int(np.argmin(np.linalg.norm(head_state.world_vertices, axis=1)))

        # Verify they share the same weld group ID
        g_body = session.topology.vertex_to_group[0][b_idx]
        g_head = session.topology.vertex_to_group[1][h_idx]
        self.assertEqual(g_body, g_head)

        # Select the seam vertex and apply 100% Smooth (なじませる)
        session.begin_selection(body_state, b_idx, None, None)
        session.apply_effect("SMOOTH", 1.0)
        session.commit_snapshot()

        # After 100% Smooth with Constant falloff, the boundary vertex normals on Body and Head must match!
        np.testing.assert_allclose(
            body_state.world_normals[b_idx],
            head_state.world_normals[h_idx],
            atol=1e-5,
        )

        # Test Restore Shape (形状を戻す) -> returns flat grid normal to (0, 0, 1)
        session.apply_effect("RESTORE_SHAPE", 1.0)
        session.commit_snapshot()
        np.testing.assert_allclose(head_state.world_normals[h_idx], [0.0, 0.0, 1.0], atol=1e-4)

        # Test Erase (元に戻す) -> clears delta on h_idx (inside 0.08 radius) while leaving outside vertices intact
        session.apply_effect("ERASE", 1.0)
        session.commit_snapshot()
        self.assertEqual(modifier_utils.count_edited_vertices(head_obj.data), 8)

        # Expand radius to 0.45m (covers whole 0.2m mesh) and Erase all
        session.brush_radius = 0.45
        session.build_influences()
        session.apply_effect("ERASE", 1.0)
        session.commit_snapshot()
        self.assertEqual(modifier_utils.count_edited_vertices(head_obj.data), 0)

    def test_05_uv_strength_mask_and_group_min(self) -> None:
        """Verify UV0 strength mask scales weights and propagates minimum across welded seam groups."""
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        obj = bpy.context.active_object
        mesh = obj.data

        # Create a 2x2 image where left half (u < 0.5) is black (0.0) and right half (u > 0.5) is white (1.0)
        img = bpy.data.images.new("TestMask", width=2, height=2)
        # Pixels in RGBA row-major: (x=0,y=0)=black, (x=1,y=0)=white, (x=0,y=1)=black, (x=1,y=1)=white
        img.pixels = [
            0.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ]

        vmask = core_math.compute_vertex_mask_from_blender_image(mesh, img)
        self.assertIsNotNone(vmask)
        self.assertEqual(len(vmask), 4)
        # Left vertices have mask ~0, right vertices have mask ~1
        self.assertLess(float(np.min(vmask)), 0.05)
        self.assertGreater(float(np.max(vmask)), 0.95)

    def test_06_bake_and_apply_with_shapekeys(self) -> None:
        """Verify Bake (_edited copy) and Apply Modifier preserve Shape Keys on the mesh."""
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
        mesh = obj.data

        # Add Basis + Blink shape keys
        obj.shape_key_add(name="Basis")
        sk = obj.shape_key_add(name="Blink")
        sk.data[0].co.z += 0.5

        modifier_utils.ensure_dne_modifier(obj)
        base_vn = modifier_utils.read_base_local_normals(mesh)
        deltas = np.zeros_like(base_vn)
        deltas[0] = np.array([0.0, 0.0, 1.0], dtype=np.float32) - base_vn[0]
        modifier_utils.write_deltas_numpy(mesh, deltas)

        # 1. Bake to _edited copy
        baked_mesh = modifier_utils.bake_to_edited_copy(obj, link_to_scene=False)
        self.assertIsNotNone(baked_mesh)
        self.assertTrue(baked_mesh.name.endswith("_edited"))
        self.assertIsNotNone(baked_mesh.shape_keys)
        self.assertEqual(len(baked_mesh.shape_keys.key_blocks), 2)
        np.testing.assert_allclose(baked_mesh.corner_normals[0].vector, [0.0, 0.0, 1.0], atol=1e-4)

        # 2. Apply directly to obj while preserving shape keys
        ok = modifier_utils.apply_dne_modifier_keep_shapekeys(obj)
        self.assertTrue(ok)
        self.assertEqual(len(obj.modifiers), 0)
        self.assertIsNotNone(obj.data.shape_keys)
        self.assertEqual(len(obj.data.shape_keys.key_blocks), 2)
        np.testing.assert_allclose(obj.data.corner_normals[0].vector, [0.0, 0.0, 1.0], atol=1e-4)

    def test_07_tilt_mirror_and_undo_redo(self) -> None:
        """Verify 3D Handle Tilt, X-Axis Mirror reflection, and modal Undo/Redo stack."""
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=2, y_subdivisions=2, size=0.2, location=(0.0, 0.0, 0.0))
        obj = bpy.context.active_object

        scene_props = bpy.context.scene.dennoko_normal_editor
        scene_props.targets.clear()
        it = scene_props.targets.add()
        it.obj = obj
        scene_props.brush_radius = 0.06
        scene_props.falloff = "CONSTANT"
        scene_props.mirror = True
        scene_props.mirror_axis = "X"

        session = modal_operator.EditSessionState(bpy.context)
        target = session.targets[0]

        # Find right-side vertex (+X > 0.05, Y ~ 0) and left-side mirror vertex (-X < -0.05, Y ~ 0)
        right_idx = int(np.argmin(np.linalg.norm(target.world_vertices - np.array([0.1, 0.0, 0.0]), axis=1)))
        left_idx = int(np.argmin(np.linalg.norm(target.world_vertices - np.array([-0.1, 0.0, 0.0]), axis=1)))

        session.begin_selection(target, right_idx, None, None)
        self.assertTrue(session.mirror_active)

        # Tilt handle toward (+X, 0, +Z) -> (1, 0, 1) normalized
        session.handle_position = session.center_world + np.array([1.0, 0.0, 1.0], dtype=np.float32)
        session.apply_tilt()
        session.push_undo_snapshot_before_commit("Tilt")
        session.commit_snapshot()

        s2 = math.sqrt(0.5)
        np.testing.assert_allclose(target.world_normals[right_idx], [s2, 0.0, s2], atol=1e-4)
        # Mirror side (-X) should reflect X component: (-s2, 0, +s2)!
        np.testing.assert_allclose(target.world_normals[left_idx], [-s2, 0.0, s2], atol=1e-4)

        # Undo should restore both sides back to (0, 0, 1)
        self.assertTrue(session.undo(None, None))
        np.testing.assert_allclose(target.world_normals[right_idx], [0.0, 0.0, 1.0], atol=1e-4)
        np.testing.assert_allclose(target.world_normals[left_idx], [0.0, 0.0, 1.0], atol=1e-4)

        # Redo should re-apply the mirrored tilt
        self.assertTrue(session.redo(None, None))
        np.testing.assert_allclose(target.world_normals[right_idx], [s2, 0.0, s2], atol=1e-4)
        np.testing.assert_allclose(target.world_normals[left_idx], [-s2, 0.0, s2], atol=1e-4)

    def test_08_refresh_targets_keeps_selection(self) -> None:
        """refresh_targets() during an active selection must not crash and must remap the selection."""
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=0.2)
        obj = bpy.context.active_object
        scene_props = bpy.context.scene.dennoko_normal_editor
        scene_props.targets.clear()
        scene_props.targets.add().obj = obj

        session = modal_operator.EditSessionState(bpy.context)
        session.begin_selection(session.targets[0], 3, None, None)
        old_state = session.targets[0]

        session.refresh_targets(bpy.context)

        self.assertTrue(session.has_selection)
        self.assertIsNot(session.selected_target, old_state)
        self.assertIs(session.selected_target, session.targets[0])
        self.assertEqual(session.selected_index, 3)
        self.assertTrue(len(session.influences) > 0)
        self.assertIs(session.influences[0].target, session.targets[0])

    def test_09_tilt_on_non_uniform_scaled_object(self) -> None:
        """World-space tilt must produce the requested world normal on non-uniformly scaled objects."""
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=2, y_subdivisions=2, size=1.0)
        obj = bpy.context.active_object
        obj.scale = (1.0, 4.0, 1.0)
        obj.rotation_euler = (0.3, 0.0, 0.0)
        bpy.context.view_layer.update()

        scene_props = bpy.context.scene.dennoko_normal_editor
        scene_props.targets.clear()
        scene_props.targets.add().obj = obj
        session = modal_operator.EditSessionState(bpy.context)
        target = session.targets[0]

        want = np.array([0.0, 0.6, 0.8], dtype=np.float32)
        idx = np.arange(len(target.world_vertices))
        local = target.world_to_local_subset(idx, np.tile(want, (len(idx), 1)))
        modifier_utils.write_deltas_numpy(obj.data, local - target.base_local_normals)

        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_mesh = obj.evaluated_get(depsgraph).to_mesh()
        try:
            local_cn = Vector(eval_mesh.corner_normals[0].vector)
        finally:
            obj.evaluated_get(depsgraph).to_mesh_clear()
        world_cn = (obj.matrix_world.to_3x3().inverted().transposed() @ local_cn).normalized()
        np.testing.assert_allclose(np.array(world_cn), want, atol=1e-3)

        # Round trip local -> world must reproduce the requested world normal as well
        np.testing.assert_allclose(target.local_to_world_subset(idx, local)[0], want, atol=1e-4)

    def test_10_cancel_interaction_reverts_preview(self) -> None:
        """Cancelling a live drag (Esc / RMB) restores the snapshot and leaves no undo entry."""
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=2, y_subdivisions=2, size=0.2)
        obj = bpy.context.active_object
        scene_props = bpy.context.scene.dennoko_normal_editor
        scene_props.targets.clear()
        scene_props.targets.add().obj = obj
        scene_props.brush_radius = 0.3
        scene_props.falloff = "CONSTANT"
        scene_props.mirror = False

        session = modal_operator.EditSessionState(bpy.context)
        target = session.targets[0]
        session.begin_selection(target, 4, None, None)

        session.dragging_handle = True
        session.grab_start_radius = session.brush_radius
        session.handle_position = session.center_world + np.array([1.0, 0.0, 1.0], dtype=np.float32)
        session.apply_tilt()
        session.brush_radius = 0.1
        session.radius_changed_during_grab = True
        self.assertGreater(modifier_utils.count_edited_vertices(obj.data), 0)

        session.cancel_interaction()
        self.assertFalse(session.is_interacting())
        self.assertEqual(modifier_utils.count_edited_vertices(obj.data), 0)
        self.assertAlmostEqual(session.brush_radius, 0.3, places=5)
        self.assertEqual(len(session.undo_stack), 0)
        np.testing.assert_allclose(target.world_normals[4], [0.0, 0.0, 1.0], atol=1e-5)

    def test_11_bake_respects_modifier_factor(self) -> None:
        """Bake / Apply must use the modifier's Factor exactly like the node tree does."""
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        obj = bpy.context.active_object
        mesh = obj.data
        mod = modifier_utils.ensure_dne_modifier(obj)

        base_vn = modifier_utils.read_base_local_normals(mesh)
        deltas = np.zeros_like(base_vn)
        deltas[0] = np.array([0.0, 0.0, 1.0], dtype=np.float32) - base_vn[0]
        modifier_utils.write_deltas_numpy(mesh, deltas)

        for item in mod.node_group.interface.items_tree:
            if getattr(item, "in_out", None) == "INPUT" and item.name == "Factor":
                mod[item.identifier] = 0.0
        obj.update_tag()
        self.assertAlmostEqual(modifier_utils.get_modifier_factor(obj), 0.0)

        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_mesh = obj.evaluated_get(depsgraph).to_mesh()
        try:
            gn_c0 = np.array(eval_mesh.corner_normals[0].vector, dtype=np.float32)
        finally:
            obj.evaluated_get(depsgraph).to_mesh_clear()

        baked = modifier_utils.bake_to_edited_copy(obj, link_to_scene=False)
        np.testing.assert_allclose(baked.corner_normals[0].vector, gn_c0, atol=1e-4)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDennokoNormalEditor)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
