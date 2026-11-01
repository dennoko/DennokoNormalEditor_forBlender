"""Geometry Nodes modifier and '_dne_delta' attribute management for Dennoko Normal Editor.

Stores normal deltas non-destructively on each target mesh's POINT attribute '_dne_delta'
and applies them via a Geometry Nodes modifier using Blender 4.5 / 5.0+ 'Set Mesh Normal'
in TANGENT_SPACE mode so that Armature posing and Shape Keys deform the custom normals
naturally.
"""

from __future__ import annotations

import bpy
import numpy as np

MODIFIER_NAME = "DennokoNormalEditor"
NODE_GROUP_NAME = "DennokoNormalEditor_GN"
DELTA_ATTR_NAME = "_dne_delta"
ZERO_THRESHOLD_SQ = 1e-8


def ensure_node_group() -> bpy.types.GeometryNodeTree:
    """Return or create the shared Geometry Nodes tree that applies '_dne_delta'."""
    ng = bpy.data.node_groups.get(NODE_GROUP_NAME)
    if ng is not None and ng.bl_idname == "GeometryNodeTree":
        # Verify it has GeometryNodeSetMeshNormal
        if any(n.bl_idname == "GeometryNodeSetMeshNormal" for n in ng.nodes):
            return ng
        bpy.data.node_groups.remove(ng)

    ng = bpy.data.node_groups.new(NODE_GROUP_NAME, "GeometryNodeTree")

    # Interface sockets
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    s_factor = ng.interface.new_socket("Factor", in_out="INPUT", socket_type="NodeSocketFloat")
    s_factor.default_value = 1.0
    s_factor.min_value = 0.0
    s_factor.max_value = 1.0
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = ng.nodes
    links = ng.links

    n_in = nodes.new("NodeGroupInput")
    n_in.location = (-800, 0)

    n_out = nodes.new("NodeGroupOutput")
    n_out.location = (600, 0)

    n_norm = nodes.new("GeometryNodeInputNormal")
    n_norm.location = (-800, -200)

    n_point_norm = nodes.new("GeometryNodeFieldOnDomain")
    n_point_norm.domain = "POINT"
    n_point_norm.data_type = "FLOAT_VECTOR"
    n_point_norm.location = (-560, -200)

    n_attr = nodes.new("GeometryNodeInputNamedAttribute")
    n_attr.data_type = "FLOAT_VECTOR"
    n_attr.inputs["Name"].default_value = DELTA_ATTR_NAME
    n_attr.location = (-800, -400)

    n_scale = nodes.new("ShaderNodeVectorMath")
    n_scale.operation = "SCALE"
    n_scale.location = (-560, -400)

    n_len = nodes.new("ShaderNodeVectorMath")
    n_len.operation = "LENGTH"
    n_len.location = (-340, -520)

    n_cmp = nodes.new("FunctionNodeCompare")
    n_cmp.data_type = "FLOAT"
    n_cmp.operation = "GREATER_THAN"
    n_cmp.inputs["B"].default_value = 1e-6
    n_cmp.location = (-140, -520)

    n_add = nodes.new("ShaderNodeVectorMath")
    n_add.operation = "ADD"
    n_add.location = (-340, -280)

    n_norm_vec = nodes.new("ShaderNodeVectorMath")
    n_norm_vec.operation = "NORMALIZE"
    n_norm_vec.location = (-140, -280)

    n_sw = nodes.new("GeometryNodeSwitch")
    n_sw.input_type = "VECTOR"
    n_sw.location = (100, -240)

    n_set = nodes.new("GeometryNodeSetMeshNormal")
    n_set.mode = "TANGENT_SPACE"
    n_set.domain = "CORNER"
    n_set.location = (280, 0)

    n_rem = nodes.new("GeometryNodeRemoveAttribute")
    n_rem.inputs["Name"].default_value = DELTA_ATTR_NAME
    n_rem.location = (440, 0)

    # Wire up links
    links.new(n_norm.outputs["Normal"], n_point_norm.inputs["Value"])
    links.new(n_attr.outputs["Attribute"], n_scale.inputs[0])
    links.new(n_in.outputs["Factor"], n_scale.inputs["Scale"])

    links.new(n_scale.outputs["Vector"], n_len.inputs[0])
    links.new(n_len.outputs["Value"], n_cmp.inputs["A"])

    links.new(n_point_norm.outputs["Value"], n_add.inputs[0])
    links.new(n_scale.outputs["Vector"], n_add.inputs[1])
    links.new(n_add.outputs["Vector"], n_norm_vec.inputs[0])

    links.new(n_cmp.outputs["Result"], n_sw.inputs["Switch"])
    links.new(n_norm.outputs["Normal"], n_sw.inputs["False"])
    links.new(n_norm_vec.outputs["Vector"], n_sw.inputs["True"])

    links.new(n_in.outputs["Geometry"], n_set.inputs["Mesh"])
    links.new(n_sw.outputs["Output"], n_set.inputs["Custom Normal"])
    links.new(n_set.outputs["Mesh"], n_rem.inputs["Geometry"])
    links.new(n_rem.outputs["Geometry"], n_out.inputs["Geometry"])

    return ng


def is_dne_modifier(m: bpy.types.Modifier) -> bool:
    """Return True if m is a DennokoNormalEditor Geometry Nodes modifier."""
    if m.type != "NODES":
        return False
    if m.name == MODIFIER_NAME:
        return True
    ng = m.node_group
    return ng is not None and ng.name == NODE_GROUP_NAME


def get_modifier_factor(obj: bpy.types.Object) -> float:
    """Return the 'Factor' input of obj's DennokoNormalEditor modifier (1.0 if not found)."""
    if obj is None:
        return 1.0
    for m in obj.modifiers:
        if not is_dne_modifier(m) or m.node_group is None:
            continue
        for item in m.node_group.interface.items_tree:
            if getattr(item, "in_out", None) == "INPUT" and item.name == "Factor":
                try:
                    return max(0.0, min(1.0, float(m[item.identifier])))
                except Exception:
                    return 1.0
    return 1.0


def ensure_delta_attribute(mesh: bpy.types.Mesh) -> bpy.types.Attribute:
    """Ensure the POINT FLOAT_VECTOR attribute '_dne_delta' exists on the mesh."""
    attr = mesh.attributes.get(DELTA_ATTR_NAME)
    if attr is not None:
        if attr.domain == "POINT" and attr.data_type == "FLOAT_VECTOR":
            return attr
        mesh.attributes.remove(attr)
    return mesh.attributes.new(name=DELTA_ATTR_NAME, type="FLOAT_VECTOR", domain="POINT")


def ensure_dne_modifier(obj: bpy.types.Object) -> bpy.types.NodesModifier | None:
    """Ensure the object has the DennokoNormalEditor modifier placed before any Armature modifier."""
    if obj is None or obj.type != "MESH" or obj.data is None:
        return None

    ensure_delta_attribute(obj.data)
    ng = ensure_node_group()

    mod = None
    for m in obj.modifiers:
        if m.type == "NODES" and (m.name == MODIFIER_NAME or m.node_group == ng):
            mod = m
            break

    created = False
    if mod is None:
        mod = obj.modifiers.new(name=MODIFIER_NAME, type="NODES")
        mod.node_group = ng
        created = True
    else:
        if mod.node_group != ng:
            mod.node_group = ng

    # Ensure Factor socket defaults to 1.0 when newly created
    if created:
        for item in ng.interface.items_tree:
            if getattr(item, "in_out", None) == "INPUT" and item.name == "Factor":
                try:
                    mod[item.identifier] = 1.0
                except Exception:
                    pass

        # Move before the first ARMATURE modifier so TANGENT_SPACE normals deform with bones
        arm_idx = next((i for i, m in enumerate(obj.modifiers) if m.type == "ARMATURE"), -1)
        mod_idx = list(obj.modifiers).index(mod)
        if arm_idx >= 0 and mod_idx > arm_idx:
            try:
                # Direct API: works from property update callbacks where operators may be unavailable
                obj.modifiers.move(mod_idx, arm_idx)
            except Exception:
                try:
                    with bpy.context.temp_override(object=obj, active_object=obj):
                        bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=arm_idx)
                except Exception:
                    pass

    return mod


def read_deltas_numpy(mesh: bpy.types.Mesh) -> np.ndarray:
    """Return an (N, 3) float32 numpy array of local normal deltas from mesh."""
    n_verts = len(mesh.vertices)
    if n_verts == 0:
        return np.zeros((0, 3), dtype=np.float32)

    attr = mesh.attributes.get(DELTA_ATTR_NAME)
    if attr is None or attr.domain != "POINT" or attr.data_type != "FLOAT_VECTOR" or len(attr.data) != n_verts:
        return np.zeros((n_verts, 3), dtype=np.float32)

    buf = np.zeros(n_verts * 3, dtype=np.float32)
    attr.data.foreach_get("vector", buf)
    return buf.reshape((n_verts, 3))


def write_deltas_numpy(mesh: bpy.types.Mesh, deltas: np.ndarray) -> None:
    """Write an (N, 3) float32 numpy array of local normal deltas to mesh and tag for update."""
    n_verts = len(mesh.vertices)
    if n_verts == 0 or deltas.shape != (n_verts, 3):
        return

    # Zero out tiny deltas below threshold (matching Unity NormalEdit.ZeroThresholdSq = 1e-8)
    clean = np.asarray(deltas, dtype=np.float32).copy()
    sq_len = np.einsum("ij,ij->i", clean, clean)
    clean[sq_len <= ZERO_THRESHOLD_SQ] = 0.0

    attr = ensure_delta_attribute(mesh)
    attr.data.foreach_set("vector", clean.ravel())
    mesh.update_tag()


def count_edited_vertices(mesh: bpy.types.Mesh) -> int:
    """Return the number of vertices on mesh whose delta squared magnitude exceeds ZERO_THRESHOLD_SQ."""
    if mesh is None or len(mesh.vertices) == 0:
        return 0
    attr = mesh.attributes.get(DELTA_ATTR_NAME)
    if attr is None or attr.domain != "POINT" or attr.data_type != "FLOAT_VECTOR":
        return 0
    deltas = read_deltas_numpy(mesh)
    sq_len = np.einsum("ij,ij->i", deltas, deltas)
    return int(np.count_nonzero(sq_len > ZERO_THRESHOLD_SQ))


def clear_deltas(obj: bpy.types.Object) -> None:
    """Reset all normal deltas on obj.data to zero."""
    if obj is None or obj.type != "MESH" or obj.data is None:
        return
    mesh = obj.data
    n_verts = len(mesh.vertices)
    if n_verts == 0:
        return
    zeros = np.zeros((n_verts, 3), dtype=np.float32)
    write_deltas_numpy(mesh, zeros)
    obj.update_tag()


def read_base_local_normals(mesh: bpy.types.Mesh) -> np.ndarray:
    """Read (N, 3) float32 base vertex normals in local rest space from mesh.vertex_normals."""
    n_verts = len(mesh.vertices)
    if n_verts == 0:
        return np.zeros((0, 3), dtype=np.float32)
    buf = np.zeros(n_verts * 3, dtype=np.float32)
    mesh.vertex_normals.foreach_get("vector", buf)
    normals = buf.reshape((n_verts, 3))
    # Ensure unit length
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = lengths > 1e-6
    normals = np.where(valid, normals / np.where(valid, lengths, 1.0), np.array([0.0, 0.0, 1.0], dtype=np.float32))
    return normals


def compute_applied_corner_normals(mesh: bpy.types.Mesh, factor: float = 1.0) -> np.ndarray:
    """Compute the (N_corners, 3) float32 custom corner normals in rest-pose local space.

    Matches the exact math of the Geometry Nodes tree (including its 'Factor' input) so we can
    bake custom normals onto meshes even when Shape Keys or Armature modifiers are present.
    """
    n_corners = len(mesh.loops)
    n_verts = len(mesh.vertices)
    if n_corners == 0 or n_verts == 0:
        return np.zeros((0, 3), dtype=np.float32)

    corner_normals = np.zeros(n_corners * 3, dtype=np.float32)
    mesh.corner_normals.foreach_get("vector", corner_normals)
    corner_normals = corner_normals.reshape((n_corners, 3))

    deltas = read_deltas_numpy(mesh) * np.float32(max(0.0, min(1.0, float(factor))))
    sq_len = np.einsum("ij,ij->i", deltas, deltas)
    # Same threshold as the node tree: length(delta * Factor) > 1e-6
    edited_mask = sq_len > 1e-12
    if not np.any(edited_mask):
        return corner_normals

    base_vn = read_base_local_normals(mesh)
    new_vn = base_vn + deltas
    norms = np.linalg.norm(new_vn, axis=1, keepdims=True)
    valid = norms > 1e-6
    new_vn = np.where(valid, new_vn / np.where(valid, norms, 1.0), base_vn)

    loop_verts = np.zeros(n_corners, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_verts)

    corner_edited = edited_mask[loop_verts]
    result = np.where(corner_edited[:, None], new_vn[loop_verts], corner_normals)
    return result


def bake_to_edited_copy(obj: bpy.types.Object, link_to_scene: bool = False) -> bpy.types.Mesh | None:
    """Create an independent '<mesh_name>_edited' Mesh datablock with custom normals baked.

    mirrors Unity's DenNormalEditorBaker: creates a baked mesh with '_edited' suffix without
    replacing the original object's mesh, and preserves all Shape Keys.
    """
    if obj is None or obj.type != "MESH" or obj.data is None:
        return None

    src_mesh = obj.data
    new_mesh = src_mesh.copy()
    base_name = f"{src_mesh.name}_edited"
    new_mesh.name = base_name
    new_mesh.use_fake_user = True

    corner_normals = compute_applied_corner_normals(src_mesh, get_modifier_factor(obj))
    if len(corner_normals) == len(new_mesh.loops):
        new_mesh.normals_split_custom_set(corner_normals.tolist())

    attr = new_mesh.attributes.get(DELTA_ATTR_NAME)
    if attr is not None:
        new_mesh.attributes.remove(attr)

    if link_to_scene and bpy.context.collection is not None:
        new_obj = obj.copy()
        new_obj.name = f"{obj.name}_edited"
        new_obj.data = new_mesh
        # Remove DNE modifier on baked object
        for m in list(new_obj.modifiers):
            if is_dne_modifier(m):
                new_obj.modifiers.remove(m)
        bpy.context.collection.objects.link(new_obj)

    return new_mesh


def apply_dne_modifier_keep_shapekeys(obj: bpy.types.Object) -> bool:
    """Apply the DennokoNormalEditor edits directly into obj.data custom normals and remove the modifier.

    Works even when the mesh has Shape Keys (where standard bpy.ops.object.modifier_apply fails).
    """
    if obj is None or obj.type != "MESH" or obj.data is None:
        return False

    mesh = obj.data
    corner_normals = compute_applied_corner_normals(mesh, get_modifier_factor(obj))
    if len(corner_normals) == len(mesh.loops) and len(corner_normals) > 0:
        mesh.normals_split_custom_set(corner_normals.tolist())

    # Remove modifier
    for m in list(obj.modifiers):
        if is_dne_modifier(m):
            obj.modifiers.remove(m)

    # Remove attribute
    attr = mesh.attributes.get(DELTA_ATTR_NAME)
    if attr is not None:
        mesh.attributes.remove(attr)

    mesh.update_tag()
    obj.update_tag()
    return True

