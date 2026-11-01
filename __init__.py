"""Dennoko Normal Editor for Blender.

Non-destructive proportional custom normal editor for VRChat avatars & 3D characters.
Stores normal edits as a Geometry Nodes modifier ('Set Mesh Normal' in TANGENT_SPACE mode)
and provides the exact Unity DennokoNormalEditor experience:
- Surface vertex click selection with BVHTree occlusion culling
- 3D normal tip handle for proportional tilting + mouse wheel radius adjustment while grabbing
- Bottom-right draggable 2D viewport overlay with Smooth, Restore Shape, Flatten, and Erase sliders
- Cross-mesh seam welding (e.g., Head + Body neck seam smoothing)
- Per-target UV0 strength mask texture support
"""

from __future__ import annotations

import bpy
from bpy.app.handlers import persistent

from . import modifier_utils
from . import modal_operator
from . import ui_panel

bl_info = {
    "name": "Dennoko Normal Editor",
    "author": "dennoko",
    "version": (1, 0, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar (N) > Dennoko / Object Context Menu > Dennoko Normal Editor",
    "description": "Non-destructive proportional custom normal editor with modifier storage",
    "category": "Mesh",
}


def _on_brush_settings_updated(self, context: bpy.types.Context) -> None:
    modal_operator.notify_brush_settings_changed()


def _on_mask_updated(self, context: bpy.types.Context) -> None:
    modal_operator.notify_mask_settings_changed()


@persistent
def _on_load_pre(*_args) -> None:
    # The session references objects of the file being closed
    modal_operator.end_active_session(push_undo=False)


@persistent
def _on_undo_redo_post(*_args) -> None:
    modal_operator.on_global_undo()


def _on_target_obj_updated(self, context: bpy.types.Context) -> None:
    if self.obj is not None and self.obj.type == "MESH":
        modifier_utils.ensure_dne_modifier(self.obj)
    session = modal_operator.get_active_session()
    if session is not None:
        session.refresh_targets(context)


def _on_add_target_picker_updated(self, context: bpy.types.Context) -> None:
    picked = self.add_target_picker
    if picked is None:
        return
    if picked.type == "MESH":
        existing = {it.obj for it in self.targets if it.obj is not None}
        if picked not in existing:
            modifier_utils.ensure_dne_modifier(picked)
            item = self.targets.add()
            item.obj = picked
            session = modal_operator.get_active_session()
            if session is not None:
                session.refresh_targets(context)
    # Clear picker slot immediately (acting like Unity's "対象を追加" drop slot)
    self["add_target_picker"] = None


def _poll_mesh_object(self, obj: bpy.types.Object) -> bool:
    return obj is not None and obj.type == "MESH"


class DNE_PG_target_item(bpy.types.PropertyGroup):
    obj: bpy.props.PointerProperty(
        name="Target Mesh",
        type=bpy.types.Object,
        poll=_poll_mesh_object,
        update=_on_target_obj_updated,
    )
    strength_mask: bpy.props.PointerProperty(
        name="Mask",
        description="UV0 brightness scales edit strength on this target (White = 100%, Black = 0%)",
        type=bpy.types.Image,
        update=_on_mask_updated,
    )


class DNE_PG_scene_settings(bpy.types.PropertyGroup):
    targets: bpy.props.CollectionProperty(type=DNE_PG_target_item)

    add_target_picker: bpy.props.PointerProperty(
        name="Add Target",
        description="Pick or drop a mesh object to add it to the editing target list",
        type=bpy.types.Object,
        poll=_poll_mesh_object,
        update=_on_add_target_picker_updated,
    )

    brush_radius: bpy.props.FloatProperty(
        name="Radius",
        description="Proportional normal editing brush radius (meters)",
        default=0.03,
        min=0.001,
        max=0.500,
        step=0.1,
        precision=3,
        update=_on_brush_settings_updated,
    )

    falloff: bpy.props.EnumProperty(
        name="Falloff",
        description="Falloff curve from brush center to edge",
        items=[
            ("SMOOTH", "Smooth (スムーズ)", "Smooth hermite falloff"),
            ("LINEAR", "Linear (直線)", "Linear falloff"),
            ("SHARP", "Sharp (シャープ)", "Quadratic sharp falloff"),
            ("CONSTANT", "Constant (一定)", "Uniform weight across radius"),
        ],
        default="SMOOTH",
        update=_on_brush_settings_updated,
    )

    mirror: bpy.props.BoolProperty(
        name="Mirror",
        description="Mirror normal edits across the selected symmetry axis",
        default=False,
        update=_on_brush_settings_updated,
    )

    mirror_axis: bpy.props.EnumProperty(
        name="Mirror Axis",
        items=[
            ("X", "X", "Left / Right symmetry"),
            ("Y", "Y", "Front / Back symmetry"),
            ("Z", "Z", "Up / Down symmetry"),
        ],
        default="X",
        update=_on_brush_settings_updated,
    )


_CLASSES = (
    DNE_PG_target_item,
    DNE_PG_scene_settings,
    modal_operator.DNE_OT_modal_edit_session,
    ui_panel.DNE_OT_quick_start,
    ui_panel.DNE_OT_add_selected_targets,
    ui_panel.DNE_OT_remove_target,
    ui_panel.DNE_OT_clear_target_edits,
    ui_panel.DNE_OT_bake_edited_meshes,
    ui_panel.DNE_OT_apply_modifiers,
    ui_panel.VIEW3D_PT_dennoko_normal_editor,
)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.dennoko_normal_editor = bpy.props.PointerProperty(type=DNE_PG_scene_settings)
    bpy.types.VIEW3D_MT_object_context_menu.append(ui_panel.draw_object_context_menu)
    _add_handler(bpy.app.handlers.load_pre, _on_load_pre)
    _add_handler(bpy.app.handlers.undo_post, _on_undo_redo_post)
    _add_handler(bpy.app.handlers.redo_post, _on_undo_redo_post)


def _add_handler(handlers: list, fn) -> None:
    if fn not in handlers:
        handlers.append(fn)


def _remove_handler(handlers: list, fn) -> None:
    while fn in handlers:
        handlers.remove(fn)


def unregister() -> None:
    modal_operator.end_active_session(push_undo=False)

    _remove_handler(bpy.app.handlers.load_pre, _on_load_pre)
    _remove_handler(bpy.app.handlers.undo_post, _on_undo_redo_post)
    _remove_handler(bpy.app.handlers.redo_post, _on_undo_redo_post)
    bpy.types.VIEW3D_MT_object_context_menu.remove(ui_panel.draw_object_context_menu)
    if hasattr(bpy.types.Scene, "dennoko_normal_editor"):
        del bpy.types.Scene.dennoko_normal_editor
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)

