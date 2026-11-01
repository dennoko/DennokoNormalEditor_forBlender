"""UI Panels, Target Management Operators, and Context Menu for Dennoko Normal Editor."""

from __future__ import annotations

import bpy

from . import modifier_utils
from . import modal_operator
from .localization import tr


class DNE_OT_quick_start(bpy.types.Operator):
    """Register selected mesh objects and immediately start Dennoko Normal Editor session."""

    bl_idname = "dne.quick_start"
    bl_label = "Dennoko Normal Editor"
    bl_description = "Register selected mesh objects (e.g., Head and Body) and start interactive normal editing"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        scene_props = context.scene.dennoko_normal_editor
        selected_meshes = [o for o in context.selected_objects if o.type == "MESH"]
        if not selected_meshes and context.active_object and context.active_object.type == "MESH":
            selected_meshes = [context.active_object]

        existing = {it.obj for it in scene_props.targets if it.obj is not None}
        for obj in selected_meshes:
            modifier_utils.ensure_dne_modifier(obj)
            if obj not in existing:
                item = scene_props.targets.add()
                item.obj = obj
                existing.add(obj)

        session = modal_operator.get_active_session()
        if session is not None:
            session.refresh_targets(context)
            return {"FINISHED"}

        # Invoke modal session in active 3D Viewport
        if context.area and context.area.type == "VIEW_3D":
            bpy.ops.dne.modal_edit_session("INVOKE_DEFAULT")
            return {"FINISHED"}

        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    region = next((r for r in area.regions if r.type == "WINDOW"), None)
                    if region is not None:
                        with context.temp_override(window=window, area=area, region=region):
                            bpy.ops.dne.modal_edit_session("INVOKE_DEFAULT")
                        return {"FINISHED"}

        return {"FINISHED"}


class DNE_OT_add_selected_targets(bpy.types.Operator):
    """Add all currently selected mesh objects to the target list."""

    bl_idname = "dne.add_selected_targets"
    bl_label = "Add Selected Meshes"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        scene_props = context.scene.dennoko_normal_editor
        existing = {it.obj for it in scene_props.targets if it.obj is not None}
        added = 0
        for obj in context.selected_objects:
            if obj.type == "MESH" and obj not in existing:
                modifier_utils.ensure_dne_modifier(obj)
                item = scene_props.targets.add()
                item.obj = obj
                existing.add(obj)
                added += 1

        session = modal_operator.get_active_session()
        if session is not None and added > 0:
            session.refresh_targets(context)
        return {"FINISHED"}


class DNE_OT_remove_target(bpy.types.Operator):
    """Remove a target entry from the target list."""

    bl_idname = "dne.remove_target"
    bl_label = "Remove Target"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(default=-1)

    def execute(self, context: bpy.types.Context):
        scene_props = context.scene.dennoko_normal_editor
        if 0 <= self.index < len(scene_props.targets):
            scene_props.targets.remove(self.index)
            session = modal_operator.get_active_session()
            if session is not None:
                session.refresh_targets(context)
        return {"FINISHED"}


class DNE_OT_clear_target_edits(bpy.types.Operator):
    """Clear normal edits on a single target or all targets."""

    bl_idname = "dne.clear_target_edits"
    bl_label = "Clear Normal Edits"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(default=-1)

    def execute(self, context: bpy.types.Context):
        scene_props = context.scene.dennoko_normal_editor
        session = modal_operator.get_active_session()
        if session is not None:
            # Make the clear undoable with the session's Ctrl+Z
            if session.is_interacting():
                session.cancel_interaction()
            session.push_undo_snapshot_before_commit("Clear Edits")

        if self.index >= 0:
            if self.index < len(scene_props.targets):
                obj = scene_props.targets[self.index].obj
                if obj is not None:
                    modifier_utils.clear_deltas(obj)
        else:
            for item in scene_props.targets:
                if item.obj is not None:
                    modifier_utils.clear_deltas(item.obj)

        if session is not None:
            session.refresh_targets(context)
        return {"FINISHED"}


class DNE_OT_bake_edited_meshes(bpy.types.Operator):
    """Bake edited normals into independent '<mesh>_edited' Mesh datablocks."""

    bl_idname = "dne.bake_edited_meshes"
    bl_label = "Bake _edited Meshes"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        scene_props = context.scene.dennoko_normal_editor
        baked_names: list[str] = []
        for item in scene_props.targets:
            if item.obj is not None and item.obj.type == "MESH":
                baked = modifier_utils.bake_to_edited_copy(item.obj, link_to_scene=True)
                if baked is not None:
                    baked_names.append(baked.name)

        if baked_names:
            self.report({"INFO"}, f"Baked: {', '.join(baked_names)}")
        else:
            self.report({"WARNING"}, "No valid target meshes to bake.")
        return {"FINISHED"}


class DNE_OT_apply_modifiers(bpy.types.Operator):
    """Apply DennokoNormalEditor modifiers directly to target meshes while keeping Shape Keys."""

    bl_idname = "dne.apply_modifiers"
    bl_label = "Apply Normal Edits to Mesh (Keep Shape Keys)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        # End active session before applying modifiers (works from any editor / F3 search;
        # this operator's own UNDO step records the result)
        modal_operator.end_active_session(push_undo=False)

        scene_props = context.scene.dennoko_normal_editor
        applied_count = 0
        for item in scene_props.targets:
            if item.obj is not None and item.obj.type == "MESH":
                if modifier_utils.apply_dne_modifier_keep_shapekeys(item.obj):
                    applied_count += 1

        if applied_count > 0:
            self.report({"INFO"}, f"Applied custom normals to {applied_count} mesh(es) (Shape Keys preserved).")
        return {"FINISHED"}


class VIEW3D_PT_dennoko_normal_editor(bpy.types.Panel):
    """N-Panel UI for Dennoko Normal Editor (mirrors Unity Inspector)."""

    bl_label = "Dennoko Normal Editor"
    bl_idname = "VIEW3D_PT_dennoko_normal_editor"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Dennoko"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        scene_props = context.scene.dennoko_normal_editor
        is_editing = modal_operator.get_active_session() is not None

        # 1. Big Start / Finish Editing Toggle Button
        row = layout.row(align=True)
        row.scale_y = 1.6
        btn_label = tr("inspector.btn_edit_end") if is_editing else tr("inspector.btn_edit_start")
        btn_icon = "PAUSE" if is_editing else "BRUSH_DATA"
        row.operator("dne.modal_edit_session", text=btn_label, icon=btn_icon, depress=is_editing)

        if is_editing:
            box_help = layout.box()
            for line in tr("inspector.edit_help").split("\n"):
                box_help.label(text=line, icon="INFO" if line == tr("inspector.edit_help").split("\n")[0] else "BLANK1")

        layout.separator(factor=0.5)

        # 2. Targets List (Multi-mesh seam smoothing supported)
        box_targets = layout.box()
        box_targets.label(text=tr("inspector.targets_header"), icon="OUTLINER_OB_MESH")

        for idx, item in enumerate(scene_props.targets):
            t_col = box_targets.column(align=True)
            row = t_col.row(align=True)
            row.prop(item, "obj", text="")

            if item.obj is not None and item.obj.type == "MESH" and item.obj.data is not None:
                mesh = item.obj.data
                v_cnt = len(mesh.vertices)
                ed_cnt = modifier_utils.count_edited_vertices(mesh)
                row.label(text=tr("inspector.vertex_count", v_cnt, ed_cnt))

                op_clr = row.operator("dne.clear_target_edits", text="", icon="LOOP_BACK")
                op_clr.index = idx

            op_rem = row.operator("dne.remove_target", text="", icon="X")
            op_rem.index = idx

            # Indented Strength Mask row (matching Unity DenNormalEditorInspector)
            if item.obj is not None and item.obj.type == "MESH" and item.obj.data is not None:
                sub = t_col.row(align=True)
                sub.separator(factor=2.0)
                sub.prop(item, "strength_mask", text=tr("inspector.mask"))
                if item.strength_mask is not None and len(item.obj.data.uv_layers) == 0:
                    warn_row = t_col.row()
                    warn_row.separator(factor=2.0)
                    warn_row.label(text=tr("inspector.mask_no_uv"), icon="ERROR")

            box_targets.separator(factor=0.3)

        # Drop / Pick Slot to add new target + Add Selected button
        add_row = box_targets.row(align=True)
        add_row.prop(scene_props, "add_target_picker", text=tr("inspector.add_target"), icon="ADD")
        box_targets.operator("dne.add_selected_targets", text=tr("inspector.add_selected"), icon="RESTRICT_SELECT_OFF")

        layout.separator(factor=0.5)

        # 3. Brush & Mirror Settings
        box_brush = layout.box()
        box_brush.label(text=tr("inspector.brush_header"), icon="PROP_ON")
        box_brush.prop(scene_props, "brush_radius", text=tr("inspector.brush_radius"), slider=True)
        box_brush.prop(scene_props, "falloff", text=tr("inspector.brush_falloff"))

        m_row = box_brush.row(align=True)
        m_label = tr("inspector.mirror_on") if scene_props.mirror else tr("inspector.mirror_off")
        m_row.prop(scene_props, "mirror", text=m_label, toggle=True, icon="MOD_MIRROR")
        sub_ax = box_brush.row(align=True)
        sub_ax.enabled = scene_props.mirror
        sub_ax.prop(scene_props, "mirror_axis", text=tr("inspector.mirror_axis"), expand=True)

        layout.separator(factor=0.5)

        # 4. Bake / Apply / Clear All
        box_bake = layout.box()
        box_bake.label(text=tr("inspector.bake_header"), icon="EXPORT")
        box_bake.operator("dne.bake_edited_meshes", text=tr("inspector.btn_bake"), icon="DUPLICATE")
        box_bake.operator("dne.apply_modifiers", text=tr("inspector.btn_apply"), icon="CHECKMARK")

        op_clear_all = box_bake.operator("dne.clear_target_edits", text=tr("inspector.btn_clear_all"), icon="TRASH")
        op_clear_all.index = -1


def draw_object_context_menu(self, context: bpy.types.Context) -> None:
    """Append 'Dennoko Normal Editor' to the 3D Viewport Object Right-Click Context Menu."""
    if context.mode == "OBJECT":
        layout = self.layout
        layout.separator()
        layout.operator("dne.quick_start", text="Dennoko Normal Editor", icon="NORMALS_VERTEX")

