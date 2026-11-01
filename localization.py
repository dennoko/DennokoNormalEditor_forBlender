"""Bilingual (Japanese / English) localization for Dennoko Normal Editor."""

import bpy

_STRINGS = {
    "inspector.targets_header": {
        "ja": "編集対象（複数指定で境界なじませ対応）",
        "en": "Targets (Multi-mesh seam smoothing supported)",
    },
    "inspector.vertex_count": {
        "ja": "{0} 頂点 ({1} 編集済)",
        "en": "{0} verts ({1} edited)",
    },
    "inspector.add_target": {
        "ja": "対象を追加",
        "en": "Add Target",
    },
    "inspector.add_selected": {
        "ja": "選択中のメッシュを対象に追加",
        "en": "Add Selected Meshes to Targets",
    },
    "inspector.btn_edit_end": {
        "ja": "編集終了",
        "en": "Finish Editing",
    },
    "inspector.btn_edit_start": {
        "ja": "編集",
        "en": "Edit",
    },
    "inspector.edit_help": {
        "ja": "3Dビューで頂点をクリックして範囲を選択し、3Dハンドルで法線を傾けるか、右下パネルのスライダーで「なじませる」「形状を戻す」「揃える」等を適用します。\nEsc で選択解除、もう一度 Esc で編集終了。",
        "en": "Click a vertex in 3D View to select an area, drag the 3D handle to tilt normals, or use the overlay sliders for Smooth, Restore Shape, Flatten, etc.\nEsc to deselect, Esc again to finish.",
    },
    "inspector.brush_header": {
        "ja": "ブラシ設定",
        "en": "Brush Settings",
    },
    "inspector.brush_radius": {
        "ja": "半径",
        "en": "Radius",
    },
    "inspector.brush_falloff": {
        "ja": "減衰",
        "en": "Falloff",
    },
    "inspector.mirror_on": {
        "ja": "ミラー: 有効 (ON)",
        "en": "Mirror: Enabled (ON)",
    },
    "inspector.mirror_off": {
        "ja": "ミラー: 無効 (OFF)",
        "en": "Mirror: Disabled (OFF)",
    },
    "inspector.mirror_axis": {
        "ja": "ミラー軸",
        "en": "Mirror Axis",
    },
    "inspector.mask": {
        "ja": "マスク",
        "en": "Mask",
    },
    "inspector.mask_no_uv": {
        "ja": "このメッシュには UV0 が無いため、マスクは使われません。",
        "en": "This mesh has no UV0, so the mask is ignored.",
    },
    "inspector.bake_header": {
        "ja": "ベイク / 適用",
        "en": "Bake / Apply",
    },
    "inspector.bake_help": {
        "ja": "編集結果を適用した _edited メッシュを生成するか、シェイプキーを保持したまま対象メッシュへ直接適用します。",
        "en": "Creates an '_edited' mesh copy with applied normals, or applies directly to the mesh while preserving Shape Keys.",
    },
    "inspector.btn_bake": {
        "ja": "ベイク (_edited 複製出力)",
        "en": "Bake (_edited Copy)",
    },
    "inspector.btn_apply": {
        "ja": "モディファイヤーを適用 (シェイプキー保持)",
        "en": "Apply Modifier (Keep Shape Keys)",
    },
    "inspector.btn_clear_all": {
        "ja": "すべての編集をクリア",
        "en": "Clear All Edits",
    },
    "overlay.title": {
        "ja": "Dennoko Normal Editor — 編集中",
        "en": "Dennoko Normal Editor — Editing",
    },
    "overlay.radius": {
        "ja": "半径",
        "en": "Radius",
    },
    "overlay.falloff": {
        "ja": "減衰",
        "en": "Falloff",
    },
    "overlay.mirror_on": {
        "ja": "ミラー: 有効 (ON)",
        "en": "Mirror: Enabled (ON)",
    },
    "overlay.mirror_off": {
        "ja": "ミラー: 無効 (OFF)",
        "en": "Mirror: Disabled (OFF)",
    },
    "overlay.effects_header": {
        "ja": "選択範囲の法線調整（スライダーで適用）",
        "en": "Selected Area Adjustments",
    },
    "overlay.effect_smooth": {
        "ja": "なじませる",
        "en": "Smooth",
    },
    "overlay.effect_restore_shape": {
        "ja": "形状を戻す",
        "en": "Shape",
    },
    "overlay.effect_flatten": {
        "ja": "揃える",
        "en": "Flatten",
    },
    "overlay.effect_erase": {
        "ja": "元に戻す",
        "en": "Erase",
    },
    "overlay.no_selection_prompt": {
        "ja": "頂点をクリックすると範囲選択＆調整できます",
        "en": "Click a vertex to select area & adjust",
    },
    "overlay.hint_wheel": {
        "ja": "※ 3Dハンドルで傾ける / ハンドル保持中ホイールで半径変更",
        "en": "※ Drag 3D handle to tilt / Wheel while grabbing to resize",
    },
    "overlay.hint_esc": {
        "ja": "※ Escキーで編集終了（選択中は選択解除）",
        "en": "※ Esc to finish editing (deselects if selected)",
    },
}


def is_japanese() -> bool:
    """Return True if Blender's UI is actually being displayed in Japanese.

    Uses the resolved locale so that 'Automatic' (DEFAULT) follows the OS language, and
    non-Japanese / non-English UIs fall back to English instead of Japanese.
    """
    try:
        prefs = bpy.context.preferences
        view = prefs.view if prefs else None
        if view is not None and not getattr(view, "use_translate_interface", True):
            return False
        locale = str(bpy.app.translations.locale or "")
        if locale:
            return locale.startswith("ja")
        if view is not None:
            return str(getattr(view, "language", "")).startswith("ja")
    except Exception:
        pass
    return False


def tr(key: str, *args) -> str:
    """Translate key to Japanese or English."""
    entry = _STRINGS.get(key)
    if not entry:
        return key
    lang = "ja" if is_japanese() else "en"
    text = entry.get(lang) or entry.get("ja") or key
    if args:
        try:
            return text.format(*args)
        except Exception:
            return text
    return text

