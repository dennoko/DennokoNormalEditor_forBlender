"""3D Viewport GPU rendering for Dennoko Normal Editor.

Draws:
1. Depth-tested 3D normal pins, vertex dots, brush circles (primary + mirror), and the
   3D interactive tilt handle gizmo (center free-move box + X/Y/Z axis arrows).
2. Bottom-right draggable 2D overlay panel matching the Unity SceneView overlay:
   - Header bar with drag grip and [×] close button
   - Radius slider (0.001 .. 0.500)
   - Falloff selector (Smooth / Linear / Sharp / Constant)
   - Mirror ON/OFF toggle + Axis selector (X / Y / Z)
   - 4 live-preview effect sliders & 1-click buttons (Smooth / Restore Shape / Flatten / Erase)
   - Footer hints
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Vector
import numpy as np

from . import core_math
from .localization import tr

MAX_DRAWN_VERTICES = 1500
NORMAL_PIN_LENGTH_RATIO = 0.35
HANDLE_SCREEN_PIXEL_LENGTH = 75.0
HANDLE_AXIS_PIXEL_LENGTH = 44.0


@dataclass
class Rect2D:
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


# ---------------------------------------------------------------------------
# 2D Primitives
# ---------------------------------------------------------------------------

def _draw_rect_fill(rect: Rect2D, color: tuple[float, float, float, float]) -> None:
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    verts = (
        (rect.x, rect.y),
        (rect.x2, rect.y),
        (rect.x2, rect.y2),
        (rect.x, rect.y2),
    )
    indices = ((0, 1, 2), (0, 2, 3))
    batch = batch_for_shader(shader, "TRIS", {"pos": verts}, indices=indices)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_rect_outline(rect: Rect2D, color: tuple[float, float, float, float], thickness: float = 2.0) -> None:
    _draw_rect_fill(Rect2D(rect.x, rect.y, rect.w, thickness), color)
    _draw_rect_fill(Rect2D(rect.x, rect.y2 - thickness, rect.w, thickness), color)
    _draw_rect_fill(Rect2D(rect.x, rect.y + thickness, thickness, max(0.0, rect.h - thickness * 2.0)), color)
    _draw_rect_fill(Rect2D(rect.x2 - thickness, rect.y + thickness, thickness, max(0.0, rect.h - thickness * 2.0)), color)


def _draw_text(text: str, x: float, y: float, size: int = 12, color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> None:
    font_id = 0
    blf.size(font_id, size)
    blf.color(font_id, *color)
    blf.position(font_id, x, y, 0.0)
    blf.draw(font_id, text)


def _text_width(text: str, size: int = 12) -> float:
    font_id = 0
    blf.size(font_id, size)
    return blf.dimensions(font_id, text)[0]


# ---------------------------------------------------------------------------
# 3D Helpers & Handle Geometry
# ---------------------------------------------------------------------------

def world_pixel_size(region: bpy.types.Region, rv3d: bpy.types.RegionView3D, world_pos: np.ndarray | Vector) -> float:
    """Return world-space size of 1 screen pixel at world_pos."""
    v = Vector((float(world_pos[0]), float(world_pos[1]), float(world_pos[2])))
    try:
        px = view3d_utils.region_2d_to_location_3d(region, rv3d, (region.width * 0.5 + 1.0, region.height * 0.5), v)
        p0 = view3d_utils.region_2d_to_location_3d(region, rv3d, (region.width * 0.5, region.height * 0.5), v)
        d = (px - p0).length
        return max(1e-5, d)
    except Exception:
        return 0.002


def get_handle_rest_length(region: bpy.types.Region, rv3d: bpy.types.RegionView3D, center_world: np.ndarray) -> float:
    """Compute world-space length for the normal handle so its screen size stays consistent."""
    px_size = world_pixel_size(region, rv3d, center_world)
    return max(0.008, min(0.25, px_size * HANDLE_SCREEN_PIXEL_LENGTH))


def get_handle_axis_length(region: bpy.types.Region, rv3d: bpy.types.RegionView3D, handle_pos: np.ndarray) -> float:
    """Compute world-space length for the X/Y/Z axis arrows of the 3D handle."""
    px_size = world_pixel_size(region, rv3d, handle_pos)
    return max(0.005, min(0.18, px_size * HANDLE_AXIS_PIXEL_LENGTH))


def hit_test_3d_handle(
    region: bpy.types.Region,
    rv3d: bpy.types.RegionView3D,
    mouse_xy: tuple[float, float],
    handle_pos: np.ndarray,
) -> str | None:
    """Return 'FREE', 'X', 'Y', 'Z', or None depending on which part of the 3D handle is under mouse_xy."""
    hp = Vector((float(handle_pos[0]), float(handle_pos[1]), float(handle_pos[2])))
    center_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, hp)
    if center_2d is None:
        return None

    mx, my = float(mouse_xy[0]), float(mouse_xy[1])
    dist_center = math.hypot(mx - center_2d.x, my - center_2d.y)
    if dist_center <= 15.0:
        return "FREE"

    axis_len = get_handle_axis_length(region, rv3d, handle_pos)
    axes = (
        ("X", Vector((axis_len, 0.0, 0.0))),
        ("Y", Vector((0.0, axis_len, 0.0))),
        ("Z", Vector((0.0, 0.0, axis_len))),
    )
    best_axis = None
    best_dist = 12.0

    for name, delta in axes:
        tip_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, hp + delta)
        if tip_2d is None:
            continue
        # Point-to-segment distance in 2D from center_2d to tip_2d
        seg_x = tip_2d.x - center_2d.x
        seg_y = tip_2d.y - center_2d.y
        seg_len_sq = seg_x * seg_x + seg_y * seg_y
        if seg_len_sq <= 1e-3:
            continue
        t = max(0.0, min(1.0, ((mx - center_2d.x) * seg_x + (my - center_2d.y) * seg_y) / seg_len_sq))
        proj_x = center_2d.x + t * seg_x
        proj_y = center_2d.y + t * seg_y
        d = math.hypot(mx - proj_x, my - proj_y)
        if d < best_dist:
            best_dist = d
            best_axis = name

    return best_axis


def _build_circle_3d_verts(center: np.ndarray, view_normal: np.ndarray, radius: float, segments: int = 48) -> list[tuple[float, float, float]]:
    n = core_math.safe_normalize_vec(view_normal)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(float(n[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = core_math.safe_normalize_vec(np.cross(n, ref))
    v = core_math.safe_normalize_vec(np.cross(n, u))

    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False, dtype=np.float32)
    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]
    pts = center[None, :] + (cos_a * u[None, :] + sin_a * v[None, :]) * float(radius)

    # Line loop pairs
    lines: list[tuple[float, float, float]] = []
    for i in range(segments):
        p0 = pts[i]
        p1 = pts[(i + 1) % segments]
        lines.append((float(p0[0]), float(p0[1]), float(p0[2])))
        lines.append((float(p1[0]), float(p1[1]), float(p1[2])))
    return lines


def draw_3d_gizmos(session) -> None:
    """Draw 3D brush circles, depth-tested normal pins, vertex dots, and the 3D tilt handle."""
    context = bpy.context
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return

    view_inv = rv3d.view_matrix.inverted_safe()
    cam_forward = np.array(-view_inv.col[2].xyz, dtype=np.float32)

    gpu.state.blend_set("ALPHA")

    # 1. Hover Mode (when no vertex is selected)
    if not session.has_selection:
        if session.hover_target is not None and 0 <= session.hover_index < len(session.hover_target.world_vertices):
            hovered = session.hover_target.world_vertices[session.hover_index]
            circle_verts = _build_circle_3d_verts(hovered, cam_forward, session.brush_radius)
            shader_u = gpu.shader.from_builtin("UNIFORM_COLOR")
            shader_u.bind()

            gpu.state.depth_test_set("NONE")
            gpu.state.line_width_set(1.8)
            shader_u.uniform_float("color", (1.0, 0.8, 0.2, 0.65))
            batch_for_shader(shader_u, "LINES", {"pos": circle_verts}).draw(shader_u)

            # Hover normal pins (depth-tested)
            radius = max(1e-5, session.brush_radius)
            radius_sq = radius * radius
            inv_radius = 1.0 / radius
            pin_len = max(0.003, min(0.04, radius * NORMAL_PIN_LENGTH_RATIO))

            line_pts: list[tuple[float, float, float]] = []
            line_cols: list[tuple[float, float, float, float]] = []
            drawn = 0

            for target in session.targets:
                verts = target.world_vertices
                normals = target.world_normals
                if verts is None or normals is None or len(verts) == 0:
                    continue
                diff = verts - hovered[None, :]
                d_sq = np.einsum("ij,ij->i", diff, diff)
                in_range = np.nonzero(d_sq < radius_sq)[0]
                for idx in in_range:
                    mask_val = 1.0 if target.mask is None else float(target.mask[idx])
                    if mask_val <= 0.0:
                        continue
                    drawn += 1
                    if drawn > MAX_DRAWN_VERTICES:
                        break
                    t = math.sqrt(float(d_sq[idx])) * inv_radius
                    w = float(core_math.falloff_weight(t, session.falloff)) * mask_val
                    alpha = 0.4 + 0.5 * w
                    p0 = verts[idx]
                    p1 = p0 + normals[idx] * pin_len
                    line_pts.append((float(p0[0]), float(p0[1]), float(p0[2])))
                    line_pts.append((float(p1[0]), float(p1[1]), float(p1[2])))
                    c = (0.15, 0.6, 1.0, alpha)
                    line_cols.append(c)
                    line_cols.append(c)
                if drawn > MAX_DRAWN_VERTICES:
                    break

            gpu.state.depth_test_set("LESS_EQUAL")
            if line_pts:
                shader_s = gpu.shader.from_builtin("SMOOTH_COLOR")
                shader_s.bind()
                gpu.state.line_width_set(1.5)
                batch_for_shader(shader_s, "LINES", {"pos": line_pts, "color": line_cols}).draw(shader_s)

            # Hovered vertex dot
            gpu.state.depth_test_set("NONE")
            gpu.state.point_size_set(8.0)
            shader_u.bind()
            shader_u.uniform_float("color", (1.0, 0.8, 0.2, 0.95))
            batch_for_shader(shader_u, "POINTS", {"pos": [(float(hovered[0]), float(hovered[1]), float(hovered[2]))]}).draw(shader_u)

        gpu.state.point_size_set(1.0)
        gpu.state.line_width_set(1.0)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")
        return

    # 2. Selection Mode
    actively_adjusting = session.dragging_handle or session.active_effect != "NONE"
    alpha_mult = 0.45 if actively_adjusting else 1.0

    shader_u = gpu.shader.from_builtin("UNIFORM_COLOR")
    shader_s = gpu.shader.from_builtin("SMOOTH_COLOR")

    # Brush circles
    gpu.state.depth_test_set("NONE")
    gpu.state.line_width_set(2.0)
    shader_u.bind()
    shader_u.uniform_float("color", (0.2, 0.7, 1.0, 0.85 * alpha_mult))
    c_verts = _build_circle_3d_verts(session.center_world, cam_forward, session.brush_radius)
    batch_for_shader(shader_u, "LINES", {"pos": c_verts}).draw(shader_u)

    if session.mirror_active:
        shader_u.uniform_float("color", (1.0, 0.35, 0.55, 0.85 * alpha_mult))
        m_verts = _build_circle_3d_verts(session.mirror_center_world, cam_forward, session.brush_radius)
        batch_for_shader(shader_u, "LINES", {"pos": m_verts}).draw(shader_u)

    # Influence normal pins (depth-tested)
    pin_len = max(0.003, min(0.04, session.brush_radius * NORMAL_PIN_LENGTH_RATIO))
    pin_pts: list[tuple[float, float, float]] = []
    pin_cols: list[tuple[float, float, float, float]] = []
    dot_pts: list[tuple[float, float, float]] = []
    dot_cols: list[tuple[float, float, float, float]] = []

    total_inf = sum(len(inf.indices) for inf in session.influences)
    step = max(1, total_inf // MAX_DRAWN_VERTICES)
    cursor = 0

    for inf in session.influences:
        verts = inf.target.world_vertices
        normals = inf.target.world_normals
        for k in range(len(inf.indices)):
            if cursor % step != 0:
                cursor += 1
                continue
            cursor += 1
            v_idx = int(inf.indices[k])
            strength = min(1.0, max(0.0, float(inf.weights[k] + inf.mirror_weights[k])))
            r = 0.15 + (1.0 - 0.15) * strength
            g = 0.55 + (0.25 - 0.55) * strength
            b = 1.0 + (0.2 - 1.0) * strength
            a = (0.65 + (0.98 - 0.65) * strength) * alpha_mult
            col = (r, g, b, a)

            p0 = verts[v_idx]
            p1 = p0 + normals[v_idx] * pin_len
            pt0 = (float(p0[0]), float(p0[1]), float(p0[2]))
            pt1 = (float(p1[0]), float(p1[1]), float(p1[2]))
            pin_pts.append(pt0)
            pin_pts.append(pt1)
            pin_cols.append(col)
            pin_cols.append(col)
            dot_pts.append(pt0)
            dot_cols.append(col)

    gpu.state.depth_test_set("LESS_EQUAL")
    if pin_pts:
        shader_s.bind()
        gpu.state.line_width_set(1.6)
        batch_for_shader(shader_s, "LINES", {"pos": pin_pts, "color": pin_cols}).draw(shader_s)
        gpu.state.point_size_set(4.5)
        batch_for_shader(shader_s, "POINTS", {"pos": dot_pts, "color": dot_cols}).draw(shader_s)

    # 3. 3D Tilt Handle Gizmo (always drawn on top for unobstructed manipulation)
    gpu.state.depth_test_set("NONE")
    cw = (float(session.center_world[0]), float(session.center_world[1]), float(session.center_world[2]))
    hp = (float(session.handle_position[0]), float(session.handle_position[1]), float(session.handle_position[2]))

    # Guideline from selected vertex to handle position
    shader_u.bind()
    gpu.state.line_width_set(2.2)
    shader_u.uniform_float("color", (1.0, 0.9, 0.2, 0.95))
    batch_for_shader(shader_u, "LINES", {"pos": [cw, hp]}).draw(shader_u)

    # Selected vertex anchor dot
    gpu.state.point_size_set(7.0)
    shader_u.uniform_float("color", (1.0, 0.9, 0.2, 1.0))
    batch_for_shader(shader_u, "POINTS", {"pos": [cw]}).draw(shader_u)

    # X / Y / Z Axis Arrows at handle_position
    axis_len = get_handle_axis_length(region, rv3d, session.handle_position)
    hover_part = session.dragging_axis if session.dragging_handle else session.hover_handle_part

    axes_draw = (
        ("X", (hp[0] + axis_len, hp[1], hp[2]), (1.0, 0.25, 0.25, 0.95), (1.0, 0.85, 0.2, 1.0)),
        ("Y", (hp[0], hp[1] + axis_len, hp[2]), (0.25, 0.95, 0.35, 0.95), (1.0, 0.85, 0.2, 1.0)),
        ("Z", (hp[0], hp[1], hp[2] + axis_len), (0.25, 0.55, 1.0, 0.95), (1.0, 0.85, 0.2, 1.0)),
    )
    for axis_name, tip_pos, base_col, hi_col in axes_draw:
        is_hi = hover_part == axis_name
        col = hi_col if is_hi else base_col
        gpu.state.line_width_set(3.2 if is_hi else 2.2)
        shader_u.uniform_float("color", col)
        batch_for_shader(shader_u, "LINES", {"pos": [hp, tip_pos]}).draw(shader_u)
        gpu.state.point_size_set(8.0 if is_hi else 6.0)
        batch_for_shader(shader_u, "POINTS", {"pos": [tip_pos]}).draw(shader_u)

    # Center free-move handle cap
    is_free_hi = hover_part == "FREE"
    gpu.state.point_size_set(13.0 if is_free_hi else 10.0)
    shader_u.uniform_float("color", (1.0, 0.95, 0.3, 1.0) if is_free_hi else (1.0, 1.0, 1.0, 0.95))
    batch_for_shader(shader_u, "POINTS", {"pos": [hp]}).draw(shader_u)

    gpu.state.point_size_set(1.0)
    gpu.state.line_width_set(1.0)
    gpu.state.depth_test_set("NONE")
    gpu.state.blend_set("NONE")


# ---------------------------------------------------------------------------
# 2D Bottom-Right Overlay Panel (EditSession.Overlay.cs equivalent)
# ---------------------------------------------------------------------------

OVERLAY_WIDTH = 348.0
OVERLAY_HEIGHT = 276.0
OVERLAY_MARGIN = 16.0


def _sidebar_overlap_width(area: bpy.types.Area | None, region: bpy.types.Region) -> float:
    """Width of the N-panel sidebar floating over the right edge of 'region' (Region Overlap ON)."""
    if area is None:
        return 0.0
    right = region.x + region.width
    inset = 0.0
    for r in area.regions:
        # Hidden regions report a 1px size; without Region Overlap the sidebar starts at 'right'
        if r.type != "UI" or r.width <= 1 or r.height <= 1:
            continue
        if region.x < r.x < right:
            inset = max(inset, float(right - r.x))
    return inset


def compute_overlay_rect(region: bpy.types.Region, session, area: bpy.types.Area | None = None) -> Rect2D:
    """Compute the bottom-right overlay rectangle in region pixel coordinates (y=0 at bottom)."""
    rw = float(region.width) - _sidebar_overlap_width(area, region)
    rh = float(region.height)
    max_x = max(OVERLAY_MARGIN, rw - OVERLAY_WIDTH - OVERLAY_MARGIN - 28.0)
    max_y = max(OVERLAY_MARGIN, rh - OVERLAY_HEIGHT - OVERLAY_MARGIN)

    if not session.has_custom_overlay_pos:
        ox = max_x
        oy = OVERLAY_MARGIN + 12.0
    else:
        ox = max(OVERLAY_MARGIN, min(max_x, session.overlay_pos[0]))
        oy = max(OVERLAY_MARGIN, min(max_y, session.overlay_pos[1]))
    session.overlay_pos = (ox, oy)
    return Rect2D(ox, oy, OVERLAY_WIDTH, OVERLAY_HEIGHT)


def build_overlay_widgets(panel: Rect2D) -> dict[str, Rect2D]:
    """Return layout rectangles for all interactive controls in the overlay panel."""
    x0 = panel.x + 12.0
    w_inner = panel.w - 24.0
    top = panel.y2

    widgets: dict[str, Rect2D] = {}
    widgets["header"] = Rect2D(panel.x, top - 28.0, panel.w - 30.0, 28.0)
    widgets["close"] = Rect2D(panel.x2 - 28.0, top - 26.0, 22.0, 22.0)

    # Row 1: Radius slider
    y_rad = top - 56.0
    widgets["radius_slider"] = Rect2D(x0 + 58.0, y_rad, w_inner - 110.0, 18.0)

    # Row 2: Falloff buttons (Smooth, Linear, Sharp, Constant)
    y_fall = top - 82.0
    btn_w = (w_inner - 58.0 - 9.0) / 4.0
    for i, f_id in enumerate(("SMOOTH", "LINEAR", "SHARP", "CONSTANT")):
        widgets[f"falloff_{f_id}"] = Rect2D(x0 + 58.0 + i * (btn_w + 3.0), y_fall, btn_w, 20.0)

    # Row 3: Mirror toggle + X/Y/Z axis buttons
    y_mir = top - 110.0
    axis_area_w = 84.0
    widgets["mirror_toggle"] = Rect2D(x0, y_mir, w_inner - axis_area_w - 6.0, 22.0)
    ax_w = (axis_area_w - 6.0) / 3.0
    for i, ax in enumerate(("X", "Y", "Z")):
        widgets[f"axis_{ax}"] = Rect2D(x0 + w_inner - axis_area_w + i * (ax_w + 3.0), y_mir, ax_w, 22.0)

    # Effect sliders & 1-click label buttons (Smooth, RestoreShape, Flatten, Erase)
    y_eff_base = top - 156.0
    eff_keys = ("SMOOTH", "RESTORE_SHAPE", "FLATTEN", "ERASE")
    for idx, eff in enumerate(eff_keys):
        ry = y_eff_base - idx * 24.0
        widgets[f"eff_btn_{eff}"] = Rect2D(x0, ry, 84.0, 19.0)
        widgets[f"eff_slider_{eff}"] = Rect2D(x0 + 90.0, ry, w_inner - 134.0, 19.0)

    return widgets


def _draw_slider(rect: Rect2D, ratio: float, enabled: bool = True, fill_color=(0.25, 0.78, 0.45, 0.9)) -> None:
    ratio = max(0.0, min(1.0, float(ratio)))
    bg = (0.09, 0.09, 0.09, 0.95) if enabled else (0.12, 0.12, 0.12, 0.5)
    _draw_rect_fill(rect, bg)
    _draw_rect_outline(rect, (0.32, 0.32, 0.32, 0.8), 1.0)
    if ratio > 0.0 and enabled:
        _draw_rect_fill(Rect2D(rect.x + 1.0, rect.y + 1.0, (rect.w - 2.0) * ratio, rect.h - 2.0), fill_color)
    # Thumb
    tx = rect.x + (rect.w - 6.0) * ratio
    thumb_col = (0.92, 0.92, 0.92, 0.98) if enabled else (0.4, 0.4, 0.4, 0.5)
    _draw_rect_fill(Rect2D(tx, rect.y - 1.0, 6.0, rect.h + 2.0), thumb_col)


def draw_2d_overlay(session) -> None:
    """Draw the bottom-right interactive overlay panel in the 3D Viewport."""
    context = bpy.context
    region = context.region
    if region is None:
        return

    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")

    panel = compute_overlay_rect(region, session, context.area)
    widgets = build_overlay_widgets(panel)

    # Background & Green Outline (matching Unity EditSession.Overlay.cs)
    _draw_rect_fill(panel, (0.15, 0.15, 0.15, 0.94))
    _draw_rect_outline(panel, (0.25, 0.88, 0.45, 0.95), 2.0)

    # Header bar
    _draw_rect_fill(Rect2D(panel.x + 2.0, panel.y2 - 28.0, panel.w - 4.0, 26.0), (0.20, 0.22, 0.20, 0.95))
    _draw_text(tr("overlay.title"), panel.x + 12.0, panel.y2 - 20.0, 12, (0.95, 0.98, 0.95, 1.0))

    # Close button [×]
    c_rect = widgets["close"]
    _draw_rect_fill(c_rect, (0.32, 0.18, 0.18, 0.9))
    _draw_text("×", c_rect.x + 6.0, c_rect.y + 5.0, 13, (1.0, 0.85, 0.85, 1.0))

    x0 = panel.x + 12.0

    # Row 1: Radius
    r_rect = widgets["radius_slider"]
    _draw_text(tr("overlay.radius"), x0, r_rect.y + 4.0, 11, (0.9, 0.9, 0.9, 1.0))
    r_ratio = (session.brush_radius - session.min_radius) / (session.max_radius - session.min_radius)
    _draw_slider(r_rect, r_ratio, enabled=True)
    _draw_text(f"{session.brush_radius:.3f}", r_rect.x2 + 6.0, r_rect.y + 4.0, 11, (0.9, 0.9, 0.9, 1.0))

    # Row 2: Falloff
    f_first = widgets["falloff_SMOOTH"]
    _draw_text(tr("overlay.falloff"), x0, f_first.y + 4.0, 11, (0.9, 0.9, 0.9, 1.0))
    falloff_labels = (
        ("SMOOTH", "Smooth"),
        ("LINEAR", "Linear"),
        ("SHARP", "Sharp"),
        ("CONSTANT", "Const"),
    )
    for f_id, f_label in falloff_labels:
        rect = widgets[f"falloff_{f_id}"]
        active = session.falloff == f_id
        bg = (0.25, 0.72, 0.42, 0.95) if active else (0.24, 0.24, 0.24, 0.9)
        _draw_rect_fill(rect, bg)
        _draw_rect_outline(rect, (0.4, 0.9, 0.55, 0.9) if active else (0.35, 0.35, 0.35, 0.7), 1.0)
        tw = _text_width(f_label, 10)
        _draw_text(f_label, rect.x + max(2.0, (rect.w - tw) * 0.5), rect.y + 5.0, 10, (1.0, 1.0, 1.0, 1.0))

    # Row 3: Mirror Toggle & Axis
    m_rect = widgets["mirror_toggle"]
    m_on = session.mirror
    m_bg = (0.26, 0.78, 0.38, 0.95) if m_on else (0.25, 0.25, 0.25, 0.9)
    _draw_rect_fill(m_rect, m_bg)
    _draw_rect_outline(m_rect, (0.45, 0.95, 0.55, 0.95) if m_on else (0.38, 0.38, 0.38, 0.8), 1.0)
    m_text = tr("overlay.mirror_on") if m_on else tr("overlay.mirror_off")
    tw = _text_width(m_text, 11)
    _draw_text(m_text, m_rect.x + max(4.0, (m_rect.w - tw) * 0.5), m_rect.y + 5.0, 11, (1.0, 1.0, 1.0, 1.0))

    for ax in ("X", "Y", "Z"):
        a_rect = widgets[f"axis_{ax}"]
        ax_active = session.mirror_axis == ax
        a_bg = (0.25, 0.65, 0.90, 0.95) if ax_active else (0.22, 0.22, 0.22, 0.9)
        _draw_rect_fill(a_rect, a_bg)
        _draw_rect_outline(a_rect, (0.45, 0.80, 1.0, 0.95) if ax_active else (0.35, 0.35, 0.35, 0.8), 1.0)
        _draw_text(ax, a_rect.x + (a_rect.w - 7.0) * 0.5, a_rect.y + 5.0, 11, (1.0, 1.0, 1.0, 1.0))

    # Row 4: Effects Section Header
    _draw_text(tr("overlay.effects_header"), x0, panel.y2 - 130.0, 11, (0.85, 0.92, 0.85, 1.0))

    eff_defs = (
        ("SMOOTH", tr("overlay.effect_smooth"), session.slider_smooth),
        ("RESTORE_SHAPE", tr("overlay.effect_restore_shape"), session.slider_restore_shape),
        ("FLATTEN", tr("overlay.effect_flatten"), session.slider_flatten),
        ("ERASE", tr("overlay.effect_erase"), session.slider_erase),
    )
    has_sel = session.has_selection
    for eff_id, label, val in eff_defs:
        b_rect = widgets[f"eff_btn_{eff_id}"]
        s_rect = widgets[f"eff_slider_{eff_id}"]
        btn_bg = (0.26, 0.30, 0.28, 0.95) if has_sel else (0.18, 0.18, 0.18, 0.5)
        _draw_rect_fill(b_rect, btn_bg)
        _draw_rect_outline(b_rect, (0.38, 0.48, 0.40, 0.9) if has_sel else (0.28, 0.28, 0.28, 0.5), 1.0)
        txt_col = (0.95, 0.95, 0.95, 1.0) if has_sel else (0.5, 0.5, 0.5, 0.6)
        _draw_text(label, b_rect.x + 6.0, b_rect.y + 4.0, 11, txt_col)

        _draw_slider(s_rect, val, enabled=has_sel)
        pct_text = f"{int(round(val * 100))}%"
        _draw_text(pct_text, s_rect.x2 + 6.0, s_rect.y + 4.0, 11, txt_col)

    # Footer Hints
    hint_col = (0.32, 0.95, 0.48, 1.0)
    _draw_text(tr("overlay.hint_wheel"), x0, panel.y + 24.0, 10, hint_col)
    _draw_text(tr("overlay.hint_esc"), x0, panel.y + 9.0, 10, hint_col)

    gpu.state.blend_set("NONE")

