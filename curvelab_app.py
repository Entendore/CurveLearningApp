#!/usr/bin/env python3
"""
CurveLab — Interactive Bezier, B-Spline & NURBS Learning Application
Supports 2D, 3D, and N-Dimensional visualization.

Controls (2D Mode):
  - Left-drag on a control point to move it
  - Shift+Left-click on canvas to add a point
  - Right-click on canvas for context menu
  - Double-click a point to edit coordinates
  - Scroll wheel to zoom in/out (toward cursor)
  - Middle-click drag to pan the view
  - Delete/Backspace to remove hovered point
  - Ctrl+Z to undo, Ctrl+Shift+Z to redo
  - Ctrl+C to copy points, Ctrl+V to paste
  - Ctrl+S to save, Ctrl+O to load configuration
  - R: Reset View, F: Fit View, G: Toggle Grid Snap, H: Help
  - Arrow keys: Nudge selected point (Shift=10x, Ctrl=grid)
  - Click on curve (no Shift): Pick nearest point, set t
"""

import sys
import os
import csv
import math
import json
import copy
import numpy as np
from numba import njit, prange

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QHBoxLayout, QVBoxLayout,
    QFormLayout, QGroupBox, QLabel, QPushButton, QSlider, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox, QLineEdit, QTextEdit,
    QListWidget, QScrollArea, QSizePolicy, QStackedWidget,
    QFileDialog, QMenu, QMenuBar, QAction, QMessageBox,
    QDialog, QDialogButtonBox, QStatusBar, QToolBar, QTextBrowser
)
from PySide6.QtCore import Qt, Signal, QPointF, QTimer, QPoint
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath, QLinearGradient,
    QKeySequence, QShortcut, QAction, QIcon, QPixmap, QClipboard
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ════════════════════════════════════════════════════════════════
# Numba-Accelerated Curve Mathematics (N-Dimensional)
# ════════════════════════════════════════════════════════════════

@njit(cache=True)
def _bspline_basis_numba(i, p, t, knots):
    if p == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    left = right = 0.0
    d1 = knots[i + p] - knots[i]
    d2 = knots[i + p + 1] - knots[i + 1]
    if abs(d1) > 1e-12:
        left = (t - knots[i]) / d1 * _bspline_basis_numba(i, p - 1, t, knots)
    if abs(d2) > 1e-12:
        right = (knots[i + p + 1] - t) / d2 * _bspline_basis_numba(i + 1, p - 1, t, knots)
    return left + right


class CurveMath:
    """Static methods for evaluating parametric curves (Numba Accelerated)."""

    # ── Bezier ──────────────────────────────────────────────────

    @staticmethod
    def de_casteljau(points, t):
        pts = np.array(points, dtype=np.float64)
        if len(pts) == 0:
            return None, []
        levels = [pts.copy()]
        current = pts
        while len(current) > 1:
            nxt = (1 - t) * current[:-1] + t * current[1:]
            levels.append(nxt.copy())
            current = nxt
        return current[0] if len(current) > 0 else None, levels

    @staticmethod
    def bezier_subdivide(points, t):
        """Subdivide Bezier curve at t, return (left_pts, right_pts)."""
        _, levels = CurveMath.de_casteljau(points, t)
        left = [levels[k][0] for k in range(len(levels))]
        right = [levels[k][-1] for k in range(len(levels) - 1, -1, -1)]
        return left, right

    @staticmethod
    def bezier_tangent(control_points, t):
        """First derivative (tangent) of Bezier at t."""
        pts = np.array(control_points, dtype=np.float64)
        n = len(pts) - 1
        if n < 1:
            return np.zeros(pts.shape[1] if len(pts.shape) > 1 else 2)
        d_pts = n * np.diff(pts, axis=0)
        if len(d_pts) == 0:
            return np.zeros(pts.shape[1])
        result, _ = CurveMath.de_casteljau(d_pts, t)
        return result if result is not None else np.zeros(pts.shape[1])

    @staticmethod
    @njit(parallel=True, nogil=True, cache=True)
    def evaluate_bezier(control_points, num_samples=300):
        n = control_points.shape[0] - 1
        dim = control_points.shape[1]
        if n < 1:
            return np.empty((0, dim), dtype=np.float64)
        curve = np.zeros((num_samples + 1, dim), dtype=np.float64)
        for i in prange(num_samples + 1):
            t = i / num_samples
            current = np.zeros((n + 1, dim), dtype=np.float64)
            for d in range(dim):
                for j in range(n + 1):
                    current[j, d] = control_points[j, d]
            for k in range(n, 0, -1):
                for j in range(k):
                    for d in range(dim):
                        current[j, d] = (1 - t) * current[j, d] + t * current[j + 1, d]
            for d in range(dim):
                curve[i, d] = current[0, d]
        return curve

    # ── B-Spline basis (Cox-de Boor) ────────────────────────────

    @staticmethod
    def bspline_basis(i, p, t, knots):
        return _bspline_basis_numba(i, p, t, knots)

    @staticmethod
    @njit(parallel=True, nogil=True, cache=True)
    def evaluate_bspline(control_points, degree, knots, num_samples=500):
        n = control_points.shape[0] - 1
        dim = control_points.shape[1]
        if n < degree:
            return np.empty((0, dim), dtype=np.float64)
        t_min = knots[degree]
        t_max = knots[n + 1]
        if t_max <= t_min:
            return np.empty((0, dim), dtype=np.float64)
        curve = np.zeros((num_samples + 1, dim), dtype=np.float64)
        for i in prange(num_samples + 1):
            t = t_min + (t_max - t_min) * i / num_samples
            if i == num_samples:
                t = t_max - 1e-10
            for j in range(n + 1):
                b = _bspline_basis_numba(j, degree, t, knots)
                for d in range(dim):
                    curve[i, d] += b * control_points[j, d]
        return curve

    # ── NURBS ───────────────────────────────────────────────────

    @staticmethod
    @njit(parallel=True, nogil=True, cache=True)
    def evaluate_nurbs(control_points, weights, degree, knots, num_samples=500):
        n = control_points.shape[0] - 1
        dim = control_points.shape[1]
        if n < degree or weights.shape[0] != n + 1:
            return np.empty((0, dim), dtype=np.float64)
        t_min = knots[degree]
        t_max = knots[n + 1]
        if t_max <= t_min:
            return np.empty((0, dim), dtype=np.float64)
        curve = np.zeros((num_samples + 1, dim), dtype=np.float64)
        for i in prange(num_samples + 1):
            t = t_min + (t_max - t_min) * i / num_samples
            if i == num_samples:
                t = t_max - 1e-10
            dw = 0.0
            temp_pt = np.zeros(dim, dtype=np.float64)
            for j in range(n + 1):
                b = _bspline_basis_numba(j, degree, t, knots)
                w = weights[j]
                wb = w * b
                dw += wb
                for d in range(dim):
                    temp_pt[d] += wb * control_points[j, d]
            if dw > 1e-12:
                for d in range(dim):
                    curve[i, d] = temp_pt[d] / dw
        return curve

    # ── Basis Data ──────────────────────────────────────────────

    @staticmethod
    @njit(parallel=True, nogil=True, cache=True)
    def evaluate_basis_data(n, degree, knots, num_samples=200):
        t_min = knots[degree]
        t_max = knots[n + 1]
        basis_data = np.zeros((n + 1, num_samples + 1), dtype=np.float64)
        t_vals = np.zeros(num_samples + 1, dtype=np.float64)
        for i in prange(num_samples + 1):
            t = t_min + (t_max - t_min) * i / num_samples
            if i == num_samples:
                t = t_max - 1e-10
            t_vals[i] = t
            for j in range(n + 1):
                basis_data[j, i] = _bspline_basis_numba(j, degree, t, knots)
        return t_vals, basis_data

    # ── Derivatives & Curvature ─────────────────────────────────

    @staticmethod
    def evaluate_derivative(curve_points):
        if len(curve_points) < 2:
            return np.zeros_like(curve_points)
        d1 = np.diff(curve_points, axis=0)
        return np.vstack([d1, d1[-1]])

    @staticmethod
    def evaluate_curvature_2d(curve_points):
        if len(curve_points) < 3:
            return np.zeros(len(curve_points))
        d1 = CurveMath.evaluate_derivative(curve_points)
        d2 = CurveMath.evaluate_derivative(d1)
        curvatures = []
        for i in range(len(curve_points)):
            dx, dy = d1[i, 0], d1[i, 1]
            ddx, ddy = d2[min(i, len(d2)-1), 0], d2[min(i, len(d2)-1), 1]
            cross = abs(dx * ddy - dy * ddx)
            speed_cubed = (dx**2 + dy**2)**1.5
            curvatures.append(cross / speed_cubed if speed_cubed > 1e-12 else 0.0)
        return np.array(curvatures)

    # ── Osculating Circle ───────────────────────────────────────

    @staticmethod
    def osculating_circle_2d(curve_points, t_idx):
        """Fit osculating circle through 3 points around t_idx. Returns (center, radius) or (None, 0)."""
        if t_idx < 1 or t_idx >= len(curve_points) - 1:
            return None, 0.0
        p0 = curve_points[t_idx - 1]
        p1 = curve_points[t_idx]
        p2 = curve_points[t_idx + 1]
        ax, ay = p0[0], p0[1]
        bx, by = p1[0], p1[1]
        cx, cy = p2[0], p2[1]
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-12:
            return None, 0.0
        ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
        uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
        r = math.hypot(ux - bx, uy - by)
        return np.array([ux, uy]), r

    # ── Arc Length & Bounding Box ────────────────────────────────

    @staticmethod
    def arc_length(curve_points):
        if len(curve_points) < 2:
            return 0.0
        diffs = np.diff(curve_points, axis=0)
        return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))

    @staticmethod
    def bounding_box_2d(curve_points):
        if len(curve_points) == 0:
            return (0, 0, 0, 0, 0, 0)
        x_min, y_min = curve_points[:, 0].min(), curve_points[:, 1].min()
        x_max, y_max = curve_points[:, 0].max(), curve_points[:, 1].max()
        return (x_min, y_min, x_max, y_max, x_max - x_min, y_max - y_min)

    # ── Knot helpers ────────────────────────────────────────────

    @staticmethod
    def generate_clamped_knots(n, degree):
        p = degree
        num_knots = n + p + 2
        knots = np.zeros(num_knots, dtype=np.float64)
        for i in range(num_knots):
            if i > p and i < n + 1:
                knots[i] = (i - p) / (n - p + 1)
            elif i >= n + 1:
                knots[i] = 1.0
        return knots

    @staticmethod
    def generate_uniform_knots(n, degree):
        m = n + degree + 1
        return np.array([i / m for i in range(m + 1)], dtype=np.float64)


# ════════════════════════════════════════════════════════════════
# Interactive 2D Curve Canvas
# ════════════════════════════════════════════════════════════════

class CurveCanvas(QWidget):
    pointMoved = Signal()
    pointDragStarted = Signal()
    pointDragFinished = Signal()
    pointAdded = Signal(float, float)
    pointRemoved = Signal(int)
    pointDoubleClicked = Signal(int)
    undoRequested = Signal()
    redoRequested = Signal()
    mouseWorldPos = Signal(float, float)
    nudgeApplied = Signal(int)  # emits index of nudged point
    nudgeFinished = Signal()  # emitted when nudge keys are released
    curvePicked = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.control_points: list = []
        self.curve_points: list = []
        self.reference_curve: list = []
        self.setMinimumSize(500, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._drag_idx = -1
        self._hover_idx = -1
        self._selected_idx = -1
        self._pt_r = 8

        self.show_control_polygon = True
        self.show_curve = True
        self.show_convex_hull = False
        self.show_curvature_comb = False
        self.show_derivatives = False
        self.show_tangent = False
        self.show_normal = False
        self.show_osculating_circle = False
        self.curvature_coloring = False
        self.curvature_comb_scale = 3000.0
        self.derivative_scale = 10.0

        self.de_casteljau_levels: list = []
        self.highlight_point = None
        self.highlight_tangent = None
        self.current_t = 0.5
        self._osculating_center = None
        self._osculating_radius = 0.0

        # Zoom & Pan
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._middle_dragging = False
        self._last_middle_pos = None

        # Grid Snap
        self.grid_snap = False
        self._grid_size = 25.0

        # Cursor world position (for HUD)
        self._cursor_world = None

        # Point-on-curve picking
        self._pick_mode = False  # True when Ctrl is held

        # Zoom indicator
        self._show_zoom_indicator = False
        self._zoom_indicator_timer = QTimer(self)
        self._zoom_indicator_timer.setSingleShot(True)
        self._zoom_indicator_timer.setInterval(1200)
        self._zoom_indicator_timer.timeout.connect(lambda: (setattr(self, '_show_zoom_indicator', False), self.update()))

        self._lvl_colors = [
            QColor(255, 200, 50, 180), QColor(50, 255, 200, 180),
            QColor(255, 100, 150, 180), QColor(150, 100, 255, 180),
            QColor(255, 180, 100, 180), QColor(100, 255, 100, 180),
        ]

    # ── Coordinate transforms ──

    def _screen_to_world(self, screen_pos):
        wx = (screen_pos.x() - self._pan.x()) / self._zoom
        wy = (screen_pos.y() - self._pan.y()) / self._zoom
        return QPointF(wx, wy)

    def _world_to_screen(self, world_pos):
        sx = world_pos.x() * self._zoom + self._pan.x()
        sy = world_pos.y() * self._zoom + self._pan.y()
        return QPointF(sx, sy)

    def _snap_to_grid(self, world_pos):
        if not self.grid_snap:
            return world_pos
        g = self._grid_size
        return QPointF(round(world_pos.x() / g) * g, round(world_pos.y() / g) * g)

    def reset_view(self):
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def set_control_points(self, pts):
        self.control_points = [QPointF(p[0], p[1]) for p in pts]
        self.update()

    def get_control_points(self):
        return [(p.x(), p.y()) for p in self.control_points]

    def set_curve_points(self, pts):
        self.curve_points = [QPointF(p[0], p[1]) for p in pts] if len(pts) > 0 else []
        self.update()

    def set_reference_curve(self, pts):
        self.reference_curve = [QPointF(p[0], p[1]) for p in pts] if len(pts) > 0 else []
        self.update()

    def set_de_casteljau_levels(self, levels):
        self.de_casteljau_levels = []
        for lv in levels[1:]:
            self.de_casteljau_levels.append([QPointF(p[0], p[1]) for p in lv])
        self.update()

    def set_highlight_point(self, pt, t=0.5, tangent=None):
        self.highlight_point = QPointF(pt[0], pt[1]) if pt is not None else None
        self.highlight_tangent = tangent
        self.current_t = t
        self.update()

    def set_osculating_circle(self, center, radius):
        self._osculating_center = QPointF(center[0], center[1]) if center is not None else None
        self._osculating_radius = radius
        self.update()

    def set_pick_mode(self, enabled):
        """Enable/disable curve-picking mode (Ctrl held)."""
        self._pick_mode = enabled
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    # ── Export ──

    def export_pixmap(self):
        pixmap = QPixmap(self.size())
        self.render(pixmap)
        return pixmap

    def export_svg(self, path):
        try:
            from PySide6.QtSvg import QSvgGenerator
        except ImportError:
            return False
        generator = QSvgGenerator()
        generator.setFileName(path)
        generator.setSize(self.size())
        generator.setViewBox(self.rect())
        p = QPainter(generator)
        p.setRenderHint(QPainter.Antialiasing)
        self._paint_content(p)
        p.end()
        return True

    # ── Painting ──

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0, QColor(20, 20, 32))
        grad.setColorAt(1, QColor(30, 30, 45))
        p.fillRect(self.rect(), grad)

        p.save()
        p.translate(self._pan.x(), self._pan.y())
        p.scale(self._zoom, self._zoom)
        self._paint_content(p)
        p.restore()

        self._paint_hud(p)
        p.end()

    def _curvature_to_color(self, curvature, max_curv):
        """Map curvature magnitude to a blue->green->yellow->red color."""
        if max_curv < 1e-12:
            return QColor(0, 200, 255)
        ratio = min(abs(curvature) / max_curv, 1.0)
        if ratio < 0.33:
            t = ratio / 0.33
            r, g, b = int(0 + t * 50), int(200 + t * 55), int(255 - t * 55)
        elif ratio < 0.66:
            t = (ratio - 0.33) / 0.33
            r, g, b = int(50 + t * 205), int(255 - t * 55), int(200 - t * 150)
        else:
            t = (ratio - 0.66) / 0.34
            r, g, b = 255, int(200 - t * 150), int(50 - t * 50)
        return QColor(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

    def _paint_content(self, p):
        self._draw_grid(p)

        if self.show_convex_hull and len(self.control_points) >= 3:
            self._draw_convex_hull(p)

        if self.show_control_polygon and len(self.control_points) > 1:
            p.setPen(QPen(QColor(100, 100, 140), 1.4 / self._zoom, Qt.DashLine))
            for i in range(len(self.control_points) - 1):
                p.drawLine(self.control_points[i], self.control_points[i + 1])

        for li, lv in enumerate(self.de_casteljau_levels):
            c = self._lvl_colors[li % len(self._lvl_colors)]
            p.setPen(QPen(c, 1 / self._zoom, Qt.DotLine))
            for i in range(len(lv) - 1):
                p.drawLine(lv[i], lv[i + 1])
            p.setPen(Qt.NoPen)
            p.setBrush(c)
            for pt in lv:
                p.drawEllipse(pt, 4 / self._zoom, 4 / self._zoom)

        if self.reference_curve and len(self.reference_curve) > 1:
            p.setPen(QPen(QColor(255, 255, 255, 60), 1.5 / self._zoom, Qt.DashLine))
            path = QPainterPath()
            path.moveTo(self.reference_curve[0])
            for pt in self.reference_curve[1:]:
                path.lineTo(pt)
            p.drawPath(path)

        # ── Curve (with optional curvature coloring) ──
        if self.show_curve and len(self.curve_points) > 1:
            if self.curvature_coloring and len(self.curve_points) > 2:
                curve_np = np.array([(pt.x(), pt.y()) for pt in self.curve_points])
                curvatures = CurveMath.evaluate_curvature_2d(curve_np)
                max_curv = max(curvatures.max(), 1e-12)
                seg_w = 2.5 / self._zoom
                for i in range(len(self.curve_points) - 1):
                    color = self._curvature_to_color(curvatures[i], max_curv)
                    p.setPen(QPen(color, seg_w))
                    p.drawLine(self.curve_points[i], self.curve_points[i + 1])
            else:
                p.setPen(QPen(QColor(0, 200, 255), 2.5 / self._zoom))
                path = QPainterPath()
                path.moveTo(self.curve_points[0])
                for pt in self.curve_points[1:]:
                    path.lineTo(pt)
                p.drawPath(path)

        # ── Curvature Comb ──
        if self.show_curvature_comb and len(self.curve_points) > 2:
            curve_np = np.array([(pt.x(), pt.y()) for pt in self.curve_points])
            curvatures = CurveMath.evaluate_curvature_2d(curve_np)
            scale = self.curvature_comb_scale
            lw = max(1, 1.0 / self._zoom)
            comb_tips = []
            p.setPen(QPen(QColor(255, 150, 50, 150), lw))
            for i, pt in enumerate(self.curve_points):
                if i < len(self.curve_points) - 1:
                    dx = self.curve_points[i+1].x() - pt.x()
                    dy = self.curve_points[i+1].y() - pt.y()
                else:
                    dx = pt.x() - self.curve_points[i-1].x()
                    dy = pt.y() - self.curve_points[i-1].y()
                length = math.hypot(dx, dy)
                if length > 1e-6:
                    nx, ny = -dy / length, dx / length
                    c = curvatures[min(i, len(curvatures)-1)]
                    end_pt = pt + QPointF(nx * c * scale, ny * c * scale)
                    p.drawLine(pt, end_pt)
                    comb_tips.append(end_pt)
            # Draw envelope connecting comb tips
            if len(comb_tips) > 1:
                p.setPen(QPen(QColor(255, 180, 80, 100), 1.5 / self._zoom))
                envelope = QPainterPath()
                envelope.moveTo(comb_tips[0])
                for tip in comb_tips[1:]:
                    envelope.lineTo(tip)
                p.drawPath(envelope)

        # ── Tangent & Normal at highlight point ──
        if self.highlight_point and len(self.curve_points) > 2:
            t_idx = min(int(self.current_t * (len(self.curve_points)-1)), len(self.curve_points)-2)
            pt = self.curve_points[t_idx]
            lw = max(1, 2.0 / self._zoom)

            if self.show_derivatives:
                # Velocity vector (green)
                vel = self.curve_points[t_idx+1] - pt
                vel_scaled = vel * self.derivative_scale
                p.setPen(QPen(QColor(100, 255, 100), lw))
                p.drawLine(pt, pt + vel_scaled)
                # Acceleration vector (red)
                if t_idx < len(self.curve_points) - 2:
                    v1 = self.curve_points[t_idx+1] - pt
                    v2 = self.curve_points[t_idx+2] - self.curve_points[t_idx+1]
                    acc = (v2 - v1) * self.derivative_scale * 5
                    p.setPen(QPen(QColor(255, 100, 100), lw))
                    p.drawLine(pt, pt + acc)

            if self.show_tangent and self.highlight_tangent is not None:
                tan = self.highlight_tangent
                tan_len = math.hypot(tan[0], tan[1])
                if tan_len > 1e-6:
                    scale_t = 80.0
                    tx, ty = tan[0] / tan_len * scale_t, tan[1] / tan_len * scale_t
                    p.setPen(QPen(QColor(0, 255, 200, 220), lw, Qt.DashLine))
                    p.drawLine(pt, pt + QPointF(tx, ty))
                    p.drawLine(pt, pt + QPointF(-tx, -ty))

            if self.show_normal and self.highlight_tangent is not None:
                tan = self.highlight_tangent
                tan_len = math.hypot(tan[0], tan[1])
                if tan_len > 1e-6:
                    scale_n = 80.0
                    nx, ny = -tan[1] / tan_len * scale_n, tan[0] / tan_len * scale_n
                    p.setPen(QPen(QColor(255, 150, 255, 220), lw, Qt.DashLine))
                    p.drawLine(pt, pt + QPointF(nx, ny))
                    p.drawLine(pt, pt + QPointF(-nx, -ny))

            if self.show_osculating_circle and self._osculating_center is not None and self._osculating_radius > 1:
                p.setPen(QPen(QColor(255, 220, 100, 120), 1.0 / self._zoom, Qt.DashLine))
                r = self._osculating_radius
                p.setBrush(QColor(255, 220, 100, 10))
                p.drawEllipse(self._osculating_center, r, r)
                # Draw radius line
                p.setPen(QPen(QColor(255, 220, 100, 60), 0.5 / self._zoom, Qt.DotLine))
                p.drawLine(pt, self._osculating_center)

        if self.highlight_point:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 100, 80))
            p.drawEllipse(self.highlight_point, 12 / self._zoom, 12 / self._zoom)
            p.setBrush(QColor(255, 255, 100))
            p.drawEllipse(self.highlight_point, 5 / self._zoom, 5 / self._zoom)

        for i, pt in enumerate(self.control_points):
            hover = i == self._hover_idx or i == self._drag_idx
            selected = i == self._selected_idx
            pen_color = QColor(255, 255, 100) if (hover or selected) else QColor(255, 255, 255)
            if selected:
                pen_color = QColor(100, 255, 100)
            p.setPen(QPen(pen_color, 1.5 / self._zoom))
            p.setBrush(QColor(255, 200, 50) if (hover or selected) else QColor(255, 100, 80))
            r = self._pt_r / self._zoom
            p.drawEllipse(pt, r, r)
            p.setPen(QColor(210, 210, 230))
            p.setFont(QFont("Consolas", max(1, int(8 / self._zoom)), QFont.Bold))
            p.drawText(pt + QPointF(10 / self._zoom, -8 / self._zoom), f"P{i}")

    def _paint_hud(self, p):
        p.setPen(QColor(90, 90, 110))
        p.setFont(QFont("Arial", 7))
        hint = "L-Drag: move | Shift+Click: add | Ctrl+Click: pick curve | Right-Click: menu | Arrows: nudge | Scroll: zoom | Mid-drag: pan"
        p.drawText(8, self.height() - 6, hint)

        if self._show_zoom_indicator:
            p.setPen(QColor(200, 200, 220, 200))
            p.setFont(QFont("Consolas", 10, QFont.Bold))
            p.drawText(self.width() - 50, 22, f"{self._zoom:.1f}x")

        if self.grid_snap:
            p.setPen(QColor(100, 255, 150, 200))
            p.setFont(QFont("Arial", 8, QFont.Bold))
            p.drawText(8, 18, "GRID SNAP")

        # Cursor world coordinate readout (near cursor)
        if self._cursor_world is not None:
            wx, wy = self._cursor_world.x(), self._cursor_world.y()
            txt = f"({wx:.1f}, {wy:.1f})"
            p.setPen(QColor(180, 180, 200, 180))
            p.setFont(QFont("Consolas", 8))
            # Draw near top-right of widget
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(txt)
            p.drawText(self.width() - tw - 10, self.height() - 8, txt)

        # Legend for tangent/normal/osc circle
        legend_y = 18
        p.setFont(QFont("Arial", 7))
        if self.show_tangent:
            p.setPen(QColor(0, 255, 200, 200))
            p.drawText(8, legend_y, "-- Tangent"); legend_y += 13
        if self.show_normal:
            p.setPen(QColor(255, 150, 255, 200))
            p.drawText(8, legend_y, "-- Normal"); legend_y += 13
        if self.show_osculating_circle:
            p.setPen(QColor(255, 220, 100, 200))
            p.drawText(8, legend_y, "o Osculating Circle"); legend_y += 13

        # Pick mode indicator
        if self._pick_mode:
            p.setPen(QColor(255, 200, 100, 220))
            p.setFont(QFont("Arial", 9, QFont.Bold))
            p.drawText(self.width() // 2 - 50, 18, "CURVE PICK MODE")

        # Minimap
        self._draw_minimap(p)

    def _draw_grid(self, p):
        # Minor grid
        p.setPen(QPen(QColor(40, 40, 55), 0.5 / self._zoom))
        s = 50
        w_tl = self._screen_to_world(QPointF(0, 0))
        w_br = self._screen_to_world(QPointF(self.width(), self.height()))
        x_start = int(w_tl.x() / s) * s
        y_start = int(w_tl.y() / s) * s
        x_end = int(w_br.x() / s + 1) * s
        y_end = int(w_br.y() / s + 1) * s
        for x in range(x_start, x_end + s, s):
            p.drawLine(QPointF(x, w_tl.y()), QPointF(x, w_br.y()))
        for y in range(y_start, y_end + s, s):
            p.drawLine(QPointF(w_tl.x(), y), QPointF(w_br.x(), y))
        # Major grid (every 4th line)
        p.setPen(QPen(QColor(55, 55, 75), 1.0 / self._zoom))
        major = 200
        mx_start = int(w_tl.x() / major) * major
        my_start = int(w_tl.y() / major) * major
        mx_end = int(w_br.x() / major + 1) * major
        my_end = int(w_br.y() / major + 1) * major
        for x in range(mx_start, mx_end + major, major):
            p.drawLine(QPointF(x, w_tl.y()), QPointF(x, w_br.y()))
        for y in range(my_start, my_end + major, major):
            p.drawLine(QPointF(w_tl.x(), y), QPointF(w_br.x(), y))

    def _draw_convex_hull(self, p):
        pts = [(pt.x(), pt.y()) for pt in self.control_points]
        hull = self._convex_hull(pts)
        if len(hull) < 3: return
        p.setPen(QPen(QColor(80, 200, 120, 70), 1 / self._zoom, Qt.DashDotLine))
        p.setBrush(QColor(80, 200, 120, 15))
        path = QPainterPath(); path.moveTo(*hull[0])
        for h in hull[1:]: path.lineTo(*h)
        path.closeSubpath(); p.drawPath(path)

    @staticmethod
    def _convex_hull(points):
        pts = sorted(set(points))
        if len(pts) <= 1: return pts
        def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        lo, up = [], []
        for q in pts:
            while len(lo)>=2 and cross(lo[-2],lo[-1],q)<=0: lo.pop()
            lo.append(q)
        for q in reversed(pts):
            while len(up)>=2 and cross(up[-2],up[-1],q)<=0: up.pop()
            up.append(q)
        return lo[:-1]+up[:-1]

    def _hit(self, screen_pos):
        for i, pt in enumerate(self.control_points):
            screen_pt = self._world_to_screen(pt)
            if (screen_pos - screen_pt).manhattanLength() < self._pt_r + 8:
                return i
        return -1

    def _hit_curve(self, screen_pos, threshold=8):
        """Find the nearest curve point to screen_pos. Returns (index, t_value) or (-1, 0)."""
        if len(self.curve_points) < 2:
            return -1, 0.0
        best_dist = float('inf')
        best_idx = -1
        for i, pt in enumerate(self.curve_points):
            sp = self._world_to_screen(pt)
            dist = (screen_pos - sp).manhattanLength()
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_dist < threshold:
            t_val = best_idx / max(len(self.curve_points) - 1, 1)
            return best_idx, t_val
        return -1, 0.0

    def _draw_minimap(self, p):
        """Draw a small minimap in the bottom-right corner showing viewport extent."""
        all_pts = list(self.control_points) + list(self.curve_points)
        if len(all_pts) < 2:
            return
        mm_w, mm_h = 120, 90
        mm_margin = 10
        mm_x = self.width() - mm_w - mm_margin
        mm_y = self.height() - mm_h - mm_margin - 14

        # Compute world bounds of all data
        xs = [pt.x() for pt in all_pts]
        ys = [pt.y() for pt in all_pts]
        wx_min, wx_max = min(xs), max(xs)
        wy_min, wy_max = min(ys), max(ys)
        ww = wx_max - wx_min
        wh = wy_max - wy_min
        if ww < 1e-6: ww = 100
        if wh < 1e-6: wh = 100
        # Add padding
        pad = max(ww, wh) * 0.1
        wx_min -= pad; wx_max += pad
        wy_min -= pad; wy_max += pad
        ww = wx_max - wx_min
        wh = wy_max - wy_min

        # Background
        p.setPen(QPen(QColor(70, 70, 100, 160), 1))
        p.setBrush(QColor(20, 20, 35, 180))
        p.drawRoundedRect(mm_x, mm_y, mm_w, mm_h, 4, 4)

        # Scale factors
        sx = mm_w / ww
        sy = mm_h / wh

        # Draw curve points in minimap
        if len(self.curve_points) > 1:
            p.setPen(QPen(QColor(0, 200, 255, 180), 1))
            for i in range(len(self.curve_points) - 1):
                x1 = mm_x + (self.curve_points[i].x() - wx_min) * sx
                y1 = mm_y + (self.curve_points[i].y() - wy_min) * sy
                x2 = mm_x + (self.curve_points[i+1].x() - wx_min) * sx
                y2 = mm_y + (self.curve_points[i+1].y() - wy_min) * sy
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # Draw viewport rectangle
        w_tl = self._screen_to_world(QPointF(0, 0))
        w_br = self._screen_to_world(QPointF(self.width(), self.height()))
        vx1 = mm_x + (w_tl.x() - wx_min) * sx
        vy1 = mm_y + (w_tl.y() - wy_min) * sy
        vx2 = mm_x + (w_br.x() - wx_min) * sx
        vy2 = mm_y + (w_br.y() - wy_min) * sy
        p.setPen(QPen(QColor(255, 255, 255, 120), 1, Qt.DashLine))
        p.setBrush(QColor(255, 255, 255, 15))
        vw = max(vx2 - vx1, 2)
        vh = max(vy2 - vy1, 2)
        p.drawRect(QPointF(vx1, vy1), QPointF(vx1 + vw, vy1 + vh))

        # Label
        p.setPen(QColor(120, 120, 150, 160))
        p.setFont(QFont("Arial", 6))
        p.drawText(mm_x + 3, mm_y + mm_h - 3, "minimap")

    # ── Mouse Events ──

    def mousePressEvent(self, ev):
        pos = ev.position()
        if ev.button() == Qt.LeftButton:
            idx = self._hit(pos)
            if idx >= 0:
                self._drag_idx = idx
                self._selected_idx = idx
                self.pointDragStarted.emit()
            elif ev.modifiers() & Qt.ShiftModifier:
                world = self._snap_to_grid(self._screen_to_world(pos))
                self.pointAdded.emit(world.x(), world.y())
            elif self._pick_mode and len(self.curve_points) > 1:
                # Point-on-curve picking
                cidx, t_val = self._hit_curve(pos, threshold=12)
                if cidx >= 0:
                    self.curvePicked.emit(t_val)
        elif ev.button() == Qt.RightButton:
            idx = self._hit(pos)
            if idx >= 0:
                self._selected_idx = idx
                self._show_point_context_menu(ev.globalPosition().toPoint(), idx)
            else:
                self._show_canvas_context_menu(ev.globalPosition().toPoint())
        elif ev.button() == Qt.MiddleButton:
            self._middle_dragging = True
            self._last_middle_pos = pos
            self.setCursor(Qt.ClosedHandCursor)
        self.update()

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self._middle_dragging and self._last_middle_pos is not None:
            delta = pos - self._last_middle_pos
            self._pan += delta
            self._last_middle_pos = pos
            self.update()
        elif self._drag_idx >= 0:
            world = self._snap_to_grid(self._screen_to_world(pos))
            self.control_points[self._drag_idx] = world
            self.pointMoved.emit()
        else:
            old = self._hover_idx
            self._hover_idx = self._hit(pos)
            if old != self._hover_idx:
                self.update()
        # Emit world position for status bar
        wp = self._screen_to_world(pos)
        self.mouseWorldPos.emit(wp.x(), wp.y())
        self._cursor_world = wp
        # Repaint for HUD coordinate readout (throttled by paint already)
        if not self._middle_dragging and self._drag_idx < 0:
            self.update()

        # Update pick mode based on Ctrl modifier
        is_ctrl = bool(ev.modifiers() & Qt.ControlModifier)
        if is_ctrl != self._pick_mode:
            self.set_pick_mode(is_ctrl)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            if self._drag_idx >= 0:
                self.pointDragFinished.emit()
            self._drag_idx = -1
        elif ev.button() == Qt.MiddleButton:
            self._middle_dragging = False
            self._last_middle_pos = None
            self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            idx = self._hit(ev.position())
            if idx >= 0:
                self._selected_idx = idx
                self.pointDoubleClicked.emit(idx)

    def wheelEvent(self, ev):
        pos = ev.position()
        old_zoom = self._zoom
        factor = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        new_zoom = max(0.1, min(50.0, self._zoom * factor))
        self._pan.setX(pos.x() - (pos.x() - self._pan.x()) * (new_zoom / old_zoom))
        self._pan.setY(pos.y() - (pos.y() - self._pan.y()) * (new_zoom / old_zoom))
        self._zoom = new_zoom
        self._show_zoom_indicator = True
        self._zoom_indicator_timer.start()
        self.update()

    def leaveEvent(self, ev):
        """Clear cursor world position when mouse leaves the widget."""
        self._cursor_world = None
        self.set_pick_mode(False)
        self.update()
        super().leaveEvent(ev)

    def keyPressEvent(self, ev):
        key = ev.key()
        modifiers = ev.modifiers()
        if key in (Qt.Key_Delete, Qt.Key_Backspace) and self._hover_idx >= 0:
            self.pointRemoved.emit(self._hover_idx)
            self._hover_idx = -1
            self._selected_idx = -1
        elif key == Qt.Key_Z and modifiers & Qt.ControlModifier and modifiers & Qt.ShiftModifier:
            self.redoRequested.emit()
        elif key == Qt.Key_Z and modifiers & Qt.ControlModifier:
            self.undoRequested.emit()
        elif key == Qt.Key_Y and modifiers & Qt.ControlModifier:
            self.redoRequested.emit()
        elif key == Qt.Key_R and not (modifiers & Qt.ControlModifier):
            self.reset_view()
        elif key == Qt.Key_G:
            self.grid_snap = not self.grid_snap
            self.update()
        elif key == Qt.Key_Escape:
            self._selected_idx = -1
            self.update()
        # Arrow key nudging
        elif key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down) and not (modifiers & Qt.ControlModifier):
            target = self._selected_idx if self._selected_idx >= 0 else self._hover_idx
            if 0 <= target < len(self.control_points):
                step = 10.0 if (modifiers & Qt.ShiftModifier) else 1.0
                if self.grid_snap:
                    step = self._grid_size
                dx, dy = 0, 0
                if key == Qt.Key_Left: dx = -step
                elif key == Qt.Key_Right: dx = step
                elif key == Qt.Key_Up: dy = -step
                elif key == Qt.Key_Down: dy = step
                self.control_points[target] += QPointF(dx, dy)
                self._selected_idx = target
                self.nudgeApplied.emit(target)
        # Update pick mode on Ctrl press
        elif key == Qt.Key_Control:
            self.set_pick_mode(True)
        else:
            super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key_Control:
            self.set_pick_mode(False)
        elif ev.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            self.nudgeFinished.emit()
        else:
            super().keyReleaseEvent(ev)

    # ── Context Menus ──

    def _show_point_context_menu(self, screen_pos, pt_idx):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu{background:#1e1e2e;color:#cdd6f4;border:1px solid #45475a;}"
                          "QMenu::item:selected{background:#45475a;}")
        act_edit = menu.addAction(f"Edit P{pt_idx} Coordinates...")
        act_remove = menu.addAction(f"Remove P{pt_idx}")
        act = menu.exec(screen_pos)
        if act == act_edit:
            self.pointDoubleClicked.emit(pt_idx)
        elif act == act_remove:
            self.pointRemoved.emit(pt_idx)
            self._selected_idx = -1

    def _show_canvas_context_menu(self, screen_pos):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu{background:#1e1e2e;color:#cdd6f4;border:1px solid #45475a;}"
                          "QMenu::item:selected{background:#45475a;}")
        world = self._screen_to_world(QPointF(screen_pos))
        act_add = menu.addAction(f"Add Point Here ({world.x():.0f}, {world.y():.0f})")
        menu.addSeparator()
        act_reset = menu.addAction("Reset View")
        act_fit = menu.addAction("Fit to View")
        act = menu.exec(screen_pos)
        if act == act_add:
            world = self._snap_to_grid(world)
            self.pointAdded.emit(world.x(), world.y())
        elif act == act_reset:
            self.reset_view()
        elif act == act_fit:
            # Signal to parent to fit
            parent = self.parent()
            while parent:
                if hasattr(parent, '_fit_view'):
                    parent._fit_view()
                    break
                parent = parent.parent()


# ════════════════════════════════════════════════════════════════
# 3D / N-D Matplotlib Canvas
# ════════════════════════════════════════════════════════════════

class NDCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(facecolor='#1e1e2e')
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = None
        self.setup_3d()

    def setup_3d(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='#1e1e2e')
        self.style_axes()
        self.draw()

    def plot_nd_curve(self, control_points, curve_points, dim=3):
        if self.ax is None: return
        self.ax.cla()
        self.style_axes()
        if len(curve_points) == 0:
            self.draw(); return

        curve = np.array(curve_points)
        ctrl = np.array(control_points)

        if dim == 3:
            self.ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color='#00c8ff', lw=2, label='Curve')
            self.ax.plot(ctrl[:, 0], ctrl[:, 1], ctrl[:, 2], '--', color='#6c7086', lw=1, label='Control Polygon')
            self.ax.scatter(ctrl[:, 0], ctrl[:, 1], ctrl[:, 2], color='#f38ba8', s=40, depthshade=True)
            for i, pt in enumerate(ctrl):
                self.ax.text(pt[0], pt[1], pt[2], f'  P{i}', color='#cdd6f4', fontsize=7)
        elif dim == 2:
            self.ax.plot(curve[:, 0], curve[:, 1], color='#00c8ff', lw=2, label='Curve')
            self.ax.plot(ctrl[:, 0], ctrl[:, 1], '--', color='#6c7086', lw=1, label='Control Polygon')
            self.ax.scatter(ctrl[:, 0], ctrl[:, 1], color='#f38ba8', s=40)
            for i, pt in enumerate(ctrl):
                self.ax.text(pt[0], pt[1], f'  P{i}', color='#cdd6f4', fontsize=7)

        self.ax.legend(fontsize=7, loc='upper left', facecolor='#313244', edgecolor='#45475a', labelcolor='#cdd6f4')
        self.draw()

    def style_axes(self):
        self.ax.set_xlabel('X', color='#89b4fa')
        self.ax.set_ylabel('Y', color='#a6e3a1')
        if hasattr(self.ax, 'set_zlabel'):
            self.ax.set_zlabel('Z', color='#cba6f7')
        self.ax.tick_params(colors='#6c7086', labelsize=7)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        if hasattr(self.ax, 'zaxis'):
            self.ax.zaxis.pane.fill = False


# ════════════════════════════════════════════════════════════════
# Basis-Function Visualisation Canvas
# ════════════════════════════════════════════════════════════════

class BasisCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150); self.setMaximumHeight(200)
        self.t_vals = np.array([]); self.basis_matrix = np.array([])
        self.knots: list = []; self.degree = 0; self.t_marker = None
        self._colors = [QColor(255,100,80), QColor(80,200,255), QColor(255,220,50), QColor(100,255,150), QColor(255,130,255)]

    def set_basis_data(self, t_vals, basis_matrix, knots, degree, t_marker=None):
        self.t_vals = t_vals
        self.basis_matrix = basis_matrix
        self.knots = knots
        self.degree = degree
        self.t_marker = t_marker
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(25, 25, 38))
        ml, mr, mt, mb = 42, 15, 22, 22
        pw = self.width() - ml - mr; ph = self.height() - mt - mb
        if pw <= 0 or ph <= 0: p.end(); return

        p.setPen(QPen(QColor(70,70,90), 1)); p.drawLine(ml, mt, ml, mt+ph); p.drawLine(ml, mt+ph, ml+pw, mt+ph)
        p.setPen(QColor(110,110,130)); p.setFont(QFont("Arial", 7))
        p.drawText(4, mt+6, "1.0"); p.drawText(8, mt+ph+2, "0"); p.drawText(ml+pw-8, mt+ph+14, "1")

        for k in self.knots:
            x = ml + k * pw
            if ml <= x <= ml + pw:
                p.setPen(QPen(QColor(60,60,80), 1, Qt.DotLine)); p.drawLine(int(x), mt, int(x), mt+ph)

        if len(self.basis_matrix) > 0 and len(self.t_vals) > 0:
            n_basis = self.basis_matrix.shape[0]
            for idx in range(n_basis):
                c = self._colors[idx % len(self._colors)]
                p.setPen(QPen(c, 1.5))
                for i in range(len(self.t_vals) - 1):
                    x1 = ml + self.t_vals[i] * pw; y1 = mt + ph - self.basis_matrix[idx, i] * ph
                    x2 = ml + self.t_vals[i+1] * pw; y2 = mt + ph - self.basis_matrix[idx, i+1] * ph
                    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        if self.t_marker is not None and 0 <= self.t_marker <= 1:
            x = ml + self.t_marker * pw
            p.setPen(QPen(QColor(255,255,100,160), 1, Qt.DashLine)); p.drawLine(int(x), mt, int(x), mt+ph)
        p.end()


# ════════════════════════════════════════════════════════════════
# Coordinate Editor Dialog (N-D)
# ════════════════════════════════════════════════════════════════

class CoordinateEditDialog(QDialog):
    def __init__(self, point_idx, coords, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit Point P{point_idx}")
        self.setMinimumWidth(260)
        layout = QFormLayout(self)
        dim_names = ['X', 'Y', 'Z', 'W', 'V', 'U']
        self.spins = []
        for i, v in enumerate(coords):
            spin = QDoubleSpinBox()
            spin.setRange(-10000, 10000)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setValue(v)
            layout.addRow(f"P{point_idx} {dim_names[i] if i < len(dim_names) else f'D{i}'}:", spin)
            self.spins.append(spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_values(self):
        return [s.value() for s in self.spins]


# ════════════════════════════════════════════════════════════════
# Help Dialog
# ════════════════════════════════════════════════════════════════

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CurveLab - Keyboard Shortcuts & Help")
        self.setMinimumSize(520, 520)
        self.resize(520, 520)
        layout = QVBoxLayout(self)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setStyleSheet("QTextBrowser{background:#181825;color:#cdd6f4;border:1px solid #45475a;border-radius:4px;padding:8px;}")
        browser.setHtml("""
        <style>
            h2 { color: #89b4fa; margin-top: 4px; }
            h3 { color: #a6e3a1; margin-top: 10px; }
            table { border-collapse: collapse; width: 100%; }
            td { padding: 3px 8px; border-bottom: 1px solid #313244; }
            td:first-child { color: #f9e2af; font-weight: bold; white-space: nowrap; }
            p { margin: 4px 0; }
        </style>
        <h2>CurveLab Help</h2>
        <h3>Mouse Controls</h3>
        <table>
        <tr><td>Left-drag</td><td>Move a control point</td></tr>
        <tr><td>Shift + Left-click</td><td>Add a new control point</td></tr>
        <tr><td>Right-click (on point)</td><td>Context menu (edit / remove)</td></tr>
        <tr><td>Right-click (on canvas)</td><td>Context menu (add point / view)</td></tr>
        <tr><td>Double-click (on point)</td><td>Edit coordinates precisely</td></tr>
        <tr><td>Scroll wheel</td><td>Zoom in/out (centered on cursor)</td></tr>
        <tr><td>Middle-click drag</td><td>Pan the view</td></tr>
        <tr><td>Ctrl + Left-click (on curve)</td><td>Pick nearest curve point, set t</td></tr>
        <tr><td>Arrow keys</td><td>Nudge selected/hovered point by 1px</td></tr>
        <tr><td>Shift + Arrow keys</td><td>Nudge point by 10px</td></tr>
        <tr><td>Click point in list</td><td>Select point and center view on it</td></tr>
        </table>

        <h3>Keyboard Shortcuts</h3>
        <table>
        <tr><td>Ctrl + Z</td><td>Undo</td></tr>
        <tr><td>Ctrl + Shift + Z / Ctrl + Y</td><td>Redo</td></tr>
        <tr><td>Delete / Backspace</td><td>Remove hovered control point</td></tr>
        <tr><td>Ctrl + C</td><td>Copy control points to clipboard</td></tr>
        <tr><td>Ctrl + V</td><td>Paste control points from clipboard</td></tr>
        <tr><td>Ctrl + S</td><td>Save configuration as JSON</td></tr>
        <tr><td>Ctrl + O</td><td>Load configuration from JSON</td></tr>
        <tr><td>Ctrl + Q</td><td>Quit</td></tr>
        <tr><td>R</td><td>Reset zoom and pan</td></tr>
        <tr><td>F</td><td>Fit all points in view</td></tr>
        <tr><td>G</td><td>Toggle grid snap</td></tr>
        <tr><td>H</td><td>Show this help dialog</td></tr>
        <tr><td>Escape</td><td>Deselect point</td></tr>
        <tr><td>Arrow keys</td><td>Nudge selected point (1px, or grid size if snap on)</td></tr>
        <tr><td>Shift + Arrows</td><td>Nudge by 10px</td></tr>
        </table>

        <h3>Curve Types</h3>
        <p><b>Bezier:</b> Polynomial curve defined by control points. The De Casteljau algorithm
        shows the recursive subdivision construction at parameter t.</p>
        <p><b>B-Spline:</b> Piecewise polynomial with local control. Controlled by degree and knot vector.
        Each basis function is shown in the basis plot.</p>
        <p><b>NURBS:</b> Non-Uniform Rational B-Spline. Adds weights to each control point,
        enabling exact representation of conic sections (circles, ellipses, etc.).</p>

        <h3>Display Options</h3>
        <p><b>Convex Hull:</b> The convex hull property guarantees the curve stays within the
        convex hull of its control points (Bezier only).</p>
        <p><b>Curvature Comb:</b> Visualizes curvature magnitude as offset lines perpendicular
        to the curve. An envelope connects the tips. Adjust the scale slider.</p>
        <p><b>Tangent/Normal:</b> Shows the tangent (cyan) and normal (pink) direction vectors
        at the current parameter t point.</p>
        <p><b>Osculating Circle:</b> The circle that best approximates the curve at the
        highlighted point. Its radius is 1/curvature.</p>
        <p><b>Curvature Coloring:</b> Colors the curve by curvature magnitude: blue=low,
        green=medium, yellow=high, red=very high.</p>
        <p><b>Derivative Vectors:</b> Shows velocity (green) and acceleration (red) at t.</p>

        <h3>Transform Tools</h3>
        <p><b>Reverse Pts:</b> Reverses the order of control points (traverses curve backwards).</p>
        <p><b>Mirror H/V:</b> Mirrors all points horizontally or vertically around their centroid.</p>
        <p><b>Scale Up/Down:</b> Scales all points 1.2x or 0.8x relative to their centroid.</p>

        <h3>Animation Modes</h3>
        <p><b>Bounce:</b> t oscillates back and forth between 0 and 1.</p>
        <p><b>Loop:</b> t wraps around from 1 back to 0 continuously.</p>
        <p><b>Single Pass:</b> t goes from 0 to 1 and stops.</p>

        <h3>Export</h3>
        <p>Export as PNG (raster image), SVG (vector), or CSV (evaluated curve data points with parameter t).</p>
        """)
        layout.addWidget(browser)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


# ════════════════════════════════════════════════════════════════
# Unified Base Tab Logic for N-D
# ════════════════════════════════════════════════════════════════

class BaseCurveTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._animating = False
        self._anim_dir = 1
        self._anim_speed = 2
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._anim_step)
        self._pts_internal = []
        self._num_samples = 500
        self._nudge_history_pushed = False

        # Undo/Redo
        self._history = []
        self._history_idx = -1
        self._max_history = 50

    def _setup_canvases(self, layout):
        self.canvas_stack = QStackedWidget()
        self.canvas_2d = CurveCanvas()
        self.canvas_2d.pointDragStarted.connect(self._on_drag_started)
        self.canvas_2d.pointDragFinished.connect(self._on_drag_finished)
        self.canvas_2d.pointMoved.connect(self._on_2d_moved)
        self.canvas_2d.pointAdded.connect(self._on_2d_added)
        self.canvas_2d.pointRemoved.connect(self._on_2d_removed)
        self.canvas_2d.pointDoubleClicked.connect(self._on_point_double_clicked)
        self.canvas_2d.undoRequested.connect(self._undo)
        self.canvas_2d.redoRequested.connect(self._redo)
        self.canvas_2d.mouseWorldPos.connect(self._on_mouse_world_pos)
        self.canvas_2d.curvePicked.connect(self._on_curve_picked)
        self.canvas_2d.nudgeApplied.connect(self._on_nudge)
        self.canvas_2d.nudgeFinished.connect(self._on_nudge_release)
        self.canvas_stack.addWidget(self.canvas_2d)
        self.canvas_3d = NDCanvas()
        self.canvas_stack.addWidget(self.canvas_3d)
        layout.addWidget(self.canvas_stack, 3)

    def _setup_common_controls(self, rl, show_tangent_options=False):
        # Info Panel
        g_info = QGroupBox("Curve Info")
        g_info_l = QVBoxLayout(g_info)
        self.info_label = QLabel("-")
        self.info_label.setFont(QFont("Consolas", 8))
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #a6adc8; padding: 2px;")
        g_info_l.addWidget(self.info_label)
        rl.addWidget(g_info)

        # View & Edit
        g_view = QGroupBox("View & Edit")
        g_view_l = QVBoxLayout(g_view)

        self.snap_cb = QCheckBox("Grid Snap (G)")
        self.snap_cb.setToolTip("Snap control points to a 25px grid when moving or adding")
        self.snap_cb.toggled.connect(self._on_grid_snap_toggled)
        g_view_l.addWidget(self.snap_cb)

        btn_row1 = QHBoxLayout()
        b_reset_view = QPushButton("Reset View (R)")
        b_reset_view.setToolTip("Reset zoom and pan to default")
        b_reset_view.clicked.connect(self.canvas_2d.reset_view)
        btn_row1.addWidget(b_reset_view)
        b_fit = QPushButton("Fit View (F)")
        b_fit.setToolTip("Zoom to fit all control points in view")
        b_fit.clicked.connect(self._fit_view)
        btn_row1.addWidget(b_fit)
        g_view_l.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        b_export_png = QPushButton("Export PNG")
        b_export_png.setToolTip("Save canvas as a PNG image file")
        b_export_png.clicked.connect(self._export_png)
        btn_row2.addWidget(b_export_png)
        b_export_svg = QPushButton("Export SVG")
        b_export_svg.setToolTip("Save canvas as a vector SVG file")
        b_export_svg.clicked.connect(self._export_svg)
        btn_row2.addWidget(b_export_svg)
        g_view_l.addLayout(btn_row2)

        btn_row3 = QHBoxLayout()
        b_save = QPushButton("Save JSON")
        b_save.setToolTip("Save control points and settings to a JSON file (Ctrl+S)")
        b_save.clicked.connect(self._save_json)
        btn_row3.addWidget(b_save)
        b_load = QPushButton("Load JSON")
        b_load.setToolTip("Load control points and settings from a JSON file (Ctrl+O)")
        b_load.clicked.connect(self._load_json)
        btn_row3.addWidget(b_load)
        g_view_l.addLayout(btn_row3)

        btn_undo_redo = QHBoxLayout()
        b_undo = QPushButton("Undo")
        b_undo.setToolTip("Undo last action (Ctrl+Z)")
        b_undo.clicked.connect(self._undo)
        btn_undo_redo.addWidget(b_undo)
        b_redo = QPushButton("Redo")
        b_redo.setToolTip("Redo last undone action (Ctrl+Shift+Z)")
        b_redo.clicked.connect(self._redo)
        btn_undo_redo.addWidget(b_redo)
        g_view_l.addLayout(btn_undo_redo)

        btn_cp = QHBoxLayout()
        b_copy = QPushButton("Copy Pts")
        b_copy.setToolTip("Copy control points to clipboard as text (Ctrl+C)")
        b_copy.clicked.connect(self._copy_points)
        btn_cp.addWidget(b_copy)
        b_paste = QPushButton("Paste Pts")
        b_paste.setToolTip("Paste control points from clipboard (Ctrl+V)")
        b_paste.clicked.connect(self._paste_points)
        btn_cp.addWidget(b_paste)
        g_view_l.addLayout(btn_cp)

        # Curve data export
        btn_export_data = QHBoxLayout()
        b_csv = QPushButton("Export Curve CSV")
        b_csv.setToolTip("Export evaluated curve points as a CSV file")
        b_csv.clicked.connect(self._export_csv)
        btn_export_data.addWidget(b_csv)
        g_view_l.addLayout(btn_export_data)

        rl.addWidget(g_view)

        # Quality
        g_quality = QGroupBox("Quality & Animation")
        g_q_l = QVBoxLayout(g_quality)

        # Sampling resolution
        sr_row = QHBoxLayout()
        sr_row.addWidget(QLabel("Samples:"))
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(50, 2000)
        self.samples_spin.setSingleStep(50)
        self.samples_spin.setValue(self._num_samples)
        self.samples_spin.setToolTip("Number of samples for curve evaluation (higher = smoother)")
        self.samples_spin.valueChanged.connect(self._on_samples_changed)
        sr_row.addWidget(self.samples_spin)
        g_q_l.addLayout(sr_row)

        # Animation speed
        as_row = QHBoxLayout()
        as_row.addWidget(QLabel("Anim Speed:"))
        self.anim_speed_slider = QSlider(Qt.Horizontal)
        self.anim_speed_slider.setRange(1, 20)
        self.anim_speed_slider.setValue(2)
        self.anim_speed_slider.setToolTip("Animation playback speed")
        self.anim_speed_slider.valueChanged.connect(self._on_anim_speed_changed)
        as_row.addWidget(self.anim_speed_slider)
        self.anim_speed_lbl = QLabel("2")
        self.anim_speed_lbl.setFixedWidth(20)
        as_row.addWidget(self.anim_speed_lbl)
        g_q_l.addLayout(as_row)

        # Animation loop mode
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Anim Mode:"))
        self.anim_mode_combo = QComboBox()
        self.anim_mode_combo.addItems(["Bounce", "Loop", "Single Pass"])
        self.anim_mode_combo.setToolTip("Bounce: back-and-forth | Loop: restart at 0 | Single: stop at 1")
        mode_row.addWidget(self.anim_mode_combo)
        g_q_l.addLayout(mode_row)

        rl.addWidget(g_quality)

    def _on_samples_changed(self, val):
        self._num_samples = val
        self._update()

    def _on_anim_speed_changed(self, val):
        self._anim_speed = val
        self.anim_speed_lbl.setText(str(val))

    def _on_mouse_world_pos(self, wx, wy):
        win = self.window()
        if isinstance(win, QMainWindow) and win.statusBar():
            idx_str = ""
            hi = self.canvas_2d._hover_idx
            if hi >= 0 and hi < len(self._pts_internal):
                pt = self._pts_internal[hi]
                idx_str = f"  |  Hovering P{hi}: ({pt[0]:.1f}, {pt[1]:.1f})"
            win.statusBar().showMessage(f"World: ({wx:.1f}, {wy:.1f})  |  Zoom: {self.canvas_2d._zoom:.1f}x{idx_str}", 100)

    def _on_curve_picked(self, t_val):
        """Called when user Ctrl+Clicks on the curve to pick a t value."""
        slider_val = int(t_val * 1000)
        slider_val = max(0, min(1000, slider_val))
        self.t_slider.setValue(slider_val)
        self.status_message(f"Picked t = {t_val:.3f} from curve")

    def _on_nudge(self, idx):
        """Called when an arrow key nudge is applied to a control point."""
        if not hasattr(self, '_nudge_history_pushed') or not self._nudge_history_pushed:
            self._push_history()
            self._nudge_history_pushed = True
        self._on_2d_nudged()

    def _on_nudge_release(self):
        """Called when arrow keys are released - reset nudge tracking."""
        self._nudge_history_pushed = False

    def _on_grid_snap_toggled(self, checked):
        self.canvas_2d.grid_snap = checked
        self.canvas_2d.update()

    def _fit_view(self):
        # Collect all relevant points (curve + control points) for a better fit
        all_pts_2d = list(self.canvas_2d.curve_points) + list(self.canvas_2d.control_points)
        if not all_pts_2d:
            return
        xs = [p.x() for p in all_pts_2d]
        ys = [p.y() for p in all_pts_2d]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w = x_max - x_min + 100
        h = y_max - y_min + 100
        if w <= 0 or h <= 0:
            return
        zoom_x = self.canvas_2d.width() / w
        zoom_y = self.canvas_2d.height() / h
        self.canvas_2d._zoom = min(zoom_x, zoom_y) * 0.85
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        self.canvas_2d._pan.setX(self.canvas_2d.width() / 2 - cx * self.canvas_2d._zoom)
        self.canvas_2d._pan.setY(self.canvas_2d.height() / 2 - cy * self.canvas_2d._zoom)
        self.canvas_2d.update()

    def _update_info_panel(self, curve_np=None, extra_info="", curve_pt_at_t=None):
        n = len(self._pts_internal)
        dim = self.dim_spin.value()
        text = f"Points: {n}  |  Dim: {dim}"
        if hasattr(self, 'deg_spin'):
            text += f"  |  Degree: {self.deg_spin.value()}"
        if curve_np is not None and len(curve_np) >= 2:
            length = CurveMath.arc_length(curve_np)
            text += f"\nArc Length: {length:.1f} px"
            if dim == 2:
                bb = CurveMath.bounding_box_2d(curve_np)
                text += f"\nBounding Box: {bb[4]:.0f} x {bb[5]:.0f} px"
        if curve_pt_at_t is not None:
            text += f"\nC(t): ({curve_pt_at_t[0]:.2f}, {curve_pt_at_t[1]:.2f})"
        if extra_info:
            text += f"\n{extra_info}"
        self.info_label.setText(text)

    # ── Undo / Redo (fixed: push on drag start, not every move) ──

    def _push_history(self):
        state = copy.deepcopy(self._pts_internal)
        self._history = self._history[:self._history_idx + 1]
        self._history.append(state)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self._history_idx = len(self._history) - 1

    def _on_drag_started(self):
        """Called once when the user starts dragging a point."""
        self._push_history()

    def _on_drag_finished(self):
        """Called when the user releases the mouse after dragging."""
        pass  # History was already saved at drag start

    def _on_nudge_started(self):
        """Called on first arrow key nudge (save history before nudge sequence)."""
        self._push_history()

    def _undo(self):
        if self._history_idx > 0:
            self._history_idx -= 1
            self._pts_internal = copy.deepcopy(self._history[self._history_idx])
            self._apply_pts_to_2d()
            self._update()
            self.status_message("Undo")

    def _redo(self):
        if self._history_idx < len(self._history) - 1:
            self._history_idx += 1
            self._pts_internal = copy.deepcopy(self._history[self._history_idx])
            self._apply_pts_to_2d()
            self._update()
            self.status_message("Redo")

    def status_message(self, msg):
        win = self.window()
        if isinstance(win, QMainWindow) and win.statusBar():
            win.statusBar().showMessage(msg, 2000)

    # ── Export ──

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "curvelab_export.png", "PNG Files (*.png)")
        if path:
            self.canvas_2d.export_pixmap().save(path)
            self.status_message(f"Exported PNG: {os.path.basename(path)}")

    def _export_svg(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export SVG", "curvelab_export.svg", "SVG Files (*.svg)")
        if path:
            if self.canvas_2d.export_svg(path):
                self.status_message(f"Exported SVG: {os.path.basename(path)}")
            else:
                QMessageBox.warning(self, "Export Error", "PySide6.QtSvg is not available.")

    def _export_csv(self):
        """Export the currently evaluated curve points as a CSV file."""
        curve_pts = self.canvas_2d.curve_points
        if not curve_pts:
            QMessageBox.information(self, "No Data", "No curve points to export. Add control points first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Curve CSV", "curvelab_curve.csv", "CSV Files (*.csv)")
        if path:
            try:
                dim = self.dim_spin.value()
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    header = [f"{'XYWVU'[i] if i < 5 else f'D{i}'}" for i in range(dim)] + ['t']
                    writer.writerow(header)
                    n = len(curve_pts)
                    for i, pt in enumerate(curve_pts):
                        row = []
                        # For 2D canvas, use the stored 2D coords; for higher dims we don't have them here
                        row.append(pt.x())
                        row.append(pt.y())
                        # Fill remaining dims with placeholder zeros if needed
                        for d in range(2, dim):
                            row.append(0.0)
                        row.append(round(i / max(n - 1, 1), 6))
                        writer.writerow(row)
                self.status_message(f"Exported CSV ({len(curve_pts)} points): {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", f"Failed to export CSV:\n{e}")

    # ── Save / Load ──

    def _get_save_data(self):
        return {
            "control_points": self._pts_internal,
            "dimension": self.dim_spin.value(),
        }

    def _apply_load_data(self, data):
        if "dimension" in data:
            self.dim_spin.blockSignals(True)
            self.dim_spin.setValue(data["dimension"])
            self.dim_spin.blockSignals(False)
        if "control_points" in data:
            self._pts_internal = copy.deepcopy(data["control_points"])
            self._apply_pts_to_2d()

    def _save_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "curvelab_config.json", "JSON Files (*.json)")
        if path:
            data = self._get_save_data()
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            self.status_message(f"Saved: {os.path.basename(path)}")

    def _load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json)")
        if path:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self._apply_load_data(data)
                self._push_history()
                self._update()
                self.status_message(f"Loaded: {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load:\n{e}")

    # ── Copy / Paste ──

    def _copy_points(self):
        text = json.dumps(self._pts_internal)
        QApplication.clipboard().setText(text)
        self.status_message(f"Copied {len(self._pts_internal)} control points to clipboard")

    def _paste_points(self):
        text = QApplication.clipboard().text()
        try:
            pts = json.loads(text)
            if not isinstance(pts, list) or len(pts) < 2:
                raise ValueError("Need at least 2 points")
            for p in pts:
                if not isinstance(p, list) or len(p) < 2:
                    raise ValueError("Each point must have at least 2 coordinates")
            self._push_history()
            dim = self.dim_spin.value()
            self._pts_internal = [[float(v) for v in p] + [0.0] * (dim - len(p)) for p in pts]
            self._apply_pts_to_2d()
            self._update()
            self.status_message(f"Pasted {len(self._pts_internal)} control points")
        except Exception as e:
            QMessageBox.warning(self, "Paste Error", f"Invalid clipboard data:\n{e}")

    # ── Point double-click ──

    def _on_point_double_clicked(self, idx):
        if 0 <= idx < len(self._pts_internal):
            pt = self._pts_internal[idx]
            dlg = CoordinateEditDialog(idx, pt, self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                vals = dlg.get_values()
                self._push_history()
                for i, v in enumerate(vals):
                    if i < len(self._pts_internal[idx]):
                        self._pts_internal[idx][i] = v
                self._apply_pts_to_2d()
                self._update()

    # ── Dim sync ──

    def _sync_dim(self, dim):
        if dim == 2:
            self.canvas_stack.setCurrentIndex(0)
        else:
            self.canvas_stack.setCurrentIndex(1)
            self.canvas_3d.setup_3d()
        self._update()

    def _apply_pts_to_2d(self):
        pts2d = [p[:2] for p in self._pts_internal]
        self.canvas_2d.set_control_points(pts2d)

    def _on_2d_moved(self):
        # History already saved at drag start - just update internal state
        pts2d = self.canvas_2d.get_control_points()
        for i, p in enumerate(pts2d):
            self._pts_internal[i][0] = p[0]
            self._pts_internal[i][1] = p[1]
        self._update()

    def _on_2d_nudged(self):
        """Called after arrow-key nudge - update internal from 2D canvas."""
        pts2d = self.canvas_2d.get_control_points()
        for i, p in enumerate(pts2d):
            self._pts_internal[i][0] = p[0]
            self._pts_internal[i][1] = p[1]
        self._update()

    def _on_2d_added(self, x, y):
        self._push_history()
        dim = self.dim_spin.value()
        new_pt = [x, y] + [250.0] * (dim - 2)
        self._pts_internal.append(new_pt)
        self._apply_pts_to_2d()
        self._update()

    def _on_2d_removed(self, idx):
        if len(self._pts_internal) > 2 and 0 <= idx < len(self._pts_internal):
            self._push_history()
            self._pts_internal.pop(idx)
            self._apply_pts_to_2d()
            self._update()

    def _toggle_anim(self):
        self._animating = not self._animating
        if self._animating:
            self.anim_btn.setText("Pause"); self._anim_dir = 1; self._anim_timer.start()
        else:
            self.anim_btn.setText("Animate"); self._anim_timer.stop()

    def _anim_step(self):
        mode = self.anim_mode_combo.currentIndex() if hasattr(self, 'anim_mode_combo') else 0
        v = self.t_slider.value() + self._anim_dir * self._anim_speed
        if mode == 0:  # Bounce
            if v >= 1000: v = 1000; self._anim_dir = -1
            elif v <= 0: v = 0; self._anim_dir = 1
        elif mode == 1:  # Loop
            if v > 1000: v = 0
        elif mode == 2:  # Single pass
            if v >= 1000:
                v = 1000
                self._animating = False
                self._anim_timer.stop()
                self.anim_btn.setText("Animate")
        self.t_slider.setValue(v)

    # ── Point List Selection ──

    def _on_pt_list_clicked(self, row):
        """When clicking a point in the list, select it on the canvas and center view."""
        if 0 <= row < len(self.canvas_2d.control_points):
            self.canvas_2d._selected_idx = row
            # Center the view on the selected point
            pt = self.canvas_2d.control_points[row]
            self.canvas_2d._pan.setX(self.canvas_2d.width() / 2 - pt.x() * self.canvas_2d._zoom)
            self.canvas_2d._pan.setY(self.canvas_2d.height() / 2 - pt.y() * self.canvas_2d._zoom)
            self.canvas_2d.update()
            self.status_message(f"Selected P{row}")

    # ── Transform Tools ──

    def _reverse_points(self):
        """Reverse the order of control points."""
        if len(self._pts_internal) < 2:
            return
        self._push_history()
        self._pts_internal.reverse()
        # For NURBS, also reverse weights
        if hasattr(self, 'w_spins') and self.w_spins:
            values = [s.value() for s in self.w_spins]
            values.reverse()
            for i, s in enumerate(self.w_spins):
                s.blockSignals(True)
                s.setValue(values[i])
                s.blockSignals(False)
        self._apply_pts_to_2d()
        self._update()
        self.status_message("Reversed control points")

    def _mirror_horizontal(self):
        """Mirror points horizontally around their centroid."""
        if not self._pts_internal:
            return
        self._push_history()
        cx = sum(p[0] for p in self._pts_internal) / len(self._pts_internal)
        for p in self._pts_internal:
            p[0] = 2 * cx - p[0]
        self._apply_pts_to_2d()
        self._update()
        self.status_message("Mirrored horizontally")

    def _mirror_vertical(self):
        """Mirror points vertically around their centroid."""
        if not self._pts_internal:
            return
        self._push_history()
        cy = sum(p[1] for p in self._pts_internal) / len(self._pts_internal)
        for p in self._pts_internal:
            p[1] = 2 * cy - p[1]
        self._apply_pts_to_2d()
        self._update()
        self.status_message("Mirrored vertically")

    def _scale_points(self, factor):
        """Scale all points toward/away from centroid by a factor."""
        if not self._pts_internal:
            return
        self._push_history()
        dim = self.dim_spin.value()
        centroids = [sum(p[d] for p in self._pts_internal) / len(self._pts_internal) for d in range(dim)]
        for p in self._pts_internal:
            for d in range(dim):
                p[d] = centroids[d] + (p[d] - centroids[d]) * factor
        self._apply_pts_to_2d()
        self._update()
        self.status_message(f"Scaled points by {factor:.2f}x")


# ════════════════════════════════════════════════════════════════
# Bezier Tab
# ════════════════════════════════════════════════════════════════

class BezierTab(BaseCurveTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._reset()

    def _build_ui(self):
        lay = QHBoxLayout(self)
        self._setup_canvases(lay)

        rp = QWidget(); rp.setMaximumWidth(370); rp.setMinimumWidth(310)
        rl = QVBoxLayout(rp)
        title = QLabel("Bezier Curves"); title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setStyleSheet("color:#00c8ff; padding:4px 0;"); rl.addWidget(title)

        g0 = QGroupBox("Dimensions"); g0l = QHBoxLayout(g0)
        self.dim_spin = QSpinBox(); self.dim_spin.setRange(2, 6); self.dim_spin.setValue(2)
        self.dim_spin.setToolTip("Number of dimensions for the curve (2D, 3D, etc.)")
        self.dim_spin.valueChanged.connect(self._sync_dim)
        g0l.addWidget(QLabel("D =")); g0l.addWidget(self.dim_spin); rl.addWidget(g0)

        g = QGroupBox("Parameter t"); gl = QVBoxLayout(g)
        self.t_slider = QSlider(Qt.Horizontal); self.t_slider.setRange(0, 1000); self.t_slider.setValue(500)
        self.t_slider.setToolTip("Current parameter value for curve evaluation")
        self.t_slider.valueChanged.connect(self._update); gl.addWidget(self.t_slider)
        self.t_lbl = QLabel("t = 0.500"); self.t_lbl.setAlignment(Qt.AlignCenter); gl.addWidget(self.t_lbl)
        self.dc_cb = QCheckBox("Show De Casteljau"); self.dc_cb.setChecked(True)
        self.dc_cb.setToolTip("Show the recursive De Casteljau construction at t")
        self.dc_cb.toggled.connect(self._update); gl.addWidget(self.dc_cb)
        self.anim_btn = QPushButton("Animate")
        self.anim_btn.setToolTip("Animate the parameter t back and forth")
        self.anim_btn.clicked.connect(self._toggle_anim); gl.addWidget(self.anim_btn)
        rl.addWidget(g)

        g2 = QGroupBox("Display"); g2l = QVBoxLayout(g2)
        self.poly_cb = QCheckBox("Control polygon"); self.poly_cb.setChecked(True)
        self.poly_cb.setToolTip("Show lines connecting consecutive control points")
        self.poly_cb.toggled.connect(self._update); g2l.addWidget(self.poly_cb)
        self.hull_cb = QCheckBox("Convex hull")
        self.hull_cb.setToolTip("Show the convex hull of control points (Bezier property)")
        self.hull_cb.toggled.connect(self._update); g2l.addWidget(self.hull_cb)
        self.curv_cb = QCheckBox("Curvature Comb")
        self.curv_cb.setToolTip("Visualize curvature as perpendicular offset lines")
        self.curv_cb.toggled.connect(self._update); g2l.addWidget(self.curv_cb)
        self.deriv_cb = QCheckBox("Derivative Vectors")
        self.deriv_cb.setToolTip("Show velocity (green) and acceleration (red) at t")
        self.deriv_cb.toggled.connect(self._update); g2l.addWidget(self.deriv_cb)
        self.tangent_cb = QCheckBox("Tangent Line")
        self.tangent_cb.setToolTip("Show the tangent direction at the curve point (cyan)")
        self.tangent_cb.toggled.connect(self._update); g2l.addWidget(self.tangent_cb)
        self.normal_cb = QCheckBox("Normal Line")
        self.normal_cb.setToolTip("Show the normal direction at the curve point (pink)")
        self.normal_cb.toggled.connect(self._update); g2l.addWidget(self.normal_cb)
        self.osc_cb = QCheckBox("Osculating Circle")
        self.osc_cb.setToolTip("Show the circle of curvature at t (yellow dashed)")
        self.osc_cb.toggled.connect(self._update); g2l.addWidget(self.osc_cb)
        self.ccolor_cb = QCheckBox("Curvature Coloring")
        self.ccolor_cb.setToolTip("Color the curve by curvature magnitude (blue=low, red=high)")
        self.ccolor_cb.toggled.connect(self._update); g2l.addWidget(self.ccolor_cb)

        # Curvature comb scale
        comb_row = QHBoxLayout()
        comb_row.addWidget(QLabel("Comb Scale:"))
        self.comb_scale_slider = QSlider(Qt.Horizontal)
        self.comb_scale_slider.setRange(100, 20000)
        self.comb_scale_slider.setValue(3000)
        self.comb_scale_slider.setToolTip("Scale factor for the curvature comb visualization")
        self.comb_scale_slider.valueChanged.connect(self._on_comb_scale)
        comb_row.addWidget(self.comb_scale_slider)
        g2l.addLayout(comb_row)

        rl.addWidget(g2)

        self._setup_common_controls(rl, show_tangent_options=True)

        # Subdivide
        g_sub = QGroupBox("Bezier Tools"); g_sub_l = QVBoxLayout(g_sub)
        b_sub = QPushButton("Subdivide at t")
        b_sub.setToolTip("Split the Bezier curve at the current t value into two sub-curves")
        b_sub.clicked.connect(self._subdivide_at_t)
        g_sub_l.addWidget(b_sub)
        rl.addWidget(g_sub)

        # Transform Tools
        g_xform = QGroupBox("Transform"); g_xform_l = QVBoxLayout(g_xform)
        xf_row1 = QHBoxLayout()
        b_rev = QPushButton("Reverse Pts"); b_rev.setToolTip("Reverse the order of control points")
        b_rev.clicked.connect(self._reverse_points); xf_row1.addWidget(b_rev)
        b_mx = QPushButton("Mirror H"); b_mx.setToolTip("Mirror points horizontally around centroid")
        b_mx.clicked.connect(self._mirror_horizontal); xf_row1.addWidget(b_mx)
        b_my = QPushButton("Mirror V"); b_my.setToolTip("Mirror points vertically around centroid")
        b_my.clicked.connect(self._mirror_vertical); xf_row1.addWidget(b_my)
        g_xform_l.addLayout(xf_row1)
        xf_row2 = QHBoxLayout()
        b_su = QPushButton("Scale Up"); b_su.setToolTip("Scale points 1.2x away from centroid")
        b_su.clicked.connect(lambda: self._scale_points(1.2)); xf_row2.addWidget(b_su)
        b_sd = QPushButton("Scale Down"); b_sd.setToolTip("Scale points 0.8x toward centroid")
        b_sd.clicked.connect(lambda: self._scale_points(0.8)); xf_row2.addWidget(b_sd)
        g_xform_l.addLayout(xf_row2)
        rl.addWidget(g_xform)

        g3 = QGroupBox("Control Points"); g3l = QVBoxLayout(g3)
        self.pt_list = QListWidget(); self.pt_list.setMaximumHeight(130); self.pt_list.currentRowChanged.connect(self._on_pt_list_clicked); g3l.addWidget(self.pt_list)
        bl = QHBoxLayout()
        ba = QPushButton("+ Add"); ba.clicked.connect(lambda: self._on_2d_added(300, 300)); bl.addWidget(ba)
        br = QPushButton("- Remove"); br.clicked.connect(lambda: self._on_2d_removed(len(self._pts_internal)-1)); bl.addWidget(br)
        bres = QPushButton("Reset"); bres.clicked.connect(self._reset); bl.addWidget(bres)
        g3l.addLayout(bl); rl.addWidget(g3)
        rl.addStretch(); lay.addWidget(rp, 1)

    def _on_comb_scale(self, val):
        self.canvas_2d.curvature_comb_scale = float(val)
        self.canvas_2d.update()

    def _subdivide_at_t(self):
        """Perform de Casteljau subdivision at current t, replace with left sub-curve."""
        if len(self._pts_internal) < 2:
            return
        t = self.t_slider.value() / 1000.0
        self._push_history()
        left, right = CurveMath.bezier_subdivide(self._pts_internal, t)
        # Replace with left half
        dim = self.dim_spin.value()
        self._pts_internal = [list(p) + [0.0] * (dim - len(p)) for p in left]
        self._apply_pts_to_2d()
        self._update()
        self.status_message(f"Subdivided at t={t:.3f}: {len(self._pts_internal)} points (left half)")

    def _reset(self):
        dim = self.dim_spin.value()
        self._pts_internal = [[120, 350] + [0]*(dim-2), [200, 100] + [150]*(dim-2),
                              [450, 100] + [150]*(dim-2), [550, 350] + [0]*(dim-2)]
        self._apply_pts_to_2d()
        self._push_history()
        self._update()

    def _update(self):
        pts = self._pts_internal
        dim = self.dim_spin.value()
        t = self.t_slider.value() / 1000.0
        self.t_lbl.setText(f"t = {t:.3f}")

        self.pt_list.clear()
        for i, p in enumerate(pts):
            self.pt_list.addItem(f"P{i}  ({', '.join(f'{v:.0f}' for v in p)})")

        pts_np = np.array(pts, dtype=np.float64)
        if len(pts) < 2:
            self.canvas_2d.set_curve_points([]); self.canvas_2d.set_de_casteljau_levels([])
            self.canvas_2d.set_highlight_point(None); self.canvas_2d.set_osculating_circle(None, 0)
            self._update_info_panel(None, "Degree: N/A (need 2+ pts)")
            return

        curve_np = CurveMath.evaluate_bezier(pts_np, num_samples=self._num_samples)

        if dim == 2:
            self.canvas_2d.show_control_polygon = self.poly_cb.isChecked()
            self.canvas_2d.show_convex_hull = self.hull_cb.isChecked()
            self.canvas_2d.show_curvature_comb = self.curv_cb.isChecked()
            self.canvas_2d.show_derivatives = self.deriv_cb.isChecked()
            self.canvas_2d.show_tangent = self.tangent_cb.isChecked()
            self.canvas_2d.show_normal = self.normal_cb.isChecked()
            self.canvas_2d.show_osculating_circle = self.osc_cb.isChecked()
            self.canvas_2d.curvature_coloring = self.ccolor_cb.isChecked()
            self.canvas_2d.set_curve_points(curve_np.tolist())

            curve_pt = None
            tangent = None

            if self.dc_cb.isChecked():
                _, levels = CurveMath.de_casteljau(pts_np, t)
                self.canvas_2d.set_de_casteljau_levels(levels)
                hp, _ = CurveMath.de_casteljau(pts_np, t)
                self.canvas_2d.set_highlight_point(hp, t)
                curve_pt = hp
                tangent = CurveMath.bezier_tangent(pts_np, t)
                self.canvas_2d.set_highlight_point(hp, t, tangent=tangent)
            else:
                self.canvas_2d.set_de_casteljau_levels([])

                # Still show highlight point and tangent even without De Casteljau
                hp, _ = CurveMath.de_casteljau(pts_np, t)
                tangent = CurveMath.bezier_tangent(pts_np, t)
                self.canvas_2d.set_highlight_point(hp, t, tangent=tangent)
                curve_pt = hp

            # Osculating circle
            if self.osc_cb.isChecked() and len(curve_np) > 2:
                t_idx = min(int(t * (len(curve_np)-1)), len(curve_np)-2)
                center, radius = CurveMath.osculating_circle_2d(curve_np, t_idx)
                self.canvas_2d.set_osculating_circle(center, radius)
            else:
                self.canvas_2d.set_osculating_circle(None, 0)

            self._update_info_panel(curve_np, f"Degree: {len(pts)-1}", curve_pt_at_t=curve_pt)
        else:
            self.canvas_3d.plot_nd_curve(pts_np, curve_np, dim)
            self._update_info_panel(curve_np, f"Degree: {len(pts)-1}")

    def _get_save_data(self):
        data = super()._get_save_data()
        data["curve_type"] = "bezier"
        return data


# ════════════════════════════════════════════════════════════════
# B-Spline Tab
# ════════════════════════════════════════════════════════════════

class BSplineTab(BaseCurveTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui(); self._reset()

    def _build_ui(self):
        lay = QHBoxLayout(self)
        left = QWidget(); ll = QVBoxLayout(left)
        self._setup_canvases(ll)
        self.basis_canvas = BasisCanvas(); ll.addWidget(self.basis_canvas, 1)
        lay.addWidget(left, 3)

        rp = QWidget(); rp.setMaximumWidth(370); rp.setMinimumWidth(310); rl = QVBoxLayout(rp)
        title = QLabel("B-Splines"); title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setStyleSheet("color:#a6e3a1; padding:4px 0;"); rl.addWidget(title)

        g0 = QGroupBox("Dimensions"); g0l = QHBoxLayout(g0)
        self.dim_spin = QSpinBox(); self.dim_spin.setRange(2, 6); self.dim_spin.setValue(2)
        self.dim_spin.setToolTip("Number of dimensions")
        self.dim_spin.valueChanged.connect(self._sync_dim)
        g0l.addWidget(QLabel("D =")); g0l.addWidget(self.dim_spin); rl.addWidget(g0)

        g1 = QGroupBox("Degree"); g1l = QHBoxLayout(g1)
        self.deg_spin = QSpinBox(); self.deg_spin.setRange(1, 10); self.deg_spin.setValue(3)
        self.deg_spin.setToolTip("Polynomial degree of the B-Spline basis functions")
        self.deg_spin.valueChanged.connect(self._on_degree); g1l.addWidget(QLabel("p =")); g1l.addWidget(self.deg_spin); rl.addWidget(g1)

        g2 = QGroupBox("Knot Vector"); g2l = QVBoxLayout(g2)
        self.knot_combo = QComboBox(); self.knot_combo.addItems(["Clamped", "Uniform", "Custom"])
        self.knot_combo.setToolTip("Type of knot vector generation")
        self.knot_combo.currentIndexChanged.connect(self._on_knot_type); g2l.addWidget(self.knot_combo)
        self.knot_edit = QLineEdit(); self.knot_edit.setPlaceholderText("0 0 0 0.33 0.67 1 1 1")
        self.knot_edit.setToolTip("Space-separated knot values (only for Custom type)")
        self.knot_edit.editingFinished.connect(self._update); self.knot_edit.setEnabled(False); g2l.addWidget(self.knot_edit)
        self.knot_lbl = QLabel("Knots: -"); self.knot_lbl.setWordWrap(True); self.knot_lbl.setFont(QFont("Consolas", 8)); g2l.addWidget(self.knot_lbl); rl.addWidget(g2)

        g3 = QGroupBox("Parameter t"); g3l = QVBoxLayout(g3)
        self.t_slider = QSlider(Qt.Horizontal); self.t_slider.setRange(0, 1000); self.t_slider.setValue(500)
        self.t_slider.setToolTip("Current parameter value for evaluation")
        self.t_slider.valueChanged.connect(self._update); g3l.addWidget(self.t_slider)
        self.t_lbl = QLabel("t = 0.500"); self.t_lbl.setAlignment(Qt.AlignCenter); g3l.addWidget(self.t_lbl)
        self.anim_btn = QPushButton("Animate")
        self.anim_btn.setToolTip("Animate t back and forth")
        self.anim_btn.clicked.connect(self._toggle_anim); g3l.addWidget(self.anim_btn); rl.addWidget(g3)

        g4 = QGroupBox("Display"); g4l = QVBoxLayout(g4)
        self.poly_cb = QCheckBox("Control polygon"); self.poly_cb.setChecked(True)
        self.poly_cb.toggled.connect(self._update); g4l.addWidget(self.poly_cb)
        self.curv_cb = QCheckBox("Curvature Comb")
        self.curv_cb.toggled.connect(self._update); g4l.addWidget(self.curv_cb)
        self.deriv_cb = QCheckBox("Derivative Vectors")
        self.deriv_cb.toggled.connect(self._update); g4l.addWidget(self.deriv_cb)
        self.tangent_cb = QCheckBox("Tangent Line")
        self.tangent_cb.toggled.connect(self._update); g4l.addWidget(self.tangent_cb)
        self.normal_cb = QCheckBox("Normal Line")
        self.normal_cb.toggled.connect(self._update); g4l.addWidget(self.normal_cb)
        self.osc_cb = QCheckBox("Osculating Circle")
        self.osc_cb.toggled.connect(self._update); g4l.addWidget(self.osc_cb)
        self.ccolor_cb = QCheckBox("Curvature Coloring")
        self.ccolor_cb.toggled.connect(self._update); g4l.addWidget(self.ccolor_cb)
        self.basis_cb = QCheckBox("Basis functions"); self.basis_cb.setChecked(True)
        self.basis_cb.setToolTip("Show the B-Spline basis functions N_i,p(t)")
        self.basis_cb.toggled.connect(self._update); g4l.addWidget(self.basis_cb); rl.addWidget(g4)

        comb_row = QHBoxLayout()
        comb_row.addWidget(QLabel("Comb Scale:"))
        self.comb_scale_slider = QSlider(Qt.Horizontal)
        self.comb_scale_slider.setRange(100, 20000)
        self.comb_scale_slider.setValue(3000)
        self.comb_scale_slider.valueChanged.connect(self._on_comb_scale)
        comb_row.addWidget(self.comb_scale_slider)
        g4l.addLayout(comb_row)
        rl.addWidget(g4)

        self._setup_common_controls(rl)

        # Transform Tools
        g_xform = QGroupBox("Transform"); g_xform_l = QVBoxLayout(g_xform)
        xf_row1 = QHBoxLayout()
        b_rev = QPushButton("Reverse Pts"); b_rev.setToolTip("Reverse the order of control points")
        b_rev.clicked.connect(self._reverse_points); xf_row1.addWidget(b_rev)
        b_mx = QPushButton("Mirror H"); b_mx.setToolTip("Mirror points horizontally around centroid")
        b_mx.clicked.connect(self._mirror_horizontal); xf_row1.addWidget(b_mx)
        b_my = QPushButton("Mirror V"); b_my.setToolTip("Mirror points vertically around centroid")
        b_my.clicked.connect(self._mirror_vertical); xf_row1.addWidget(b_my)
        g_xform_l.addLayout(xf_row1)
        xf_row2 = QHBoxLayout()
        b_su = QPushButton("Scale Up"); b_su.setToolTip("Scale points 1.2x away from centroid")
        b_su.clicked.connect(lambda: self._scale_points(1.2)); xf_row2.addWidget(b_su)
        b_sd = QPushButton("Scale Down"); b_sd.setToolTip("Scale points 0.8x toward centroid")
        b_sd.clicked.connect(lambda: self._scale_points(0.8)); xf_row2.addWidget(b_sd)
        g_xform_l.addLayout(xf_row2)
        rl.addWidget(g_xform)

        g5 = QGroupBox("Control Points"); g5l = QVBoxLayout(g5)
        self.pt_list = QListWidget(); self.pt_list.setMaximumHeight(100); self.pt_list.currentRowChanged.connect(self._on_pt_list_clicked); g5l.addWidget(self.pt_list)
        bl = QHBoxLayout()
        ba = QPushButton("+ Add"); ba.clicked.connect(lambda: self._on_2d_added(350, 300)); bl.addWidget(ba)
        br = QPushButton("- Remove"); br.clicked.connect(lambda: self._on_2d_removed(len(self._pts_internal)-1)); bl.addWidget(br)
        bres = QPushButton("Reset"); bres.clicked.connect(self._reset); bl.addWidget(bres)
        g5l.addLayout(bl); rl.addWidget(g5); rl.addStretch(); lay.addWidget(rp, 1)

    def _on_comb_scale(self, val):
        self.canvas_2d.curvature_comb_scale = float(val)
        self.canvas_2d.update()

    def _reset(self):
        dim = self.dim_spin.value()
        self._pts_internal = [[80, 350] + [0]*(dim-2), [200, 120] + [150]*(dim-2),
                              [370, 120] + [150]*(dim-2), [450, 350] + [0]*(dim-2),
                              [530, 180] + [250]*(dim-2), [650, 280] + [0]*(dim-2)]
        self._apply_pts_to_2d()
        self._push_history()
        self._update()

    def _on_degree(self):
        if len(self._pts_internal) < self.deg_spin.value() + 1:
            self._push_history()
            self._pts_internal = self._pts_internal[:self.deg_spin.value() + 1]
            self._apply_pts_to_2d()
        self._update()

    def _on_knot_type(self):
        self.knot_edit.setEnabled(self.knot_combo.currentIndex() == 2); self._update()

    def _get_knots(self):
        n = len(self._pts_internal) - 1; deg = self.deg_spin.value()
        if self.knot_combo.currentIndex() == 0: return CurveMath.generate_clamped_knots(n, deg)
        elif self.knot_combo.currentIndex() == 1: return CurveMath.generate_uniform_knots(n, deg)
        else:
            try: return np.array([float(x) for x in self.knot_edit.text().split()], dtype=np.float64)
            except: return None

    def _update(self):
        pts = self._pts_internal; dim = self.dim_spin.value()
        n = len(pts) - 1; deg = self.deg_spin.value()
        t_norm = self.t_slider.value() / 1000.0; self.t_lbl.setText(f"t = {t_norm:.3f}")

        self.pt_list.clear()
        for i, p in enumerate(pts): self.pt_list.addItem(f"P{i}  ({', '.join(f'{v:.0f}' for v in p)})")

        if n < deg:
            self.canvas_2d.set_curve_points([]); self.knot_lbl.setText("Need >= p+1 pts")
            self.basis_canvas.set_basis_data(np.array([]), np.array([]), [], deg)
            self.canvas_2d.set_osculating_circle(None, 0)
            self._update_info_panel(None, f"Need >= {deg+1} control points")
            return

        knots = self._get_knots()
        if knots is None or len(knots) != n + deg + 2:
            self.canvas_2d.set_curve_points([]); self.knot_lbl.setText(f"Need {n+deg+2} knots")
            self.basis_canvas.set_basis_data(np.array([]), np.array([]), [], deg)
            self.canvas_2d.set_osculating_circle(None, 0)
            self._update_info_panel(None, f"Knot vector error: need {n+deg+2} knots")
            return

        ks = "  ".join(f"{k:.3f}" for k in knots); self.knot_lbl.setText(f"[{ks}]")
        pts_np = np.array(pts, dtype=np.float64)
        curve_np = CurveMath.evaluate_bspline(pts_np, deg, knots, num_samples=self._num_samples)

        t_min = knots[deg]; t_max = knots[n + 1]

        if dim == 2:
            self.canvas_2d.show_control_polygon = self.poly_cb.isChecked()
            self.canvas_2d.show_curvature_comb = self.curv_cb.isChecked()
            self.canvas_2d.show_derivatives = self.deriv_cb.isChecked()
            self.canvas_2d.show_tangent = self.tangent_cb.isChecked()
            self.canvas_2d.show_normal = self.normal_cb.isChecked()
            self.canvas_2d.show_osculating_circle = self.osc_cb.isChecked()
            self.canvas_2d.curvature_coloring = self.ccolor_cb.isChecked()
            self.canvas_2d.set_curve_points(curve_np.tolist())

            curve_pt = None
            tangent = None

            if t_max > t_min:
                t_val = t_min + t_norm * (t_max - t_min)
                hp = np.zeros(2);
                for j in range(n + 1):
                    b = CurveMath.bspline_basis(j, deg, t_val, knots)
                    hp += b * pts_np[j, :2]
                self.canvas_2d.set_highlight_point(hp, t_norm)
                curve_pt = hp

                # Approximate tangent via finite difference
                dt = 1e-5
                t_val2 = min(t_val + dt, t_max - 1e-10)
                hp2 = np.zeros(2)
                for j in range(n + 1):
                    b2 = CurveMath.bspline_basis(j, deg, t_val2, knots)
                    hp2 += b2 * pts_np[j, :2]
                tangent = (hp2 - hp) / dt if dt > 0 else None
                self.canvas_2d.set_highlight_point(hp, t_norm, tangent=tangent)
            else:
                self.canvas_2d.set_highlight_point(None)

            if self.osc_cb.isChecked() and len(curve_np) > 2 and t_max > t_min:
                t_idx = min(int(t_norm * (len(curve_np)-1)), len(curve_np)-2)
                center, radius = CurveMath.osculating_circle_2d(curve_np, t_idx)
                self.canvas_2d.set_osculating_circle(center, radius)
            else:
                self.canvas_2d.set_osculating_circle(None, 0)

            self._update_info_panel(curve_np, f"Knot span: [{t_min:.3f}, {t_max:.3f}]", curve_pt_at_t=curve_pt)
        else:
            self.canvas_3d.plot_nd_curve(pts_np, curve_np, dim)
            self._update_info_panel(curve_np, f"Knot span: [{t_min:.3f}, {t_max:.3f}]")

        if self.basis_cb.isChecked() and t_max > t_min:
            t_vals, b_data = CurveMath.evaluate_basis_data(n, deg, knots)
            self.basis_canvas.set_basis_data(t_vals, b_data, knots, deg, t_norm)
        else: self.basis_canvas.set_basis_data(np.array([]), np.array([]), [], deg)

    def _get_save_data(self):
        data = super()._get_save_data()
        data["curve_type"] = "bspline"
        data["degree"] = self.deg_spin.value()
        data["knot_type"] = self.knot_combo.currentText()
        data["knot_custom"] = self.knot_edit.text()
        return data

    def _apply_load_data(self, data):
        super()._apply_load_data(data)
        if "degree" in data:
            self.deg_spin.blockSignals(True)
            self.deg_spin.setValue(data["degree"])
            self.deg_spin.blockSignals(False)
        if "knot_type" in data:
            idx = self.knot_combo.findText(data["knot_type"])
            if idx >= 0:
                self.knot_combo.blockSignals(True)
                self.knot_combo.setCurrentIndex(idx)
                self.knot_combo.blockSignals(False)
        if "knot_custom" in data:
            self.knot_edit.setText(data["knot_custom"])


# ════════════════════════════════════════════════════════════════
# NURBS Tab
# ════════════════════════════════════════════════════════════════

class NURBSTab(BaseCurveTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui(); self._reset()

    def _build_ui(self):
        lay = QHBoxLayout(self)
        left = QWidget(); ll = QVBoxLayout(left)
        self._setup_canvases(ll)
        self.basis_canvas = BasisCanvas(); ll.addWidget(self.basis_canvas, 1)
        lay.addWidget(left, 3)

        rp = QWidget(); rp.setMaximumWidth(370); rp.setMinimumWidth(310); rl = QVBoxLayout(rp)
        title = QLabel("NURBS"); title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setStyleSheet("color:#cba6f7; padding:4px 0;"); rl.addWidget(title)

        g0 = QGroupBox("Dimensions"); g0l = QHBoxLayout(g0)
        self.dim_spin = QSpinBox(); self.dim_spin.setRange(2, 6); self.dim_spin.setValue(2)
        self.dim_spin.setToolTip("Number of dimensions")
        self.dim_spin.valueChanged.connect(self._sync_dim)
        g0l.addWidget(QLabel("D =")); g0l.addWidget(self.dim_spin); rl.addWidget(g0)

        g1 = QGroupBox("Degree"); g1l = QHBoxLayout(g1)
        self.deg_spin = QSpinBox(); self.deg_spin.setRange(1, 10); self.deg_spin.setValue(3)
        self.deg_spin.setToolTip("Polynomial degree")
        self.deg_spin.valueChanged.connect(self._on_degree); g1l.addWidget(QLabel("p =")); g1l.addWidget(self.deg_spin); rl.addWidget(g1)

        g2 = QGroupBox("Knot Vector"); g2l = QVBoxLayout(g2)
        self.knot_combo = QComboBox(); self.knot_combo.addItems(["Clamped", "Uniform", "Custom"])
        self.knot_combo.setToolTip("Knot vector type")
        self.knot_combo.currentIndexChanged.connect(self._on_knot_type); g2l.addWidget(self.knot_combo)
        self.knot_edit = QLineEdit(); self.knot_edit.setPlaceholderText("0 0 0 0.33 0.67 1 1 1")
        self.knot_edit.editingFinished.connect(self._update); self.knot_edit.setEnabled(False); g2l.addWidget(self.knot_edit)
        self.knot_lbl = QLabel("Knots: -"); self.knot_lbl.setWordWrap(True); self.knot_lbl.setFont(QFont("Consolas", 8)); g2l.addWidget(self.knot_lbl); rl.addWidget(g2)

        g3 = QGroupBox("Weights"); g3l = QVBoxLayout(g3)
        self.w_equal_btn = QPushButton("All = 1"); self.w_equal_btn.clicked.connect(self._equal_weights); g3l.addWidget(self.w_equal_btn)
        self.w_circle_btn = QPushButton("Preset: Circle (2D)"); self.w_circle_btn.clicked.connect(self._circle_preset); g3l.addWidget(self.w_circle_btn)
        self.w_scroll = QScrollArea(); self.w_scroll.setWidgetResizable(True)
        self.w_container = QWidget(); self.w_container_l = QVBoxLayout(self.w_container); self.w_container_l.setContentsMargins(0,0,0,0)
        self.w_scroll.setWidget(self.w_container); self.w_scroll.setMaximumHeight(150); g3l.addWidget(self.w_scroll); rl.addWidget(g3)

        g4 = QGroupBox("Parameter t"); g4l = QVBoxLayout(g4)
        self.t_slider = QSlider(Qt.Horizontal); self.t_slider.setRange(0, 1000); self.t_slider.setValue(500)
        self.t_slider.valueChanged.connect(self._update); g4l.addWidget(self.t_slider)
        self.t_lbl = QLabel("t = 0.500"); self.t_lbl.setAlignment(Qt.AlignCenter); g4l.addWidget(self.t_lbl)
        self.anim_btn = QPushButton("Animate"); self.anim_btn.clicked.connect(self._toggle_anim); g4l.addWidget(self.anim_btn); rl.addWidget(g4)

        g5 = QGroupBox("Display"); g5l = QVBoxLayout(g5)
        self.poly_cb = QCheckBox("Control polygon"); self.poly_cb.setChecked(True); self.poly_cb.toggled.connect(self._update); g5l.addWidget(self.poly_cb)
        self.ref_cb = QCheckBox("B-Spline overlay (w=1)"); self.ref_cb.setChecked(True)
        self.ref_cb.setToolTip("Show the unweighted B-Spline for comparison")
        self.ref_cb.toggled.connect(self._update); g5l.addWidget(self.ref_cb)
        self.curv_cb = QCheckBox("Curvature Comb"); self.curv_cb.toggled.connect(self._update); g5l.addWidget(self.curv_cb)
        self.deriv_cb = QCheckBox("Derivative Vectors"); self.deriv_cb.toggled.connect(self._update); g5l.addWidget(self.deriv_cb)
        self.tangent_cb = QCheckBox("Tangent Line"); self.tangent_cb.toggled.connect(self._update); g5l.addWidget(self.tangent_cb)
        self.normal_cb = QCheckBox("Normal Line"); self.normal_cb.toggled.connect(self._update); g5l.addWidget(self.normal_cb)
        self.osc_cb = QCheckBox("Osculating Circle"); self.osc_cb.toggled.connect(self._update); g5l.addWidget(self.osc_cb)
        self.ccolor_cb = QCheckBox("Curvature Coloring"); self.ccolor_cb.toggled.connect(self._update); g5l.addWidget(self.ccolor_cb)
        self.basis_cb2 = QCheckBox("Basis functions"); self.basis_cb2.setChecked(True); self.basis_cb2.toggled.connect(self._update); g5l.addWidget(self.basis_cb2); rl.addWidget(g5)

        comb_row = QHBoxLayout()
        comb_row.addWidget(QLabel("Comb Scale:"))
        self.comb_scale_slider = QSlider(Qt.Horizontal)
        self.comb_scale_slider.setRange(100, 20000)
        self.comb_scale_slider.setValue(3000)
        self.comb_scale_slider.valueChanged.connect(self._on_comb_scale)
        comb_row.addWidget(self.comb_scale_slider)
        g5l.addLayout(comb_row)
        rl.addWidget(g5)

        self._setup_common_controls(rl)

        # Transform Tools
        g_xform = QGroupBox("Transform"); g_xform_l = QVBoxLayout(g_xform)
        xf_row1 = QHBoxLayout()
        b_rev = QPushButton("Reverse Pts"); b_rev.setToolTip("Reverse the order of control points (and weights)")
        b_rev.clicked.connect(self._reverse_points); xf_row1.addWidget(b_rev)
        b_mx = QPushButton("Mirror H"); b_mx.setToolTip("Mirror points horizontally around centroid")
        b_mx.clicked.connect(self._mirror_horizontal); xf_row1.addWidget(b_mx)
        b_my = QPushButton("Mirror V"); b_my.setToolTip("Mirror points vertically around centroid")
        b_my.clicked.connect(self._mirror_vertical); xf_row1.addWidget(b_my)
        g_xform_l.addLayout(xf_row1)
        xf_row2 = QHBoxLayout()
        b_su = QPushButton("Scale Up"); b_su.setToolTip("Scale points 1.2x away from centroid")
        b_su.clicked.connect(lambda: self._scale_points(1.2)); xf_row2.addWidget(b_su)
        b_sd = QPushButton("Scale Down"); b_sd.setToolTip("Scale points 0.8x toward centroid")
        b_sd.clicked.connect(lambda: self._scale_points(0.8)); xf_row2.addWidget(b_sd)
        g_xform_l.addLayout(xf_row2)
        rl.addWidget(g_xform)

        g6 = QGroupBox("Control Points"); g6l = QVBoxLayout(g6)
        self.pt_list = QListWidget(); self.pt_list.setMaximumHeight(90); self.pt_list.currentRowChanged.connect(self._on_pt_list_clicked); g6l.addWidget(self.pt_list)
        bl = QHBoxLayout()
        ba = QPushButton("+ Add"); ba.clicked.connect(lambda: self._on_2d_added(350, 300)); bl.addWidget(ba)
        br = QPushButton("- Remove"); br.clicked.connect(lambda: self._on_2d_removed(len(self._pts_internal)-1)); bl.addWidget(br)
        bres = QPushButton("Reset"); bres.clicked.connect(self._reset); bl.addWidget(bres)
        g6l.addLayout(bl); rl.addWidget(g6); rl.addStretch(); lay.addWidget(rp, 1)

        self.w_spins = []

    def _on_comb_scale(self, val):
        self.canvas_2d.curvature_comb_scale = float(val)
        self.canvas_2d.update()

    def _rebuild_weight_spins(self):
        while self.w_container_l.count():
            item = self.w_container_l.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.w_spins = []
        for i in range(len(self._pts_internal)):
            row = QHBoxLayout(); lbl = QLabel(f"w{i}:"); lbl.setFixedWidth(26); row.addWidget(lbl)
            spin = QDoubleSpinBox(); spin.setRange(0.01, 50.0); spin.setSingleStep(0.1); spin.setDecimals(2); spin.setValue(1.0)
            spin.setToolTip(f"Weight for control point P{i}")
            spin.valueChanged.connect(self._on_weight_changed); row.addWidget(spin); self.w_spins.append(spin)
            self.w_container_l.addLayout(row)

    def _on_weight_changed(self):
        self._push_history()
        self._update()

    def _reset(self):
        dim = self.dim_spin.value()
        self._pts_internal = [[80, 350] + [0]*(dim-2), [200, 120] + [150]*(dim-2),
                              [370, 120] + [150]*(dim-2), [450, 350] + [0]*(dim-2),
                              [530, 180] + [250]*(dim-2), [650, 280] + [0]*(dim-2)]
        self._apply_pts_to_2d(); self._rebuild_weight_spins()
        self._push_history()
        self._update()

    def _on_degree(self):
        if len(self._pts_internal) < self.deg_spin.value() + 1:
            self._push_history()
            self._pts_internal = self._pts_internal[:self.deg_spin.value() + 1]
            self._apply_pts_to_2d(); self._rebuild_weight_spins()
        self._update()

    def _on_knot_type(self):
        self.knot_edit.setEnabled(self.knot_combo.currentIndex() == 2); self._update()

    def _equal_weights(self):
        self._push_history()
        for s in self.w_spins: s.blockSignals(True); s.setValue(1.0); s.blockSignals(False)
        self._update()

    def _circle_preset(self):
        self._push_history()
        self.dim_spin.setValue(2)
        cx, cy, r = 350, 250, 180; w = math.cos(math.pi / 8); pts = []
        for i in range(9):
            angle = math.pi / 2 - i * math.pi / 4
            pts.append([cx + r * math.cos(angle), cy + r * math.sin(angle)])
        self._pts_internal = pts; self._apply_pts_to_2d()
        self.deg_spin.blockSignals(True); self.deg_spin.setValue(2); self.deg_spin.blockSignals(False)
        self.knot_combo.blockSignals(True); self.knot_combo.setCurrentIndex(2); self.knot_combo.blockSignals(False)
        self.knot_edit.setEnabled(True); self.knot_edit.setText("0 0 0 0.25 0.25 0.5 0.5 0.75 0.75 1 1 1")
        self._rebuild_weight_spins()
        weights = [1, w, 1, w, 1, w, 1, w, 1]
        for i, s in enumerate(self.w_spins): s.blockSignals(True); s.setValue(weights[i]); s.blockSignals(False)
        self._update()

    def _get_knots(self):
        n = len(self._pts_internal) - 1; deg = self.deg_spin.value()
        if self.knot_combo.currentIndex() == 0: return CurveMath.generate_clamped_knots(n, deg)
        elif self.knot_combo.currentIndex() == 1: return CurveMath.generate_uniform_knots(n, deg)
        else:
            try: return np.array([float(x) for x in self.knot_edit.text().split()], dtype=np.float64)
            except: return None

    def _get_weights(self):
        return np.array([s.value() for s in self.w_spins], dtype=np.float64)

    def _update(self):
        pts = self._pts_internal; dim = self.dim_spin.value()
        n = len(pts) - 1; deg = self.deg_spin.value()
        t_norm = self.t_slider.value() / 1000.0; self.t_lbl.setText(f"t = {t_norm:.3f}")

        self.pt_list.clear(); ws = self._get_weights()
        for i, p in enumerate(pts):
            wi = ws[i] if i < len(ws) else 1.0
            self.pt_list.addItem(f"P{i} ({', '.join(f'{v:.0f}' for v in p)}) w={wi:.2f}")

        if n < deg:
            self.canvas_2d.set_curve_points([]); self.canvas_2d.set_reference_curve([])
            self.knot_lbl.setText("Need >= p+1 pts"); self.basis_canvas.set_basis_data(np.array([]), np.array([]), [], deg)
            self.canvas_2d.set_osculating_circle(None, 0)
            self._update_info_panel(None, f"Need >= {deg+1} control points")
            return

        knots = self._get_knots()
        if knots is None or len(knots) != n + deg + 2:
            self.canvas_2d.set_curve_points([]); self.canvas_2d.set_reference_curve([])
            self.knot_lbl.setText(f"Need {n+deg+2} knots"); self.basis_canvas.set_basis_data(np.array([]), np.array([]), [], deg)
            self.canvas_2d.set_osculating_circle(None, 0)
            self._update_info_panel(None, f"Knot vector error: need {n+deg+2} knots")
            return

        ks = "  ".join(f"{k:.3f}" for k in knots); self.knot_lbl.setText(f"[{ks}]")
        if len(ws) != n + 1: self.canvas_2d.set_curve_points([]); return

        pts_np = np.array(pts, dtype=np.float64)
        curve_np = CurveMath.evaluate_nurbs(pts_np, ws, deg, knots, num_samples=self._num_samples)

        t_min = knots[deg]; t_max = knots[n + 1]

        if dim == 2:
            self.canvas_2d.show_control_polygon = self.poly_cb.isChecked()
            self.canvas_2d.show_curvature_comb = self.curv_cb.isChecked()
            self.canvas_2d.show_derivatives = self.deriv_cb.isChecked()
            self.canvas_2d.show_tangent = self.tangent_cb.isChecked()
            self.canvas_2d.show_normal = self.normal_cb.isChecked()
            self.canvas_2d.show_osculating_circle = self.osc_cb.isChecked()
            self.canvas_2d.curvature_coloring = self.ccolor_cb.isChecked()
            self.canvas_2d.set_curve_points(curve_np.tolist())

            if self.ref_cb.isChecked():
                ref_np = CurveMath.evaluate_bspline(pts_np, deg, knots, num_samples=self._num_samples)
                self.canvas_2d.set_reference_curve(ref_np.tolist())
            else: self.canvas_2d.set_reference_curve([])

            curve_pt = None
            if t_max > t_min:
                t_val = t_min + t_norm * (t_max - t_min)
                hp = np.zeros(2); dw = 0.0
                for j in range(n + 1):
                    b = CurveMath.bspline_basis(j, deg, t_val, knots); w = ws[j]
                    hp += w * b * pts_np[j, :2]; dw += w * b
                if dw > 1e-12:
                    hp = hp / dw
                    self.canvas_2d.set_highlight_point(hp, t_norm)
                    curve_pt = hp

                    # Approximate tangent
                    dt = 1e-5
                    t_val2 = min(t_val + dt, t_max - 1e-10)
                    hp2 = np.zeros(2); dw2 = 0.0
                    for j in range(n + 1):
                        b2 = CurveMath.bspline_basis(j, deg, t_val2, knots); w2 = ws[j]
                        hp2 += w2 * b2 * pts_np[j, :2]; dw2 += w2 * b2
                    if dw2 > 1e-12:
                        hp2 = hp2 / dw2
                        tangent = (hp2 - hp) / dt
                        self.canvas_2d.set_highlight_point(hp, t_norm, tangent=tangent)
                else:
                    self.canvas_2d.set_highlight_point(None)
            else: self.canvas_2d.set_highlight_point(None)

            if self.osc_cb.isChecked() and len(curve_np) > 2 and t_max > t_min:
                t_idx = min(int(t_norm * (len(curve_np)-1)), len(curve_np)-2)
                center, radius = CurveMath.osculating_circle_2d(curve_np, t_idx)
                self.canvas_2d.set_osculating_circle(center, radius)
            else:
                self.canvas_2d.set_osculating_circle(None, 0)

            self._update_info_panel(curve_np, f"Knot span: [{t_min:.3f}, {t_max:.3f}]", curve_pt_at_t=curve_pt)
        else:
            self.canvas_3d.plot_nd_curve(pts_np, curve_np, dim)
            self.canvas_2d.set_reference_curve([])
            self._update_info_panel(curve_np, f"Knot span: [{t_min:.3f}, {t_max:.3f}]")

        if self.basis_cb2.isChecked() and t_max > t_min:
            t_vals, b_data = CurveMath.evaluate_basis_data(n, deg, knots)
            self.basis_canvas.set_basis_data(t_vals, b_data, knots, deg, t_norm)
        else: self.basis_canvas.set_basis_data(np.array([]), np.array([]), [], deg)

    def _get_save_data(self):
        data = super()._get_save_data()
        data["curve_type"] = "nurbs"
        data["degree"] = self.deg_spin.value()
        data["knot_type"] = self.knot_combo.currentText()
        data["knot_custom"] = self.knot_edit.text()
        data["weights"] = [s.value() for s in self.w_spins]
        return data

    def _apply_load_data(self, data):
        super()._apply_load_data(data)
        if "degree" in data:
            self.deg_spin.blockSignals(True)
            self.deg_spin.setValue(data["degree"])
            self.deg_spin.blockSignals(False)
        if "knot_type" in data:
            idx = self.knot_combo.findText(data["knot_type"])
            if idx >= 0:
                self.knot_combo.blockSignals(True)
                self.knot_combo.setCurrentIndex(idx)
                self.knot_combo.blockSignals(False)
        if "knot_custom" in data:
            self.knot_edit.setText(data["knot_custom"])
        if "weights" in data:
            self._rebuild_weight_spins()
            for i, w in enumerate(data["weights"]):
                if i < len(self.w_spins):
                    self.w_spins[i].blockSignals(True)
                    self.w_spins[i].setValue(w)
                    self.w_spins[i].blockSignals(False)


# ════════════════════════════════════════════════════════════════
# Main Window
# ════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CurveLab - Bezier / B-Spline / NURBS  (N-Dimensional)")
        self.resize(1350, 900)

        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI', Arial; font-size: 11px; }
            QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 10px; padding: 10px 6px 6px 6px; font-weight: bold; color: #89b4fa; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { background-color: #313244; border: 1px solid #45475a; border-radius: 4px; padding: 5px 10px; color: #cdd6f4; }
            QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
            QPushButton:pressed { background-color: #585b70; }
            QSlider::groove:horizontal { border: 1px solid #45475a; height: 6px; background: #313244; border-radius: 3px; }
            QSlider::handle:horizontal { background: #89b4fa; border: 1px solid #74c7ec; width: 14px; margin: -5px 0; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #89b4fa; border-radius: 3px; }
            QSpinBox, QDoubleSpinBox, QLineEdit { background-color: #313244; border: 1px solid #45475a; border-radius: 4px; padding: 3px; color: #cdd6f4; }
            QComboBox { background-color: #313244; border: 1px solid #45475a; border-radius: 4px; padding: 3px 6px; color: #cdd6f4; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView { background-color: #313244; color: #cdd6f4; selection-background-color: #45475a; }
            QListWidget { background-color: #313244; border: 1px solid #45475a; border-radius: 4px; color: #cdd6f4; }
            QListWidget::item:selected { background-color: #45475a; }
            QCheckBox { color: #cdd6f4; spacing: 6px; }
            QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #45475a; background-color: #313244; }
            QCheckBox::indicator:checked { background-color: #89b4fa; border-color: #74c7ec; }
            QTabWidget::pane { border: 1px solid #45475a; border-radius: 4px; }
            QTabBar::tab { background-color: #313244; border: 1px solid #45475a; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; padding: 8px 20px; color: #a6adc8; font-weight: bold; font-size: 12px; }
            QTabBar::tab:selected { background-color: #1e1e2e; color: #cdd6f4; }
            QTabBar::tab:hover { background-color: #45475a; }
            QScrollArea { border: none; }
            QLabel { color: #cdd6f4; }
            QStatusBar { color: #a6adc8; font-size: 10px; }
            QMenuBar { background-color: #181825; color: #cdd6f4; border-bottom: 1px solid #313244; }
            QMenuBar::item:selected { background-color: #313244; }
            QMenu { background-color: #1e1e2e; color: #cdd6f4; border: 1px solid #45475a; }
            QMenu::item:selected { background-color: #45475a; }
            QDialog { background-color: #1e1e2e; color: #cdd6f4; }
            QToolTip { background-color: #313244; color: #cdd6f4; border: 1px solid #585b70; padding: 4px; border-radius: 3px; }
        """)

        self._build_menu_bar()
        self.statusBar().showMessage("Ready  |  H: Help  |  R: Reset View  |  G: Grid Snap  |  Scroll: Zoom  |  Mid-drag: Pan")

        self.tabs = QTabWidget()
        self.bezier_tab = BezierTab()
        self.bspline_tab = BSplineTab()
        self.nurbs_tab = NURBSTab()
        self.tabs.addTab(self.bezier_tab, "Bezier Curves")
        self.tabs.addTab(self.bspline_tab, "B-Splines")
        self.tabs.addTab(self.nurbs_tab, "NURBS")
        self.setCentralWidget(self.tabs)

    def _build_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        self._add_action(file_menu, "Save Configuration...", "Ctrl+S", self._save_current_tab)
        self._add_action(file_menu, "Load Configuration...", "Ctrl+O", self._load_current_tab)
        file_menu.addSeparator()
        self._add_action(file_menu, "Export as PNG...", "", self._export_current_png)
        self._add_action(file_menu, "Export as SVG...", "", self._export_current_svg)
        file_menu.addSeparator()
        self._add_action(file_menu, "Quit", "Ctrl+Q", self.close)

        edit_menu = menubar.addMenu("&Edit")
        self._add_action(edit_menu, "Undo", "Ctrl+Z", self._undo_current_tab)
        self._add_action(edit_menu, "Redo", "Ctrl+Shift+Z", self._redo_current_tab)
        edit_menu.addSeparator()
        self._add_action(edit_menu, "Copy Control Points", "Ctrl+C", self._copy_current)
        self._add_action(edit_menu, "Paste Control Points", "Ctrl+V", self._paste_current)
        edit_menu.addSeparator()
        self._add_action(edit_menu, "Reset View", "R", self._reset_current_view)
        self._add_action(edit_menu, "Fit to View", "F", self._fit_current_view)
        self._add_action(edit_menu, "Toggle Grid Snap", "G", self._toggle_current_snap)

        presets_menu = menubar.addMenu("&Presets")
        for name in ["line", "quadratic", "cubic", "heart", "star", "wave"]:
            label = name.replace("_", " ").title()
            self._add_action(presets_menu, label, "", lambda n=name: self._apply_preset(n))

        help_menu = menubar.addMenu("&Help")
        self._add_action(help_menu, "Keyboard Shortcuts && Help", "H", self._show_help)
        help_menu.addSeparator()
        self._add_action(help_menu, "About CurveLab", "", self._show_about)

    def _add_action(self, menu, text, shortcut, callback):
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(callback)
        menu.addAction(action)

    def _current_tab_widget(self):
        idx = self.tabs.currentIndex()
        if idx == 0: return self.bezier_tab
        elif idx == 1: return self.bspline_tab
        elif idx == 2: return self.nurbs_tab
        return None

    def _save_current_tab(self):
        tab = self._current_tab_widget()
        if tab: tab._save_json()

    def _load_current_tab(self):
        tab = self._current_tab_widget()
        if tab: tab._load_json()

    def _export_current_png(self):
        tab = self._current_tab_widget()
        if tab: tab._export_png()

    def _export_current_svg(self):
        tab = self._current_tab_widget()
        if tab: tab._export_svg()

    def _undo_current_tab(self):
        tab = self._current_tab_widget()
        if tab: tab._undo()

    def _redo_current_tab(self):
        tab = self._current_tab_widget()
        if tab: tab._redo()

    def _reset_current_view(self):
        tab = self._current_tab_widget()
        if tab: tab.canvas_2d.reset_view()

    def _fit_current_view(self):
        tab = self._current_tab_widget()
        if tab: tab._fit_view()

    def _toggle_current_snap(self):
        tab = self._current_tab_widget()
        if tab:
            tab.canvas_2d.grid_snap = not tab.canvas_2d.grid_snap
            tab.snap_cb.setChecked(tab.canvas_2d.grid_snap)
            tab.canvas_2d.update()

    def _copy_current(self):
        tab = self._current_tab_widget()
        if tab: tab._copy_points()

    def _paste_current(self):
        tab = self._current_tab_widget()
        if tab: tab._paste_points()

    def _show_help(self):
        HelpDialog(self).exec()

    def _show_about(self):
        QMessageBox.about(self, "About CurveLab",
            "<h3>CurveLab</h3>"
            "<p>Interactive Bezier, B-Spline & NURBS Learning Application</p>"
            "<p>Supports 2D, 3D, and N-Dimensional visualization with "
            "Numba-accelerated curve evaluation.</p>"
            "<p>Press <b>H</b> for the full keyboard shortcut reference.</p>")

    def _apply_preset(self, name):
        tab = self._current_tab_widget()
        if tab is None:
            return
        tab._push_history()
        dim = tab.dim_spin.value()
        cx, cy = 350, 250

        presets = {
            "line": [[100, 400] + [0]*(dim-2), [600, 100] + [0]*(dim-2)],
            "quadratic": [[100, 400] + [0]*(dim-2), [350, 80] + [0]*(dim-2), [600, 400] + [0]*(dim-2)],
            "cubic": [[100, 400] + [0]*(dim-2), [200, 100] + [0]*(dim-2), [500, 100] + [0]*(dim-2), [600, 400] + [0]*(dim-2)],
            "heart": [
                [cx, cy + 60] + [0]*(dim-2),
                [cx - 40, cy + 10] + [0]*(dim-2),
                [cx - 140, cy - 30] + [0]*(dim-2),
                [cx - 100, cy - 130] + [0]*(dim-2),
                [cx, cy - 60] + [0]*(dim-2),
                [cx + 100, cy - 130] + [0]*(dim-2),
                [cx + 140, cy - 30] + [0]*(dim-2),
                [cx + 40, cy + 10] + [0]*(dim-2),
                [cx, cy + 60] + [0]*(dim-2),
            ],
            "star": [
                [cx, cy - 180] + [0]*(dim-2),
                [cx + 50, cy - 60] + [0]*(dim-2),
                [cx + 180, cy - 60] + [0]*(dim-2),
                [cx + 80, cy + 10] + [0]*(dim-2),
                [cx + 110, cy + 140] + [0]*(dim-2),
                [cx, cy + 70] + [0]*(dim-2),
                [cx - 110, cy + 140] + [0]*(dim-2),
                [cx - 80, cy + 10] + [0]*(dim-2),
                [cx - 180, cy - 60] + [0]*(dim-2),
                [cx - 50, cy - 60] + [0]*(dim-2),
                [cx, cy - 180] + [0]*(dim-2),
            ],
            "wave": [],
        }
        if name == "wave":
            pts = []
            n_pts = 12
            for i in range(n_pts):
                x = 60 + i * (580 / (n_pts - 1))
                y = cy + 120 * math.sin(i * math.pi / 2.5)
                pts.append([x, y] + [0]*(dim-2))
            presets["wave"] = pts

        if name in presets:
            tab._pts_internal = presets[name]
            tab._apply_pts_to_2d()
            if hasattr(tab, '_rebuild_weight_spins'):
                tab._rebuild_weight_spins()
            tab._update()
            self.statusBar().showMessage(f"Preset applied: {name}", 2000)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_H and not (ev.modifiers() & Qt.ControlModifier):
            self._show_help()
        else:
            super().keyPressEvent(ev)


# ════════════════════════════════════════════════════════════════
# Entry Point & JIT Warmup
# ════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    print("Warming up Numba JIT Compiler...")
    dummy_pts = np.array([[0,0], [1,1], [2,0]], dtype=np.float64)
    dummy_w = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    dummy_knots = np.array([0,0,0,1,1,1], dtype=np.float64)

    CurveMath.evaluate_bezier(dummy_pts, num_samples=10)
    CurveMath.evaluate_bspline(dummy_pts, 2, dummy_knots, num_samples=10)
    CurveMath.evaluate_nurbs(dummy_pts, dummy_w, 2, dummy_knots, num_samples=10)
    CurveMath.evaluate_basis_data(2, 2, dummy_knots, num_samples=10)
    print("JIT Compiler ready! Launching application.")

    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()