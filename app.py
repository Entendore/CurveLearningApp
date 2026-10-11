#!/usr/bin/env python3
"""
CurveLab — Interactive Bézier, B-Spline & NURBS Learning Application
Supports 2D, 3D, and N-Dimensional visualization.

Controls (2D Mode):
  - Left-drag on a control point to move it
  - Shift+Left-click on canvas to add a point
  - Right-click on a control point to remove it
"""

import sys
import math
import numpy as np
from numba import njit, prange

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QHBoxLayout, QVBoxLayout,
    QFormLayout, QGroupBox, QLabel, QPushButton, QSlider, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox, QLineEdit, QTextEdit,
    QListWidget, QScrollArea, QSizePolicy, QStackedWidget
)
from PySide6.QtCore import Qt, Signal, QPointF, QTimer
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath, QLinearGradient
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ════════════════════════════════════════════════════════════════
# Numba-Accelerated Curve Mathematics (N-Dimensional)
# ════════════════════════════════════════════════════════════════

# Must be defined at module level so Numba can resolve it natively
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

    # ── Bézier ──────────────────────────────────────────────────

    @staticmethod
    def de_casteljau(points, t):
        """N-Dimensional De Casteljau."""
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
    @njit(parallel=True, nogil=True, cache=True)
    def evaluate_bezier(control_points, num_samples=300):
        n = control_points.shape[0] - 1
        dim = control_points.shape[1]
        if n < 1:
            return np.empty((0, dim), dtype=np.float64)
        curve = np.zeros((num_samples + 1, dim), dtype=np.float64)
        for i in prange(num_samples + 1):
            t = i / num_samples
            # Inline De Casteljau for performance
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
        # Wrapper for external calls
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
        # Pad to match length
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
    pointAdded = Signal(float, float)
    pointRemoved = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.control_points: list = []
        self.curve_points: list = []
        self.reference_curve: list = []
        self.setMinimumSize(500, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self._drag_idx = -1
        self._hover_idx = -1
        self._pt_r = 8

        self.show_control_polygon = True
        self.show_curve = True
        self.show_convex_hull = False
        self.show_curvature_comb = False
        self.show_derivatives = False
        
        self.de_casteljau_levels: list = []
        self.highlight_point = None
        self.current_t = 0.5

        self._lvl_colors = [
            QColor(255, 200, 50, 180), QColor(50, 255, 200, 180),
            QColor(255, 100, 150, 180), QColor(150, 100, 255, 180),
            QColor(255, 180, 100, 180), QColor(100, 255, 100, 180),
        ]

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

    def set_highlight_point(self, pt, t=0.5):
        self.highlight_point = QPointF(pt[0], pt[1]) if pt is not None else None
        self.current_t = t
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0, QColor(20, 20, 32))
        grad.setColorAt(1, QColor(30, 30, 45))
        p.fillRect(self.rect(), grad)
        self._draw_grid(p)

        if self.show_convex_hull and len(self.control_points) >= 3:
            self._draw_convex_hull(p)

        if self.show_control_polygon and len(self.control_points) > 1:
            p.setPen(QPen(QColor(100, 100, 140), 1.4, Qt.DashLine))
            for i in range(len(self.control_points) - 1):
                p.drawLine(self.control_points[i], self.control_points[i + 1])

        for li, lv in enumerate(self.de_casteljau_levels):
            c = self._lvl_colors[li % len(self._lvl_colors)]
            p.setPen(QPen(c, 1, Qt.DotLine))
            for i in range(len(lv) - 1):
                p.drawLine(lv[i], lv[i + 1])
            p.setPen(Qt.NoPen)
            p.setBrush(c)
            for pt in lv:
                p.drawEllipse(pt, 4, 4)

        if self.reference_curve and len(self.reference_curve) > 1:
            p.setPen(QPen(QColor(255, 255, 255, 60), 1.5, Qt.DashLine))
            path = QPainterPath()
            path.moveTo(self.reference_curve[0])
            for pt in self.reference_curve[1:]:
                path.lineTo(pt)
            p.drawPath(path)

        if self.show_curve and len(self.curve_points) > 1:
            p.setPen(QPen(QColor(0, 200, 255), 2.5))
            path = QPainterPath()
            path.moveTo(self.curve_points[0])
            for pt in self.curve_points[1:]:
                path.lineTo(pt)
            p.drawPath(path)

        # Curvature Comb
        if self.show_curvature_comb and len(self.curve_points) > 2:
            curve_np = np.array([(pt.x(), pt.y()) for pt in self.curve_points])
            curvatures = CurveMath.evaluate_curvature_2d(curve_np)
            scale = 3000
            p.setPen(QPen(QColor(255, 150, 50, 150), 1))
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

        # Derivative Vectors
        if self.show_derivatives and self.highlight_point and len(self.curve_points) > 2:
            t_idx = min(int(self.current_t * (len(self.curve_points)-1)), len(self.curve_points)-2)
            pt = self.curve_points[t_idx]
            # Velocity
            vel = self.curve_points[t_idx+1] - pt
            vel *= 10
            p.setPen(QPen(QColor(100, 255, 100), 2))
            p.drawLine(pt, pt + vel)
            # Acceleration
            if t_idx < len(self.curve_points) - 2:
                v1 = self.curve_points[t_idx+1] - pt
                v2 = self.curve_points[t_idx+2] - self.curve_points[t_idx+1]
                acc = (v2 - v1) * 50
                p.setPen(QPen(QColor(255, 100, 100), 2))
                p.drawLine(pt, pt + acc)

        if self.highlight_point:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 100, 80))
            p.drawEllipse(self.highlight_point, 12, 12)
            p.setBrush(QColor(255, 255, 100))
            p.drawEllipse(self.highlight_point, 5, 5)

        for i, pt in enumerate(self.control_points):
            hover = i == self._hover_idx or i == self._drag_idx
            p.setPen(QPen(QColor(255, 255, 255) if not hover else QColor(255, 255, 100), 1.5))
            p.setBrush(QColor(255, 200, 50) if hover else QColor(255, 100, 80))
            p.drawEllipse(pt, self._pt_r, self._pt_r)
            p.setPen(QColor(210, 210, 230))
            p.setFont(QFont("Consolas", 8, QFont.Bold))
            p.drawText(pt + QPointF(10, -8), f"P{i}")

        p.setPen(QColor(90, 90, 110))
        p.setFont(QFont("Arial", 7))
        p.drawText(8, self.height() - 6, "Drag: move  |  Shift+Click: add  |  Right-click: remove")
        p.end()

    def _draw_grid(self, p):
        p.setPen(QPen(QColor(40, 40, 55), 0.5))
        s = 50
        for x in range(0, self.width(), s): p.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), s): p.drawLine(0, y, self.width(), y)

    def _draw_convex_hull(self, p):
        pts = [(pt.x(), pt.y()) for pt in self.control_points]
        hull = self._convex_hull(pts)
        if len(hull) < 3: return
        p.setPen(QPen(QColor(80, 200, 120, 70), 1, Qt.DashDotLine))
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

    def _hit(self, pos):
        for i, pt in enumerate(self.control_points):
            if (pos - pt).manhattanLength() < self._pt_r + 8: return i
        return -1

    def mousePressEvent(self, ev):
        pos = ev.position()
        if ev.button() == Qt.LeftButton:
            idx = self._hit(pos)
            if idx >= 0: self._drag_idx = idx
            elif ev.modifiers() & Qt.ShiftModifier: self.pointAdded.emit(pos.x(), pos.y())
        elif ev.button() == Qt.RightButton:
            idx = self._hit(pos)
            if idx >= 0: self.pointRemoved.emit(idx)
        self.update()

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self._drag_idx >= 0:
            self.control_points[self._drag_idx] = pos
            self.pointMoved.emit()
        else:
            old = self._hover_idx
            self._hover_idx = self._hit(pos)
            if old != self._hover_idx: self.update()

    def mouseReleaseEvent(self, ev): self._drag_idx = -1


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
            self.ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color='#00c8ff', lw=2)
            self.ax.plot(ctrl[:, 0], ctrl[:, 1], ctrl[:, 2], '--', color='#6c7086', lw=1)
            self.ax.scatter(ctrl[:, 0], ctrl[:, 1], ctrl[:, 2], color='#f38ba8', s=40, depthshade=True)
        elif dim == 2:
            self.ax.plot(curve[:, 0], curve[:, 1], color='#00c8ff', lw=2)
            self.ax.plot(ctrl[:, 0], ctrl[:, 1], '--', color='#6c7086', lw=1)
            self.ax.scatter(ctrl[:, 0], ctrl[:, 1], color='#f38ba8', s=40)

        self.draw()

    def style_axes(self):
        self.ax.set_xlabel('X', color='#89b4fa')
        self.ax.set_ylabel('Y', color='#a6e3a1')
        if hasattr(self.ax, 'set_zlabel'):
            self.ax.set_zlabel('Z', color='#cba6f7')
        self.ax.tick_params(colors='#6c7086')
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
# Unified Base Tab Logic for N-D
# ════════════════════════════════════════════════════════════════

class BaseCurveTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._animating = False
        self._anim_dir = 1
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._anim_step)
        self._pts_internal = []  # List of lists (N-D)

    def _setup_canvases(self, layout):
        self.canvas_stack = QStackedWidget()
        
        # 2D Canvas
        self.canvas_2d = CurveCanvas()
        self.canvas_2d.pointMoved.connect(self._on_2d_moved)
        self.canvas_2d.pointAdded.connect(self._on_2d_added)
        self.canvas_2d.pointRemoved.connect(self._on_2d_removed)
        self.canvas_stack.addWidget(self.canvas_2d)
        
        # 3D/N-D Canvas
        self.canvas_3d = NDCanvas()
        self.canvas_stack.addWidget(self.canvas_3d)
        
        layout.addWidget(self.canvas_stack, 3)

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
        pts2d = self.canvas_2d.get_control_points()
        for i, p in enumerate(pts2d):
            self._pts_internal[i][0] = p[0]
            self._pts_internal[i][1] = p[1]
        self._update()

    def _on_2d_added(self, x, y):
        dim = self.dim_spin.value()
        new_pt = [x, y] + [250.0] * (dim - 2)  # Default Z/W to middle
        self._pts_internal.append(new_pt)
        self._apply_pts_to_2d()
        self._update()

    def _on_2d_removed(self, idx):
        if len(self._pts_internal) > 2 and 0 <= idx < len(self._pts_internal):
            self._pts_internal.pop(idx)
            self._apply_pts_to_2d()
            self._update()

    def _toggle_anim(self):
        self._animating = not self._animating
        if self._animating:
            self.anim_btn.setText("⏸  Pause"); self._anim_dir = 1; self._anim_timer.start()
        else:
            self.anim_btn.setText("▶  Animate"); self._anim_timer.stop()

    def _anim_step(self):
        v = self.t_slider.value() + self._anim_dir * 2
        if v >= 1000: v = 1000; self._anim_dir = -1
        elif v <= 0: v = 0; self._anim_dir = 1
        self.t_slider.setValue(v)


# ════════════════════════════════════════════════════════════════
# Bézier Tab
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
        title = QLabel("🔵  Bézier Curves"); title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setStyleSheet("color:#00c8ff; padding:4px 0;"); rl.addWidget(title)

        # Dim
        g0 = QGroupBox("Dimensions"); g0l = QHBoxLayout(g0)
        self.dim_spin = QSpinBox(); self.dim_spin.setRange(2, 6); self.dim_spin.setValue(2)
        self.dim_spin.valueChanged.connect(self._sync_dim)
        g0l.addWidget(QLabel("D =")); g0l.addWidget(self.dim_spin); rl.addWidget(g0)

        # t parameter
        g = QGroupBox("Parameter t"); gl = QVBoxLayout(g)
        self.t_slider = QSlider(Qt.Horizontal); self.t_slider.setRange(0, 1000); self.t_slider.setValue(500)
        self.t_slider.valueChanged.connect(self._update); gl.addWidget(self.t_slider)
        self.t_lbl = QLabel("t = 0.500"); self.t_lbl.setAlignment(Qt.AlignCenter); gl.addWidget(self.t_lbl)
        self.dc_cb = QCheckBox("Show De Casteljau"); self.dc_cb.setChecked(True); self.dc_cb.toggled.connect(self._update); gl.addWidget(self.dc_cb)
        self.anim_btn = QPushButton("▶  Animate"); self.anim_btn.clicked.connect(self._toggle_anim); gl.addWidget(self.anim_btn)
        rl.addWidget(g)

        # display
        g2 = QGroupBox("Display"); g2l = QVBoxLayout(g2)
        self.poly_cb = QCheckBox("Control polygon"); self.poly_cb.setChecked(True); self.poly_cb.toggled.connect(self._update); g2l.addWidget(self.poly_cb)
        self.hull_cb = QCheckBox("Convex hull"); self.hull_cb.toggled.connect(self._update); g2l.addWidget(self.hull_cb)
        self.curv_cb = QCheckBox("Curvature Comb"); self.curv_cb.toggled.connect(self._update); g2l.addWidget(self.curv_cb)
        self.deriv_cb = QCheckBox("Derivative Vectors"); self.deriv_cb.toggled.connect(self._update); g2l.addWidget(self.deriv_cb)
        rl.addWidget(g2)

        # points
        g3 = QGroupBox("Control Points"); g3l = QVBoxLayout(g3)
        self.pt_list = QListWidget(); self.pt_list.setMaximumHeight(130); g3l.addWidget(self.pt_list)
        bl = QHBoxLayout()
        ba = QPushButton("➕ Add"); ba.clicked.connect(lambda: self._on_2d_added(300, 300)); bl.addWidget(ba)
        br = QPushButton("➖ Remove"); br.clicked.connect(lambda: self._on_2d_removed(len(self._pts_internal)-1)); bl.addWidget(br)
        bres = QPushButton("🔄 Reset"); bres.clicked.connect(self._reset); bl.addWidget(bres)
        g3l.addLayout(bl); rl.addWidget(g3)
        rl.addStretch(); lay.addWidget(rp, 1)

    def _reset(self):
        dim = self.dim_spin.value()
        self._pts_internal = [[120, 350] + [0]*(dim-2), [200, 100] + [150]*(dim-2), 
                              [450, 100] + [150]*(dim-2), [550, 350] + [0]*(dim-2)]
        self._apply_pts_to_2d(); self._update()

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
            self.canvas_2d.set_highlight_point(None); return

        curve_np = CurveMath.evaluate_bezier(pts_np)
        
        if dim == 2:
            self.canvas_2d.show_control_polygon = self.poly_cb.isChecked()
            self.canvas_2d.show_convex_hull = self.hull_cb.isChecked()
            self.canvas_2d.show_curvature_comb = self.curv_cb.isChecked()
            self.canvas_2d.show_derivatives = self.deriv_cb.isChecked()
            self.canvas_2d.set_curve_points(curve_np.tolist())

            if self.dc_cb.isChecked():
                _, levels = CurveMath.de_casteljau(pts_np, t)
                self.canvas_2d.set_de_casteljau_levels(levels)
                hp, _ = CurveMath.de_casteljau(pts_np, t)
                self.canvas_2d.set_highlight_point(hp, t)
            else:
                self.canvas_2d.set_de_casteljau_levels([]); self.canvas_2d.set_highlight_point(None)
        else:
            self.canvas_3d.plot_nd_curve(pts_np, curve_np, dim)


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
        title = QLabel("🟢  B-Splines"); title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setStyleSheet("color:#a6e3a1; padding:4px 0;"); rl.addWidget(title)

        g0 = QGroupBox("Dimensions"); g0l = QHBoxLayout(g0)
        self.dim_spin = QSpinBox(); self.dim_spin.setRange(2, 6); self.dim_spin.setValue(2)
        self.dim_spin.valueChanged.connect(self._sync_dim)
        g0l.addWidget(QLabel("D =")); g0l.addWidget(self.dim_spin); rl.addWidget(g0)

        g1 = QGroupBox("Degree"); g1l = QHBoxLayout(g1)
        self.deg_spin = QSpinBox(); self.deg_spin.setRange(1, 10); self.deg_spin.setValue(3)
        self.deg_spin.valueChanged.connect(self._on_degree); g1l.addWidget(QLabel("p =")); g1l.addWidget(self.deg_spin); rl.addWidget(g1)

        g2 = QGroupBox("Knot Vector"); g2l = QVBoxLayout(g2)
        self.knot_combo = QComboBox(); self.knot_combo.addItems(["Clamped", "Uniform", "Custom"])
        self.knot_combo.currentIndexChanged.connect(self._on_knot_type); g2l.addWidget(self.knot_combo)
        self.knot_edit = QLineEdit(); self.knot_edit.setPlaceholderText("0 0 0 0.33 0.67 1 1 1")
        self.knot_edit.editingFinished.connect(self._update); self.knot_edit.setEnabled(False); g2l.addWidget(self.knot_edit)
        self.knot_lbl = QLabel("Knots: —"); self.knot_lbl.setWordWrap(True); self.knot_lbl.setFont(QFont("Consolas", 8)); g2l.addWidget(self.knot_lbl); rl.addWidget(g2)

        g3 = QGroupBox("Parameter t"); g3l = QVBoxLayout(g3)
        self.t_slider = QSlider(Qt.Horizontal); self.t_slider.setRange(0, 1000); self.t_slider.setValue(500)
        self.t_slider.valueChanged.connect(self._update); g3l.addWidget(self.t_slider)
        self.t_lbl = QLabel("t = 0.500"); self.t_lbl.setAlignment(Qt.AlignCenter); g3l.addWidget(self.t_lbl)
        self.anim_btn = QPushButton("▶  Animate"); self.anim_btn.clicked.connect(self._toggle_anim); g3l.addWidget(self.anim_btn); rl.addWidget(g3)

        g4 = QGroupBox("Display"); g4l = QVBoxLayout(g4)
        self.poly_cb = QCheckBox("Control polygon"); self.poly_cb.setChecked(True); self.poly_cb.toggled.connect(self._update); g4l.addWidget(self.poly_cb)
        self.curv_cb = QCheckBox("Curvature Comb"); self.curv_cb.toggled.connect(self._update); g4l.addWidget(self.curv_cb)
        self.deriv_cb = QCheckBox("Derivative Vectors"); self.curv_cb.toggled.connect(self._update); g4l.addWidget(self.deriv_cb)
        self.basis_cb = QCheckBox("Basis functions"); self.basis_cb.setChecked(True); self.basis_cb.toggled.connect(self._update); g4l.addWidget(self.basis_cb); rl.addWidget(g4)

        g5 = QGroupBox("Control Points"); g5l = QVBoxLayout(g5)
        self.pt_list = QListWidget(); self.pt_list.setMaximumHeight(100); g5l.addWidget(self.pt_list)
        bl = QHBoxLayout()
        ba = QPushButton("➕ Add"); ba.clicked.connect(lambda: self._on_2d_added(350, 300)); bl.addWidget(ba)
        br = QPushButton("➖ Remove"); br.clicked.connect(lambda: self._on_2d_removed(len(self._pts_internal)-1)); bl.addWidget(br)
        bres = QPushButton("🔄 Reset"); bres.clicked.connect(self._reset); bl.addWidget(bres)
        g5l.addLayout(bl); rl.addWidget(g5); rl.addStretch(); lay.addWidget(rp, 1)

    def _reset(self):
        dim = self.dim_spin.value()
        self._pts_internal = [[80, 350] + [0]*(dim-2), [200, 120] + [150]*(dim-2), 
                              [370, 120] + [150]*(dim-2), [450, 350] + [0]*(dim-2),
                              [530, 180] + [250]*(dim-2), [650, 280] + [0]*(dim-2)]
        self._apply_pts_to_2d(); self._update()

    def _on_degree(self):
        if len(self._pts_internal) < self.deg_spin.value() + 1:
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
            self.canvas_2d.set_curve_points([]); self.knot_lbl.setText("Need ≥ p+1 pts")
            self.basis_canvas.set_basis_data(np.array([]), np.array([]), [], deg); return

        knots = self._get_knots()
        if knots is None or len(knots) != n + deg + 2:
            self.canvas_2d.set_curve_points([]); self.knot_lbl.setText(f"Need {n+deg+2} knots")
            self.basis_canvas.set_basis_data(np.array([]), np.array([]), [], deg); return

        ks = "  ".join(f"{k:.3f}" for k in knots); self.knot_lbl.setText(f"[{ks}]")
        pts_np = np.array(pts, dtype=np.float64)
        curve_np = CurveMath.evaluate_bspline(pts_np, deg, knots)

        t_min = knots[deg]; t_max = knots[n + 1]

        if dim == 2:
            self.canvas_2d.show_control_polygon = self.poly_cb.isChecked()
            self.canvas_2d.show_curvature_comb = self.curv_cb.isChecked()
            self.canvas_2d.show_derivatives = self.deriv_cb.isChecked()
            self.canvas_2d.set_curve_points(curve_np.tolist())
            
            if t_max > t_min:
                t_val = t_min + t_norm * (t_max - t_min)
                hp = np.zeros(2); 
                for j in range(n + 1):
                    b = CurveMath.bspline_basis(j, deg, t_val, knots)
                    hp += b * pts_np[j, :2]
                self.canvas_2d.set_highlight_point(hp, t_norm)
            else: self.canvas_2d.set_highlight_point(None)
        else:
            self.canvas_3d.plot_nd_curve(pts_np, curve_np, dim)

        if self.basis_cb.isChecked() and t_max > t_min:
            t_vals, b_data = CurveMath.evaluate_basis_data(n, deg, knots)
            self.basis_canvas.set_basis_data(t_vals, b_data, knots, deg, t_norm)
        else: self.basis_canvas.set_basis_data(np.array([]), np.array([]), [], deg)


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
        title = QLabel("🟣  NURBS"); title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setStyleSheet("color:#cba6f7; padding:4px 0;"); rl.addWidget(title)

        g0 = QGroupBox("Dimensions"); g0l = QHBoxLayout(g0)
        self.dim_spin = QSpinBox(); self.dim_spin.setRange(2, 6); self.dim_spin.setValue(2)
        self.dim_spin.valueChanged.connect(self._sync_dim)
        g0l.addWidget(QLabel("D =")); g0l.addWidget(self.dim_spin); rl.addWidget(g0)

        g1 = QGroupBox("Degree"); g1l = QHBoxLayout(g1)
        self.deg_spin = QSpinBox(); self.deg_spin.setRange(1, 10); self.deg_spin.setValue(3)
        self.deg_spin.valueChanged.connect(self._on_degree); g1l.addWidget(QLabel("p =")); g1l.addWidget(self.deg_spin); rl.addWidget(g1)

        g2 = QGroupBox("Knot Vector"); g2l = QVBoxLayout(g2)
        self.knot_combo = QComboBox(); self.knot_combo.addItems(["Clamped", "Uniform", "Custom"])
        self.knot_combo.currentIndexChanged.connect(self._on_knot_type); g2l.addWidget(self.knot_combo)
        self.knot_edit = QLineEdit(); self.knot_edit.setPlaceholderText("0 0 0 0.33 0.67 1 1 1")
        self.knot_edit.editingFinished.connect(self._update); self.knot_edit.setEnabled(False); g2l.addWidget(self.knot_edit)
        self.knot_lbl = QLabel("Knots: —"); self.knot_lbl.setWordWrap(True); self.knot_lbl.setFont(QFont("Consolas", 8)); g2l.addWidget(self.knot_lbl); rl.addWidget(g2)

        g3 = QGroupBox("Weights"); g3l = QVBoxLayout(g3)
        self.w_equal_btn = QPushButton("All = 1"); self.w_equal_btn.clicked.connect(self._equal_weights); g3l.addWidget(self.w_equal_btn)
        self.w_circle_btn = QPushButton("⚖️ Preset: Circle (2D)"); self.w_circle_btn.clicked.connect(self._circle_preset); g3l.addWidget(self.w_circle_btn)
        self.w_scroll = QScrollArea(); self.w_scroll.setWidgetResizable(True)
        self.w_container = QWidget(); self.w_container_l = QVBoxLayout(self.w_container); self.w_container_l.setContentsMargins(0,0,0,0)
        self.w_scroll.setWidget(self.w_container); self.w_scroll.setMaximumHeight(150); g3l.addWidget(self.w_scroll); rl.addWidget(g3)

        g4 = QGroupBox("Parameter t"); g4l = QVBoxLayout(g4)
        self.t_slider = QSlider(Qt.Horizontal); self.t_slider.setRange(0, 1000); self.t_slider.setValue(500)
        self.t_slider.valueChanged.connect(self._update); g4l.addWidget(self.t_slider)
        self.t_lbl = QLabel("t = 0.500"); self.t_lbl.setAlignment(Qt.AlignCenter); g4l.addWidget(self.t_lbl)
        self.anim_btn = QPushButton("▶  Animate"); self.anim_btn.clicked.connect(self._toggle_anim); g4l.addWidget(self.anim_btn); rl.addWidget(g4)

        g5 = QGroupBox("Display"); g5l = QVBoxLayout(g5)
        self.poly_cb = QCheckBox("Control polygon"); self.poly_cb.setChecked(True); self.poly_cb.toggled.connect(self._update); g5l.addWidget(self.poly_cb)
        self.ref_cb = QCheckBox("B-Spline overlay (w=1)"); self.ref_cb.setChecked(True); self.ref_cb.toggled.connect(self._update); g5l.addWidget(self.ref_cb)
        self.curv_cb = QCheckBox("Curvature Comb"); self.curv_cb.toggled.connect(self._update); g5l.addWidget(self.curv_cb)
        self.deriv_cb = QCheckBox("Derivative Vectors"); self.deriv_cb.toggled.connect(self._update); g5l.addWidget(self.deriv_cb)
        self.basis_cb2 = QCheckBox("Basis functions"); self.basis_cb2.setChecked(True); self.basis_cb2.toggled.connect(self._update); g5l.addWidget(self.basis_cb2); rl.addWidget(g5)

        g6 = QGroupBox("Control Points"); g6l = QVBoxLayout(g6)
        self.pt_list = QListWidget(); self.pt_list.setMaximumHeight(90); g6l.addWidget(self.pt_list)
        bl = QHBoxLayout()
        ba = QPushButton("➕ Add"); ba.clicked.connect(lambda: self._on_2d_added(350, 300)); bl.addWidget(ba)
        br = QPushButton("➖ Remove"); br.clicked.connect(lambda: self._on_2d_removed(len(self._pts_internal)-1)); bl.addWidget(br)
        bres = QPushButton("🔄 Reset"); bres.clicked.connect(self._reset); bl.addWidget(bres)
        g6l.addLayout(bl); rl.addWidget(g6); rl.addStretch(); lay.addWidget(rp, 1)

        self.w_spins = []

    def _rebuild_weight_spins(self):
        while self.w_container_l.count():
            item = self.w_container_l.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.w_spins = []
        for i in range(len(self._pts_internal)):
            row = QHBoxLayout(); lbl = QLabel(f"w{i}:"); lbl.setFixedWidth(26); row.addWidget(lbl)
            spin = QDoubleSpinBox(); spin.setRange(0.01, 50.0); spin.setSingleStep(0.1); spin.setDecimals(2); spin.setValue(1.0)
            spin.valueChanged.connect(self._update); row.addWidget(spin); self.w_spins.append(spin)
            self.w_container_l.addLayout(row)

    def _reset(self):
        dim = self.dim_spin.value()
        self._pts_internal = [[80, 350] + [0]*(dim-2), [200, 120] + [150]*(dim-2), 
                              [370, 120] + [150]*(dim-2), [450, 350] + [0]*(dim-2),
                              [530, 180] + [250]*(dim-2), [650, 280] + [0]*(dim-2)]
        self._apply_pts_to_2d(); self._rebuild_weight_spins(); self._update()

    def _on_degree(self):
        if len(self._pts_internal) < self.deg_spin.value() + 1:
            self._pts_internal = self._pts_internal[:self.deg_spin.value() + 1]
            self._apply_pts_to_2d(); self._rebuild_weight_spins()
        self._update()

    def _on_knot_type(self):
        self.knot_edit.setEnabled(self.knot_combo.currentIndex() == 2); self._update()

    def _equal_weights(self):
        for s in self.w_spins: s.blockSignals(True); s.setValue(1.0); s.blockSignals(False)
        self._update()

    def _circle_preset(self):
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
            self.knot_lbl.setText("Need ≥ p+1 pts"); self.basis_canvas.set_basis_data(np.array([]), np.array([]), [], deg); return

        knots = self._get_knots()
        if knots is None or len(knots) != n + deg + 2:
            self.canvas_2d.set_curve_points([]); self.canvas_2d.set_reference_curve([])
            self.knot_lbl.setText(f"Need {n+deg+2} knots"); self.basis_canvas.set_basis_data(np.array([]), np.array([]), [], deg); return

        ks = "  ".join(f"{k:.3f}" for k in knots); self.knot_lbl.setText(f"[{ks}]")
        if len(ws) != n + 1: self.canvas_2d.set_curve_points([]); return

        pts_np = np.array(pts, dtype=np.float64)
        curve_np = CurveMath.evaluate_nurbs(pts_np, ws, deg, knots)

        t_min = knots[deg]; t_max = knots[n + 1]

        if dim == 2:
            self.canvas_2d.show_control_polygon = self.poly_cb.isChecked()
            self.canvas_2d.show_curvature_comb = self.curv_cb.isChecked()
            self.canvas_2d.show_derivatives = self.deriv_cb.isChecked()
            self.canvas_2d.set_curve_points(curve_np.tolist())

            if self.ref_cb.isChecked():
                ref_np = CurveMath.evaluate_bspline(pts_np, deg, knots)
                self.canvas_2d.set_reference_curve(ref_np.tolist())
            else: self.canvas_2d.set_reference_curve([])

            if t_max > t_min:
                t_val = t_min + t_norm * (t_max - t_min)
                hp = np.zeros(2); dw = 0.0
                for j in range(n + 1):
                    b = CurveMath.bspline_basis(j, deg, t_val, knots); w = ws[j]
                    hp += w * b * pts_np[j, :2]; dw += w * b
                if dw > 1e-12: self.canvas_2d.set_highlight_point(hp / dw, t_norm)
                else: self.canvas_2d.set_highlight_point(None)
            else: self.canvas_2d.set_highlight_point(None)
        else:
            self.canvas_3d.plot_nd_curve(pts_np, curve_np, dim)
            self.canvas_2d.set_reference_curve([])

        if self.basis_cb2.isChecked() and t_max > t_min:
            t_vals, b_data = CurveMath.evaluate_basis_data(n, deg, knots)
            self.basis_canvas.set_basis_data(t_vals, b_data, knots, deg, t_norm)
        else: self.basis_canvas.set_basis_data(np.array([]), np.array([]), [], deg)


# ════════════════════════════════════════════════════════════════
# Main Window
# ════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CurveLab — Bézier · B-Spline · NURBS  (N-Dimensional)")
        self.resize(1250, 800)

        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI', Arial; font-size: 11px; }
            QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 10px; padding: 10px 6px 6px 6px; font-weight: bold; color: #89b4fa; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { background-color: #313244; border: 1px solid #45475a; border-radius: 4px; padding: 5px 10px; color: #cdd6f4; }
            QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
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
        """)

        tabs = QTabWidget()
        tabs.addTab(BezierTab(), "🔵  Bézier Curves")
        tabs.addTab(BSplineTab(), "🟢  B-Splines")
        tabs.addTab(NURBSTab(), "🟣  NURBS")
        self.setCentralWidget(tabs)


# ════════════════════════════════════════════════════════════════
# Entry Point & JIT Warmup
# ════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Warmup Numba JIT to prevent UI stutter on first interaction
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