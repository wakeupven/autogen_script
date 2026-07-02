#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор синтетического датасета 2D планов помещений в стиле ГОСТ.
~50 000 изображений + маски семантической сегментации.

Зависимости: pip install shapely pillow tqdm
"""

import random
import os
import math
import sys
import argparse
import warnings
from typing import List, Tuple, Optional

from PIL import Image, ImageDraw
from shapely.geometry import Polygon, LineString, Point, box, MultiPolygon, JOIN_STYLE
from shapely.ops import unary_union
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="shapely")

# =============================================================================
# ПАРАМЕТРЫ
# =============================================================================
NUM_IMAGES = 10
SEED = 42
MIN_CANVAS = 1200
MAX_CANVAS = 2500
WALL_T_MIN = 12
WALL_T_MAX = 40
DOOR_W_MIN = 100
DOOR_W_MAX = 120
WIND_W_MIN = 150
WIND_W_MAX = 250
# Диапазон смещения УГО окна относительно центра стены по глубине
# (доля от wall_t; отрицательное = ближе к помещению, положительное = наружу)
WINDOW_SHIFT_RANGE = (-0.2, 0.2)
# Толщина линий УГО (делитель wall_t; чем меньше число, тем толще линия)
DOOR_LINE_DIV = 4
WINDOW_LINE_DIV = 6
# Диапазон яркости заливки стен (0 = чёрный, 255 = белый)
WALL_FILL_RANGE = (60, 220)
# Вероятность скоса (шамфера) внешнего угла комнаты (0 = выкл, т.к. внешние стены д.б. прямыми)
CHAMFER_PROB = 0.0
# Диапазон длины скоса относительно меньшей прилегающей стены
CHAMFER_SIZE_RANGE = (0.12, 0.35)
# Доля комнат, получающих наклонные внутренние стены (не под 90°)
ANGLED_WALL_FRACTION = 0.45

# =============================================================================
# ПАРАМЕТРЫ АУГМЕНТАЦИИ (Augraphy)
# =============================================================================
USE_AUGMENTATIONS = True          # Включить/выключить имитацию артефактов сканирования
AUGMENTATION_PROB = 0.75          # Вероятность применения pipeline к изображению
MIN_ROOMS = 2
MAX_ROOMS = 8
MIN_OPENINGS = 1
MAX_OPENINGS = 6
DOOR_PROB = 0.5         # Вероятность, что случайный проём — дверь (вес; нормируется с WINDOW_PROB)
WINDOW_PROB = 0.5       # Вероятность, что случайный проём — окно (вес; нормируется с DOOR_PROB)
                        # Первый проём всегда дверь (гарантия хотя бы одной двери).
CANVAS_MARGIN = 300

OUTPUT_DIR = "dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")

# =============================================================================
# ЦВЕТА
# =============================================================================
BG = (255, 255, 255)
WALL_LINE = (20, 20, 20)
WALL_FILL = (180, 180, 180)
ROOM_FILL = (245, 245, 245)
DOOR_COL = (20, 20, 20)
WIND_COL = (20, 20, 20)
FURN_COL = (160, 160, 160)

M_ROOM = (0, 255, 255)
M_WALL = (255, 255, 0)
M_WIND = (255, 0, 0)
M_DOOR = (255, 127, 80)
M_DOORW = (128, 0, 128)
M_WINDW = (0, 255, 0)

# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================
def rand_wall_t() -> int:
    return random.randint(WALL_T_MIN, WALL_T_MAX)

def rand_door_w() -> int:
    return random.randint(DOOR_W_MIN, DOOR_W_MAX)

def rand_wind_w() -> int:
    return random.randint(WIND_W_MIN, WIND_W_MAX)

def dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def point_along(p1: Tuple[float, float], p2: Tuple[float, float], d: float) -> Tuple[float, float]:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return p1
    return (p1[0] + dx * d / length, p1[1] + dy * d / length)

def line_offset(p1, p2, offset):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (p1, p2)
    nx, ny = -dy / length, dx / length
    return ((p1[0] + nx * offset, p1[1] + ny * offset),
            (p2[0] + nx * offset, p2[1] + ny * offset))

def rect_from_line(p1, p2, thickness):
    half = thickness / 2.0
    a1, a2 = line_offset(p1, p2, half)
    b1, b2 = line_offset(p1, p2, -half)
    return [a1, a2, b2, b1]

def draw_shapely_poly(draw, polygon, fill, outline=None, width=1):
    if polygon is None:
        return
    if polygon.geom_type == "Polygon":
        exterior = [(int(x), int(y)) for x, y in polygon.exterior.coords]
        if len(exterior) >= 3:
            draw.polygon(exterior, fill=fill, outline=outline, width=width)
    elif polygon.geom_type == "MultiPolygon":
        for poly in polygon.geoms:
            draw_shapely_poly(draw, poly, fill, outline, width)
    elif polygon.geom_type == "GeometryCollection":
        for geom in polygon.geoms:
            draw_shapely_poly(draw, geom, fill, outline, width)


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ НЕПРЯМОУГОЛЬНЫХ СТЕН
# =============================================================================
def _is_ccw(poly_pts):
    """Check if polygon points are counter-clockwise (shoelace)."""
    s = 0
    n = len(poly_pts)
    for i in range(n):
        x1, y1 = poly_pts[i]
        x2, y2 = poly_pts[(i + 1) % n]
        s += (x2 - x1) * (y2 + y1)
    return s > 0


def _normal_outward(edge_p1, edge_p2, poly_pts):
    """
    For an edge from edge_p1 to edge_p2 that is part of poly_pts (CCW-order),
    return the outward-pointing unit normal (pointing outside the polygon).
    """
    dx = edge_p2[0] - edge_p1[0]
    dy = edge_p2[1] - edge_p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (0, 0)
    nx, ny = -dy / length, dx / length  # left normal (CCW = inward)
    if _is_ccw(poly_pts):
        return (-nx, -ny)  # flip to outward
    return (nx, ny)  # already outward


def _polygon_edges(poly_pts):
    """Return list of ((x1,y1),(x2,y2)) edges from polygon points (list of tuples)."""
    n = len(poly_pts)
    edges = []
    for i in range(n):
        p1 = poly_pts[i]
        p2 = poly_pts[(i + 1) % n]
        if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) > 1:
            edges.append((p1, p2))
    return edges


def _edge_key(p1, p2):
    """Hashable key for an edge (order-independent, rounded)."""
    x1, y1 = p1
    x2, y2 = p2
    return (round(min(x1, x2), 6), round(min(y1, y2), 6),
            round(max(x1, x2), 6), round(max(y1, y2), 6))


def _room_to_poly(room_data):
    """Convert room data (tuple or list of pts) to Shapely Polygon."""
    if isinstance(room_data, tuple) and len(room_data) == 4:
        x1, y1, x2, y2 = room_data
        return box(x1, y1, x2, y2)
    return Polygon(room_data)


def _room_edges(room_data):
    """Extract edges from room data (tuple or list of pts)."""
    if isinstance(room_data, tuple) and len(room_data) == 4:
        x1, y1, x2, y2 = room_data
        return [
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ]
    return _polygon_edges(room_data)


def _normal_outward_for_room(room_data, edge_p1, edge_p2):
    """Outward normal for an edge of a room (pointing away from room interior)."""
    if isinstance(room_data, tuple) and len(room_data) == 4:
        x1, y1, x2, y2 = room_data
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    else:
        poly = Polygon(room_data)
        cx, cy = poly.centroid.x, poly.centroid.y
    mx, my = (edge_p1[0] + edge_p2[0]) / 2, (edge_p1[1] + edge_p2[1]) / 2
    dx = edge_p2[0] - edge_p1[0]
    dy = edge_p2[1] - edge_p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (0, 0)
    # Left normal (-dy, dx) points inward for CCW polygons
    nx, ny = -dy / length, dx / length
    # If normal points toward centroid, it's inward — flip to outward
    to_center = math.atan2(cy - my, cx - mx)
    normal_angle = math.atan2(ny, nx)
    if abs((normal_angle - to_center) % (2 * math.pi)) <= math.pi / 2:
        nx, ny = -nx, -ny
    return nx, ny


def _are_adjacent(room_a, room_b, eps=2.0):
    """Check if two rooms share an edge (rooms are separated by wall_t gap).
    Returns (edge_a_p1, edge_a_p2) for room_a's side, or None."""
    poly_a = _room_to_poly(room_a)
    poly_b = _room_to_poly(room_b)
    # Buffer room_a by wall_t to reach room_b, then intersect to find the shared line
    inter = poly_a.buffer(eps, join_style=2, cap_style=2).intersection(poly_b)
    if inter.is_empty:
        return None
    # Convert Polygon/rectangle intersection to a line
    if inter.geom_type == "LineString":
        line = inter
    elif inter.geom_type == "MultiLineString":
        lines = [ls for ls in inter.geoms if ls.length > eps]
        if not lines:
            return None
        line = max(lines, key=lambda ls: ls.length)
    elif inter.geom_type in ("Polygon", "MultiPolygon"):
        # Two buffered rectangles can intersect in a thin rectangle;
        # extract its longer axis as the shared line
        if inter.geom_type == "MultiPolygon":
            polys = list(inter.geoms)
        else:
            polys = [inter]
        best = None
        best_len = 0
        for poly in polys:
            if poly.area < 1:
                continue
            minx, miny, maxx, maxy = poly.bounds
            w, h = maxx - minx, maxy - miny
            if w > h and w > best_len:
                best = LineString([(minx, (miny + maxy) / 2), (maxx, (miny + maxy) / 2)])
                best_len = w
            elif h > best_len:
                best = LineString([((minx + maxx) / 2, miny), ((minx + maxx) / 2, maxy)])
                best_len = h
        line = best
        if line is None:
            return None
    else:
        return None
    if line.length < eps:
        return None
    return (line.coords[0], line.coords[-1])


def _perturb_shared_wall(part_a, part_b, edge_p1, edge_p2, wall_t):
    """
    Perturb the shared wall between two rooms using a V-shaped offset.
    Shifts the MIDPOINT of each room's edge TOWARD the other room,
    keeping both endpoints fixed so external walls stay straight.
    Each room's V-vertex is offset by 's' from its original edge,
    maintaining a minimum wall gap of ~ 0.4 * wall_t at the vertex.
    edge_p1, edge_p2 is the centerline between the two rooms (from _are_adjacent).
    Returns (new_a, new_b) or None if perturbation fails.
    """
    cdx = edge_p2[0] - edge_p1[0]
    cdy = edge_p2[1] - edge_p1[1]
    edge_len = math.hypot(cdx, cdy)
    if edge_len < wall_t * 6:
        return None

    cx, cy = (edge_p1[0] + edge_p2[0]) / 2, (edge_p1[1] + edge_p2[1]) / 2

    def _pts(room):
        if isinstance(room, tuple) and len(room) == 4:
            x1, y1, x2, y2 = room
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return list(room)

    def _find_edge(room, cx, cy, cdx, cdy):
        """Find edge parallel to and closest to the centerline midpoint."""
        pts = _pts(room)
        n = len(pts)
        best_i = None
        best_dist = float('inf')
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            edx, edy = b[0] - a[0], b[1] - a[1]
            e_len = math.hypot(edx, edy)
            if e_len < 1:
                continue
            dot = (edx * cdx + edy * cdy) / (e_len * edge_len)
            if abs(abs(dot) - 1) > 0.01:
                continue
            edge_nx, edge_ny = -edy / e_len, edx / e_len
            dist = abs((cx - a[0]) * edge_nx + (cy - a[1]) * edge_ny)
            if dist < best_dist:
                best_dist = dist
                best_i = i
        return best_i

    def _midpoint_at(room, idx):
        pts = _pts(room)
        n = len(pts)
        a, b = pts[idx], pts[(idx + 1) % n]
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

    def _insert_at_index(room, idx, v):
        pts = _pts(room)
        n = len(pts)
        if idx >= n:
            return None
        return pts[:idx + 1] + [v] + pts[idx + 1:]

    idx_a = _find_edge(part_a, cx, cy, cdx, cdy)
    idx_b = _find_edge(part_b, cx, cy, cdx, cdy)
    if idx_a is None or idx_b is None:
        return None

    mid_a = _midpoint_at(part_a, idx_a)
    mid_b = _midpoint_at(part_b, idx_b)

    # Direction from room A's edge midpoint to room B's edge midpoint
    mdx = mid_b[0] - mid_a[0]
    mdy = mid_b[1] - mid_a[1]
    md_len = math.hypot(mdx, mdy)
    if md_len < 1:
        return None

    # Shift magnitude: push each room's midpoint toward the other
    shift = random.uniform(0.08, 0.25) * edge_len
    shift = max(shift, wall_t * 0.5)
    # Cap so minimum gap at vertex >= wall_t * 0.4
    max_shift = md_len * 0.5 - wall_t * 0.2
    shift = min(shift, max_shift)
    if shift < wall_t * 0.15:
        return None

    # Unit direction from A to B and B to A
    udx = mdx / md_len
    udy = mdy / md_len

    new_mid_a = (mid_a[0] + udx * shift, mid_a[1] + udy * shift)
    new_mid_b = (mid_b[0] - udx * shift, mid_b[1] - udy * shift)

    new_a = _insert_at_index(part_a, idx_a, new_mid_a)
    new_b = _insert_at_index(part_b, idx_b, new_mid_b)
    if new_a is None or new_b is None:
        return None

    poly_a = Polygon(new_a)
    poly_b = Polygon(new_b)
    if (not poly_a.is_valid or not poly_a.is_simple
            or not poly_b.is_valid or not poly_b.is_simple):
        return None
    if poly_a.area < 200 or poly_b.area < 200:
        return None
    return new_a, new_b


def apply_angled_walls(parts, wall_t):
    """
    Post-process BSP room parts to create non-90-degree walls.
    Selects random adjacent room pairs and perturbs their shared wall.
    Returns modified parts list (some entries become polygon point lists).
    """
    if len(parts) < 2:
        return parts

    num_pairs = max(1, int(len(parts) * ANGLED_WALL_FRACTION))
    candidates = list(range(len(parts)))
    random.shuffle(candidates)

    modified = set()
    pairs_done = 0
    attempts = 0

    while pairs_done < num_pairs and attempts < len(parts) * 4:
        attempts += 1
        idx = random.choice(candidates)
        if idx in modified:
            continue
        room_a = parts[idx]

        # Find an adjacent room not yet modified
        neighbors = []
        for j, room_b in enumerate(parts):
            if j == idx or j in modified:
                continue
            shared = _are_adjacent(room_a, room_b, wall_t)
            if shared is not None:
                neighbors.append((j, shared))
        if not neighbors:
            continue

        j, (ep1, ep2) = random.choice(neighbors)
        result = _perturb_shared_wall(room_a, parts[j], ep1, ep2, wall_t)
        if result is None:
            continue

        new_a, new_b = result
        if new_a is not None and new_b is not None:
            # Validate new polygons have positive area
            poly_a = Polygon(new_a)
            poly_b = Polygon(new_b)
            if poly_a.area > 100 and poly_b.area > 100:
                parts[idx] = new_a
                parts[j] = new_b
                modified.add(idx)
                modified.add(j)
                pairs_done += 1

    return parts


# =============================================================================
# ГЕНЕРАЦИЯ ГЕОМЕТРИИ ПОМЕЩЕНИЙ
# =============================================================================
def generate_rooms(
    canvas_w: int, canvas_h: int, wall_t: int, num_rooms: int
) -> Tuple[List[Polygon], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """
    Сгенерировать комнаты (interior) и стены (midlines) через BSP
    со случайными скошенными (шамферными) внешними углами
    и опциональными непрямоугольными (наклонными) стенами.
    Возвращает (room_polygons, wall_midlines).
    """
    margin = CANVAS_MARGIN
    half_t = wall_t / 2.0
    bb_x1, bb_y1 = margin, margin
    bb_x2, bb_y2 = canvas_w - margin, canvas_h - margin

    if bb_x2 - bb_x1 < 300 or bb_y2 - bb_y1 < 300:
        bb_x1, bb_y1 = 100, 100
        bb_x2, bb_y2 = canvas_w - 100, canvas_h - 100

    min_room = 150

    parts: List = [
        (bb_x1 + half_t, bb_y1 + half_t, bb_x2 - half_t, bb_y2 - half_t)
    ]

    while len(parts) < num_rooms:
        candidates = []
        for i, p in enumerate(parts):
            if not (isinstance(p, tuple) and len(p) == 4):
                continue
            rx1, ry1, rx2, ry2 = p
            w, h = rx2 - rx1, ry2 - ry1
            min_dim_needed = min_room * 2 + wall_t
            if w > min_dim_needed or h > min_dim_needed:
                candidates.append(i)
        if not candidates:
            break
        idx = random.choice(candidates)
        rx1, ry1, rx2, ry2 = parts[idx]
        w, h = rx2 - rx1, ry2 - ry1

        split_ok = False
        if w > h * 1.2:
            dirs = ['v', 'h']
        elif h > w * 1.2:
            dirs = ['h', 'v']
        else:
            dirs = ['v', 'h'] if random.random() < 0.5 else ['h', 'v']

        for d in dirs:
            if d == 'v':
                lo = int(rx1 + min_room + half_t)
                hi = int(rx2 - min_room - half_t)
            else:
                lo = int(ry1 + min_room + half_t)
                hi = int(ry2 - min_room - half_t)
            if lo >= hi:
                continue
            split = random.randint(lo, hi)
            if d == 'v':
                a, b = (rx1, ry1, split - half_t, ry2), (split + half_t, ry1, rx2, ry2)
            else:
                a, b = (rx1, ry1, rx2, split - half_t), (rx1, split + half_t, rx2, ry2)
            parts.pop(idx)
            parts.append(a)
            parts.append(b)
            split_ok = True
            break

        if not split_ok:
            break

    parts = parts[:num_rooms]

    # --- Применяем наклонные стены к части прямоугольных комнат ---
    parts = apply_angled_walls(parts, wall_t)

    # --- Шаг 1: подсчёт рёбер (универсальный, для rect и polygon комнат) ---
    edge_counts: dict = {}
    for room_data in parts:
        for p1, p2 in _room_edges(room_data):
            key = _edge_key(p1, p2)
            edge_counts[key] = edge_counts.get(key, 0) + 1

    external_edges = {k for k, v in edge_counts.items() if v == 1}

    # Для прямоугольных комнат считаем ориентированные рёбра (нужно для шамферов)
    oriented_counts: dict = {}
    for room_data in parts:
        if isinstance(room_data, tuple) and len(room_data) == 4:
            rx1, ry1, rx2, ry2 = room_data
            elist = [
                ("h", rx1, ry1, rx2, ry1),
                ("h", rx1, ry2, rx2, ry2),
                ("v", rx1, ry1, rx1, ry2),
                ("v", rx2, ry1, rx2, ry2),
            ]
            for etype, ex1, ey1, ex2, ey2 in elist:
                key = (etype, round(min(ex1, ex2), 6), round(min(ey1, ey2), 6),
                       round(max(ex1, ex2), 6), round(max(ey1, ey2), 6))
                oriented_counts[key] = oriented_counts.get(key, 0) + 1

    oriented_ext_keys = {k for k, v in oriented_counts.items() if v == 1}

    def _is_ext_corner(rx1, ry1, rx2, ry2, cx, cy):
        eps = 0.01
        horiz_key = None
        vert_key = None
        if abs(cy - ry1) < eps:
            horiz_key = ("h", round(min(rx1, rx2), 6), round(cy, 6),
                         round(max(rx1, rx2), 6), round(cy, 6))
        elif abs(cy - ry2) < eps:
            horiz_key = ("h", round(min(rx1, rx2), 6), round(cy, 6),
                         round(max(rx1, rx2), 6), round(cy, 6))
        if abs(cx - rx1) < eps:
            vert_key = ("v", round(cx, 6), round(min(ry1, ry2), 6),
                        round(cx, 6), round(max(ry1, ry2), 6))
        elif abs(cx - rx2) < eps:
            vert_key = ("v", round(cx, 6), round(min(ry1, ry2), 6),
                        round(cx, 6), round(max(ry1, ry2), 6))
        return (horiz_key in oriented_ext_keys and vert_key in oriented_ext_keys)

    # --- Шаг 2: строим полигоны комнат со скосами + наклонными стенами ---
    room_polys: List[Polygon] = []
    inner_wall_midlines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    for room_data in parts:
        if isinstance(room_data, tuple) and len(room_data) == 4:
            rx1, ry1, rx2, ry2 = room_data
            corners = [(rx1, ry1), (rx2, ry1), (rx2, ry2), (rx1, ry2)]
            neighbors = [(3, 1), (0, 2), (1, 3), (2, 0)]

            chamfer_at = {}
            for ci, (cx, cy) in enumerate(corners):
                if not _is_ext_corner(rx1, ry1, rx2, ry2, cx, cy):
                    continue
                if random.random() >= CHAMFER_PROB:
                    continue
                ni, pi = neighbors[ci]
                nx, ny = corners[ni]
                px, py = corners[pi]
                len_n = math.hypot(cx - nx, cy - ny)
                len_p = math.hypot(cx - px, cy - py)
                min_len = min(len_n, len_p)
                if min_len < wall_t * 3:
                    continue
                chamfer_dist = random.uniform(CHAMFER_SIZE_RANGE[0], CHAMFER_SIZE_RANGE[1]) * min_len
                chamfer_dist = max(chamfer_dist, wall_t)
                a = point_along((cx, cy), (nx, ny), chamfer_dist)
                b = point_along((cx, cy), (px, py), chamfer_dist)
                chamfer_at[ci] = (a, b)

            poly_points = []
            for ci in range(4):
                cx, cy = corners[ci]
                if ci in chamfer_at:
                    a, b = chamfer_at[ci]
                    poly_points.append(a)
                    poly_points.append(b)
                else:
                    poly_points.append((cx, cy))

            if len(poly_points) >= 3:
                poly = Polygon(poly_points).buffer(0)
                if poly.is_valid and poly.area > 100:
                    room_polys.append(poly)
                else:
                    room_polys.append(box(rx1, ry1, rx2, ry2))
            else:
                room_polys.append(box(rx1, ry1, rx2, ry2))

            for ci, (a, b) in chamfer_at.items():
                dx = b[0] - a[0]
                dy = b[1] - a[1]
                edge_len = math.hypot(dx, dy)
                if edge_len < wall_t * 2:
                    continue
                nx, ny = -dy / edge_len, dx / edge_len
                cx, cy = corners[ci]
                room_cx = (rx1 + rx2) / 2.0
                room_cy = (ry1 + ry2) / 2.0
                to_center = math.atan2(room_cy - cy, room_cx - cx)
                normal_angle = math.atan2(ny, nx)
                angle_diff = (normal_angle - to_center) % (2 * math.pi)
                if angle_diff > math.pi:
                    nx, ny = -nx, -ny
                mid_a = (a[0] + nx * half_t, a[1] + ny * half_t)
                mid_b = (b[0] + nx * half_t, b[1] + ny * half_t)
                inner_wall_midlines.append((mid_a, mid_b))
        else:
            # Непрямоугольная комната (наклонные стены)
            poly = Polygon(room_data).buffer(0)
            if poly.is_valid and poly.area > 100:
                room_polys.append(poly)
            else:
                room_polys.append(box(room_data[0][0], room_data[0][1], room_data[2][0], room_data[2][1]))

    # --- Шаг 3: Стены (midlines) для расстановки проёмов ---
    # Строим edge -> список комнат
    edge_to_rooms = {}
    for i, room_data in enumerate(parts):
        for p1, p2 in _room_edges(room_data):
            key = _edge_key(p1, p2)
            if key not in edge_to_rooms:
                edge_to_rooms[key] = []
            edge_to_rooms[key].append(i)

    wall_midlines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    for key, room_indices in edge_to_rooms.items():
        count = len(room_indices)
        # Восстанавливаем концы ребра из любой комнаты
        ep1 = ep2 = None
        for idx in room_indices:
            for p1, p2 in _room_edges(parts[idx]):
                if _edge_key(p1, p2) == key:
                    ep1, ep2 = p1, p2
                    break
            if ep1 is not None:
                break
        if ep1 is None:
            continue

        if count >= 2:
            # Внутренняя стена (общая для 2+ комнат) — центральная линия на ребре
            wall_midlines.append((ep1, ep2))
        else:
            # Внешняя стена — смещаем наружу на half_t
            idx = room_indices[0]
            nx, ny = _normal_outward_for_room(parts[idx], ep1, ep2)
            mid_a = (ep1[0] + nx * half_t, ep1[1] + ny * half_t)
            mid_b = (ep2[0] + nx * half_t, ep2[1] + ny * half_t)
            wall_midlines.append((mid_a, mid_b))

    wall_midlines.extend(inner_wall_midlines)
    wall_midlines = [(a, b) for a, b in wall_midlines if dist(a, b) > 50]
    return room_polys, wall_midlines

# =============================================================================
# РАЗМЕЩЕНИЕ ДВЕРЕЙ И ОКОН
# =============================================================================
class Opening:
    def __init__(self, open_type: str, wall_idx: int, position: float,
                 width: float, p1: Tuple[float, float], p2: Tuple[float, float],
                 wall_p1: Tuple[float, float], wall_p2: Tuple[float, float]):
        self.type = open_type
        self.wall_idx = wall_idx
        self.position = position
        self.width = width
        self.p1 = p1
        self.p2 = p2
        self.wall_p1 = wall_p1
        self.wall_p2 = wall_p2

def place_openings(
    wall_midlines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    wall_t: float, num_openings: int
) -> List[Opening]:
    """Разместить двери/окна на стенах с проверкой пересечений через Shapely."""
    # Сортируем стены по длине (от длинных к коротким)
    wall_data = []
    for i, (p1, p2) in enumerate(wall_midlines):
        length = dist(p1, p2)
        wall_data.append((i, p1, p2, length))
    wall_data.sort(key=lambda x: x[3], reverse=True)

    # Вычисляем точки пересечений стен (T-образные / крестовые стыки)
    wall_lines = [LineString([p1, p2]) for p1, p2 in wall_midlines]
    junction_pts: List[Tuple[float, float]] = []
    for i, l1 in enumerate(wall_lines):
        for j, l2 in enumerate(wall_lines):
            if j <= i:
                continue
            inter = l1.intersection(l2)
            if inter.geom_type == "Point":
                junction_pts.append((inter.x, inter.y))
    min_junction_dist = wall_t * 1.5

    openings: List[Opening] = []
    door_placed = False

    def try_place(otype: str, wlen: float, wp1, wp2, idx: int,
                  attempt_idx: int) -> Optional[Opening]:
        """Попробовать разместить проём на стене. Возвращает Opening или None."""
        nonlocal door_placed
        ow = rand_door_w() if otype == "door" else rand_wind_w()
        if ow >= wlen * 0.85:
            return None
        # На коротких стенах уменьшаем отступ
        margin = min(ow * 0.6, wlen * 0.15)
        jm = max(margin, min_junction_dist)
        max_pos = wlen - ow - jm
        if max_pos <= jm:
            return None
        pos = random.uniform(jm, max_pos)
        op1 = point_along(wp1, wp2, pos)
        op2 = point_along(wp1, wp2, pos + ow)

        # Проверяем, что проём не на стыке стен
        mid_x = (op1[0] + op2[0]) / 2.0
        mid_y = (op1[1] + op2[1]) / 2.0
        for jx, jy in junction_pts:
            if math.hypot(mid_x - jx, mid_y - jy) < min_junction_dist:
                return None

        ol = LineString([op1, op2])
        for ex in openings:
            el = LineString([ex.p1, ex.p2])
            if ol.distance(el) < ow * 0.3:
                return None

        opening = Opening(otype, idx, pos, ow, op1, op2, wp1, wp2)
        if otype == "door":
            door_placed = True
        return opening

    # Основные попытки: случайный выбор из всех стен
    attempts = 0
    max_attempts = max(num_openings * 2, 10)
    while len(openings) < num_openings and attempts < max_attempts:
        attempts += 1
        # С вероятностью 70% выбираем любую стену, иначе самую длинную
        if random.random() < 0.7 or not wall_data:
            idx, wp1, wp2, wlen = random.choice(wall_data)
        else:
            idx, wp1, wp2, wlen = wall_data[0]

        if not door_placed:
            otype = "door"
        else:
            door_chance = DOOR_PROB / (DOOR_PROB + WINDOW_PROB)
            otype = "door" if random.random() < door_chance else "window"

        op = try_place(otype, wlen, wp1, wp2, idx, attempts)
        if op is not None:
            openings.append(op)

    # Если ни одного проёма не разместили — форсируем дверь на самой длинной стене
    if not openings and wall_data:
        idx, wp1, wp2, wlen = wall_data[0]
        for ow in sorted([rand_door_w() for _ in range(5)]):
            margin = min(ow * 0.3, wlen * 0.1)
            jm = max(margin, min_junction_dist * 0.5)
            max_pos = wlen - ow - jm
            if max_pos > jm:
                pos = (jm + max_pos) / 2.0
                op1 = point_along(wp1, wp2, pos)
                op2 = point_along(wp1, wp2, pos + ow)
                openings.append(Opening("door", idx, pos, ow, op1, op2, wp1, wp2))
                break

    return openings

# =============================================================================
# ВЫЧИСЛЕНИЕ ПОЛИГОНА СТЕН (GAP-FREE ЧЕРЕЗ BUFFER)
# =============================================================================
def compute_wall_polygon(
    rooms: List[Polygon], wall_t: float
) -> Polygon:
    """
    Вычислить единый полигон стен через buffer — гарантирует 
    замкнутость углов без зазоров.
    """
    if len(rooms) == 1:
        union = rooms[0]
    else:
        union = unary_union(rooms).buffer(0)
    outer = union.buffer(wall_t, join_style=2, mitre_limit=5.0)
    walls = outer.difference(union)
    return walls.buffer(0)

def compute_opening_rect(
    op: Opening, wall_t: float
) -> List[Tuple[float, float]]:
    """Прямоугольник проёма (поперёк стены, на всю толщину)."""
    dx = op.p2[0] - op.p1[0]
    dy = op.p2[1] - op.p1[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return []
    perp_x, perp_y = -dy / length, dx / length
    half_t = wall_t / 2.0
    return [
        (op.p1[0] + perp_x * half_t, op.p1[1] + perp_y * half_t),
        (op.p2[0] + perp_x * half_t, op.p2[1] + perp_y * half_t),
        (op.p2[0] - perp_x * half_t, op.p2[1] - perp_y * half_t),
        (op.p1[0] - perp_x * half_t, op.p1[1] - perp_y * half_t),
    ]


def door_clear(
    rooms: List[Polygon], op: Opening, wall_t: int,
    nx: float, ny: float, dw: float,
    drawn_swings: Optional[List[Polygon]] = None,
    outward: bool = False,
    room_union: Optional[Polygon] = None,
    wall_polygon: Optional[Polygon] = None,
) -> bool:
    """Проверить, что створка и дуга двери не пересекают стены и другие двери.
    outward=True — створка наружу (вне комнат);
    outward=False — створка внутрь (внутри комнаты).
    """
    leaf_end = (op.p1[0] + nx * dw, op.p1[1] + ny * dw)
    cx, cy = op.p1
    a0 = math.atan2(op.p2[1] - cy, op.p2[0] - cx)
    a1 = math.atan2(leaf_end[1] - cy, leaf_end[0] - cx)
    sweep = (a1 - a0) % (2 * math.pi)
    if sweep > math.pi:
        a0, a1 = a1, a0

    # Полигон створки (линия + дуга)
    num_steps = 24
    arc_points = [(cx + dw * math.cos(a0 + (a1 - a0) * i / num_steps),
                   cy + dw * math.sin(a0 + (a1 - a0) * i / num_steps))
                  for i in range(num_steps + 1)]
    swing = Polygon([op.p1, leaf_end] + arc_points)
    if not swing.is_valid:
        swing = swing.buffer(0)
    if swing.is_empty or swing.area < 1:
        return False

    # Площадь створки, которая пересекается с прямоугольником проёма (стеной),
    # вычитаем — она всегда накладывается на стену у петли
    opening_rect_pts = compute_opening_rect(op, wall_t)
    if opening_rect_pts and swing.area > 0:
        opening_poly = Polygon(opening_rect_pts).buffer(0)
        try:
            swing_clean = swing.difference(opening_poly).buffer(0)
        except Exception:
            swing_clean = swing.buffer(0)
        if swing_clean.is_empty or swing_clean.area < 1:
            swing_clean = swing
        # Если difference развалился на мультиполигон, берём самую большую часть
        if swing_clean.geom_type == "MultiPolygon":
            parts = sorted(swing_clean.geoms, key=lambda p: p.area, reverse=True)
            swing_clean = parts[0] if parts else swing
        if swing_clean.area < swing.area * 0.1:
            swing_clean = swing  # разность не удалась — считаем по полной
    else:
        swing_clean = swing

    # Проверка пересечения с комнатами
    if room_union is not None and swing_clean.area > 0:
        try:
            overlap = swing_clean.intersection(room_union).area
        except Exception:
            overlap = 0
        ratio = overlap / swing_clean.area
        if outward:
            if ratio > 0.15:
                return False
        else:
            if ratio < 0.85:
                return False

    # Проверка пересечения со стенами (створка не должна задевать стены)
    if wall_polygon is not None and swing_clean.area > 0:
        try:
            wall_overlap = swing_clean.intersection(wall_polygon).area
        except Exception:
            wall_overlap = 0
        if wall_overlap / swing_clean.area > 0.05:
            return False

    # Проверка пересечения с уже нарисованными створками
    if drawn_swings:
        for existing in drawn_swings:
            try:
                if swing.intersects(existing):
                    return False
            except Exception:
                return False

    return True


# =============================================================================
# ГЕНЕРАЦИЯ ОДНОГО ПЛАНА (КООРДИНАТЫ)
# =============================================================================
def generate_plan(
    canvas_w: int, canvas_h: int
) -> Tuple[List[Polygon], List[Opening], int, int]:
    """Сгенерировать один план: комнаты, проёмы, wall_t, num_rooms."""
    wall_t = rand_wall_t()
    num_rooms = random.randint(MIN_ROOMS, MAX_ROOMS)
    num_openings = random.randint(MIN_OPENINGS, MAX_OPENINGS)

    rooms, wall_midlines = generate_rooms(canvas_w, canvas_h, wall_t, num_rooms)
    openings = place_openings(wall_midlines, wall_t, num_openings)
    return rooms, openings, wall_t, num_rooms


# =============================================================================
# РИСОВАНИЕ ПЛАНА (ОСНОВНОЕ ИЗОБРАЖЕНИЕ)
# =============================================================================
def _clean_union(rooms):
    if len(rooms) == 1:
        return rooms[0]
    return unary_union(rooms).buffer(0)


def draw_outer_buffer(draw, rooms, wall_t, fill_color):
    """Нарисовать внешний буфер стен (без дыр — комнаты перерисовываются сверху)."""
    union = _clean_union(rooms)
    outer = union.buffer(wall_t, join_style=2, mitre_limit=5.0).buffer(0)
    draw_shapely_poly(draw, outer, fill=fill_color, outline=None)
    return outer


def draw_mask_outer_buffer(draw, rooms, wall_t):
    """Нарисовать внешний буфер стен для маски."""
    union = _clean_union(rooms)
    outer = union.buffer(wall_t, join_style=2, mitre_limit=5.0).buffer(0)
    draw_shapely_poly(draw, outer, fill=M_WALL, outline=None)
    return outer


def draw_plan(
    img: Image.Image,
    rooms: List[Polygon],
    openings: List[Opening],
    wall_t: int,
) -> None:
    """Нарисовать основной план (цвета BG/WALL/ROOM/DOOR)."""
    draw = ImageDraw.Draw(img)

    # Случайный цвет заливки стен для этого изображения
    wall_brightness = random.randint(WALL_FILL_RANGE[0], WALL_FILL_RANGE[1])
    wall_fill = (wall_brightness, wall_brightness, wall_brightness)

    # 1. Заливка комнат
    for room in rooms:
        draw_shapely_poly(draw, room, fill=ROOM_FILL, outline=None)

    # 2. Стены (через внешний буфер — комнаты временно закрашиваются)
    outer = draw_outer_buffer(draw, rooms, wall_t, wall_fill)

    # 3. Перерисовка комнат (прорезаем дыры в стенах)
    for room in rooms:
        draw_shapely_poly(draw, room, fill=ROOM_FILL, outline=None)

    # 4. Двери/окна — вырезаем прямоугольники поверх стен
    for op in openings:
        rect = compute_opening_rect(op, wall_t)
        if len(rect) < 4:
            continue
        draw.polygon(rect, fill=ROOM_FILL)

    # 5. Обводки комнат
    for room in rooms:
        draw_shapely_poly(draw, room, fill=None, outline=WALL_LINE, width=2)

    # 6. Обводка внешнего периметра стен
    if outer is not None and not outer.is_empty:
        draw_shapely_poly(draw, outer, fill=None, outline=WALL_LINE, width=1)

    # 7. Проёмы поверх обводок (чтобы линии не перечеркивали проём)
    for op in openings:
        rect = compute_opening_rect(op, wall_t)
        if len(rect) < 4:
            continue
        draw.polygon(rect, fill=ROOM_FILL)

    # 8. Дверные дуги и линии
    drawn_swings: List[Polygon] = []
    room_union = _clean_union(rooms)
    wall_polygon = compute_wall_polygon(rooms, wall_t)
    for op in openings:
        if op.type == "door":
            wx = op.wall_p2[0] - op.wall_p1[0]
            wy = op.wall_p2[1] - op.wall_p1[1]
            wlen = math.hypot(wx, wy)
            if wlen < 1:
                continue
            nx = -wy / wlen
            ny = wx / wlen
            dw = math.hypot(op.p2[0] - op.p1[0], op.p2[1] - op.p1[1])
            if dw < 1:
                continue
            # Выбираем направление и сторону створки (внутрь/наружу)
            modes = [False, True]
            random.shuffle(modes)
            chosen_nx, chosen_ny = nx, ny
            fits = False
            for outward in modes:
                for dxn, dyn in [(nx, ny), (-nx, -ny)]:
                    if door_clear(rooms, op, wall_t, dxn, dyn, dw, drawn_swings, outward, room_union, wall_polygon):
                        chosen_nx, chosen_ny = dxn, dyn
                        fits = True
                        break
                if fits:
                    break
                if fits:
                    break
            if not fits:
                continue  # пропускаем — налетает на стену/другую дверь
            nx, ny = chosen_nx, chosen_ny
            # Точка створки (конец линии, перпендикулярной стене)
            swing = dw
            leaf_end = (int(op.p1[0] + nx * swing), int(op.p1[1] + ny * swing))
            leaf_end = (max(0, min(leaf_end[0], img.width)), max(0, min(leaf_end[1], img.height)))
            # Линия створки от края проёма (петля) под 90° к стене
            draw.line([(int(op.p1[0]), int(op.p1[1])), leaf_end], fill=DOOR_COL, width=max(2, wall_t // DOOR_LINE_DIV))
            # Дуга 90° от другого края проёма до конца створки
            r = int(dw)
            bbox = (int(op.p1[0] - r), int(op.p1[1] - r), int(op.p1[0] + r), int(op.p1[1] + r))
            ang_start = math.degrees(math.atan2(op.p2[1] - op.p1[1], op.p2[0] - op.p1[0]))
            ang_end = math.degrees(math.atan2(leaf_end[1] - op.p1[1], leaf_end[0] - op.p1[0]))
            sweep_d = (ang_end - ang_start) % 360
            if sweep_d > 180:
                ang_start, ang_end = ang_end, ang_start
            draw.arc(bbox, ang_start, ang_end, fill=DOOR_COL, width=max(2, wall_t // (DOOR_LINE_DIV + 1)))
            # Сохраняем полигон створки для проверки пересечений
            cx, cy = op.p1
            a0 = math.atan2(op.p2[1] - cy, op.p2[0] - cx)
            a1 = math.atan2(leaf_end[1] - cy, leaf_end[0] - cx)
            sweep_a = (a1 - a0) % (2 * math.pi)
            if sweep_a > math.pi:
                a0, a1 = a1, a0
            pts = [(cx + dw * math.cos(a0 + (a1 - a0) * i / 24),
                    cy + dw * math.sin(a0 + (a1 - a0) * i / 24))
                   for i in range(25)]
            drawn_swings.append(Polygon([op.p1, leaf_end] + pts))
        else:
            # Окно — две параллельные линии (всегда внутри толщины стены)
            dx, dy = op.p2[0] - op.p1[0], op.p2[1] - op.p1[1]
            length = math.hypot(dx, dy)
            if length < 1:
                continue
            nx, ny = -dy / length, dx / length
            half_spread = wall_t * 0.3
            line_w = max(1, wall_t // WINDOW_LINE_DIV)
            margin = 1  # запас на погрешность растеризации
            max_shift = wall_t * 0.5 - margin - line_w * 0.5 - half_spread
            if max_shift < 0:
                max_shift = 0
            raw_shift = random.uniform(WINDOW_SHIFT_RANGE[0], WINDOW_SHIFT_RANGE[1]) * wall_t
            shift = max(-max_shift, min(raw_shift, max_shift))
            e1 = (op.p1[0] + nx * (half_spread + shift), op.p1[1] + ny * (half_spread + shift))
            e2 = (op.p2[0] + nx * (half_spread + shift), op.p2[1] + ny * (half_spread + shift))
            draw.line([e1, e2], fill=WIND_COL, width=line_w)
            i1 = (op.p1[0] + nx * (-half_spread + shift), op.p1[1] + ny * (-half_spread + shift))
            i2 = (op.p2[0] + nx * (-half_spread + shift), op.p2[1] + ny * (-half_spread + shift))
            draw.line([i1, i2], fill=WIND_COL, width=line_w)


# =============================================================================
# РИСОВАНИЕ МАСКИ (СЕМАНТИЧЕСКАЯ СЕГМЕНТАЦИЯ)
# =============================================================================
def draw_mask(
    mask: Image.Image,
    rooms: List[Polygon],
    openings: List[Opening],
    wall_t: int,
) -> None:
    """Нарисовать маску (цвета M_ROOM/M_WALL/M_DOOR/M_DOORW/M_WIND/M_WINDW)."""
    draw = ImageDraw.Draw(mask)

    # 1. Заливка комнат
    for room in rooms:
        draw_shapely_poly(draw, room, fill=M_ROOM, outline=None)

    # 2. Стены (через внешний буфер — комнаты временно закрашиваются)
    draw_mask_outer_buffer(draw, rooms, wall_t)

    # 3. Перерисовка комнат (прорезаем дыры в стенах)
    for room in rooms:
        draw_shapely_poly(draw, room, fill=M_ROOM, outline=None)

    # 4. Проёмы поверх стен
    drawn_swings: List[Polygon] = []
    room_union = _clean_union(rooms)
    wall_polygon = compute_wall_polygon(rooms, wall_t)
    for op in openings:
        rect = compute_opening_rect(op, wall_t)
        if len(rect) < 4:
            continue
        if op.type == "door":
            draw.polygon(rect, fill=M_DOORW)
            wx = op.wall_p2[0] - op.wall_p1[0]
            wy = op.wall_p2[1] - op.wall_p1[1]
            wlen = math.hypot(wx, wy)
            if wlen >= 1:
                nx = -wy / wlen
                ny = wx / wlen
                dw = math.hypot(op.p2[0] - op.p1[0], op.p2[1] - op.p1[1])
                if dw >= 1:
                    modes = [False, True]
                    random.shuffle(modes)
                    chosen_nx, chosen_ny = nx, ny
                    fits = False
                    for outward in modes:
                        for dxn, dyn in [(nx, ny), (-nx, -ny)]:
                            if door_clear(rooms, op, wall_t, dxn, dyn, dw, drawn_swings, outward, room_union, wall_polygon):
                                chosen_nx, chosen_ny = dxn, dyn
                                fits = True
                                break
                        if fits:
                            break
                        if fits:
                            break
                    if not fits:
                        continue
                    nx, ny = chosen_nx, chosen_ny
                    swing = dw
                    leaf_end = (op.p1[0] + nx * swing, op.p1[1] + ny * swing)
                    leaf_end = (max(0, min(leaf_end[0], mask.width)), max(0, min(leaf_end[1], mask.height)))
                    # Линия створки
                    draw.line([(op.p1[0], op.p1[1]), leaf_end], fill=M_DOOR, width=max(2, wall_t // DOOR_LINE_DIV))
                    # Дуга 90°
                    r = int(dw)
                    bbox = (int(op.p1[0] - r), int(op.p1[1] - r), int(op.p1[0] + r), int(op.p1[1] + r))
                    ang_start = math.degrees(math.atan2(op.p2[1] - op.p1[1], op.p2[0] - op.p1[0]))
                    ang_end = math.degrees(math.atan2(leaf_end[1] - op.p1[1], leaf_end[0] - op.p1[0]))
                    sweep_d = (ang_end - ang_start) % 360
                    if sweep_d > 180:
                        ang_start, ang_end = ang_end, ang_start
                    draw.arc(bbox, ang_start, ang_end, fill=M_DOOR, width=max(2, wall_t // (DOOR_LINE_DIV + 1)))
                    # Сохраняем полигон створки для проверки пересечений
                    cx, cy = op.p1
                    a0 = math.atan2(op.p2[1] - cy, op.p2[0] - cx)
                    a1 = math.atan2(leaf_end[1] - cy, leaf_end[0] - cx)
                    sweep_a = (a1 - a0) % (2 * math.pi)
                    if sweep_a > math.pi:
                        a0, a1 = a1, a0
                    pts = [(cx + dw * math.cos(a0 + (a1 - a0) * i / 24),
                            cy + dw * math.sin(a0 + (a1 - a0) * i / 24))
                           for i in range(25)]
                    drawn_swings.append(Polygon([op.p1, leaf_end] + pts))
        else:
            draw.polygon(rect, fill=M_WINDW)
            # Две поперечные линии окна (всегда внутри толщины стены)
            wx = op.wall_p2[0] - op.wall_p1[0]
            wy = op.wall_p2[1] - op.wall_p1[1]
            wlen = math.hypot(wx, wy)
            if wlen >= 1:
                nx = -wy / wlen
                ny = wx / wlen
                half_spread = wall_t * 0.3
                line_w = max(1, wall_t // WINDOW_LINE_DIV)
                margin = 1
                max_shift = wall_t * 0.5 - margin - line_w * 0.5 - half_spread
                if max_shift < 0:
                    max_shift = 0
                raw_shift = random.uniform(WINDOW_SHIFT_RANGE[0], WINDOW_SHIFT_RANGE[1]) * wall_t
                shift = max(-max_shift, min(raw_shift, max_shift))
                e1 = (op.p1[0] + nx * (half_spread + shift), op.p1[1] + ny * (half_spread + shift))
                e2 = (op.p2[0] + nx * (half_spread + shift), op.p2[1] + ny * (half_spread + shift))
                draw.line([e1, e2], fill=M_WIND, width=line_w)
                i1 = (op.p1[0] + nx * (-half_spread + shift), op.p1[1] + ny * (-half_spread + shift))
                i2 = (op.p2[0] + nx * (-half_spread + shift), op.p2[1] + ny * (-half_spread + shift))
                draw.line([i1, i2], fill=M_WIND, width=line_w)


# =============================================================================
# АУГМЕНТАЦИЯ (ИМИТАЦИЯ СКАНИРОВАНИЯ) ЧЕРЕЗ AUGRAPHY
# =============================================================================

def create_augmentation_pipeline() -> Optional[object]:
    """Создать pipeline Augraphy для имитации артефактов сканирования."""
    if not USE_AUGMENTATIONS:
        return None
    try:
        from augraphy import (
            AugraphyPipeline,
            SubtleNoise, LowLightNoise,
            DirtyScreen, DirtyDrum, Dithering,
            Brightness, Jpeg,
            ColorPaper, Folding, InkBleed,
        )
        pipeline = AugraphyPipeline([
            SubtleNoise(subtle_range=5, p=0.5),
            LowLightNoise(p=0.6),
            DirtyScreen(p=0.2),
            DirtyDrum(line_width_range=(1, 2), p=0.3),
            Dithering(p=0.15),
            InkBleed(p=0.2),
            Brightness(brightness_range=(0.88, 1.12), p=0.7),
            Jpeg(quality_range=(65, 92), p=0.75),
            ColorPaper(p=0.15),
            Folding(p=0.15),
        ])
        return pipeline
    except ImportError:
        print("[WARNING] Augraphy не установлен. Установите: pip install augraphy")
        return None


def apply_scanning_artifacts(
    image: Image.Image,
    pipeline: object,
) -> Tuple[Image.Image, Tuple[str, ...]]:
    """
    Применить pipeline аугментаций к изображению.
    Возвращает (изображение, кортеж имён применившихся аугментаций).
    """
    if pipeline is None or random.random() > AUGMENTATION_PROB:
        return image, ()
    import numpy as np
    img_np = np.array(image)
    result = pipeline.augment(img_np)
    # Извлекаем названия сработавших аугментаций
    log = result.get("log", {})
    names = log.get("augmentation_name", [])
    statuses = log.get("augmentation_status", [])
    applied = tuple(n for n, s in zip(names, statuses) if s)
    return Image.fromarray(result["output"]), applied


# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Генератор синтетического датасета 2D планов помещений")
    parser.add_argument("-n", "--num-images", type=int, default=NUM_IMAGES,
                        help=f"Количество изображений (по умолч. {NUM_IMAGES})")
    parser.add_argument("-o", "--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Директория вывода (по умолч. '{OUTPUT_DIR}')")
    parser.add_argument("--seed", type=int, default=None,
                        help="Сид для воспроизводимости")
    return parser.parse_args()


def main():
    args = parse_args()
    num_images = args.num_images
    output_dir = args.output_dir
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Pipeline аугментации (создаётся один раз)
    pipeline = create_augmentation_pipeline()

    # Определяем следующий индекс, чтобы не перезаписывать существующие файлы
    existing = [f for f in os.listdir(images_dir) if f.startswith("plan_") and f.endswith(".png")]
    max_idx = -1
    for f in existing:
        try:
            num = int(f.split("_")[1])  # plan_000123_XYZ.png -> 000123 -> 123
            if num > max_idx:
                max_idx = num
        except (ValueError, IndexError):
            pass
    start_idx = max_idx + 1

    print(f"Генерация {num_images} изображений...")
    print(f"  -> Изображения: {images_dir}  (пропущено {start_idx} существующих)")
    print(f"  -> Маски:      {masks_dir}")
    print(f"  -> Аугментация: {'вкл' if pipeline else 'выкл'}")

    for i in tqdm(range(num_images), desc="Планы", unit="img"):
        idx = start_idx + i
        canvas_w = random.randint(MIN_CANVAS, MAX_CANVAS)
        canvas_h = random.randint(MIN_CANVAS, MAX_CANVAS)

        rooms, openings, wall_t, num_rooms = generate_plan(canvas_w, canvas_h)

        # Основное изображение
        img = Image.new("RGB", (canvas_w, canvas_h), BG)
        draw_plan(img, rooms, openings, wall_t)

        # Маска (чистая, без аугментаций)
        mask = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
        draw_mask(mask, rooms, openings, wall_t)

        # Применяем артефакты сканирования только к изображению
        img, augs = apply_scanning_artifacts(img, pipeline)

        # Суффикс из названий сработавших аугментаций (подчёркивания в именах заменяем на дефисы)
        aug_suffix = "_" + "_".join(a.replace("_", "-") for a in augs) if augs else "_clean"

        img.save(os.path.join(images_dir, f"plan_{idx:06d}{aug_suffix}.png"))
        mask.save(os.path.join(masks_dir, f"mask_{idx:06d}{aug_suffix}.png"))


if __name__ == "__main__":
    main()
