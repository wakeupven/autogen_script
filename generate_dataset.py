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

from PIL import Image, ImageDraw, ImageFont
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

# Параметры простановки размеров стен
SCALE_MM_PER_PX = 10         # 1 пиксель = 10 мм
DIM_OFFSET = 35              # отступ размерной линии от края стены (px)
DIM_TICK_SIZE = 8            # размер засечки (px)
DIM_TEXT_SIZE = 14           # размер шрифта текста размеров
DIM_MIN_LENGTH = 30          # мин. длина стены для простановки (px)

# Параметры штриховки несущих стен
HATCH_LINE_WIDTH = 1         # толщина линии штриховки (px)
HATCH_COLOR = (60, 60, 60)   # цвет штриховки на плане (тёмно-серый)
LOAD_BEARING_INTERIOR_PROB = 0.3  # вероятность несущей для внутренней стены

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
M_DIM = (255, 255, 255)  # размерные элементы в маске
M_WALL_HATCH = (255, 128, 0)  # несущие стены в маске (оранжевый)

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
# ГЕНЕРАЦИЯ ГЕОМЕТРИИ ПОМЕЩЕНИЙ
# =============================================================================
def generate_rooms(
    canvas_w: int, canvas_h: int, wall_t: int, num_rooms: int
) -> Tuple[List[Polygon], List[Tuple[Tuple[float, float], Tuple[float, float]]], List[dict]]:
    """
    Сгенерировать комнаты (interior), стены (midlines) и инфо о внешних стенах для размеров.
    Возвращает (room_polygons, wall_midlines, wall_info).
    """
    margin = CANVAS_MARGIN
    half_t = wall_t / 2.0
    bb_x1, bb_y1 = margin, margin
    bb_x2, bb_y2 = canvas_w - margin, canvas_h - margin

    if bb_x2 - bb_x1 < 300 or bb_y2 - bb_y1 < 300:
        bb_x1, bb_y1 = 100, 100
        bb_x2, bb_y2 = canvas_w - 100, canvas_h - 100

    min_room = 150

    parts: List[Tuple[float, float, float, float]] = [
        (bb_x1 + half_t, bb_y1 + half_t, bb_x2 - half_t, bb_y2 - half_t)
    ]

    while len(parts) < num_rooms:
        candidates = []
        for i, (rx1, ry1, rx2, ry2) in enumerate(parts):
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
    rooms = [box(rx1, ry1, rx2, ry2) for rx1, ry1, rx2, ry2 in parts]

    # Стены (midlines) для расстановки проёмов
    edges: dict = {}
    for rx1, ry1, rx2, ry2 in parts:
        edge_list = [
            ("h", rx1, ry1, rx2, ry1),
            ("h", rx1, ry2, rx2, ry2),
            ("v", rx1, ry1, rx1, ry2),
            ("v", rx2, ry1, rx2, ry2),
        ]
        for etype, ex1, ey1, ex2, ey2 in edge_list:
            key = (etype, round(min(ex1, ex2), 6), round(min(ey1, ey2), 6),
                   round(max(ex1, ex2), 6), round(max(ey1, ey2), 6))
            edges[key] = edges.get(key, 0) + 1

    wall_midlines = []
    wall_info = []
    for (etype, ex1, ey1, ex2, ey2), count in edges.items():
        ext = wall_t
        if etype == "h":
            if count >= 2:
                midline = ((ex1 - ext, ey1), (ex2 + ext, ey1))
                wall_midlines.append(midline)
                wall_info.append({"midline": midline, "normal": (0, -1)})
            else:
                is_top = any(abs(ey1 - ry1) < 1 for _, ry1, _, _ in parts)
                if is_top:
                    midline = ((ex1 - ext, ey1 - half_t), (ex2 + ext, ey1 - half_t))
                    normal = (0, -1)
                else:
                    midline = ((ex1 - ext, ey1 + half_t), (ex2 + ext, ey1 + half_t))
                    normal = (0, 1)
                wall_midlines.append(midline)
                wall_info.append({"midline": midline, "normal": normal})
        else:
            if count >= 2:
                midline = ((ex1, ey1 - ext), (ex2, ey2 + ext))
                wall_midlines.append(midline)
                wall_info.append({"midline": midline, "normal": (-1, 0)})
            else:
                is_left = any(abs(ex1 - rx1) < 1 for rx1, _, _, _ in parts)
                if is_left:
                    midline = ((ex1 - half_t, ey1 - ext), (ex1 - half_t, ey2 + ext))
                    normal = (-1, 0)
                else:
                    midline = ((ex1 + half_t, ey1 - ext), (ex1 + half_t, ey2 + ext))
                    normal = (1, 0)
                wall_midlines.append(midline)
                wall_info.append({"midline": midline, "normal": normal})

    wall_midlines = [(a, b) for a, b in wall_midlines if dist(a, b) > 50]
    wall_info = [w for w in wall_info if dist(w["midline"][0], w["midline"][1]) > 50]
    return rooms, wall_midlines, wall_info

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
        union = unary_union(rooms)
    outer = union.buffer(wall_t, join_style=2, mitre_limit=5.0)
    walls = outer.difference(union)
    return walls

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

    # Площадь створки, которая пересекается с прямоугольником проёма (стеной),
    # вычитаем — она всегда накладывается на стену у петли
    opening_rect_pts = compute_opening_rect(op, wall_t)
    if opening_rect_pts and swing.area > 0:
        opening_poly = Polygon(opening_rect_pts)
        swing_clean = swing.difference(opening_poly)
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
        overlap = swing_clean.intersection(room_union).area
        ratio = overlap / swing_clean.area
        if outward:
            if ratio > 0.15:
                return False
        else:
            if ratio < 0.85:
                return False

    # Проверка пересечения со стенами (створка не должна задевать стены)
    if wall_polygon is not None and swing_clean.area > 0:
        wall_overlap = swing_clean.intersection(wall_polygon).area
        if wall_overlap / swing_clean.area > 0.05:
            return False

    # Проверка пересечения с уже нарисованными створками
    if drawn_swings:
        for existing in drawn_swings:
            if swing.intersects(existing):
                return False

    return True

    return True


# =============================================================================
# ГЕНЕРАЦИЯ ОДНОГО ПЛАНА (КООРДИНАТЫ)
# =============================================================================
def generate_plan(
    canvas_w: int, canvas_h: int
) -> Tuple[List[Polygon], List[Opening], List[Tuple[Tuple[float, float], Tuple[float, float]]], List[dict], int, int]:
    """Сгенерировать один план: комнаты, проёмы, wall_midlines, wall_info, wall_t, num_rooms."""
    wall_t = rand_wall_t()
    num_rooms = random.randint(MIN_ROOMS, MAX_ROOMS)
    num_openings = random.randint(MIN_OPENINGS, MAX_OPENINGS)

    rooms, wall_midlines, wall_info = generate_rooms(canvas_w, canvas_h, wall_t, num_rooms)
    openings = place_openings(wall_midlines, wall_t, num_openings)
    return rooms, openings, wall_midlines, wall_info, wall_t, num_rooms


# =============================================================================
# РИСОВАНИЕ ПЛАНА (ОСНОВНОЕ ИЗОБРАЖЕНИЕ)
# =============================================================================
def draw_outer_buffer(draw, rooms, wall_t, fill_color):
    """Нарисовать внешний буфер стен (без дыр — комнаты перерисовываются сверху)."""
    if len(rooms) == 1:
        union = rooms[0]
    else:
        union = unary_union(rooms)
    outer = union.buffer(wall_t, join_style=2, mitre_limit=5.0)
    draw_shapely_poly(draw, outer, fill=fill_color, outline=None)
    return outer


def draw_mask_outer_buffer(draw, rooms, wall_t):
    """Нарисовать внешний буфер стен для маски."""
    if len(rooms) == 1:
        union = rooms[0]
    else:
        union = unary_union(rooms)
    outer = union.buffer(wall_t, join_style=2, mitre_limit=5.0)
    draw_shapely_poly(draw, outer, fill=M_WALL, outline=None)
    return outer


# =============================================================================
# ПРОСТАНОВКА РАЗМЕРОВ СТЕН
# =============================================================================
try:
    _dim_font = ImageFont.truetype("arial.ttf", DIM_TEXT_SIZE)
except IOError:
    try:
        _dim_font = ImageFont.truetype("DejaVuSans.ttf", DIM_TEXT_SIZE)
    except IOError:
        _dim_font = ImageFont.load_default()


def draw_dimensions(
    draw: ImageDraw.ImageDraw,
    wall_info: List[dict],
    wall_t: int,
    scale_mm_per_px: float,
    line_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int],
    img_size: Tuple[int, int],
) -> None:
    """Нарисовать размерные линии с засечками и текст длины для внешних стен."""
    for w in wall_info:
        (x1, y1), (x2, y2) = w["midline"]
        nx, ny = w["normal"]
        length_px = dist((x1, y1), (x2, y2))
        if length_px < DIM_MIN_LENGTH:
            continue
        # Длина стены в мм
        length_mm = round(length_px * scale_mm_per_px)

        # Смещение размерной линии наружу
        offset = wall_t / 2.0 + DIM_OFFSET
        # Выносные линии от концов стены перпендикулярно наружу
        ex1 = x1 + nx * offset
        ey1 = y1 + ny * offset
        ex2 = x2 + nx * offset
        ey2 = y2 + ny * offset

        # Ограничиваем координаты в пределах изображения
        margin = 5
        ex1 = max(margin, min(ex1, img_size[0] - margin))
        ey1 = max(margin, min(ey1, img_size[1] - margin))
        ex2 = max(margin, min(ex2, img_size[0] - margin))
        ey2 = max(margin, min(ey2, img_size[1] - margin))

        # Выносные линии (от концов стены до размерной линии)
        draw.line([(int(x1), int(y1)), (int(ex1), int(ey1))], fill=line_color, width=1)
        draw.line([(int(x2), int(y2)), (int(ex2), int(ey2))], fill=line_color, width=1)

        # Размерная линия между выносными
        draw.line([(int(ex1), int(ey1)), (int(ex2), int(ey2))], fill=line_color, width=1)

        # Направление размерной линии (единичный вектор)
        dx = ex2 - ex1
        dy = ey2 - ey1
        dl = math.hypot(dx, dy)
        if dl < 1:
            continue
        ux, uy = dx / dl, dy / dl

        # Перпендикуляр для засечек
        px, py = -uy, ux
        half_tick = DIM_TICK_SIZE / 2.0

        # Засечки на концах размерной линии (перпендикулярные штрихи)
        draw.line([(int(ex1 - px * half_tick), int(ey1 - py * half_tick)),
                   (int(ex1 + px * half_tick), int(ey1 + py * half_tick))],
                  fill=line_color, width=1)
        draw.line([(int(ex2 - px * half_tick), int(ey2 - py * half_tick)),
                   (int(ex2 + px * half_tick), int(ey2 + py * half_tick))],
                  fill=line_color, width=1)

        # Текст размера над размерной линией
        text = str(length_mm)
        bbox = draw.textbbox((0, 0), text, font=_dim_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Центр размерной линии + смещение вдоль normal для текста
        cx = (ex1 + ex2) / 2.0
        cy = (ey1 + ey2) / 2.0
        text_offset = DIM_TICK_SIZE * 0.5 + 2
        tx = cx + nx * text_offset - tw / 2.0
        ty = cy + ny * text_offset - th / 2.0
        draw.text((int(tx), int(ty)), text, fill=text_color, font=_dim_font)


# =============================================================================
# ШТРИХОВКА НЕСУЩИХ СТЕН
# =============================================================================
def classify_walls(
    wall_info: List[dict],
    rooms: List[Polygon],
    wall_t: float,
) -> None:
    """Классифицировать стены на внешние (периметр) и внутренние.
    Добавляет ключ 'is_exterior' в каждый словарь wall_info.
    Внешняя = нормаль выходит за пределы union комнат, buffered на wall_t.
    """
    if len(rooms) == 1:
        room_union = rooms[0]
    else:
        room_union = unary_union(rooms)
    outer = room_union.buffer(wall_t, join_style=2, mitre_limit=5.0)

    for w in wall_info:
        (x1, y1), (x2, y2) = w["midline"]
        nx, ny = w["normal"]
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0

        # Точка на расстоянии wall_t вдоль normal от средней линии
        test = Point(mx + nx * wall_t, my + ny * wall_t)

        # Если точка снаружи outer (комнаты + стены) — стена внешняя
        w["is_exterior"] = not outer.covers(test)


def get_load_bearing_regions(
    rooms: List[Polygon],
    wall_info: List[dict],
    wall_t: float,
) -> Tuple[List[Polygon], List[Polygon]]:
    """Определить несущие стены и их регионы.
    Возвращает (load_bearing_regions, non_load_bearing_regions) — списки полигонов.
    """
    from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon

    # Сначала классифицируем стены геометрически
    classify_walls(wall_info, rooms, wall_t)

    wall_poly = compute_wall_polygon(rooms, wall_t)
    half_t = wall_t / 2.0

    lb_regions: List[Polygon] = []
    nlb_regions: List[Polygon] = []

    for w in wall_info:
        (x1, y1), (x2, y2) = w["midline"]
        nx, ny = w["normal"]
        is_ext = w["is_exterior"]

        is_lb = is_ext or (not is_ext and random.random() < LOAD_BEARING_INTERIOR_PROB)

        # Прямоугольник стены: midline + wall_t/2 в обе стороны вдоль normal
        pts = [
            (x1 + nx * half_t, y1 + ny * half_t),
            (x2 + nx * half_t, y2 + ny * half_t),
            (x2 - nx * half_t, y2 - ny * half_t),
            (x1 - nx * half_t, y1 - ny * half_t),
        ]
        seg_poly = ShapelyPolygon(pts)
        clipped = seg_poly.intersection(wall_poly)

        if clipped.is_empty:
            continue
        if clipped.geom_type == "MultiPolygon":
            for part in clipped.geoms:
                if is_lb:
                    lb_regions.append(part)
                else:
                    nlb_regions.append(part)
        elif clipped.geom_type == "Polygon":
            if is_lb:
                lb_regions.append(clipped)
            else:
                nlb_regions.append(clipped)

    return lb_regions, nlb_regions


def draw_hatching(
    draw: ImageDraw.ImageDraw,
    regions: List[Polygon],
    wall_t: float,
    color: Tuple[int, int, int],
    line_width: int,
) -> None:
    """Нарисовать диагональную штриховку (45°) для заданных регионов стен."""
    from shapely.geometry import LineString, MultiLineString

    inv_sqrt2 = 0.7071067811865475

    for poly in regions:
        minx, miny, maxx, maxy = poly.bounds
        w = maxx - minx
        h = maxy - miny
        step = max(4, wall_t * 0.3)
        if w < 1 and h < 1:
            continue

        # Диагональ bbox — достаточно, чтобы линия гарантированно пересекла полигон
        diag = math.sqrt(w * w + h * h)
        if diag < 1:
            continue
        span = diag * 2.0  # запас, чтобы линия выходила за пределы полигона

        num_lines = int(diag / step) + 2

        for i in range(num_lines):
            perp = (i - num_lines * 0.5) * step
            cx = (minx + maxx) / 2.0 + perp * (-inv_sqrt2)
            cy = (miny + maxy) / 2.0 + perp * inv_sqrt2

            # Линия вдоль направления (1,1) с центром в (cx, cy)
            p1 = (cx - span * inv_sqrt2, cy - span * inv_sqrt2)
            p2 = (cx + span * inv_sqrt2, cy + span * inv_sqrt2)

            line = LineString([p1, p2])
            clipped = poly.intersection(line)

            if clipped.is_empty:
                continue

            segments = []
            if clipped.geom_type == "LineString":
                segments = [list(clipped.coords)]
            elif clipped.geom_type == "MultiLineString":
                segments = [list(ls.coords) for ls in clipped.geoms]

            for coords in segments:
                for j in range(len(coords) - 1):
                    draw.line([(int(coords[j][0]), int(coords[j][1])),
                               (int(coords[j+1][0]), int(coords[j+1][1]))],
                              fill=color, width=line_width)


def draw_plan(
    img: Image.Image,
    rooms: List[Polygon],
    openings: List[Opening],
    wall_t: int,
    wall_info: List[dict],
    lb_regions: List[Polygon],
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

    # 2.5 Штриховка несущих стен (поверх стен, обрезается комнатами на шаге 3)
    draw_hatching(draw, lb_regions, wall_t, HATCH_COLOR, HATCH_LINE_WIDTH)

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
    room_union = unary_union(rooms) if len(rooms) > 1 else rooms[0]
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

    # 9. Размеры стен
    draw_dimensions(draw, wall_info, wall_t, SCALE_MM_PER_PX, (0, 0, 0), (0, 0, 0), img.size)


# =============================================================================
# РИСОВАНИЕ МАСКИ (СЕМАНТИЧЕСКАЯ СЕГМЕНТАЦИЯ)
# =============================================================================
def draw_mask(
    mask: Image.Image,
    rooms: List[Polygon],
    openings: List[Opening],
    wall_t: int,
    wall_info: List[dict],
    lb_regions: List[Polygon],
) -> None:
    """Нарисовать маску (цвета M_ROOM/M_WALL/M_WALL_HATCH/M_DOOR/M_DOORW/M_WIND/M_WINDW)."""
    draw = ImageDraw.Draw(mask)

    # 1. Заливка комнат
    for room in rooms:
        draw_shapely_poly(draw, room, fill=M_ROOM, outline=None)

    # 2. Стены (через внешний буфер — комнаты временно закрашиваются)
    draw_mask_outer_buffer(draw, rooms, wall_t)

    # 3. Перерисовка комнат (прорезаем дыры в стенах)
    for room in rooms:
        draw_shapely_poly(draw, room, fill=M_ROOM, outline=None)

    # 3.5 Несущие стены — перерисовка оранжевым (M_WALL_HATCH) поверх жёлтого M_WALL
    for region in lb_regions:
        draw_shapely_poly(draw, region, fill=M_WALL_HATCH, outline=None)

    # 4. Проёмы поверх стен
    drawn_swings: List[Polygon] = []
    room_union = unary_union(rooms) if len(rooms) > 1 else rooms[0]
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

    # 5. Размеры стен в маске
    draw_dimensions(draw, wall_info, wall_t, SCALE_MM_PER_PX, M_DIM, M_DIM, mask.size)


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

        rooms, openings, wall_midlines, wall_info, wall_t, num_rooms = generate_plan(canvas_w, canvas_h)

        # Регионы несущих стен (один раз для плана и маски)
        lb_regions, _ = get_load_bearing_regions(rooms, wall_info, wall_t)

        # Основное изображение
        img = Image.new("RGB", (canvas_w, canvas_h), BG)
        draw_plan(img, rooms, openings, wall_t, wall_info, lb_regions)

        # Маска (чистая, без аугментаций)
        mask = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
        draw_mask(mask, rooms, openings, wall_t, wall_info, lb_regions)

        # Применяем артефакты сканирования только к изображению
        img, augs = apply_scanning_artifacts(img, pipeline)

        # Суффикс из названий сработавших аугментаций (подчёркивания в именах заменяем на дефисы)
        aug_suffix = "_" + "_".join(a.replace("_", "-") for a in augs) if augs else "_clean"

        img.save(os.path.join(images_dir, f"plan_{idx:06d}{aug_suffix}.png"))
        mask.save(os.path.join(masks_dir, f"mask_{idx:06d}{aug_suffix}.png"))


if __name__ == "__main__":
    main()
