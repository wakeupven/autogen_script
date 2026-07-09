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
from typing import List, Tuple, Optional, Dict, Set

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
DOOR_DIM_EXTRA_MARGIN = 20   # доп. отступ размерной линии, если дверь открывается в её сторону

# Параметры маркировки помещений (ГОСТ 21.501-2011)
ROOM_LABEL_CIRCLE_R = 28          # базовый радиус окружности вокруг номера (px)
ROOM_LABEL_CIRCLE_R_RANGE = 3     # диапазон рандомизации радиуса (±px)
ROOM_LABEL_NUM_FONT_SIZE = 15     # шрифт номера в круге
ROOM_LABEL_TEXT_FONT_SIZE = 12    # шрифт типа/площади
ROOM_LABEL_OFFSET_RANGE = 30      # макс. случайный сдвиг метки от центроида (px)
ROOM_LABEL_MARGIN = 15            # мин. отступ метки от края комнаты (px)

# Параметры штриховки несущих стен
HATCH_LINE_WIDTH = 1         # толщина линии штриховки (px)
HATCH_COLOR = (60, 60, 60)   # цвет штриховки на плане (тёмно-серый)
LOAD_BEARING_INTERIOR_PROB = 0.3  # вероятность несущей для внутренней стены

# Параметры генерации этажа с квартирами
WALL_T_EXT_SCALE = 1.5       # внешние стены: wall_t * scale
WALL_T_PARTY_SCALE = 1.2     # межквартирные стены: wall_t * scale
MIN_APARTMENTS = 2
MAX_APARTMENTS = 6
MIN_APARTMENT_AREA = 50000   # минимальная площадь квартиры (px²)
CORRIDOR_W_MIN = 120
CORRIDOR_W_MAX = 200
STAIRWELL_SIZE = 180         # размер лестнично-лифтового узла (px)

OUTPUT_DIR = "dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")

# Параметры Scribbles (рукописные пометки на изображении)
SCRIBBLES_ENABLED = True
SCRIBBLES_PROB = 0.3

# Параметры WaterMark (водяной знак)
WATERMARK_ENABLED = False
WATERMARK_PROB = 0.0

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
M_DIM = (255, 0, 0)  # размерные элементы в маске
M_WALL_HATCH = (255, 128, 0)  # несущие стены в маске (оранжевый)
M_CORRIDOR = (0, 128, 255)     # коридор/общественная зона (голубой)
M_WALL_EXTERIOR = (255, 128, 0)   # внешние стены (оранжевый)
M_WALL_PARTY = (128, 128, 0)      # межквартирные стены (оливковый)
M_ROOM_LABEL = (0, 0, 255)        # маркировка помещения в маске (синий)

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
        self.swing_dir: Optional[Tuple[float, float]] = None

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


def compute_extra_offset_map(
    wall_info: List[dict],
    openings: List[Opening],
    rooms: List[Polygon],
    wall_t: float,
    room_union: Polygon,
    wall_polygon: Polygon,
) -> Dict[int, float]:
    """Вернуть словарь {idx: extra_offset} для стен, где дверь открывается
    в ту же сторону, что и размерная линия.
    Выноска отодвигается так, чтобы быть на DOOR_DIM_EXTRA_MARGIN от дуги двери.
    Также добавляет offset коллинеарным стенам, лежащим на одной линии."""
    extra: Dict[int, float] = {}
    _EPS = 1e-3

    # 1. Собираем стены-источники с макс. шириной двери
    src_map: Dict[int, float] = {}  # wall_idx -> max door width (dw)
    for op in openings:
        if op.type != "door" or op.swing_dir is None:
            continue
        if op.wall_idx >= len(wall_info):
            continue
        nx, ny = wall_info[op.wall_idx]["normal"]
        sx, sy = op.swing_dir
        if nx * sx + ny * sy > 0.5:
            dw = math.hypot(op.p2[0] - op.p1[0], op.p2[1] - op.p1[1])
            if dw > src_map.get(op.wall_idx, 0):
                src_map[op.wall_idx] = dw

    if not src_map:
        return extra

    base_offset = wall_t / 2.0 + DIM_OFFSET
    n = len(wall_info)
    for src_idx, max_dw in src_map.items():
        (sx1, sy1), (sx2, sy2) = wall_info[src_idx]["midline"]
        ny = wall_info[src_idx]["normal"][1]
        is_h = abs(ny) > 0.5

        # Целевое положение выноски: дуга двери + отступ
        target = max_dw + DOOR_DIM_EXTRA_MARGIN
        need = max(0.0, target - base_offset)

        for i in range(n):
            (ix1, iy1), (ix2, iy2) = wall_info[i]["midline"]
            is_h2 = abs(iy1 - iy2) < _EPS
            if is_h2 != is_h:
                continue

            if is_h:
                if abs(iy1 - sy1) > _EPS:
                    continue
                lo = max(min(ix1, ix2), min(sx1, sx2))
                hi = min(max(ix1, ix2), max(sx1, sx2))
            else:
                if abs(ix1 - sx1) > _EPS:
                    continue
                lo = max(min(iy1, iy2), min(sy1, sy2))
                hi = min(max(iy1, iy2), max(sy1, sy2))

            if hi - lo >= -1 and need > extra.get(i, 0):
                extra[i] = need

    return extra


def _count_walls_per_side(wall_info: List[dict]) -> Set[int]:
    """Для каждой стены подсчитать количество пересекающих стен с каждой стороны нормали.
    Возвращает {idx: use_flipped} — True, если стен больше с противоположной стороны."""
    flip_map: Set[int] = set()
    n = len(wall_info)
    _EPS = 1e-3

    for i in range(n):
        (x1, y1), (x2, y2) = wall_info[i]["midline"]
        nx, ny = wall_info[i]["normal"]
        is_h = abs(ny) > 0.5

        cnt_norm = 0
        cnt_opp = 0

        wx_min, wx_max = min(x1, x2), max(x1, x2)
        wy_min, wy_max = min(y1, y2), max(y1, y2)

        for j in range(n):
            if j == i:
                continue
            (vx1, vy1), (vx2, vy2) = wall_info[j]["midline"]
            is_h2 = abs(vy1 - vy2) < _EPS
            if is_h2 != is_h:
                continue

            if is_h:
                wy_w = (vy1 + vy2) / 2.0
                if abs(wy_w - y1) > _EPS:
                    continue
                vx_min, vx_max = min(vx1, vx2), max(vx1, vx2)
                if vx_min > wx_max - 1 or vx_max < wx_min + 1:
                    continue
                for ptx, pty in [(vx1, vy1), (vx2, vy2)]:
                    sd = (ptx - x1) * nx + (pty - y1) * ny
                    if abs(sd) < _EPS:
                        continue
                    if sd > 0:
                        cnt_norm += 1
                    else:
                        cnt_opp += 1
            else:
                vx_w = (vx1 + vx2) / 2.0
                if abs(vx_w - x1) > _EPS:
                    continue
                vy_min, vy_max = min(vy1, vy2), max(vy1, vy2)
                if vy_min > wy_max - 1 or vy_max < wy_min + 1:
                    continue
                for ptx, pty in [(vx1, vy1), (vx2, vy2)]:
                    sd = (ptx - x1) * nx + (pty - y1) * ny
                    if abs(sd) < _EPS:
                        continue
                    if sd > 0:
                        cnt_norm += 1
                    else:
                        cnt_opp += 1

        if cnt_opp < cnt_norm:
            flip_map.add(i)

    return flip_map


def draw_dimensions(
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    wall_info: List[dict],
    wall_t: int,
    scale_mm_per_px: float,
    line_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int],
    img_size: Tuple[int, int],
    extra_offset_map: Optional[dict] = None,
    flip_map: Optional[set] = None,
) -> None:
    """Нарисовать размерные линии с засечками и текст длины для внешних стен."""
    if extra_offset_map is None:
        extra_offset_map = {}
    if flip_map is None:
        flip_map = set()

    # --- Feature 2: дедупликация параллельных стен с одинаковым размером ---
    # Группируем стены по (ориентация, длина_в_мм)
    groups: dict = {}  # (orient, length_mm) -> list of (idx, total_offset)
    for i, w in enumerate(wall_info):
        (x1, y1), (x2, y2) = w["midline"]
        length_px = dist((x1, y1), (x2, y2))
        if length_px < DIM_MIN_LENGTH:
            continue
        length_mm = round(length_px * scale_mm_per_px)
        nx, ny = w["normal"]
        orient = 'h' if abs(nx) < 0.5 else 'v'
        total_offset = wall_t / 2.0 + DIM_OFFSET + extra_offset_map.get(i, 0)
        key = (orient, length_mm)
        groups.setdefault(key, []).append((i, total_offset))

    skip: set = set()
    for key, items in groups.items():
        if len(items) > 1:
            items.sort(key=lambda x: x[1], reverse=True)
            for idx, _ in items[1:]:
                skip.add(idx)

    # --- Рисуем размеры ---
    for i, w in enumerate(wall_info):
        if i in skip:
            continue
        (x1, y1), (x2, y2) = w["midline"]
        nx, ny = w["normal"]
        if i in flip_map:
            nx, ny = -nx, -ny
        length_px = dist((x1, y1), (x2, y2))
        if length_px < DIM_MIN_LENGTH:
            continue
        length_mm = round(length_px * scale_mm_per_px)

        # Смещение размерной линии наружу
        offset = wall_t / 2.0 + DIM_OFFSET + extra_offset_map.get(i, 0)
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

        # Наклонные (45°) штрихи на пересечениях выносных и размерной линии
        slant_x = ux - px
        slant_y = uy - py
        sl = math.hypot(slant_x, slant_y)
        if sl > 1e-6:
            sx, sy = slant_x / sl, slant_y / sl
            draw.line([(int(ex1 - sx * half_tick * 2), int(ey1 - sy * half_tick * 2)),
                       (int(ex1 + sx * half_tick * 2), int(ey1 + sy * half_tick * 2))],
                      fill=line_color, width=1)
            draw.line([(int(ex2 - sx * half_tick * 2), int(ey2 - sy * half_tick * 2)),
                       (int(ex2 + sx * half_tick * 2), int(ey2 + sy * half_tick * 2))],
                      fill=line_color, width=1)

        # Текст размера — повёрнут вдоль размерной линии
        text = str(length_mm)

        # Угол размерной линии (нормализуем для читаемости)
        angle_deg = math.degrees(math.atan2(dy, dx))
        _EPS_ANG = 1e-6
        if abs(dx) < _EPS_ANG:                     # вертикальная выноска → 90° (CW)
            angle_deg = -90.0
        elif angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180

        bbox = draw.textbbox((0, 0), text, font=_dim_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Временное RGBA-изображение с текстом
        padding = 4
        txt_img = Image.new("RGBA", (tw + padding * 2, th + padding * 2), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((padding, padding), text, fill=text_color + (255,), font=_dim_font)
        rotated = txt_img.rotate(-angle_deg, expand=True, resample=Image.BICUBIC)
        rw, rh = rotated.size

        # Центр размерной линии
        cx = (ex1 + ex2) / 2.0
        cy = (ey1 + ey2) / 2.0
        text_offset = DIM_TICK_SIZE * 1.5 + 6

        # Текст всегда «над» размерной линией (линия — подчёркивание)
        # Для гориз. стен — выше (снаружи/внутри выноски), для верт. — левее
        text_dir_x = -abs(nx)
        text_dir_y = -abs(ny)
        tx = cx + text_dir_x * text_offset - rw / 2.0
        ty = cy + text_dir_y * text_offset - rh / 2.0

        paste_x = int(tx)
        paste_y = int(ty)

        if isinstance(img, Image.Image):
            img.paste(rotated, (paste_x, paste_y), rotated)


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


# =============================================================================
# МАРКИРОВКА ПОМЕЩЕНИЙ (ГОСТ 21.501-2011)
# =============================================================================
def assign_room_type(area_m2: float) -> str:
    """Вернуть тип помещения по площади."""
    if area_m2 < 15:
        return random.choice(["Санузел", "Кладовая", "Коридор"])
    elif area_m2 < 25:
        return random.choice(["Кухня", "Спальня", "Кабинет"])
    elif area_m2 < 50:
        return random.choice(["Жилая", "Спальня", "Кухня-гостиная"])
    elif area_m2 < 80:
        return random.choice(["Гостиная", "Жилая"])
    else:
        return random.choice(["Гостиная", "Зал"])


def _dimension_exclusion_lines(
    wall_info: List[dict],
    wall_t: float,
    extra_offset_map: Dict[int, float],
    flip_map: set,
) -> List[LineString]:
    """Построить список LineString геометрии размерных линий для проверки пересечений."""
    from shapely.geometry import LineString as SLine

    lines: List[SLine] = []

    for i, w in enumerate(wall_info):
        (x1, y1), (x2, y2) = w["midline"]
        nx, ny = w["normal"]
        if i in flip_map:
            nx, ny = -nx, -ny
        length_px = math.hypot(x2 - x1, y2 - y1)
        if length_px < DIM_MIN_LENGTH:
            continue
        offset = wall_t / 2.0 + DIM_OFFSET + extra_offset_map.get(i, 0)
        ex1 = x1 + nx * offset
        ey1 = y1 + ny * offset
        ex2 = x2 + nx * offset
        ey2 = y2 + ny * offset
        lines.append(SLine([(x1, y1), (ex1, ey1)]))
        lines.append(SLine([(x2, y2), (ex2, ey2)]))
        lines.append(SLine([(ex1, ey1), (ex2, ey2)]))

    return lines


def compute_room_labels(
    rooms: List[Polygon],
    wall_info: List[dict],
    wall_t: float,
    extra_offset_map: Dict[int, float],
    flip_map: set,
) -> List[dict]:
    """Вычислить метки помещений: номер, тип, площадь, позиция (cx, cy).
    Позиции детерминированы (seeded random) — одинаковы для плана и маски."""
    from shapely.geometry import Point as SPoint

    dim_lines = _dimension_exclusion_lines(wall_info, wall_t, extra_offset_map, flip_map)
    labels: List[dict] = []

    for room_idx, poly in enumerate(rooms):
        rng = random.Random(room_idx * 7 + 42)
        point = poly.representative_point()
        cx, cy = point.x, point.y

        # Случайный сдвиг от центроида
        for _ in range(20):
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(0, ROOM_LABEL_OFFSET_RANGE)
            dx = math.cos(angle) * dist
            dy = math.sin(angle) * dist
            nx, ny = cx + dx, cy + dy
            test_point = SPoint(nx, ny)
            if not poly.contains(test_point):
                continue
            # Отступ от края полигона
            if poly.boundary.distance(test_point) < ROOM_LABEL_MARGIN:
                continue
            # Отступ от стен (midline)
            too_close = False
            for w in wall_info:
                (wx1, wy1), (wx2, wy2) = w["midline"]
                seg = LineString([(wx1, wy1), (wx2, wy2)])
                if test_point.distance(seg) < wall_t:
                    too_close = True
                    break
            if too_close:
                continue
            # Проверка пересечения с размерными линиями
            conflict = False
            for dl in dim_lines:
                if test_point.distance(dl) < ROOM_LABEL_CIRCLE_R + 10:
                    conflict = True
                    break
            if conflict:
                continue
            cx, cy = nx, ny
            break

        area_m2 = round(poly.area * SCALE_MM_PER_PX * SCALE_MM_PER_PX / 1_000_000, 1)
        room_type = assign_room_type(area_m2)

        labels.append({
            "number": room_idx + 1,
            "type": room_type,
            "area_m2": area_m2,
            "cx": cx,
            "cy": cy,
        })

    return labels


def draw_room_labels(
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    room_labels: List[dict],
    text_color: Tuple[int, int, int],
    *,
    mask_mode: bool = False,
) -> None:
    """Нарисовать метки помещений по ГОСТ.
    В режиме mask_mode — фиксированный формат A без дефектов."""
    try:
        num_font = ImageFont.truetype("arial.ttf", ROOM_LABEL_NUM_FONT_SIZE)
        text_font = ImageFont.truetype("arial.ttf", ROOM_LABEL_TEXT_FONT_SIZE)
    except IOError:
        try:
            num_font = ImageFont.truetype("DejaVuSans.ttf", ROOM_LABEL_NUM_FONT_SIZE)
            text_font = ImageFont.truetype("DejaVuSans.ttf", ROOM_LABEL_TEXT_FONT_SIZE)
        except IOError:
            num_font = ImageFont.load_default()
            text_font = ImageFont.load_default()

    _ROOM_SEP_CHOICES = [".", ","]

    for lbl in room_labels:
        num = lbl["number"]
        room_type = lbl["type"]
        area = lbl["area_m2"]
        cx, cy = lbl["cx"], lbl["cy"]

        rng = random.Random(num * 13 + 7)

        if mask_mode:
            fmt = "A"
            circle_r = ROOM_LABEL_CIRCLE_R
            circle_width = 2
            angle = 0.0
            num_fs = ROOM_LABEL_NUM_FONT_SIZE
            txt_fs = ROOM_LABEL_TEXT_FONT_SIZE
            sep = "."
            area_suffix = "m2"
        else:
            fmt = rng.choices(["A", "B", "C", "D"], weights=[40, 30, 15, 15])[0]
            circle_r = ROOM_LABEL_CIRCLE_R + rng.randint(-ROOM_LABEL_CIRCLE_R_RANGE, ROOM_LABEL_CIRCLE_R_RANGE)
            circle_width = rng.randint(1, 3)
            angle = rng.uniform(-5, 5)
            num_fs = ROOM_LABEL_NUM_FONT_SIZE + rng.randint(-2, 2)
            txt_fs = ROOM_LABEL_TEXT_FONT_SIZE + rng.randint(-1, 1)
            sep = rng.choice(_ROOM_SEP_CHOICES)
            area_suffix = rng.choice(["m2", "kv.m", ""])

        # Номер в окружности
        num_str = str(num)
        bbox_n = draw.textbbox((0, 0), num_str, font=num_font)
        nw = bbox_n[2] - bbox_n[0]
        nh = bbox_n[3] - bbox_n[1]
        tx_n = cx - nw / 2.0 - bbox_n[0]
        ty_n = cy - nh / 2.0 - bbox_n[1]

        # Окружность
        draw.ellipse(
            [cx - circle_r, cy - circle_r, cx + circle_r, cy + circle_r],
            outline=text_color, width=circle_width,
        )
        # Номер
        draw.text((tx_n, ty_n), num_str, fill=text_color, font=num_font)

        # Текст метки (тип и/или площадь) — под окружностью
        area_str = f"{area:.1f}".replace(".", sep) + (f" {area_suffix}" if area_suffix else "")

        if fmt == "A":
            lines = [room_type, area_str]
        elif fmt == "B":
            lines = [f"{room_type} {area_str}"]
        elif fmt == "C":
            lines = [area_str]
        else:  # fmt == "D"
            lines = [room_type]

        # Рендерим каждую строку текста
        y_off = circle_r + 6
        for line in lines:
            bbox_t = draw.textbbox((0, 0), line, font=text_font)
            tw = bbox_t[2] - bbox_t[0]
            th = bbox_t[3] - bbox_t[1]

            if mask_mode:
                # Прямой рендеринг в маску (без RGBA-оверлея)
                paste_x = int(cx - tw / 2.0 - bbox_t[0])
                paste_y = int(cy + y_off - th / 2.0 - bbox_t[1])
                draw.text((paste_x, paste_y), line, fill=text_color, font=text_font)
                y_off += th + 4
                continue

            # План: RGBA-оверлей с поворотом
            padding = 3
            txt_img = Image.new("RGBA", (tw + padding * 2, th + padding * 2), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((padding - bbox_t[0], padding - bbox_t[1]), line,
                          fill=text_color + (255,), font=text_font)

            rotated = txt_img.rotate(angle, expand=True, resample=Image.BICUBIC)
            rw, rh = rotated.size

            paste_x = int(cx - rw / 2.0)
            paste_y = int(cy + y_off - rh / 2.0)

            img.paste(rotated, (paste_x, paste_y), rotated)

            y_off += th + 4


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


# =============================================================================
# РУКОПИСНЫЙ ТЕКСТ НА ПЛАНЕ
# =============================================================================
# Попытка загрузить шрифт, имитирующий рукописный текст
_handwriting_font: Optional[ImageFont.ImageFont] = None
for _hw_font_candidate in ["fonts/Caveat-VariableFont_wght.ttf", "Caveat.ttf", "comic.ttf",
                           "Caveat-VariableFont_wght.ttf", "caveat.ttf"]:
    try:
        _handwriting_font = ImageFont.truetype(_hw_font_candidate, 28)
        break
    except IOError:
        continue
if _handwriting_font is None:
    try:
        _handwriting_font = ImageFont.truetype("arial.ttf", 28)
    except IOError:
        _handwriting_font = ImageFont.load_default()


_HANDWRITING_TEXTS = [
    "утв.", "этаж 3", "кв. 45", "лист 1", "подпись",
    "Проверил", "Арх. Иванов", "В.О.",
    "Не для печати", "Черновик", "Копия",
    "Сантехника", "Электрика", "План этажа",
]


def draw_handwriting(
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    wall_polygon: Optional[Polygon] = None,
) -> bool:
    """Нанести случайные рукописные пометки на план.
    Возвращает True, если хотя бы одна надпись была размещена."""
    import random
    num_labels = random.randint(2, 5)
    margin = 40
    placed = 0

    for attempt in range(num_labels * 8):
        if placed >= num_labels:
            break

        text = random.choice(_HANDWRITING_TEXTS)
        font_size = random.randint(20, 32)
        try:
            font = ImageFont.truetype(_handwriting_font.font.path, font_size)
        except Exception:
            font = _handwriting_font

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        angle = random.uniform(-15, 15)
        color_val = random.randint(40, 80)
        color = (color_val, color_val, color_val)

        # Рисуем текст на временном полотне
        txt_img = Image.new("RGBA", (tw + 12, th + 12), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((6 - bbox[0], 6 - bbox[1]), text, fill=(*color, 200), font=font)
        txt_img = txt_img.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

        # Позиционирование с учётом реального размера после поворота
        rw, rh = txt_img.size
        x = random.randint(margin, img.width - margin - rw)
        y = random.randint(margin, img.height - margin - rh)

        # Если задан полигон стен — не ставить текст на стены
        if wall_polygon is not None and not wall_polygon.is_empty:
            try:
                from shapely.geometry import Point as ShPt
                cx = x + rw // 2
                cy = y + rh // 2
                if 0 <= cx < img.width and 0 <= cy < img.height:
                    if wall_polygon.covers(ShPt(cx, cy)):
                        continue
            except Exception:
                pass

        img.paste(txt_img, (x, y), txt_img)
        placed += 1
    return placed > 0


def draw_plan(
    img: Image.Image,
    rooms: List[Polygon],
    openings: List[Opening],
    wall_t: int,
    wall_info: List[dict],
    lb_regions: List[Polygon],
) -> bool:
    """Нарисовать основной план (цвета BG/WALL/ROOM/DOOR).
    Возвращает True, если на план нанесён рукописный текст."""
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
            if not fits:
                continue  # пропускаем — налетает на стену/другую дверь
            nx, ny = chosen_nx, chosen_ny
            op.swing_dir = (nx, ny)
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
    flip_map = _count_walls_per_side(wall_info)
    wall_poly = compute_variable_wall_polygon(wall_info)
    extra_map = compute_extra_offset_map(wall_info, openings, rooms, wall_t, room_union, wall_poly)
    draw_dimensions(draw, img, wall_info, wall_t, SCALE_MM_PER_PX, (0, 0, 0), (0, 0, 0), img.size, extra_map, flip_map)

    # 9.5. Маркировка помещений (ГОСТ 21.501-2011)
    room_labels = compute_room_labels(rooms, wall_info, wall_t, extra_map, flip_map)
    draw_room_labels(draw, img, room_labels, (0, 0, 0))

    # 10. Рукописные пометки (с вероятностью 60%)
    has_hw = False
    if random.random() < 0.6:
        try:
            wp = compute_variable_wall_polygon(wall_info)
        except Exception:
            wp = None
        has_hw = draw_handwriting(draw, img, wp)

    return has_hw


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
    wall_poly = compute_variable_wall_polygon(wall_info)
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
                            if door_clear(rooms, op, wall_t, dxn, dyn, dw, drawn_swings, outward, room_union, wall_poly):
                                chosen_nx, chosen_ny = dxn, dyn
                                fits = True
                                break
                        if fits:
                            break
                    if not fits:
                        continue
                    nx, ny = chosen_nx, chosen_ny
                    op.swing_dir = (nx, ny)
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
    flip_map = _count_walls_per_side(wall_info)
    extra_map = compute_extra_offset_map(wall_info, openings, rooms, wall_t, room_union, wall_poly)
    draw_dimensions(draw, mask, wall_info, wall_t, SCALE_MM_PER_PX, M_DIM, M_DIM, mask.size, extra_map, flip_map)

    # 5.5. Маркировка помещений в маске
    room_labels = compute_room_labels(rooms, wall_info, wall_t, extra_map, flip_map)
    draw_room_labels(draw, mask, room_labels, M_ROOM_LABEL, mask_mode=True)


# =============================================================================
# ГЕНЕРАЦИЯ ЭТАЖА С КВАРТИРАМИ И ОБЩЕСТВЕННЫМИ ЗОНАМИ
# =============================================================================
def generate_floor_layout(
    canvas_w: int, canvas_h: int, wall_t_int: float
) -> Tuple[List[Polygon], List[dict], List[dict], List[str]]:
    """Генерировать этаж: коридор, ЛЛУ, квартиры.
    Возвращает (rooms, wall_info, wall_segments, room_types).
    room_types[i] — тип i-й комнаты: 'room', 'corridor', 'stairwell'.
    wall_info расширен полями wall_type и thickness.
    """
    margin = CANVAS_MARGIN
    bb_x1, bb_y1 = margin, margin
    bb_x2, bb_y2 = canvas_w - margin, canvas_h - margin

    if bb_x2 - bb_x1 < 600 or bb_y2 - bb_y1 < 600:
        bb_x1, bb_y1 = 100, 100
        bb_x2, bb_y2 = canvas_w - 100, canvas_h - 100

    half_ext = wall_t_int * WALL_T_EXT_SCALE / 2.0
    half_party = wall_t_int * WALL_T_PARTY_SCALE / 2.0
    half_int = wall_t_int / 2.0

    def thick(wt: str) -> float:
        if wt == "exterior":
            return wall_t_int * WALL_T_EXT_SCALE
        if wt == "party":
            return wall_t_int * WALL_T_PARTY_SCALE
        return wall_t_int

    # --- 1. Коридор ---
    corr_w = random.randint(CORRIDOR_W_MIN, CORRIDOR_W_MAX)
    is_horizontal = random.random() < 0.5

    bbox_w = bb_x2 - bb_x1
    bbox_h = bb_y2 - bb_y1

    if is_horizontal:
        # Горизонтальный коридор — делит этаж по вертикали
        corr_y = random.randint(int(bb_y1 + bbox_h * 0.3), int(bb_y2 - bbox_h * 0.3))
        corr_y = max(bb_y1 + corr_w, min(corr_y, bb_y2 - corr_w))
        corr_rect = (bb_x1, corr_y - corr_w // 2, bb_x2, corr_y + corr_w // 2 + corr_w % 2)
        # ЛЛУ — слева
        stair_rect = (bb_x1, corr_y - corr_w // 2, bb_x1 + STAIRWELL_SIZE, corr_y + corr_w // 2)
        # Лоты: верхний ряд, нижний ряд
        top_y1, top_y2 = bb_y1, corr_rect[1]
        bot_y1, bot_y2 = corr_rect[3], bb_y2
        row_specs = [
            ("top", top_y1, top_y2),
            ("bot", bot_y1, bot_y2),
        ]
    else:
        # Вертикальный коридор — делит этаж по горизонтали
        corr_x = random.randint(int(bb_x1 + bbox_w * 0.3), int(bb_x2 - bbox_w * 0.3))
        corr_x = max(bb_x1 + corr_w, min(corr_x, bb_x2 - corr_w))
        corr_rect = (corr_x - corr_w // 2, bb_y1, corr_x + corr_w // 2 + corr_w % 2, bb_y2)
        # ЛЛУ — сверху
        stair_rect = (corr_x - corr_w // 2, bb_y1, corr_x + corr_w // 2, bb_y1 + STAIRWELL_SIZE)
        # Лоты: левый ряд, правый ряд
        left_x1, left_x2 = bb_x1, corr_rect[0]
        right_x1, right_x2 = corr_rect[2], bb_x2
        row_specs = [
            ("left", left_x1, left_x2),
            ("right", right_x1, right_x2),
        ]

    # --- 2. Полигоны коридора и ЛЛУ ---
    corridor_poly = box(*corr_rect)
    stair_poly = box(*stair_rect)

    # --- 3. Деление рядов на лоты квартир ---
    num_aparts = random.randint(MIN_APARTMENTS, MAX_APARTMENTS)

    all_rooms: List[Polygon] = []
    all_room_types: List[str] = []
    all_wall_info: List[dict] = []
    all_wall_midlines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    # Добавляем коридор и ЛЛУ как комнаты
    all_rooms.append(corridor_poly)
    all_room_types.append("corridor")
    all_rooms.append(stair_poly)
    all_room_types.append("stairwell")

    apt_bboxes: List[Tuple[float, float, float, float, str]] = []  # (x1,y1,x2,y2,side)

    for side, s1, s2 in row_specs:
        if is_horizontal:
            available = bb_x2 - bb_x1
            min_apt_w = max(300, int(math.sqrt(MIN_APARTMENT_AREA * (bbox_w / bbox_h))))

            if num_aparts * min_apt_w > available:
                n = max(1, int(available / min_apt_w))
            else:
                n = max(1, num_aparts // len(row_specs))

            # Распределяем лоты равномерно
            part_w = (available - (n - 1) * half_party * 2) / n
            if part_w < min_apt_w:
                part_w = min_apt_w
                n = max(1, int(available / (part_w + half_party * 2)))

            for j in range(n):
                ax1 = bb_x1 + j * (part_w + half_party * 2)
                ax2 = ax1 + part_w
                if ax2 > bb_x2:
                    ax2 = bb_x2
                    ax1 = ax2 - part_w
                if ax1 < bb_x1:
                    ax1 = bb_x1
                apt_bboxes.append((ax1, s1, ax2, s2, side))
        else:
            available = bb_y2 - bb_y1
            min_apt_h = max(300, int(math.sqrt(MIN_APARTMENT_AREA * (bbox_h / bbox_w))))
            n = max(1, num_aparts // len(row_specs))
            part_h = (available - (n - 1) * half_party * 2) / n
            if part_h < min_apt_h:
                part_h = min_apt_h
                n = max(1, int(available / (part_h + half_party * 2)))

            for j in range(n):
                ay1 = bb_y1 + j * (part_h + half_party * 2)
                ay2 = ay1 + part_h
                if ay2 > bb_y2:
                    ay2 = bb_y2
                    ay1 = ay2 - part_h
                if ay1 < bb_y1:
                    ay1 = bb_y1
                apt_bboxes.append((s1, ay1, s2, ay2, side))

    if not apt_bboxes:
        apt_bboxes.append((bb_x1 + corr_w, bb_y1 + corr_w, bb_x2 - corr_w, bb_y2 - corr_w, "top" if is_horizontal else "left"))

    # --- 4. BSP внутри каждой квартиры ---
    for ax1, ay1, ax2, ay2, side in apt_bboxes:
        if ax2 - ax1 < 200 or ay2 - ay1 < 200:
            # Квартира-студия (одна комната)
            room_poly = box(ax1 + half_int, ay1 + half_int, ax2 - half_int, ay2 - half_int)
            all_rooms.append(room_poly)
            all_room_types.append("room")
            continue

        apt_rooms, apt_walls = _bsp_apartment(ax1, ay1, ax2, ay2, wall_t_int)
        for r in apt_rooms:
            all_rooms.append(r)
            all_room_types.append("room")

        for w in apt_walls:
            w["wall_type"] = "interior"
            all_wall_info.append(w)
            all_wall_midlines.append(w["midline"])

        # --- 5. Стены по периметру квартиры (межквартирные и внешние) ---
        # Верхняя, нижняя, левая, правая границы квартиры
        perim_edges = [
            ("h", ax1, ay1, ax2, ay1, "top"),
            ("h", ax1, ay2, ax2, ay2, "bottom"),
            ("v", ax1, ay1, ax1, ay2, "left"),
            ("v", ax2, ay1, ax2, ay2, "right"),
        ]
        for etype, ex1, ey1, ex2, ey2, pos in perim_edges:
            # Пропускаем границу, смежную с коридором
            if is_horizontal and ((side == "top" and pos == "bottom") or (side == "bot" and pos == "top")):
                continue
            if not is_horizontal and ((side == "left" and pos == "right") or (side == "right" and pos == "left")):
                continue

            midline = ((ex1, ey1), (ex2, ey2)) if etype == "h" else ((ex1, ey1), (ex2, ey2))

            # Определяем тип стены: внешняя или межквартирная
            is_ext = False
            if etype == "h":
                if pos == "top" and side == "top":
                    is_ext = True
                elif pos == "bottom" and side == "bot":
                    is_ext = True
                else:
                    is_ext = any(ay1 == bb_y1 or ay2 == bb_y2 for ax1, ay1, ax2, ay2, _ in apt_bboxes if (ex1, ey1, ex2, ey2) == (ax1, ay1, ax2, ay1))
                    # Упрощённо: если на границе building bbox
                    is_ext = (pos == "top" and ay1 == bb_y1) or (pos == "bottom" and ay2 == bb_y2)
            else:
                if pos == "left" and side == "left":
                    is_ext = True
                elif pos == "right" and side == "right":
                    is_ext = True
                else:
                    is_ext = (pos == "left" and ax1 == bb_x1) or (pos == "right" and ax2 == bb_x2)

            wt = "exterior" if is_ext else "party"
            half_t = thick(wt) / 2.0

            # Смещаем midline для внешних/межквартирных стен
            if etype == "h":
                if wt == "exterior":
                    if pos == "top":
                        midline = ((ex1 - half_t, ey1 + (thick(wt) - wall_t_int) / 2), (ex2 + half_t, ey1 + (thick(wt) - wall_t_int) / 2))
                        normal = (0, -1)
                    else:  # bottom
                        midline = ((ex1 - half_t, ey1 - (thick(wt) - wall_t_int) / 2), (ex2 + half_t, ey1 - (thick(wt) - wall_t_int) / 2))
                        normal = (0, 1)
                else:
                    if pos == "top":
                        midline = ((ex1 - half_t, ey1 + half_int), (ex2 + half_t, ey1 + half_int))
                        normal = (0, -1)
                    else:
                        midline = ((ex1 - half_t, ey1 - half_int), (ex2 + half_t, ey1 - half_int))
                        normal = (0, 1)
            else:  # vertical
                if wt == "exterior":
                    if pos == "left":
                        midline = ((ex1 + (thick(wt) - wall_t_int) / 2, ey1 - half_t), (ex2 + (thick(wt) - wall_t_int) / 2, ey2 + half_t))
                        normal = (-1, 0)
                    else:  # right
                        midline = ((ex1 - (thick(wt) - wall_t_int) / 2, ey1 - half_t), (ex2 - (thick(wt) - wall_t_int) / 2, ey2 + half_t))
                        normal = (1, 0)
                else:
                    if pos == "left":
                        midline = ((ex1 + half_int, ey1 - half_t), (ex2 + half_int, ey2 + half_t))
                        normal = (-1, 0)
                    else:
                        midline = ((ex1 - half_int, ey1 - half_t), (ex2 - half_int, ey2 + half_t))
                        normal = (1, 0)

            w_entry = {"midline": midline, "normal": normal, "wall_type": wt, "thickness": thick(wt)}
            all_wall_info.append(w_entry)
            all_wall_midlines.append(midline)

    return all_rooms, all_wall_info, all_wall_midlines, all_room_types


def _bsp_apartment(
    ax1: float, ay1: float, ax2: float, ay2: float, wall_t: float
) -> Tuple[List[Polygon], List[dict]]:
    """BSP-разбиение внутри квартиры. Возвращает (rooms, wall_info)."""
    half_t = wall_t / 2.0
    num_rooms = random.randint(2, 5)
    min_room = 100

    parts: List[Tuple[float, float, float, float]] = [
        (ax1 + half_t, ay1 + half_t, ax2 - half_t, ay2 - half_t)
    ]

    while len(parts) < num_rooms:
        candidates = []
        for i, (rx1, ry1, rx2, ry2) in enumerate(parts):
            w, h = rx2 - rx1, ry2 - ry1
            min_dim = min_room * 2 + wall_t
            if w > min_dim or h > min_dim:
                candidates.append(i)
        if not candidates:
            break
        idx = random.choice(candidates)
        rx1, ry1, rx2, ry2 = parts[idx]
        w, h = rx2 - rx1, ry2 - ry1

        split_ok = False
        if w > h * 1.2:
            dirs = ["v", "h"]
        elif h > w * 1.2:
            dirs = ["h", "v"]
        else:
            dirs = ["v", "h"] if random.random() < 0.5 else ["h", "v"]

        for d in dirs:
            if d == "v":
                lo = int(rx1 + min_room + half_t)
                hi = int(rx2 - min_room - half_t)
            else:
                lo = int(ry1 + min_room + half_t)
                hi = int(ry2 - min_room - half_t)
            if lo >= hi:
                continue
            split = random.randint(lo, hi)
            if d == "v":
                a = (rx1, ry1, split - half_t, ry2)
                b = (split + half_t, ry1, rx2, ry2)
            else:
                a = (rx1, ry1, rx2, split - half_t)
                b = (rx1, split + half_t, rx2, ry2)
            parts.pop(idx)
            parts.append(a)
            parts.append(b)
            split_ok = True
            break
        if not split_ok:
            break

    rooms = [box(rx1, ry1, rx2, ry2) for rx1, ry1, rx2, ry2 in parts]

    # Стены (midlines)
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

    wall_info = []
    for (etype, ex1, ey1, ex2, ey2), count in edges.items():
        ext = wall_t
        if etype == "h":
            if count >= 2:
                midline = ((ex1 - ext, ey1), (ex2 + ext, ey1))
                normal = (0, -1)
            else:
                is_top = any(abs(ey1 - ry1) < 1 for _, ry1, _, _ in parts)
                if is_top:
                    midline = ((ex1 - ext, ey1 - half_t), (ex2 + ext, ey1 - half_t))
                    normal = (0, -1)
                else:
                    midline = ((ex1 - ext, ey1 + half_t), (ex2 + ext, ey1 + half_t))
                    normal = (0, 1)
        else:
            if count >= 2:
                midline = ((ex1, ey1 - ext), (ex2, ey2 + ext))
                normal = (-1, 0)
            else:
                is_left = any(abs(ex1 - rx1) < 1 for rx1, _, _, _ in parts)
                if is_left:
                    midline = ((ex1 - half_t, ey1 - ext), (ex1 - half_t, ey2 + ext))
                    normal = (-1, 0)
                else:
                    midline = ((ex1 + half_t, ey1 - ext), (ex1 + half_t, ey2 + ext))
                    normal = (1, 0)

        length = math.hypot(midline[1][0] - midline[0][0], midline[1][1] - midline[0][1])
        if length > 50:
            wall_info.append({"midline": midline, "normal": normal})

    return rooms, wall_info


def compute_variable_wall_polygon(
    wall_info: List[dict],
) -> Polygon:
    """Построить единый полигон стен с переменной толщиной."""
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    polys = []
    for w in wall_info:
        (x1, y1), (x2, y2) = w["midline"]
        nx, ny = w["normal"]
        half_t = w.get("thickness", 20) / 2.0

        pts = [
            (x1 + nx * half_t, y1 + ny * half_t),
            (x2 + nx * half_t, y2 + ny * half_t),
            (x2 - nx * half_t, y2 - ny * half_t),
            (x1 - nx * half_t, y1 - ny * half_t),
        ]
        try:
            polys.append(ShapelyPolygon(pts))
        except Exception:
            continue

    if not polys:
        return Polygon()
    if len(polys) == 1:
        return polys[0]
    return unary_union(polys)


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


def apply_scribbles(
    img: Image.Image,
) -> Tuple[Image.Image, bool]:
    """Нанести случайные чернильные пометки (Scribbles) на изображение.
    Возвращает (изображение, True если пометки были нанесены)."""
    if not SCRIBBLES_ENABLED or random.random() >= SCRIBBLES_PROB:
        return img, False

    draw = ImageDraw.Draw(img)
    w, h = img.size
    placed = False

    num_marks = random.randint(1, 4)
    for _ in range(num_marks):
        x1 = random.randint(10, w - 10)
        y1 = random.randint(10, h - 10)
        x2 = x1 + random.randint(-60, 60)
        y2 = y1 + random.randint(-60, 60)
        width = random.randint(1, 3)
        gray = random.randint(30, 100)
        draw.line([(x1, y1), (x2, y2)], fill=(gray, gray, gray), width=width)
        placed = True

    return img, placed


# =============================================================================
# ДЕДУПЛИКАЦИЯ СТЕН
# =============================================================================

def deduplicate_wall_info(
    wall_info: List[dict],
    wall_t: float,
    openings: List[Opening],
) -> List[dict]:
    """Удалить back-to-back дубликаты стен (два сегмента одной стены с разных сторон).
    Обновляет wall_idx в openings при удалении дубликатов."""
    keep = [True] * len(wall_info)
    n = len(wall_info)
    max_dist = wall_t * 1.5

    for i in range(n):
        if not keep[i]:
            continue
        (x1, y1), (x2, y2) = wall_info[i]["midline"]
        nx_i, ny_i = wall_info[i]["normal"]
        horiz_i = abs(y1 - y2) < 1e-6

        for j in range(i + 1, n):
            if not keep[j]:
                continue
            (vx1, vy1), (vx2, vy2) = wall_info[j]["midline"]
            nx_j, ny_j = wall_info[j]["normal"]
            horiz_j = abs(vy1 - vy2) < 1e-6

            if horiz_i != horiz_j:
                continue

            if nx_i * nx_j + ny_i * ny_j > -0.5:
                continue

            if horiz_i:
                dist_y = abs(y1 - vy1)
                if dist_y > max_dist:
                    continue
                overlap = max(0, min(x2, vx2) - max(x1, vx1))
                if overlap < min(x2 - x1, vx2 - vx1) * 0.3:
                    continue
            else:
                dist_x = abs(x1 - vx1)
                if dist_x > max_dist:
                    continue
                overlap = max(0, min(y2, vy2) - max(y1, vy1))
                if overlap < min(y2 - y1, vy2 - vy1) * 0.3:
                    continue

            keep[j] = False

    old_to_new = {}
    new_wall_info = []
    for i, w in enumerate(wall_info):
        if keep[i]:
            old_to_new[i] = len(new_wall_info)
            new_wall_info.append(w)

    for op in openings:
        if op.wall_idx in old_to_new:
            op.wall_idx = old_to_new[op.wall_idx]

    return new_wall_info


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
    parser.add_argument("--scribbles-prob", type=float, default=None,
                        help="Вероятность Scribbles (0-1), 0=отключено")
    parser.add_argument("--watermark-prob", type=float, default=None,
                        help="Вероятность WaterMark (0-1), 0=отключено")
    return parser.parse_args()


def main():
    args = parse_args()
    num_images = args.num_images
    output_dir = args.output_dir
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")

    if args.seed is not None:
        random.seed(args.seed)

    # CLI-параметры Scribbles / WaterMark
    global SCRIBBLES_ENABLED, SCRIBBLES_PROB, WATERMARK_ENABLED, WATERMARK_PROB
    if args.scribbles_prob is not None:
        if args.scribbles_prob == 0:
            SCRIBBLES_ENABLED = False
            SCRIBBLES_PROB = 0.0
        else:
            SCRIBBLES_PROB = args.scribbles_prob
    if args.watermark_prob is not None:
        if args.watermark_prob == 0:
            WATERMARK_ENABLED = False
            WATERMARK_PROB = 0.0
        else:
            WATERMARK_PROB = args.watermark_prob

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
    print(f"  -> Аугментация: {'вкл' if pipeline else 'выкл'}"
          f"  Scribbles:{SCRIBBLES_PROB if SCRIBBLES_ENABLED else 0}  WaterMark:{WATERMARK_PROB if WATERMARK_ENABLED else 0}")

    for i in tqdm(range(num_images), desc="Планы", unit="img"):
        idx = start_idx + i
        canvas_w = random.randint(MIN_CANVAS, MAX_CANVAS)
        canvas_h = random.randint(MIN_CANVAS, MAX_CANVAS)

        rooms, openings, wall_midlines, wall_info, wall_t, num_rooms = generate_plan(canvas_w, canvas_h)

        # Удаляем back-to-back дубликаты (межквартирные стены из двух половинок)
        wall_info = deduplicate_wall_info(wall_info, wall_t, openings)

        # Регионы несущих стен (один раз для плана и маски)
        lb_regions, _ = get_load_bearing_regions(rooms, wall_info, wall_t)

        # Основное изображение
        img = Image.new("RGB", (canvas_w, canvas_h), BG)
        has_handwriting = draw_plan(img, rooms, openings, wall_t, wall_info, lb_regions)

        # Маска (чистая, без аугментаций)
        mask = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
        draw_mask(mask, rooms, openings, wall_t, wall_info, lb_regions)

        # Рукописные пометки (Scribbles) — на план, до pipeline
        has_scribbles = False
        img, has_scribbles = apply_scribbles(img)

        # Применяем артефакты сканирования только к изображению
        img, augs = apply_scanning_artifacts(img, pipeline)

        # Суффикс из названий сработавших аугментаций (подчёркивания в именах заменяем на дефисы)
        aug_list = list(augs)
        if has_scribbles:
            aug_list.append("Scribbles")
        aug_suffix = "_" + "_".join(a.replace("_", "-") for a in aug_list) if aug_list else "_clean"

        # Суффикс для рукописного текста
        hw_suffix = "_Handwriting" if has_handwriting else ""

        img.save(os.path.join(images_dir, f"plan_{idx:06d}{aug_suffix}{hw_suffix}.png"))
        mask.save(os.path.join(masks_dir, f"mask_{idx:06d}{aug_suffix}{hw_suffix}.png"))


if __name__ == "__main__":
    main()
