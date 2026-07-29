#!/usr/bin/env python3
"""Small dependency-free reader for the ASCII DXF geometry used by CyberRunner.

This intentionally supports only the model-space primitives present in the
maze exports: LINE, CIRCLE, ARC, and LWPOLYLINE.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Line:
    start: tuple[float, float]
    end: tuple[float, float]


@dataclass(frozen=True)
class Circle:
    center: tuple[float, float]
    radius: float


@dataclass(frozen=True)
class Arc:
    center: tuple[float, float]
    radius: float
    start_angle_deg: float
    end_angle_deg: float


@dataclass(frozen=True)
class Polyline:
    vertices: tuple[tuple[float, float], ...]
    closed: bool


@dataclass(frozen=True)
class Geometry:
    lines: tuple[Line, ...]
    circles: tuple[Circle, ...]
    arcs: tuple[Arc, ...]
    polylines: tuple[Polyline, ...]

    def points(self) -> np.ndarray:
        points = []
        for line in self.lines:
            points.extend((line.start, line.end))
        for circle in self.circles:
            x, y = circle.center
            r = circle.radius
            points.extend(((x - r, y - r), (x + r, y + r)))
        for arc in self.arcs:
            points.extend(sample_arc(arc, max_step=pi / 18.0))
        for polyline in self.polylines:
            points.extend(polyline.vertices)
        return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def _pairs(path: str | Path) -> list[tuple[int, str]]:
    lines = Path(path).read_text(encoding="utf-8", errors="strict").splitlines()
    if len(lines) % 2:
        raise ValueError(f"{path}: malformed DXF with an odd number of lines")
    result = []
    for index in range(0, len(lines), 2):
        try:
            code = int(lines[index].strip())
        except ValueError as exc:
            raise ValueError(
                f"{path}: invalid group code at line {index + 1}"
            ) from exc
        result.append((code, lines[index + 1].strip()))
    return result


def _entities(pairs: Iterable[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    in_entities = False
    expect_section_name = False
    current = None
    entities = []
    for code, value in pairs:
        if code == 0 and value == "SECTION":
            expect_section_name = True
            continue
        if expect_section_name:
            in_entities = code == 2 and value == "ENTITIES"
            expect_section_name = False
            continue
        if code == 0 and value == "ENDSEC":
            if in_entities and current:
                entities.append(current)
            in_entities = False
            current = None
            continue
        if not in_entities:
            continue
        if code == 0:
            if current:
                entities.append(current)
            current = [(code, value)]
        elif current is not None:
            current.append((code, value))
    return entities


def _first(entity: list[tuple[int, str]], code: int, default=None):
    for item_code, value in entity:
        if item_code == code:
            return value
    return default


def read_geometry(path: str | Path) -> Geometry:
    lines = []
    circles = []
    arcs = []
    polylines = []
    for entity in _entities(_pairs(path)):
        kind = entity[0][1]
        if kind == "LINE":
            lines.append(
                Line(
                    (float(_first(entity, 10)), float(_first(entity, 20))),
                    (float(_first(entity, 11)), float(_first(entity, 21))),
                )
            )
        elif kind == "CIRCLE":
            circles.append(
                Circle(
                    (float(_first(entity, 10)), float(_first(entity, 20))),
                    float(_first(entity, 40)),
                )
            )
        elif kind == "ARC":
            arcs.append(
                Arc(
                    (float(_first(entity, 10)), float(_first(entity, 20))),
                    float(_first(entity, 40)),
                    float(_first(entity, 50)),
                    float(_first(entity, 51)),
                )
            )
        elif kind == "LWPOLYLINE":
            vertices = []
            x = None
            for code, value in entity[1:]:
                if code == 10:
                    x = float(value)
                elif code == 20 and x is not None:
                    vertices.append((x, float(value)))
                    x = None
            flags = int(_first(entity, 70, "0"))
            polylines.append(Polyline(tuple(vertices), bool(flags & 1)))
    return Geometry(
        tuple(lines),
        tuple(circles),
        tuple(arcs),
        tuple(polylines),
    )


def sample_arc(arc: Arc, max_step: float = 0.5) -> np.ndarray:
    start = arc.start_angle_deg * pi / 180.0
    end = arc.end_angle_deg * pi / 180.0
    while end < start:
        end += 2.0 * pi
    count = max(2, int(np.ceil((end - start) * arc.radius / max_step)) + 1)
    angles = np.linspace(start, end, count)
    cx, cy = arc.center
    return np.column_stack(
        (cx + arc.radius * np.cos(angles), cy + arc.radius * np.sin(angles))
    )


def polyline_segments(polyline: Polyline) -> list[Line]:
    vertices = list(polyline.vertices)
    if polyline.closed and len(vertices) > 1:
        vertices.append(vertices[0])
    return [
        Line(vertices[index], vertices[index + 1])
        for index in range(len(vertices) - 1)
    ]
