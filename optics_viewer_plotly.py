"""
optics_viewer_plotly.py

Viewer для оптических задач:
- трассировка лучей через несколько поверхностей (отражение/преломление);
- 3D-визуализация лучей и поверхностей (аналитически и численно) на Plotly.

Ожидается, что в той же папке лежит файл ядра (например, try.py),
в котором определены классы Vector, Ray, Surface, Plane, Plate, Glass
и функции get_reflected_ray, get_refrected_ray.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import importlib
import numpy as np
import plotly.graph_objects as go


# === Импорт ядра (файл optics_core.py / твой основной модуль) ===
_core = importlib.import_module("optics_core")

Vector = _core.Vector
Ray = _core.Ray
Surface = _core.Surface
Plane = _core.Plane
Plate = getattr(_core, "Plate", None)
Glass = getattr(_core, "Glass", None)

get_reflected_ray = _core.get_reflected_ray
get_refrected_ray = _core.get_refrected_ray  # да, с опечаткой, как в ядре


# === Структуры для трассировки ===

@dataclass
class OpticalElement:
    """
    Один оптический элемент:
    - surface: объект Surface/Plane/Plate/Glass и т.п.
    - interaction:
        "reflect"  – отражение (зеркало)
        "refract"  – преломление (граница сред)
        "none"     – пересечение считаем, но луч не продолжаем (экраны, детекторы)
    - n1, n2: опциональные показатели преломления для interaction="refract".
              Если у поверхности есть свой метод refract, они могут не понадобиться.
    - name: подпись для легенды/подписей.
    """
    surface: Surface
    interaction: str = "reflect"
    n1: Optional[float] = None
    n2: Optional[float] = None
    name: Optional[str] = None


@dataclass
class RaySegment:
    """
    Один отрезок траектории луча между двумя событиями (или до "улёта").
    """
    start: Vector
    end: Vector
    ray: Ray                     # входной луч на этом участке
    element: Optional[OpticalElement]  # на чём закончился сегмент (или None, если "улетел")
    hit_point: Optional[Vector]        # точка попадания (обычно совпадает с end, если element не None)


# === Базовая 3D-фигура ===

def create_figure_3d(
    title: str = "Optical viewer",
    width: int = 900,
    height: int = 700,
) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# === Вспомогательные функции для геометрии ===

def _vector_from_array(arr: Union[np.ndarray, Vector]) -> Vector:
    arr = np.asarray(arr, dtype=float)
    return Vector(arr[0], arr[1], arr[2])


def _distance(p1: Vector, p2: Vector) -> float:
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


# === Трассировка луча через набор элементов ===

def trace_ray_through_elements(
    ray: Ray,
    elements: Sequence[OpticalElement],
    max_events: int = 10,
    escape_distance: float = 100.0,
) -> List[RaySegment]:
    """
    Трассирует один луч через список элементов.

    Алгоритм:
    - на каждом шаге ищем БЛИЖАЙШЕЕ пересечение луча с любой поверхностью;
    - строим сегмент от текущего начала до точки пересечения;
    - в зависимости от interaction у элемента продолжаем луч (reflect / refract) или останавливаем.
    - если пересечений нет — добавляем сегмент "улетел вперёд" и выходим.
    """
    segments: List[RaySegment] = []
    current_ray = ray

    for _ in range(max_events):
        origin = current_ray.origin

        closest_elem: Optional[OpticalElement] = None
        closest_point: Optional[Vector] = None
        closest_dist: Optional[float] = None

        # ищем ближайшее пересечение
        for elem in elements:
            point = elem.surface.intersect(current_ray)
            if point is None:
                continue

            dist = _distance(origin, point)
            # отбрасываем слишком близкие точки (численный шум)
            if dist < 1e-8:
                continue

            if closest_dist is None or dist < closest_dist:
                closest_dist = dist
                closest_point = point
                closest_elem = elem

        # пересечений нет — продолжаем луч "в никуда" и выходим
        if closest_elem is None or closest_point is None:
            d = current_ray.direction_normalized()
            arr_end = np.asarray(origin) + np.asarray(d) * escape_distance
            end = _vector_from_array(arr_end)
            segments.append(
                RaySegment(
                    start=origin,
                    end=end,
                    ray=current_ray,
                    element=None,
                    hit_point=None,
                )
            )
            break

        # добавляем сегмент до точки пересечения
        segments.append(
            RaySegment(
                start=origin,
                end=closest_point,
                ray=current_ray,
                element=closest_elem,
                hit_point=closest_point,
            )
        )

        # решаем, как продолжается луч
        if closest_elem.interaction == "none":
            # например, экран: луч дошёл и всё
            break

        elif closest_elem.interaction == "reflect":
            # отражение
            if hasattr(closest_elem.surface, "reflect"):
                new_ray = closest_elem.surface.reflect(current_ray)  # type: ignore[call-arg]
            else:
                new_ray = get_reflected_ray(current_ray, closest_elem.surface)

        elif closest_elem.interaction == "refract":
            # преломление
            if hasattr(closest_elem.surface, "refract"):
                # если у элемента заданы n1/n2, используем их, иначе даём поверхности решать самой
                if closest_elem.n1 is not None or closest_elem.n2 is not None:
                    new_ray = closest_elem.surface.refract(  # type: ignore[call-arg]
                        current_ray,
                        closest_elem.n1,
                        closest_elem.n2,
                    )
                else:
                    new_ray = closest_elem.surface.refract(current_ray)  # type: ignore[call-arg]
            else:
                # нет собственного метода refract -> используем глобальную функцию,
                # тогда n1 и n2 ОБЯЗАТЕЛЬНО должны быть заданы
                if closest_elem.n1 is None or closest_elem.n2 is None:
                    raise ValueError(
                        "For refracting element without 'refract' method "
                        "you must specify n1 and n2."
                    )
                new_ray = get_refrected_ray(
                    current_ray,
                    closest_elem.surface,
                    closest_elem.n1,
                    closest_elem.n2,
                )
        else:
            raise ValueError(f"Unknown interaction type: {closest_elem.interaction!r}")

        if new_ray is None:
            # например, полное внутреннее отражение для преломления -> луч не продолжаем
            break

        current_ray = new_ray

    return segments


# === Рисование траекторий лучей ===

def add_ray_path(
    fig: go.Figure,
    segments: Sequence[RaySegment],
    name: Optional[str] = None,
    show_points: bool = True,
    **line_kwargs,
) -> go.Figure:
    """
    Добавляет на фигуру одну непрерывную траекторию луча (несколько сегментов).
    """
    if not segments:
        return fig

    # строим полилинию: первая точка = start первого сегмента,
    # дальше берём end каждого сегмента по порядку
    xs = [segments[0].start.x]
    ys = [segments[0].start.y]
    zs = [segments[0].start.z]

    for seg in segments:
        xs.append(seg.end.x)
        ys.append(seg.end.y)
        zs.append(seg.end.z)

    line_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name=name or "ray",
        line=dict(**line_kwargs),
    )
    fig.add_trace(line_trace)

    if show_points:
        # точки пересечений (кроме последнего "улёта")
        hit_x, hit_y, hit_z = [], [], []
        for seg in segments:
            if seg.hit_point is not None and seg.element is not None:
                hit_x.append(seg.hit_point.x)
                hit_y.append(seg.hit_point.y)
                hit_z.append(seg.hit_point.z)

        if hit_x:
            points_trace = go.Scatter3d(
                x=hit_x,
                y=hit_y,
                z=hit_z,
                mode="markers",
                name=(name or "ray") + " hits",
                marker=dict(size=4),
            )
            fig.add_trace(points_trace)

    return fig


def add_multiple_ray_paths(
    fig: go.Figure,
    rays: Sequence[Ray],
    elements: Sequence[OpticalElement],
    max_events: int = 10,
    escape_distance: float = 100.0,
    base_name: str = "ray",
    show_points: bool = True,
    **line_kwargs,
) -> go.Figure:
    """
    Трассирует и рисует несколько лучей.
    """
    for i, ray in enumerate(rays):
        segments = trace_ray_through_elements(
            ray,
            elements,
            max_events=max_events,
            escape_distance=escape_distance,
        )
        name = f"{base_name} {i}"
        add_ray_path(fig, segments, name=name, show_points=show_points, **line_kwargs)
    return fig


# === Визуализация поверхностей ===

def _plane_parametrization(
    plane: Plane,
    u_range: Tuple[float, float],
    v_range: Tuple[float, float],
    resolution: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Задаём сетку в подходящих координатах и выражаем третью координату
    из уравнения плоскости A x + B y + C z + D = 0.
    """
    A, B, C, D = plane.A, plane.B, plane.C, plane.D

    u_min, u_max = u_range
    v_min, v_max = v_range

    u = np.linspace(u_min, u_max, resolution)
    v = np.linspace(v_min, v_max, resolution)
    U, V = np.meshgrid(u, v)

    eps = 1e-12

    if abs(C) > eps:
        X = U
        Y = V
        Z = -(A * X + B * Y + D) / C
    elif abs(B) > eps:
        X = U
        Z = V
        Y = -(A * X + C * Z + D) / B
    elif abs(A) > eps:
        Y = U
        Z = V
        X = -(B * Y + C * Z + D) / A
    else:
        raise ValueError("Degenerate plane: A=B=C=0")

    return X, Y, Z


def add_plane(
    fig: go.Figure,
    plane: Plane,
    u_range: Tuple[float, float] = (-10.0, 10.0),
    v_range: Tuple[float, float] = (-10.0, 10.0),
    resolution: int = 30,
    name: Optional[str] = None,
    opacity: float = 0.4,
    showscale: bool = False,
    **surface_kwargs,
) -> go.Figure:
    """
    Рисует плоскость как гладкую поверхность (go.Surface).
    """
    X, Y, Z = _plane_parametrization(plane, u_range, v_range, resolution)

    surf = go.Surface(
        x=X,
        y=Y,
        z=Z,
        name=name or "plane",
        opacity=opacity,
        showscale=showscale,
        **surface_kwargs,
    )
    fig.add_trace(surf)
    return fig


def add_implicit_surface_points(
    fig: go.Figure,
    surface: Surface,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    resolution: int = 25,
    iso_tol: float = 0.05,
    name: Optional[str] = None,
    **marker_kwargs,
) -> go.Figure:
    """
    Численно визуализирует поверхность F(x,y,z)=0:
    берём 3D-сетку и рисуем точки, где |F| < iso_tol.

    Это не гладкая поверхность, а "облако" точек, но даёт хорошее
    представление о форме поверхности для численных экспериментов.
    """
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    zs = np.linspace(z_range[0], z_range[1], resolution)

    pts_x, pts_y, pts_z = [], [], []

    for x in xs:
        for y in ys:
            for z in zs:
                val = surface.F(x, y, z)
                if abs(val) <= iso_tol:
                    pts_x.append(x)
                    pts_y.append(y)
                    pts_z.append(z)

    if not pts_x:
        # ничего не нашли – либо нужен другой диапазон, либо tol
        return fig

    scatter = go.Scatter3d(
        x=pts_x,
        y=pts_y,
        z=pts_z,
        mode="markers",
        name=name or "surface",
        marker=dict(size=2, **marker_kwargs),
    )
    fig.add_trace(scatter)
    return fig


# === Высокоуровневый viewer ===

def plot_system(
    rays: Sequence[Ray],
    elements: Sequence[OpticalElement],
    *,
    max_events: int = 10,
    escape_distance: float = 100.0,
    plane_ranges: Tuple[Tuple[float, float], Tuple[float, float]] = ((-10, 10), (-10, 10)),
    implicit_range: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-10, 10),
        (-10, 10),
        (-10, 10),
    ),
    implicit_resolution: int = 25,
    implicit_iso_tol: float = 0.05,
    title: str = "Optical system",
    show_points: bool = True,
) -> go.Figure:
    """
    Один вызов, чтобы:
    - нарисовать все поверхности (плоские как поверхности, остальные как численные точки),
    - трассировать и нарисовать все лучи.

    plane_ranges: диапазоны для параметризации плоскостей (u_range, v_range).
    implicit_range: диапазоны (x_range, y_range, z_range) для численных поверхностей.
    """
    fig = create_figure_3d(title=title)

    u_range, v_range = plane_ranges
    x_range, y_range, z_range = implicit_range

    # 1) Рисуем поверхности
    for elem in elements:
        surf = elem.surface
        label = elem.name or elem.interaction

        if isinstance(surf, Plane):
            add_plane(
                fig,
                surf,
                u_range=u_range,
                v_range=v_range,
                name=label,
                opacity=0.4,
            )
        else:
            # произвольная поверхность -> численный level set
            add_implicit_surface_points(
                fig,
                surf,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                resolution=implicit_resolution,
                iso_tol=implicit_iso_tol,
                name=label,
            )

    # 2) Трассируем и рисуем лучи
    add_multiple_ray_paths(
        fig,
        rays,
        elements,
        max_events=max_events,
        escape_distance=escape_distance,
        base_name="ray",
        show_points=show_points,
    )

    return fig


def show_figure(fig: go.Figure) -> None:
    fig.show()


# === Пример использования ===

if __name__ == "__main__":
    # Пример: луч отражается от плоскости и дальше летит.
    # Здесь предполагается, что в try.py есть класс Plate.
    if Plate is None:
        raise SystemExit("В ядре нет Plate; добавь или поправь пример в optics_viewer_plotly.py")

    # Плоское зеркало z = 5
    mirror_surface = Plate(0.0, 0.0, 1.0, -5.0)
    mirror = OpticalElement(surface=mirror_surface, interaction="reflect", name="mirror")

    # Луч, идущий из (0,0,0) в сторону (1, 0, 2)
    ray0 = Ray(0.0, 1.0, 0.0, 0.0, 0.0, 2.0)

    fig = plot_system(
        rays=[ray0],
        elements=[mirror],
        max_events=5,
        escape_distance=50.0,
        plane_ranges=((-10, 10), (-10, 10)),
        implicit_range=((-10, 10), (-10, 10), (-10, 10)),
        title="Simple reflection example",
    )

    show_figure(fig)
