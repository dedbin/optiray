import numpy as np
from typing import Callable, Optional, Tuple, Union

#TODO: добавить объект экран (плоскость), на который падают лучи, и вычисление точек падения лучей на экран

class Vector(np.ndarray):
    def __new__(cls, x: float, y: float, z: float):
        arr = np.asarray([x, y, z], dtype=float).view(cls)
        return arr

    @property
    def x(self) -> float:
        return float(self[0])

    @property
    def y(self) -> float:
        return float(self[1])

    @property
    def z(self) -> float:
        return float(self[2])

    def __repr__(self) -> str:
        return f"Vector({self.x:.6g}, {self.y:.6g}, {self.z:.6g})"


class Ray:
    def __init__(self, x0: float, a: float,
                 y0: float, b: float,
                 z0: float, c: float,
                 frec: float = 0.0):
        self.x0 = float(x0)
        self.a = float(a)
        self.y0 = float(y0)
        self.b = float(b)
        self.z0 = float(z0)
        self.c = float(c)
        self.frec = frec

    def __repr__(self) -> str:
        return (
            f"ray((x-{self.x0})/{self.a} = (y-{self.y0})/{self.b} = "
            f"(z-{self.z0})/{self.c}, frec={self.frec})"
        )

    @property
    def origin(self) -> Vector:
        return Vector(self.x0, self.y0, self.z0)

    @property
    def direction(self) -> Vector:
        return Vector(self.a, self.b, self.c)

    def direction_normalized(self) -> Vector:
        d = self.direction
        norm = np.linalg.norm(d)
        if norm == 0.0:
            return d
        return Vector(*(d / norm))

    def point_at(self, t: float) -> Vector:
        return Vector(
            self.x0 + self.a * t,
            self.y0 + self.b * t,
            self.z0 + self.c * t,
        )


class Surface:
    def __init__(
        self,
        func: Optional[Callable[[float, float, float], float]] = None,
        gradient: Optional[Callable[[float, float, float], Tuple[float, float, float]]] = None,
    ):
        self._func = func
        self._gradient = gradient

    def F(self, x: float, y: float, z: float) -> float:
        if self._func is None:
            raise NotImplementedError("F(x,y,z) is not defined for this Surface")
        return float(self._func(x, y, z))

    def get_norm(self, x: float, y: float, z: float) -> Vector:
        if self._gradient is not None:
            gx, gy, gz = self._gradient(x, y, z)
        else:
            eps = 1e-6
            gx = (self.F(x + eps, y, z) - self.F(x - eps, y, z)) / (2 * eps)
            gy = (self.F(x, y + eps, z) - self.F(x, y - eps, z)) / (2 * eps)
            gz = (self.F(x, y, z + eps) - self.F(x, y, z - eps)) / (2 * eps)
        n = Vector(gx, gy, gz)
        norm = np.linalg.norm(n)
        if norm == 0.0:
            return n
        return Vector(*(n / norm))

    def intersect(
        self,
        ray: Ray,
        t_max: float = 100.0,
        steps: int = 10000,
    ) -> Optional[Vector]:
        x0, y0, z0 = ray.x0, ray.y0, ray.z0
        a, b, c = ray.a, ray.b, ray.c

        def F_of_t(t: float) -> float:
            x = x0 + a * t
            y = y0 + b * t
            z = z0 + c * t
            return self.F(x, y, z)

        t_prev = 0.0
        f_prev = F_of_t(t_prev)

        # если стартовая точка уже на поверхности
        if abs(f_prev) < 1e-12:
            return ray.point_at(t_prev)

        for i in range(1, steps + 1):
            t = t_max * i / steps
            f = F_of_t(t)

            if abs(f) < 1e-12:
                return ray.point_at(t)

            if f_prev * f < 0.0:
                lo, hi = t_prev, t
                flo, fhi = f_prev, f

                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    fm = F_of_t(mid)
                    if flo * fm <= 0.0:
                        hi, fhi = mid, fm
                    else:
                        lo, flo = mid, fm

                t_root = 0.5 * (lo + hi)
                return ray.point_at(t_root)

            t_prev, f_prev = t, f

        return None


class Plane(Surface):
    def __init__(self, A: float, B: float, C: float, D: float):
        super().__init__(func=None, gradient=None)
        self.A = float(A)
        self.B = float(B)
        self.C = float(C)
        self.D = float(D)

        n = Vector(self.A, self.B, self.C)
        norm = np.linalg.norm(n)
        if norm == 0.0:
            self._normal = n
        else:
            self._normal = Vector(*(n / norm))

    def F(self, x: float, y: float, z: float) -> float:
        return self.A * x + self.B * y + self.C * z + self.D

    def get_norm(self, x: float, y: float, z: float) -> Vector:
        # нормаль везде одинакова
        return self._normal

    def intersect(self, ray: Ray, *_, **__) -> Optional[Vector]:
        denom = self.A * ray.a + self.B * ray.b + self.C * ray.c
        if abs(denom) < 1e-12:
            # луч параллелен плоскости
            return None

        t = -(self.A * ray.x0 + self.B * ray.y0 + self.C * ray.z0 + self.D) / denom
        if t < 0.0:
            # пересечение "за спиной" луча
            return None

        return ray.point_at(t)


class ReflectiveSurfaceMixin:
    def reflect(
        self,
        ray: Ray,
        with_point: bool = False,
    ) -> Optional[Union[Ray, Tuple[Ray, Vector]]]:
        result = _compute_reflection(self, ray)
        if result is None:
            return None
        point, v_ref = result

        if any(map(np.isnan, v_ref)):
            return None

        new_ray = Ray(point.x, v_ref[0], point.y, v_ref[1], point.z, v_ref[2], ray.frec)
        if with_point:
            return new_ray, point
        return new_ray


class Mirror(ReflectiveSurfaceMixin, Surface):
    def __init__(
        self,
        func: Callable[[float, float, float], float],
        gradient: Optional[Callable[[float, float, float], Tuple[float, float, float]]] = None,
    ):
        super().__init__(func, gradient=gradient)


class Plate(ReflectiveSurfaceMixin, Plane):
    def __init__(self, A: float, B: float, C: float, D: float):
        Plane.__init__(self, A, B, C, D)


class Glass(Surface):
    def __init__(
        self,
        func: Callable[[float, float, float], float],
        n_inside: float,
        n_outside: float = 1.0,
        gradient: Optional[Callable[[float, float, float], Tuple[float, float, float]]] = None,
    ):
        super().__init__(func, gradient=gradient)
        self.n_inside = float(n_inside)
        self.n_outside = float(n_outside)

    def refract(
        self,
        ray: Ray,
        n1: Optional[float] = None,
        n2: Optional[float] = None,
        with_point: bool = False,
    ) -> Optional[Union[Ray, Tuple[Ray, Vector]]]:
        if n1 is None:
            n1 = self.n_outside
        if n2 is None:
            n2 = self.n_inside

        result = _compute_refraction(self, ray, n1, n2)
        if result is None:
            return None

        point, v_refr = result
        new_ray = Ray(point.x, v_refr[0], point.y, v_refr[1], point.z, v_refr[2], ray.frec)
        if with_point:
            return new_ray, point
        return new_ray


def _compute_reflection(
    surface: Surface,
    ray: Ray,
) -> Optional[Tuple[Vector, Vector]]:
    point = surface.intersect(ray)
    if point is None:
        return None

    n = surface.get_norm(point.x, point.y, point.z)
    n = np.asarray(n, dtype=float)
    norm_n = np.linalg.norm(n)
    if norm_n == 0.0:
        return None
    n = n / norm_n

    d = np.asarray(ray.direction_normalized(), dtype=float)
    # формула отражения: r = d - 2 (d·n) n
    dot_dn = float(np.dot(d, n))
    r = d - 2.0 * dot_dn * n

    return point, Vector(*r)


def _compute_refraction(
    surface: Surface,
    ray: Ray,
    n1: float,
    n2: float,
) -> Optional[Tuple[Vector, Vector]]:
    point = surface.intersect(ray)
    if point is None:
        return None

    n = surface.get_norm(point.x, point.y, point.z)
    n = np.asarray(n, dtype=float)
    norm_n = np.linalg.norm(n)
    if norm_n == 0.0:
        return None
    n = n / norm_n

    d = np.asarray(ray.direction_normalized(), dtype=float)

    # cos(theta_i) = -n·d (n направлена в сторону из второй среды в первую)
    cosi = -float(np.dot(n, d))
    # если cosi < 0, значит нормаль "с той стороны", разворачиваем
    if cosi < 0.0:
        n = -n
        cosi = -cosi
        n1, n2 = n2, n1

    eta = n1 / n2
    k = 1.0 - eta**2 * (1.0 - cosi**2)

    # полное внутреннее отражение
    if k < 0.0:
        return None

    import math

    cost = math.sqrt(k)
    t = eta * d + (eta * cosi - cost) * n
    return point, Vector(*t)


def get_angle_from_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("Zero-length vector given to get_angle_from_vectors")

    cos_angle = float(np.dot(v1, v2) / (norm1 * norm2))
    # из-за численных ошибок cos может слегка вылезать за [-1,1]
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return float(np.arccos(cos_angle))


def intersect_ray_surface(ray: Ray, surface: Surface) -> Optional[Vector]:
    if not hasattr(surface, "intersect"):
        raise TypeError("surface must implement intersect(ray)")
    return surface.intersect(ray)


def get_reflected_ray(ray: Ray, surface: Surface) -> Optional[Ray]:
    if hasattr(surface, "reflect"):
        return surface.reflect(ray)  # type: ignore[arg-type]

    result = _compute_reflection(surface, ray)
    if result is None:
        return None
    point, v_ref = result
    return Ray(point.x, v_ref[0], point.y, v_ref[1], point.z, v_ref[2], ray.frec)


def get_refrected_ray(
    ray: Ray,
    surface: Surface,
    n1: float,
    n2: float,
) -> Optional[Ray]:
    if hasattr(surface, "refract"):
        return surface.refract(ray, n1, n2)  # type: ignore[arg-type]

    result = _compute_refraction(surface, ray, n1, n2)
    if result is None:
        return None
    point, v_refr = result
    return Ray(point.x, v_refr[0], point.y, v_refr[1], point.z, v_refr[2], ray.frec)


def get_reflected_ray_from_surface(
    ray: Ray,
    surface: Surface,
) -> Optional[Tuple[Vector, Vector]]:
    return _compute_reflection(surface, ray)


if __name__ == "__main__":
    plane = Plane(0.0, 0.0, 1.0, -5.0)  # z = 5
    ray = Ray(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    p = intersect_ray_surface(ray, plane)
    print("Пересечение луча с плоскостью:", p)

    reflected = get_reflected_ray(ray, Plate(0.0, 0.0, 1.0, -5.0))
    print("Отражённый луч:", reflected)
