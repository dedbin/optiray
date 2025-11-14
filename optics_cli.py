"""
Простой консольный интерфейс для работы с оптической системой.

Зависит от:
    - optics_core.py        (Ray, Plane, Plate, Mirror, Glass, ...)
    - optics_viewer_plotly.py (OpticalElement, plot_system, show_figure)
"""

import math
import numpy as np

import optics_core as core
from optics_viewer_plotly import OpticalElement, plot_system, show_figure


# --- Удобные псевдонимы --- #

Ray = core.Ray
Plane = core.Plane
Plate = core.Plate
Mirror = core.Mirror
Glass = core.Glass


# --- Глобальные списки объектов --- #

rays = []       # список объектов Ray
elements = []   # список объектов OpticalElement


# --- Вспомогательные функции ввода --- #

def input_float(prompt, default=None):
    """
    Ввод числа с плавающей точкой.
    Если ничего не ввели и есть default — возвращаем default.
    """
    while True:
        if default is None:
            s = input(f"{prompt}: ")
        else:
            s = input(f"{prompt} [{default}]: ")

        s = s.strip()
        if s == "":
            return default

        try:
            return float(s)
        except ValueError:
            print("Нужно ввести число.")


def input_int(prompt, default=None):
    while True:
        if default is None:
            s = input(f"{prompt}: ")
        else:
            s = input(f"{prompt} [{default}]: ")

        s = s.strip()
        if s == "":
            return default

        try:
            return int(s)
        except ValueError:
            print("Нужно ввести целое число.")


def input_str(prompt, default=None):
    if default is None:
        s = input(f"{prompt}: ")
    else:
        s = input(f"{prompt} [{default}]: ")
    s = s.strip()
    if s == "":
        return default
    return s


def choose_index(n, what="элемент"):
    """
    Выбрать индекс от 0 до n-1.
    Возвращает индекс или None.
    """
    if n == 0:
        print(f"Нет ни одного {what}.")
        return None

    while True:
        i = input_int(f"Введите номер {what} (0..{n-1})", default=0)
        if i is None:
            return None
        if 0 <= i < n:
            return i
        print("Неверный номер.")


# --- F(x,y,z) из строкового выражения --- #

def make_F_from_expr(expr):
    """
    expr: строка, например 'x**2 + y**2 + z**2 - 25'.
    Возвращает функцию F(x, y, z), которая считает это выражение
    в окружении x, y, z, np.
    """
    expr = expr.strip()

    def F(x, y, z):
        local_vars = {"x": x, "y": y, "z": z, "np": np}
        return eval(expr, {"__builtins__": {}}, local_vars)

    return F


# --- Работа с лучами --- #

def list_rays():
    if not rays:
        print("Лучей нет.")
        return
    print("Список лучей:")
    for i, r in enumerate(rays):
        print(f"  [{i}] (x-{r.x0})/{r.a} = (y-{r.y0})/{r.b} = (z-{r.z0})/{r.c}, frec={r.frec}")


def add_ray():
    print("Добавление нового луча.")
    x0 = input_float("x0", 0.0)
    a = input_float("a", 0.0)
    y0 = input_float("y0", 0.0)
    b = input_float("b", 0.0)
    z0 = input_float("z0", 0.0)
    c = input_float("c", 1.0)
    frec = input_float("frec", 0.0)

    ray = Ray(x0, a, y0, b, z0, c, frec)
    rays.append(ray)
    print("Луч добавлен.")


def edit_ray():
    if not rays:
        print("Лучей нет.")
        return
    list_rays()
    idx = choose_index(len(rays), "луча")
    if idx is None:
        return

    r = rays[idx]
    print("Редактирование луча:", r)

    x0 = input_float("x0", r.x0)
    a = input_float("a", r.a)
    y0 = input_float("y0", r.y0)
    b = input_float("b", r.b)
    z0 = input_float("z0", r.z0)
    c = input_float("c", r.c)
    frec = input_float("frec", r.frec)

    rays[idx] = Ray(x0, a, y0, b, z0, c, frec)
    print("Луч обновлён.")


def delete_ray():
    if not rays:
        print("Лучей нет.")
        return
    list_rays()
    idx = choose_index(len(rays), "луча")
    if idx is None:
        return
    rays.pop(idx)
    print("Луч удалён.")


# --- Работа с оптическими элементами --- #

def list_elements():
    if not elements:
        print("Элементов нет.")
        return
    print("Список оптических элементов:")
    for i, elem in enumerate(elements):
        surf = elem.surface
        kind = type(surf).__name__
        name = elem.name or ""
        print(f"  [{i}] {kind}, interaction={elem.interaction}, name='{name}'")


def add_element():
    print("Добавление нового оптического элемента.")
    print("Тип элемента:")
    print("  1) Plane (A x + B y + C z + D = 0)")
    print("  2) Plate (отражающая плоскость)")
    print("  3) Mirror (зеркало общего вида, F(x,y,z)=0)")
    print("  4) Glass (преломляющая поверхность, F(x,y,z)=0)")

    choice = input_int("Выберите тип (1-4)", 1)

    if choice == 1:
        kind = "Plane"
    elif choice == 2:
        kind = "Plate"
    elif choice == 3:
        kind = "Mirror"
    elif choice == 4:
        kind = "Glass"
    else:
        print("Неверный выбор.")
        return

    # общие параметры
    name = input_str("Имя элемента", "")
    print("Тип взаимодействия с лучом:")
    print("  1) reflect (отражение)")
    print("  2) refract (преломление)")
    print("  3) none    (просто пересечение, луч дальше не идёт)")
    inter_choice = input_int("Выбор (1-3)", 1)
    if inter_choice == 1:
        interaction = "reflect"
    elif inter_choice == 2:
        interaction = "refract"
    else:
        interaction = "none"

    # показатели преломления для общего случая refract
    n1 = input_float("n1 (для refract, можно пусто)", None)
    n2 = input_float("n2 (для refract, можно пусто)", None)

    # создаём поверхность
    if kind in ("Plane", "Plate"):
        print("Введи коэффициенты уравнения плоскости: A x + B y + C z + D = 0")
        A = input_float("A", 0.0)
        B = input_float("B", 0.0)
        C = input_float("C", 1.0)
        D = input_float("D", 0.0)

        if kind == "Plane":
            surface = Plane(A, B, C, D)
        else:
            surface = Plate(A, B, C, D)

    elif kind == "Mirror":
        print("Задай функцию F(x,y,z) = 0, например: x**2 + y**2 + z**2 - 25")
        expr = input_str("F(x,y,z)", "x**2 + y**2 + z**2 - 25")
        func = make_F_from_expr(expr)
        surface = Mirror(func)

    elif kind == "Glass":
        print("Задай функцию F(x,y,z) = 0 для поверхности границы стекла")
        expr = input_str("F(x,y,z)", "x**2 + y**2 + z**2 - 25")
        func = make_F_from_expr(expr)
        n_inside = input_float("n_inside (показатель внутри)", 1.5)
        n_outside = input_float("n_outside (снаружи)", 1.0)
        surface = Glass(func, n_inside=n_inside, n_outside=n_outside)
    else:
        print("Неизвестный тип.")
        return

    elem = OpticalElement(
        surface=surface,
        interaction=interaction,
        n1=n1,
        n2=n2,
        name=name,
    )
    elements.append(elem)
    print("Элемент добавлен.")


def edit_element():
    if not elements:
        print("Элементов нет.")
        return
    list_elements()
    idx = choose_index(len(elements), "элемента")
    if idx is None:
        return

    elem = elements[idx]
    surf = elem.surface
    kind = type(surf).__name__

    print(f"Редактирование элемента [{idx}], тип {kind}.")

    # общие параметры
    name = input_str("Имя элемента", elem.name or "")
    print("Тип взаимодействия с лучом:")
    print("  1) reflect (отражение)")
    print("  2) refract (преломление)")
    print("  3) none    (просто пересечение)")
    if elem.interaction == "reflect":
        default_inter = 1
    elif elem.interaction == "refract":
        default_inter = 2
    else:
        default_inter = 3
    inter_choice = input_int("Выбор (1-3)", default_inter)
    if inter_choice == 1:
        interaction = "reflect"
    elif inter_choice == 2:
        interaction = "refract"
    else:
        interaction = "none"

    n1 = input_float("n1 (для refract, можно пусто)", elem.n1)
    n2 = input_float("n2 (для refract, можно пусто)", elem.n2)

    # специфические параметры
    if isinstance(surf, Plane) and not isinstance(surf, Plate):
        print("Plane: A x + B y + C z + D = 0")
        A = input_float("A", surf.A)
        B = input_float("B", surf.B)
        C = input_float("C", surf.C)
        D = input_float("D", surf.D)
        surface = Plane(A, B, C, D)

    elif isinstance(surf, Plate):
        print("Plate (зеркальная плоскость): A x + B y + C z + D = 0")
        A = input_float("A", surf.A)
        B = input_float("B", surf.B)
        C = input_float("C", surf.C)
        D = input_float("D", surf.D)
        surface = Plate(A, B, C, D)

    elif isinstance(surf, Mirror):
        print("Mirror: F(x,y,z) = 0")
        # мы не храним исходную строку expr, поэтому просто даём дефолт
        expr = input_str("Новая F(x,y,z)", "x**2 + y**2 + z**2 - 25")
        func = make_F_from_expr(expr)
        surface = Mirror(func)

    elif isinstance(surf, Glass):
        print("Glass: F(x,y,z) = 0")
        expr = input_str("Новая F(x,y,z)", "x**2 + y**2 + z**2 - 25")
        func = make_F_from_expr(expr)
        n_inside = input_float("n_inside", surf.n_inside)
        n_outside = input_float("n_outside", surf.n_outside)
        surface = Glass(func, n_inside=n_inside, n_outside=n_outside)
    else:
        print("Неизвестный тип поверхности, переписываем как Mirror.")
        expr = input_str("F(x,y,z)", "x**2 + y**2 + z**2 - 25")
        func = make_F_from_expr(expr)
        surface = Mirror(func)

    elements[idx] = OpticalElement(
        surface=surface,
        interaction=interaction,
        n1=n1,
        n2=n2,
        name=name,
    )
    print("Элемент обновлён.")


def delete_element():
    if not elements:
        print("Элементов нет.")
        return
    list_elements()
    idx = choose_index(len(elements), "элемента")
    if idx is None:
        return
    elements.pop(idx)
    print("Элемент удалён.")


# --- Показ системы --- #

def show_system():
    if not rays:
        print("Нет ни одного луча, но всё равно покажем поверхности.")
    fig = plot_system(
        rays=rays,
        elements=elements,
        max_events=10,
        escape_distance=50.0,
        plane_ranges=((-10, 10), (-10, 10)),
        implicit_range=((-10, 10), (-10, 10), (-10, 10)),
        implicit_resolution=20,
        implicit_iso_tol=0.05,
        title="Optical system",
        show_points=True,
    )
    show_figure(fig)


# --- Главное меню --- #

def main_menu():
    while True:
        print("\n=== Оптический viewer ===")
        print("1) Показать список лучей")
        print("2) Добавить луч")
        print("3) Изменить луч")
        print("4) Удалить луч")
        print("5) Показать список оптических элементов")
        print("6) Добавить элемент")
        print("7) Изменить элемент")
        print("8) Удалить элемент")
        print("9) Показать систему (Plotly)")
        print("0) Выход")

        choice = input_int("Выбор", 9)

        if choice == 1:
            list_rays()
        elif choice == 2:
            add_ray()
        elif choice == 3:
            edit_ray()
        elif choice == 4:
            delete_ray()
        elif choice == 5:
            list_elements()
        elif choice == 6:
            add_element()
        elif choice == 7:
            edit_element()
        elif choice == 8:
            delete_element()
        elif choice == 9:
            show_system()
        elif choice == 0:
            print("Выход.")
            break
        else:
            print("Неверный пункт меню.")


if __name__ == "__main__":
    main_menu()
