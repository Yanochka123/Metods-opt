import numpy as np
import matplotlib.pyplot as plt
import time
import re
import math

def parse_function(func_str):
    """
    Поддерживает: x, sin, cos, tan, exp, log, sqrt, pi, e, и арифметику.
    """
    # Безопасная замена
    func_str = func_str.replace('^', '**')
    func_str = re.sub(r'\b(\d+\.?\d*)\s*\*\s*x\b', r'\1*x', func_str)  # не нужно
    allowed_names = {
        "x": None,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "ln": np.log,
        "sqrt": np.sqrt,
        "pi": np.pi,
        "e": np.e,
        "abs": np.abs,
    }

    def func(x):
        allowed_names["x"] = x
        try:
            return eval(func_str, {"__builtins__": {}}, allowed_names)
        except Exception as e:
            raise ValueError(f"Ошибка при вычислении функции: {e}")

    return func

def global_minimum_characteristic_method(f, a, b, eps, max_iter=10000):
    """
    Поиск глобального минимума методом ломаных (характеристик).
    Возвращает: xmin, fmin, iterations, time, history
    """
    start_time = time.time()

     # === ШАГ 1: Грубая оценка L на начальной сетке ===
    n_init = 7  # нечётное, чтобы включить a и b
    x_init = np.linspace(a, b, n_init)
    y_init = [f(x) for x in x_init]
    
    # Оцениваем L0
    L0 = 0.0
    for i in range(n_init - 1):
        dx = x_init[i+1] - x_init[i]
        if dx > 1e-12:
            slope = abs(y_init[i+1] - y_init[i]) / dx
            if slope > L0:
                L0 = slope
    
    # Если функция константа на сетке — ставим L = 1
    if L0 < 1e-12:
        L = 1.0
    else:
        L = 2.0 * L0  # запас γ = 2
    
    # Начальные точки
    x = [a, b]
    y = [f(a), f(b)]
    
    # История для визуализации
    history = {'x': x.copy(), 'y': y.copy(), 'L': [L]}


    # старт итерационных вычислений
    for iteration in range(2, max_iter + 1):
        # Вычисляем характеристики для каждого интервала
        R = []  # характеристики
        for i in range(len(x) - 1):
            dx = x[i+1] - x[i]
            # Характеристика: R_i = (y_i + y_{i+1})/2 - L * dx / 2
            # характеристикой является значение по y точек пересечения ломаных
            R_i = 0.5 * (y[i] + y[i+1]) - 0.5 * L * dx
            R.append(R_i) # пополням массив нижних вершин
        
        # Находим индекс в массиве точки c минимальным значением
        min_R_idx = int(np.argmin(R))
        
        # Точка нового испытания — середина интервала (можно точнее, но достаточно)
        x_new = 0.5 * (x[min_R_idx] + x[min_R_idx + 1]) + 0.5 * (y[min_R_idx] - y[min_R_idx+1])/L
        y_new = f(x_new)
        
        # Вставляем новую точку
        x.insert(min_R_idx + 1, x_new)
        y.insert(min_R_idx + 1, y_new)

        # так как массив нижних точек каждый раз выычисляется по формулам итеративно
        # из имеющихся точек, удалять из него переменную не нужно
        
        history['x'].append(x_new)
        history['y'].append(y_new)
        
        # Обновляем оценку L (максимальный наклон)
        L_candidates = []
        # Только соседние с новой точкой интервалы (для эффективности)
        i_left = min_R_idx      # (x[i_left], x_new)
        i_right = min_R_idx + 1 # (x_new, x[i_right+1])
        
        # Левый интервал
        dx = x_new - x[i_left]
        if dx > 1e-12:
            slope = abs(y_new - y[i_left]) / dx
            L_candidates.append(slope)
        
        # Правый интервал
        if i_right + 1 < len(x):
            dx = x[i_right + 1] - x_new
            if dx > 1e-12:
                slope = abs(y[i_right + 1] - y_new) / dx
                L_candidates.append(slope)
        
        if L_candidates:
            L_local = max(L_candidates)
            if L_local > L:
                L = L_local
        
        history['L'].append(L)

        # Условие останова: минимальная длина интервала < eps
        min_interval = y_new - R[min_R_idx]
        if min_interval < eps and iteration > 2:
            break
        
    
    # Найдём минимум
    min_idx = int(np.argmin(y))
    xmin = x[min_idx]
    fmin = y[min_idx]
    
    end_time = time.time()
    return xmin, fmin, iteration, end_time - start_time, history, x, y, L

def main():
    # Ввод данных
    func_str = input("Введите функцию f(x) (например, x + sin(3.14159*x)): ").strip()
    a = float(input("Введите левый конец отрезка a: "))
    b = float(input("Введите правый конец отрезка b: "))
    eps = float(input("Введите точность eps (например, 0.01): "))
    
    if a >= b:
        raise ValueError("Должно быть a < b")
    
    # Парсинг функции
    f = parse_function(func_str)
    
    # Вычисление минимума
    print("\nВыполняется поиск глобального минимума...")
    xmin, fmin, n_iter, exec_time, history, x_grid, y_grid, L_final = \
        global_minimum_characteristic_method(f, a, b, eps)
    
    # Вывод результатов
    print("\n✅ Результаты:")
    print(f"Приближённый аргумент минимума: x* = {xmin:.6f}")
    print(f"Минимальное значение функции:   f(x*) = {fmin:.6f}")
    print(f"Число итераций:                 {n_iter}")
    print(f"Затраченное время:              {exec_time:.4f} сек")
    print(f"Оценка константы Липшица:       L ≈ {L_final:.4f}")
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    
    # Плотная сетка для исходной функции
    x_dense = np.linspace(a, b, 1000)
    y_dense = np.array([f(x) for x in x_dense])
    plt.plot(x_dense, y_dense, 'b-', label='f(x)', linewidth=2)
    
    # Ломаная (нижняя оценка)
    # Строим ломаную: в каждой точке x[i] значение = y[i], соединяем прямыми
    plt.plot(x_grid, y_grid, 'ro--', markersize=4, label='Точки испытаний', alpha=0.7)
    
    # Найденный минимум
    plt.plot(xmin, fmin, 'go', markersize=10, label=f'Найденный минимум ({xmin:.3f}, {fmin:.3f})')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Поиск глобального минимума методом ломаных')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()