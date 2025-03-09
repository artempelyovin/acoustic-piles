import numpy as np
import matplotlib.pyplot as plt


def generate_parabola(
    vertex: tuple[float, float], width: float, num_points: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """
    Рисует параболу с вершиной (h, k), где k > 0, ветви направлены вниз и график заканчивается при y = 0.

    Параметры:
      vertex     - кортеж (h, k), где h - абсцисса вершины, k - ордината вершины (обязательно k > 0)
      width      - расстояние от вершины до точки пересечения оси y (где y = 0)
      num_points - (опционально) число точек для построения графика (по умолчанию 500)
    """
    h, k = vertex
    if k < 0:
        raise ValueError("Ордината вершины должна быть больше нуля (k > 0).")
    if width <= 0:
        raise ValueError("Параметр width должен быть положительным.")

    # Вычисление коэффициента параболы a: a = - k / width^2 так, что f(h ± width) = 0.
    a = -k / (width**2)

    # Определяем диапазон по оси x от h - width до h + width
    x = np.linspace(h - width, h + width, num_points)
    y = a * (x - h) ** 2 + k
    return x, y


def generate_acoustic_signal() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Генерирует "фейковый" акустический сигнал удара молотка по бетонной свае
    :return: x, y координаты функции, изображающей сигнал и нулевые точки x, в которых функция меняет знак
    """
    segments = np.random.randint(15, 20)
    y_coefficient = 1.0
    cur_width = np.random.uniform(2, 5)
    cur_y = np.random.uniform(70, 120)
    cur_x = cur_width

    x_all = []
    y_all = []
    zero_crossings = []
    last_sign = "plus"

    for segment in range(1, segments + 1):
        cur_y *= y_coefficient - (segment / segments)
        cur_y = max(cur_y, 0.75)
        x, y = generate_parabola(vertex=(cur_x, cur_y), width=cur_width, num_points=100)
        if np.random.random() < 0.5:  # Разворачиваем ветви параболы с вероятностью 50%
            y = -y
            cur_sign = "minus"
        else:
            cur_sign = "plus"

        x_all.append(x)
        y_all.append(y)
        if cur_sign != last_sign:
            zero_crossings.append(cur_x-cur_width)

        new_width = cur_width + cur_width * 0.07
        cur_x = cur_x + new_width + cur_width
        cur_width = new_width
        last_sign = cur_sign

    return np.concatenate(x_all), np.concatenate(y_all), np.array(zero_crossings)


x, y, zero_crossings_xs = generate_acoustic_signal()

plt.plot(x, y, "black")
for zero_crossings_x in zero_crossings_xs:
    plt.axvline(x=zero_crossings_x, color='red', linestyle='dotted')

plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.show()
