import json
import os

import numpy as np
from keras import Sequential, Input
from keras.src.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, MaxPooling2D
from matplotlib.axes import Axes

X_SHAPE_RAW = 4000
X_SHAPE_GPH = (369, 496, 1)  # изображение размером 369x496 в одноканале
Y_SHAPE = 20


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
            zero_crossings.append(cur_x - cur_width)

        new_width = cur_width + cur_width * 0.07
        cur_x = cur_x + new_width + cur_width
        cur_width = new_width
        last_sign = cur_sign

    return np.concatenate(x_all), np.concatenate(y_all), np.array(zero_crossings)


def draw_acoustic_signal(ax: Axes, x: np.ndarray, y: np.ndarray) -> None:
    ax.plot(x, y, "black")
    ax.axhline(0, color="black", linewidth=0.5)


def draw_zero_crossings(
    ax: Axes, zero_crossings_xs: np.ndarray, color: str = "red", linestyle: str = "dotted", alpha: float = 1.0
) -> None:
    for zero_crossings_x in zero_crossings_xs:
        ax.axvline(x=zero_crossings_x, color=color, linestyle=linestyle, alpha=alpha)


def load_dataset__raw(dirpath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Загружает датасет "сырых" (в формате json) данных
    :param dirpath: путь до датасета
    :return: X и Y значения, где:
        - X - точки функции в формате (x;y)
        - Y - координаты x нулевых точек, в которых функция меняет знак
    """

    def load_raw_file(filepath: str) -> tuple[np.ndarray, np.ndarray]:
        with open(filepath, "r") as f:
            data = json.load(f)
            return np.array(data["points"]), np.array(data["answers"])

    all_x = []
    all_y = []
    for path_ in os.listdir(dirpath):
        points, answers = load_raw_file(f"{dirpath}/{path_}")
        # добиваем значением `-1` до нужного shape
        points = np.pad(points, (0, X_SHAPE_RAW - points.shape[0]), mode="constant", constant_values=(0, -1))
        answers = np.pad(answers, (0, Y_SHAPE - answers.shape[0]), mode="constant", constant_values=(0, -1))
        all_x.append(np.array(points))  # вход нейросети в виде точек [x1, y1, x2, y2, ..., xn, yn, -1, -1, ...., -1]
        all_y.append(answers)  # выход в виде координат x, где функция меняет знак
    return np.array(all_x), np.array(all_y)


def load_dataset__gph(dirpath_gph: str, dirpath_raw: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Загружает датасет изображений данных и датасет "сырых" данных
    :param dirpath_gph: путь до датасета с изображениями
    :param dirpath_raw: путь до датасета с "сырыми" данными
    :return: X и Y значения, где:
        - X - изображения
        - Y - координаты x нулевых точек, в которых функция меняет знак
    """

    def load_raw_file(filepath: str) -> tuple[np.ndarray, np.ndarray]:
        with open(filepath, "r") as f:
            data = json.load(f)
            return np.array(data["points"]), np.array(data["answers"])

    all_x = []
    all_y = []
    for path_ in os.listdir(dirpath_gph):
        points, answers = load_raw_file(f"{dirpath_raw}/{path_}")
        # добиваем значением `0` до нужного shape
        points = np.pad(points, (0, X_SHAPE_RAW - points.shape[0]), mode="constant", constant_values=(0, 0))
        answers = np.pad(answers, (0, Y_SHAPE - answers.shape[0]), mode="constant", constant_values=(0, 0))
        all_x.append(np.array(points))  # вход нейросети в виде точек [x1, y1, x2, y2, ..., xn, yn, 0, 0, ...., 0]
        all_y.append(answers)  # выход в виде координат x, где функция меняет знак
    return np.array(all_x), np.array(all_y)


def normalize_x(x: np.ndarray) -> np.ndarray:
    """Нормализует вектор чисел в диапазон [0;1]"""
    return (x - x.min()) / (x.max() - x.min())


def denormalize_x(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Денормализует вектор чисел в диапазоне [0;1] обратно в диапазон [x_min;x_max]"""
    return x * (x_max - x_min) + x_min


def generate_model__raw() -> Sequential:
    return Sequential(
        [
            Input(shape=(X_SHAPE_RAW,)),
            Dense(512),
            Dropout(0.2),
            Dense(256),
            Dropout(0.2),
            Dense(128),
            Dense(Y_SHAPE, activation="linear"),
        ]
    )


def generate_model__gph() -> Sequential:
    return Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=X_SHAPE_GPH),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            # Преобразование в вектор
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            # Выходной слой. Используем сигмоиду, чтобы возвращать [0,1]
            Dense(Y_SHAPE, activation="sigmoid"),
        ]
    )
