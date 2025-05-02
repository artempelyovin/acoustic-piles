import json
import os
from typing import Callable

import numpy as np
from keras import Sequential, Input
from keras.src.callbacks import Callback
from keras.src.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Conv1D,
    MaxPooling1D,
    Reshape,
    GlobalAveragePooling1D,
)
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

X_SHAPE_RAW = 3000
X_SHAPE_GPH = (369, 496, 1)  # изображение размером 369x496 в одноканале
Y_SHAPE = 2  # ответ - количество точек в предсказании


def _generate_simple_pulse_signal(
    fs: int = 1000,
    duration: float = 1.0,
    frequency: float = 50,
    decay: float = 5,
    start_time: float = 0.1,
    pulse_duration: float = 0.1,
    reflection_delay: float = 0.3,
    reflection_amp: float = 0.6,
    with_noise: bool = False,
    noise_level: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Генерация простейшего акустического сигнала (синусоида) + отражения

    :param fs: частота дискретизации, Гц
    :param duration: общая длительность сигнала, сек
    :param frequency: частота основной синусоиды, Гц
    :param decay: коэффициент экспоненциального затухания
    :param start_time: время начала основного удара, сек
    :param pulse_duration: длительность основного импульса, сек
    :param reflection_delay: задержка отражения относительно удара, сек
    :param reflection_amp: коэффициент ослабления отражения
    :param with_noise: добавлять ли шум к полученному сигналу?
    :param noise_level: стандартное отклонение шума (при with_noise=True)

    :return t: массив времени
    :return pulse: сигнал
    :return start_x: координата начала удара
    :return reflection_x: координата начала отражения
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    pulse = np.zeros_like(t)

    start_x = start_time
    pulse_samples = int(pulse_duration * fs)

    start_idx = int(start_x * fs)

    # Ограничение на длину импульса
    if start_idx + pulse_samples > len(t):
        pulse_samples = len(t) - start_idx

    signal = np.sin(2 * np.pi * frequency * t[:pulse_samples]) * np.exp(-decay * t[:pulse_samples])
    pulse[start_idx : start_idx + pulse_samples] += signal

    reflection_x = start_x + reflection_delay
    reflection_idx = int(reflection_x * fs)

    if reflection_idx + pulse_samples <= len(t):
        pulse[reflection_idx : reflection_idx + pulse_samples] += reflection_amp * signal

    if with_noise:
        assert (
            isinstance(noise_level, float) and noise_level >= 0.0
        ), f"Необходимо указать noise_level в диапазоне [0;∞)"
        noise = np.random.normal(0, noise_level, size=t.shape)
        pulse += noise

    return t, pulse, start_x, reflection_x


def generate_simple_pulse_signal_without_noice() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Генерация простейшего акустического сигнала (синусоида) + отражения"""
    return _generate_simple_pulse_signal(
        fs=1000,
        duration=1.5,
        frequency=np.random.randint(35, 75),
        decay=np.random.randint(3, 15),
        start_time=np.random.uniform(0.05, 0.2),
        pulse_duration=np.random.uniform(0.03, 0.1),
        reflection_delay=np.random.uniform(0.25, 0.8),
        reflection_amp=np.random.uniform(0.3, 0.8),
        with_noise=False,
    )


def generate_simple_pulse_signal_with_noice() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Генерация простейшего акустического сигнала (синусоида) + отражения + шум"""
    return _generate_simple_pulse_signal(
        fs=1000,
        duration=1.5,
        frequency=np.random.randint(35, 75),
        decay=np.random.randint(3, 15),
        start_time=np.random.uniform(0.05, 0.2),
        pulse_duration=np.random.uniform(0.03, 0.1),
        reflection_delay=np.random.uniform(0.25, 0.8),
        reflection_amp=np.random.uniform(0.3, 0.8),
        with_noise=True,
        noise_level=np.random.uniform(0.05, 0.15),
    )


def _generate_complex_pulse_signal(
    fs: int = 1000,
    duration: float = 1.0,
    frequencies: tuple[float, ...] = (30, 60, 90, 120),
    decay: float = 5.0,
    start_time: float = 0.1,
    pulse_duration: float = 0.1,
    reflection_delay: float = 0.3,
    reflection_amp: float = 0.5,
    distortion_level: float = 0.05,
    with_noise: bool = False,
    noise_level: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Генерация сложного акустического сигнала (сумма затухающих синусоид) + отражения

    :param fs: частота дискретизации, Гц
    :param duration: общая длительность сигнала, сек
    :param frequencies: список частот для композиции сигнала, Гц
    :param decay: коэффициент экспоненциального затухания
    :param start_time: время начала основного удара, сек
    :param pulse_duration: длительность основного импульса, сек
    :param reflection_delay: задержка отражения относительно удара, сек
    :param reflection_amp: коэффициент ослабления отражения
    :param distortion_level: уровень искажения отражения (Гауссов шум)
    :param with_noise: добавлять ли шум к полученному сигналу?
    :param noise_level: стандартное отклонение шума (при with_noise=True)

    :return t: массив времени
    :return pulse: сигнал
    :return start_x: координата начала удара (сек)
    :return reflection_x: координата начала отражения (сек)
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    pulse = np.zeros_like(t)

    start_idx = int(start_time * fs)
    pulse_samples = int(pulse_duration * fs)

    # Время для одного импульса
    segment_t = t[:pulse_samples]

    # Генерация случайных фаз
    phases = np.random.uniform(0, 2 * np.pi, size=len(frequencies))

    # Комплексный затухающий сигнал
    multi_signal = sum(np.sin(2 * np.pi * f * segment_t + phi) for f, phi in zip(frequencies, phases))
    multi_signal *= np.exp(-decay * segment_t)

    # Добавление основного импульса
    pulse[start_idx : start_idx + pulse_samples] += multi_signal

    # Параметры отражения
    reflection_x = start_time + reflection_delay
    reflection_idx = int(reflection_x * fs)

    if reflection_idx + pulse_samples <= len(t):
        distortion = np.random.normal(0, distortion_level, size=pulse_samples)
        pulse[reflection_idx : reflection_idx + pulse_samples] += reflection_amp * (multi_signal + distortion)

    # Добавление шума
    if with_noise:
        assert isinstance(noise_level, float) and noise_level >= 0.0, "Необходимо указать noise_level в диапазоне [0;∞)"
        pulse += np.random.normal(0, noise_level, size=len(t))

    return t, pulse, start_time, reflection_x


def generate_complex_pulse_signal_without_noice() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Генерация сложного акустического сигнала (сумма затухающих синусоид) + отражения"""
    return _generate_complex_pulse_signal(
        fs=1000,
        duration=1.5,
        frequencies=(
            np.random.uniform(25, 35),
            np.random.uniform(55, 65),
            np.random.uniform(85, 95),
            np.random.uniform(115, 125),
        ),
        decay=np.random.randint(3, 15),
        start_time=np.random.uniform(0.05, 0.2),
        pulse_duration=np.random.uniform(0.03, 0.1),
        reflection_delay=np.random.uniform(0.25, 0.8),
        reflection_amp=np.random.uniform(0.3, 0.8),
        distortion_level=np.random.uniform(0.02, 0.08),
        with_noise=False,
    )


def generate_complex_pulse_signal_with_noice() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Генерация сложного акустического сигнала (сумма затухающих синусоид) + отражения + шум"""
    return _generate_complex_pulse_signal(
        fs=1000,
        duration=1.5,
        frequencies=(
            np.random.uniform(25, 35),
            np.random.uniform(55, 65),
            np.random.uniform(85, 95),
            np.random.uniform(115, 125),
        ),
        decay=np.random.randint(3, 15),
        start_time=np.random.uniform(0.05, 0.2),
        pulse_duration=np.random.uniform(0.03, 0.1),
        reflection_delay=np.random.uniform(0.25, 0.8),
        reflection_amp=np.random.uniform(0.3, 0.8),
        distortion_level=np.random.uniform(0.02, 0.08),
        with_noise=True,
        noise_level=np.random.uniform(0.15, 0.25),
    )


def _generate_gaussian_pulse_signal(
    fs: int = 1000,
    duration: float = 1.0,
    start_time: float = 0.1,
    pulse_duration: float = 0.05,
    reflection_delay: float = 0.3,
    distortion_level: float = 0.05,
    num_false_echoes: int = 2,
    with_noise: bool = False,
    noise_level: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Генерация рандомизированного сигнала (гауссовые огибающие) + отражение

    :param fs: частота дискретизации, Гц
    :param duration: общая длительность сигнала, сек
    :param start_time: время начала основного удара, сек
    :param pulse_duration: длительность основного импульса, сек
    :param reflection_delay: задержка отражения относительно удара, сек
    :param distortion_level: уровень искажения отражения
    :param num_false_echoes: количества ложных всплесков
    :param with_noise: добавлять ли шум к полученному сигналу?
    :param noise_level: стандартное отклонение шума (при with_noise=True)

    :return t: массив времени
    :return pulse: сигнал
    :return start_x: координата начала удара (сек)
    :return reflection_x: координата начала отражения (сек)
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    pulse = np.zeros_like(t)

    start_x = start_time
    pulse_samples = int(pulse_duration * fs)
    center = pulse_samples // 2
    std_dev = pulse_samples / 6

    start_idx = int(start_x * fs)
    gauss = np.exp(-0.5 * ((np.arange(pulse_samples) - center) / std_dev) ** 2)
    gauss *= np.random.uniform(0.8, 1.2)
    pulse[start_idx : start_idx + pulse_samples] += gauss

    reflection_x = start_time + reflection_delay
    reflection_idx = int(reflection_delay * fs)
    reflection_amp = np.random.uniform(0.3, 0.7)
    distortion = np.random.normal(0, distortion_level, size=pulse_samples)
    reflected = reflection_amp * (gauss + distortion)
    if reflection_idx + pulse_samples < len(pulse):
        pulse[reflection_idx : reflection_idx + pulse_samples] += reflected

    # Ложные всплески
    for _ in range(num_false_echoes):
        false_idx = np.random.randint(0, len(t) - pulse_samples)
        if abs(false_idx - start_idx) > pulse_samples and abs(false_idx - reflection_idx) > pulse_samples:
            false_pulse = np.exp(-0.5 * ((np.arange(pulse_samples) - center) / std_dev) ** 2)
            false_pulse *= np.random.uniform(0.2, 0.5)
            pulse[false_idx : false_idx + pulse_samples] += false_pulse

    if with_noise:
        assert (
            isinstance(noise_level, float) and noise_level >= 0.0
        ), f"Необходимо указать noise_level в диапазоне [0;∞)"
        noise = np.random.normal(0, noise_level, size=t.shape)
        pulse += noise

    return t, pulse, start_x, reflection_x


def generate_gaussian_pulse_signal_without_noise() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Генерация рандомизированного сигнала (гауссовые огибающие) + отражение"""
    return _generate_gaussian_pulse_signal(
        fs=1000,
        duration=1.5,
        start_time=np.random.uniform(0.05, 0.2),
        pulse_duration=np.random.uniform(0.03, 0.1),
        reflection_delay=np.random.uniform(0.25, 0.8),
        distortion_level=np.random.uniform(0.02, 0.08),
        num_false_echoes=np.random.randint(1, 6),
        with_noise=False,
    )


def generate_gaussian_pulse_signal_with_noise() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Генерация рандомизированного сигнала (гауссовые огибающие) + отражение + шум"""
    return _generate_gaussian_pulse_signal(
        fs=1000,
        duration=1.5,
        start_time=np.random.uniform(0.05, 0.2),
        pulse_duration=np.random.uniform(0.03, 0.1),
        reflection_delay=np.random.uniform(0.25, 0.8),
        distortion_level=np.random.uniform(0.02, 0.08),
        num_false_echoes=np.random.randint(1, 6),
        with_noise=True,
        noise_level=np.random.uniform(0.05, 0.15),
    )


def get_generator_function_by_model_number(model_number: int) -> Callable:
    """
    Возвращает функцию генерации сигнала в зависимости от номера модели.

    :param model_number: Номер модели, для которой требуется функция генерации.
    """
    generator_function_by_model_number = {
        10: generate_simple_pulse_signal_without_noice,
        20: generate_simple_pulse_signal_with_noice,
        30: generate_complex_pulse_signal_without_noice,
        40: generate_complex_pulse_signal_with_noice,
        50: generate_gaussian_pulse_signal_without_noise,
        60: generate_gaussian_pulse_signal_with_noise,
    }
    return generator_function_by_model_number[model_number]


def draw_acoustic_signal(ax: Axes, x: np.ndarray, y: np.ndarray) -> None:
    """Рисует акустический сигнал на заданной оси"""
    ax.plot(x, y, "black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)


def save_acoustic_signal_as_image(fig: Figure, filename: str) -> None:
    """Сохраняет акустический сигнал в виде изображения"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, bbox_inches="tight", dpi=300, pad_inches=0)


def save_acoustic_signal_as_json(
    x: np.ndarray, y: np.ndarray, start_x: float, reflection_x: float, filename: str
) -> None:
    """Сохраняет акустический сигнал в виде JSON файла"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        points = np.column_stack((x, y)).reshape(-1)  # делаем массив точек формата [x1, y1, x2, y2, ..., xn, yn]
        points = {
            "points": points.tolist(),
            "answers": [start_x, reflection_x],
        }
        json.dump(points, f, indent=4)


def load_dataset__raw(dirpath: str) -> tuple[list[list[float]], list[list[float]]]:
    """
    Загружает датасет "сырых" (в формате json) данных
    :param dirpath: путь до датасета
    :return: X и Y значения, где:
        - X - точки функции в формате (x;y)
        - Y - координаты x точек начала сигнала и отражения
    """

    def load_raw_file(filepath: str) -> tuple[list[float], list[float]]:
        with open(filepath, "r") as f:
            data = json.load(f)
            return data["points"], data["answers"]

    all_x = []
    all_y = []
    for path_ in os.listdir(dirpath):
        points, answers = load_raw_file(f"{dirpath}/{path_}")
        all_x.append(points)  # вход нейросети в виде точек [x1, y1, x2, y2, ..., xn, yn]
        all_y.append(answers)  # выход в виде координат x, где функция меняет знак
    return all_x, all_y


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


def generate_model__raw() -> Sequential:
    return Sequential(
        [
            Input(shape=(X_SHAPE_RAW,)),
            Reshape((X_SHAPE_RAW, 1)),
            Conv1D(32, 5, activation="relu", padding="same"),
            Conv1D(64, 5, activation="relu", padding="same"),
            MaxPooling1D(pool_size=2),
            Conv1D(128, 5, activation="relu", padding="same"),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(128, activation="relu"),
            Dropout(0.2),
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


class HistoryToFile(Callback):
    def __init__(self, history_file):
        super().__init__()
        self.history_file = history_file

    def on_epoch_end(self, epoch, logs=None):
        if not self.model.history.history:
            return
        with open(self.history_file, "w") as f:
            json.dump(self.model.history.history, f, indent=4)


class PlotHistory(Callback):
    def __init__(self, image_file):
        super().__init__()
        self.image_file = image_file

    def on_epoch_end(self, epoch, logs=None):
        # Отсекаем первую эпоху, т.к. там очень большие ошибки
        loss = self.model.history.history.get("loss", [])[1:]
        val_loss = self.model.history.history.get("val_loss", [])[1:]

        if not loss or not val_loss:
            return

        plt.figure()
        plt.plot(loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, max(max(loss), max(val_loss)) * 1.1)  # Установка пределов по оси Y
        plt.savefig(self.image_file)
        plt.close()  # Закрываем фигуру, чтобы не перегружать память
