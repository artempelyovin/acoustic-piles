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
    frequency: float = 50.0,
    pulse_half_cycles: int = 3,
    pulse_start: float = 0.1,
    pulse_decay: float = 5.0,
    reflection_delay: float = 0.3,
    reflection_amplitude: float = 0.6,
    reflection_decay: float = 10.0,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Генерация акустического импульса с отражением

    Args:
        fs: частота дискретизации (Гц)
        duration: длительность сигнала (сек)
        frequency: частота синусоиды (Гц)
        pulse_half_cycles: количество полупериодов основного импульса
        pulse_start: время начала импульса (сек)
        pulse_decay: коэффициент затухания основного импульса
        reflection_delay: задержка отражения после окончания импульса (сек)
        reflection_amplitude: амплитуда отражения (0-1)
        reflection_decay: коэффициент затухания отражения
        noise_std: стандартное отклонение шума (0 = без шума)

    Returns:
        time_ms: массив времени (мс)
        signal: результирующий сигнал
        pulse_start_ms: начало импульса (мс)
        reflection_start_ms: начало отражения (мс)
    """
    # Создаем временную ось
    time = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.zeros_like(time)

    # Параметры основного импульса
    pulse_duration = pulse_half_cycles / (2 * frequency)
    pulse_end = pulse_start + pulse_duration

    # Генерируем основной импульс
    pulse_mask = (time >= pulse_start) & (time < pulse_end)
    pulse_time = time[pulse_mask] - pulse_start
    pulse_signal = np.sin(2 * np.pi * frequency * pulse_time) * np.exp(-pulse_decay * pulse_time)
    signal[pulse_mask] = pulse_signal

    # Генерируем отражение
    reflection_start = pulse_end + reflection_delay
    reflection_mask = time >= reflection_start
    reflection_time = time[reflection_mask] - reflection_start

    if np.any(reflection_mask):
        reflection_signal = (
            reflection_amplitude
            * np.sin(2 * np.pi * frequency * reflection_time)
            * np.exp(-reflection_decay * reflection_time)
        )
        signal[reflection_mask] += reflection_signal

    # Добавляем шум если нужно
    if noise_std > 0:
        signal += np.random.normal(0, noise_std, signal.shape)

    # Возвращаем время в миллисекундах, поэтому умножаем на 1000
    return time * 1000, signal, pulse_start * 1000, reflection_start * 1000


def generate_simple_pulse_signal_without_noice() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Генерация простейшего акустического сигнала (синусоида) + отражения"""
    return _generate_simple_pulse_signal(
        fs=1000,
        duration=1.5,
        frequency=np.random.uniform(3.5, 7),
        pulse_half_cycles=np.random.choice([2, 3, 4]),
        pulse_start=np.random.uniform(0.01, 0.2),
        pulse_decay=np.random.uniform(4, 7),
        reflection_delay=np.random.uniform(0.25, 0.45),
        reflection_amplitude=np.random.uniform(0.15, 0.25),
        reflection_decay=np.random.uniform(8, 11),
        noise_std=0.0,
    )


def generate_simple_pulse_signal_with_noice() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Генерация простейшего акустического сигнала (синусоида) + отражения + шум"""
    return _generate_simple_pulse_signal(
        fs=1000,
        duration=1.5,
        frequency=np.random.uniform(3.5, 7),
        pulse_half_cycles=np.random.choice([2, 3, 4]),
        pulse_start=np.random.uniform(0.01, 0.2),
        pulse_decay=np.random.uniform(4, 7),
        reflection_delay=np.random.uniform(0.25, 0.45),
        reflection_amplitude=np.random.uniform(0.15, 0.25),
        reflection_decay=np.random.uniform(8, 11),
        noise_std=np.random.uniform(0.025, 0.05),
    )


def _generate_complex_pulse_signal(
    fs: int = 1000,
    duration: float = 1.0,
    frequencies: tuple[float, ...] = (30, 60, 90, 120),
    pulse_start: float = 0.1,
    pulse_duration: float = 0.1,
    pulse_decay: float = 5.0,
    reflection_delay: float = 0.3,
    reflection_amplitude: float = 0.5,
    reflection_decay: float = 5.0,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Генерация сложного акустического импульса с отражением

    Args:
        fs: частота дискретизации (Гц)
        duration: длительность сигнала (сек)
        frequencies: частоты для композиции сигнала (Гц)
        pulse_start: время начала импульса (сек)
        pulse_duration: длительность основного импульса (сек)
        pulse_decay: коэффициент затухания основного импульса
        reflection_delay: задержка отражения после окончания импульса (сек)
        reflection_amplitude: амплитуда отражения (0-1)
        reflection_decay: коэффициент затухания отражения
        noise_std: стандартное отклонение общего шума (0 = без шума)

    Returns:
        time_ms: массив времени (мс)
        signal: результирующий сигнал
        pulse_start_ms: начало импульса (мс)
        reflection_start_ms: начало отражения (мс)
    """
    time = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.zeros_like(time)

    # Параметры импульса
    pulse_end_sec = pulse_start + pulse_duration

    # Генерируем основной импульс (композиция синусоид со случайными фазами)
    pulse_mask = (time >= pulse_start) & (time < pulse_end_sec)
    pulse_time = time[pulse_mask] - pulse_start

    if np.any(pulse_mask):
        # Генерируем случайные фазы для каждой частоты
        phases = np.random.uniform(0, 2 * np.pi, size=len(frequencies))

        # Создаем композицию синусоид
        multi_signal = sum(np.sin(2 * np.pi * freq * pulse_time + phase) for freq, phase in zip(frequencies, phases))
        multi_signal *= np.exp(-pulse_decay * pulse_time)
        signal[pulse_mask] = multi_signal

    # Генерируем отражение (продолжается до конца)
    reflection_start_sec = pulse_end_sec + reflection_delay
    reflection_mask = time >= reflection_start_sec
    reflection_time = time[reflection_mask] - reflection_start_sec

    if np.any(reflection_mask):
        # Используем те же фазы для отражения
        reflection_signal = sum(
            np.sin(2 * np.pi * freq * reflection_time + phase) for freq, phase in zip(frequencies, phases)
        )
        reflection_signal *= np.exp(-reflection_decay * reflection_time)
        signal[reflection_mask] += reflection_amplitude * reflection_signal

    # Добавляем общий шум
    if noise_std > 0:
        signal += np.random.normal(0, noise_std, signal.shape)

    # Возвращаем время в миллисекундах
    return time * 1000, signal, pulse_start * 1000, reflection_start_sec * 1000


def generate_complex_pulse_signal_without_noice() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Генерация сложного акустического сигнала (сумма затухающих синусоид) + отражения"""
    return _generate_complex_pulse_signal(
        fs=1000,
        duration=1.5,
        frequencies=(
            np.random.uniform(3, 5),
            np.random.uniform(10, 12),
        ),
        pulse_start=np.random.uniform(0.01, 0.2),
        pulse_duration=np.random.uniform(0.03, 0.1),
        pulse_decay=np.random.uniform(4, 7),
        reflection_delay=np.random.uniform(0.25, 0.45),
        reflection_amplitude=np.random.uniform(0.15, 0.25),
        reflection_decay=np.random.uniform(8, 11),
        noise_std=0.0,
    )


def generate_complex_pulse_signal_with_noice() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Генерация сложного акустического сигнала (сумма затухающих синусоид) + отражения + шум"""
    return _generate_complex_pulse_signal(
        fs=1000,
        duration=1.5,
        frequencies=(
            np.random.uniform(3, 5),
            np.random.uniform(10, 12),
        ),
        pulse_start=np.random.uniform(0.01, 0.2),
        pulse_duration=np.random.uniform(0.03, 0.1),
        pulse_decay=np.random.uniform(4, 7),
        reflection_delay=np.random.uniform(0.25, 0.45),
        reflection_amplitude=np.random.uniform(0.15, 0.25),
        reflection_decay=np.random.uniform(8, 11),
        noise_std=np.random.uniform(0.025, 0.05),
    )


def _generate_gaussian_pulse_signal(
    fs: int = 1000,
    duration: float = 1000.0,
    start_time: float = 100.0,
    pulse_duration: float = 50.0,
    reflection_delay: float = 300.0,
    distortion_level: float = 0.05,
    num_false_echoes: int = 2,
    with_noise: bool = False,
    noise_level: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Генерация рандомизированного сигнала (гауссовые огибающие) + отражение

    :param fs: частота дискретизации, Гц
    :param duration: общая длительность сигнала, мс
    :param start_time: время начала основного удара, мс
    :param pulse_duration: длительность основного импульса, мс
    :param reflection_delay: задержка отражения относительно удара, мс
    :param distortion_level: уровень искажения отражения
    :param num_false_echoes: количество ложных всплесков
    :param with_noise: добавлять ли шум к полученному сигналу?
    :param noise_level: стандартное отклонение шума (при with_noise=True)

    :return t_ms: массив времени (мс)
    :return pulse: сигнал
    :return start_x: координата начала удара (мс)
    :return reflection_x: координата начала отражения (мс)
    """
    start_x = start_time
    reflection_x = start_time + reflection_delay

    # Перевод параметров в секунды
    duration = duration / 1000
    start_time = start_time / 1000
    pulse_duration = pulse_duration / 1000
    reflection_delay = reflection_delay / 1000

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    t_ms = t * 1000
    pulse = np.zeros_like(t)

    pulse_samples = int(pulse_duration * fs)
    center = pulse_samples // 2
    std_dev = pulse_samples / 6

    start_idx = int(start_time * fs)
    gauss = np.exp(-0.5 * ((np.arange(pulse_samples) - center) / std_dev) ** 2)
    gauss *= np.random.uniform(0.8, 1.2)
    pulse[start_idx : start_idx + pulse_samples] += gauss

    reflection_x = start_time + reflection_delay
    reflection_idx = int(reflection_x * fs)
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

    return t_ms, pulse, start_x, reflection_x


def generate_gaussian_pulse_signal_without_noise() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Генерация рандомизированного сигнала (гауссовые огибающие) + отражение"""
    return _generate_gaussian_pulse_signal(
        fs=1000,
        duration=1500,
        start_time=np.random.uniform(50, 200),
        pulse_duration=np.random.uniform(30, 100),
        reflection_delay=np.random.uniform(250, 800),
        distortion_level=np.random.uniform(0.02, 0.08),
        num_false_echoes=np.random.randint(1, 6),
        with_noise=False,
    )


def generate_gaussian_pulse_signal_with_noise() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Генерация рандомизированного сигнала (гауссовые огибающие) + отражение + шум"""
    return _generate_gaussian_pulse_signal(
        fs=1000,
        duration=1500,
        start_time=np.random.uniform(50, 200),
        pulse_duration=np.random.uniform(30, 100),
        reflection_delay=np.random.uniform(250, 800),
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
        1: generate_simple_pulse_signal_without_noice,
        2: generate_simple_pulse_signal_with_noice,
        3: generate_complex_pulse_signal_without_noice,
        4: generate_complex_pulse_signal_with_noice,
    }
    return generator_function_by_model_number[model_number]


def set_plt_style() -> None:
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.grid"] = True


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
        points = {
            "x": x.tolist(),
            "y": y.tolist(),
            "answers": [start_x, reflection_x],
        }
        json.dump(points, f, indent=4)


def load_dataset__raw(dirpath: str) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """
    Загружает датасет "сырых" (в формате json) данных
    :param dirpath: путь до датасета
    :return: tuple(X, Y, answers), где:
        - X - x-координаты функции
        - X - y-координаты функции
        - answers - координаты двух x точек - начала сигнала и отражения
    """

    def load_raw_file(filepath: str) -> tuple[list[float], list[float], list[float]]:
        with open(filepath, "r") as f:
            data = json.load(f)
            return data["x"], data["y"], data["answers"]

    all_x = []
    all_y = []
    all_answers = []
    for path_ in os.listdir(dirpath):
        x, y, answers = load_raw_file(f"{dirpath}/{path_}")
        all_x.append(x)
        all_y.append(y)
        all_answers.append(answers)
    return all_x, all_y, all_answers


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


def normalize(x: np.ndarray, x_min: float | int | None = None, x_max: float | int | None = None) -> np.ndarray:
    """Нормализует вектор чисел в диапазон [0;1]"""
    if x_min is not None or x_max is not None:
        assert isinstance(x_min, float | int) and isinstance(x_max, float | int)
        return (x - x_min) / (x_max - x_min)
    return (x - x.min()) / (x.max() - x.min())


def denormalize(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Денормализует вектор чисел в диапазоне [0;1] обратно в диапазон [x_min;x_max]"""
    return x * (x_max - x_min) + x_min


def generate_model__raw() -> Sequential:
    return Sequential(
        [
            Input(shape=(X_SHAPE_RAW,)),
            Reshape((X_SHAPE_RAW, 1)),
            Conv1D(16, 5, activation="relu", padding="same"),
            MaxPooling1D(pool_size=2),
            Conv1D(32, 5, activation="relu", padding="same"),
            MaxPooling1D(pool_size=2),
            Conv1D(64, 5, activation="relu", padding="same"),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(256, activation="relu"),
            # Dropout(0.3),
            Dense(128, activation="relu"),
            # Dropout(0.2),
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
        plt.savefig(self.image_file, dpi=300)
        plt.close()  # Закрываем фигуру, чтобы не перегружать память
