import json
import os
from typing import Callable

import numpy as np
from PIL import Image
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
    GlobalAveragePooling2D,
)
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import keras

X_MAX = 0
X_MIN = 1500
X_SHAPE_RAW = 3000
X_SHAPE_GPH = (1830, 1350, 1)  # изображение размером 1830x1350 в одноканале
Y_SHAPE = 2  # ответ - количество точек в предсказании


def _generate_simple_pulse_signal(
    fs: int = 1000,
    duration: float = 1000.0,
    frequency: float = 50,
    decay: float = 5,
    start_time: float = 100.0,
    pulse_duration: float = 100.0,
    reflection_delay: float = 300.0,
    reflection_amp: float = 0.6,
    with_noise: bool = False,
    noise_level: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Генерация простейшего акустического сигнала (синусоида) + отражения

    :param fs: частота дискретизации, Гц
    :param duration: общая длительность сигнала, мс
    :param frequency: частота основной синусоиды, Гц
    :param decay: коэффициент экспоненциального затухания
    :param start_time: время начала основного удара, мс
    :param pulse_duration: длительность основного импульса, мс
    :param reflection_delay: задержка отражения относительно удара, мс
    :param reflection_amp: коэффициент ослабления отражения
    :param with_noise: добавлять ли шум к полученному сигналу?
    :param noise_level: стандартное отклонение шума (при with_noise=True)

    :return t_ms: массив времени (мс)
    :return pulse: сигнал
    :return start_x: координата начала удара (мс)
    :return reflection_x: координата начала отражения (мс)
    """
    start_x = start_time
    reflection_x = start_time + reflection_delay

    # Перевод из миллисекунд в секунды
    duration = duration / 1000
    start_time = start_time / 1000
    pulse_duration = pulse_duration / 1000
    reflection_delay = reflection_delay / 1000

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    t_ms = t * 1000  # массив времени в мс
    pulse = np.zeros_like(t)

    start_idx = int(start_time * fs)
    pulse_samples = int(pulse_duration * fs)

    if start_idx + pulse_samples > len(t):
        pulse_samples = len(t) - start_idx

    signal = np.sin(2 * np.pi * frequency * t[:pulse_samples]) * np.exp(-decay * t[:pulse_samples])
    pulse[start_idx : start_idx + pulse_samples] += signal

    reflection_idx = int((start_time + reflection_delay) * fs)

    if reflection_idx + pulse_samples <= len(t):
        pulse[reflection_idx : reflection_idx + pulse_samples] += reflection_amp * signal

    if with_noise:
        assert (
            isinstance(noise_level, float) and noise_level >= 0.0
        ), f"Необходимо указать noise_level в диапазоне [0;∞)"
        noise = np.random.normal(0, noise_level, size=t.shape)
        pulse += noise

    return t_ms, pulse, start_x, reflection_x


def generate_simple_pulse_signal_without_noice() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Генерация простейшего акустического сигнала (синусоида) + отражения"""
    return _generate_simple_pulse_signal(
        fs=1000,
        duration=1500,
        frequency=np.random.randint(35, 75),
        decay=np.random.randint(3, 15),
        start_time=np.random.uniform(50, 200),
        pulse_duration=np.random.uniform(30, 100),
        reflection_delay=np.random.uniform(250, 800),
        reflection_amp=np.random.uniform(0.3, 0.8),
        with_noise=False,
    )


def generate_simple_pulse_signal_with_noice() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Генерация простейшего акустического сигнала (синусоида) + отражения + шум"""
    return _generate_simple_pulse_signal(
        fs=1000,
        duration=1500,
        frequency=np.random.randint(35, 75),
        decay=np.random.randint(3, 15),
        start_time=np.random.uniform(50, 200),
        pulse_duration=np.random.uniform(30, 100),
        reflection_delay=np.random.uniform(250, 800),
        reflection_amp=np.random.uniform(0.3, 0.8),
        with_noise=True,
        noise_level=np.random.uniform(0.05, 0.15),
    )


def _generate_complex_pulse_signal(
    fs: int = 1000,
    duration: float = 1000.0,
    frequencies: tuple[float, ...] = (30, 60, 90, 120),
    decay: float = 5.0,
    start_time: float = 100.0,
    pulse_duration: float = 100.0,
    reflection_delay: float = 300.0,
    reflection_amp: float = 0.5,
    distortion_level: float = 0.05,
    with_noise: bool = False,
    noise_level: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Генерация сложного акустического сигнала (сумма затухающих синусоид) + отражения

    :param fs: частота дискретизации, Гц
    :param duration: общая длительность сигнала, мс
    :param frequencies: список частот для композиции сигнала, Гц
    :param decay: коэффициент экспоненциального затухания
    :param start_time: время начала основного удара, мс
    :param pulse_duration: длительность основного импульса, мс
    :param reflection_delay: задержка отражения относительно удара, мс
    :param reflection_amp: коэффициент ослабления отражения
    :param distortion_level: уровень искажения отражения (Гауссов шум)
    :param with_noise: добавлять ли шум к полученному сигналу?
    :param noise_level: стандартное отклонение шума (при with_noise=True)

    :return t_ms: массив времени (мс)
    :return pulse: сигнал
    :return start_x: координата начала удара (мс)
    :return reflection_x: координата начала отражения (мс)
    """
    start_x = start_time
    reflection_x = start_time + reflection_delay

    # Перевод из миллисекунд в секунды
    duration = duration / 1000
    start_time = start_time / 1000
    pulse_duration = pulse_duration / 1000
    reflection_delay = reflection_delay / 1000

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    t_ms = t * 1000  # массив времени в мс
    pulse = np.zeros_like(t)

    start_idx = int(start_time * fs)
    pulse_samples = int(pulse_duration * fs)

    # Время для одного импульса
    segment_t = t[:pulse_samples]

    # Генерация случайных фаз
    phases = np.random.uniform(0, 2 * np.pi, size=len(frequencies))

    # Композиция затухающих синусоид
    multi_signal = sum(np.sin(2 * np.pi * f * segment_t + phi) for f, phi in zip(frequencies, phases))
    multi_signal *= np.exp(-decay * segment_t)

    # Добавление основного импульса
    pulse[start_idx : start_idx + pulse_samples] += multi_signal

    # Параметры отражения
    reflection_idx = int((start_time + reflection_delay) * fs)

    if reflection_idx + pulse_samples <= len(t):
        distortion = np.random.normal(0, distortion_level, size=pulse_samples)
        pulse[reflection_idx : reflection_idx + pulse_samples] += reflection_amp * (multi_signal + distortion)

    # Добавление шума
    if with_noise:
        assert isinstance(noise_level, float) and noise_level >= 0.0, "Необходимо указать noise_level в диапазоне [0;∞)"
        pulse += np.random.normal(0, noise_level, size=len(t))

    return t_ms, pulse, start_x, reflection_x


def generate_complex_pulse_signal_without_noice() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Генерация сложного акустического сигнала (сумма затухающих синусоид) + отражения"""
    return _generate_complex_pulse_signal(
        fs=1000,
        duration=1500,
        frequencies=(
            np.random.uniform(25, 35),
            np.random.uniform(55, 65),
            np.random.uniform(85, 95),
            np.random.uniform(115, 125),
        ),
        decay=np.random.randint(3, 15),
        start_time=np.random.uniform(50, 200),
        pulse_duration=np.random.uniform(30, 100),
        reflection_delay=np.random.uniform(250, 800),
        reflection_amp=np.random.uniform(0.3, 0.8),
        distortion_level=np.random.uniform(0.02, 0.08),
        with_noise=False,
    )


def generate_complex_pulse_signal_with_noice() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Генерация сложного акустического сигнала (сумма затухающих синусоид) + отражения + шум"""
    return _generate_complex_pulse_signal(
        fs=1000,
        duration=1500,
        frequencies=(
            np.random.uniform(25, 35),
            np.random.uniform(55, 65),
            np.random.uniform(85, 95),
            np.random.uniform(115, 125),
        ),
        decay=np.random.randint(3, 15),
        start_time=np.random.uniform(50, 200),
        pulse_duration=np.random.uniform(30, 100),
        reflection_delay=np.random.uniform(250, 800),
        reflection_amp=np.random.uniform(0.3, 0.8),
        distortion_level=np.random.uniform(0.02, 0.08),
        with_noise=True,
        noise_level=np.random.uniform(0.15, 0.25),
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
        10: generate_simple_pulse_signal_without_noice,
        20: generate_simple_pulse_signal_with_noice,
        30: generate_complex_pulse_signal_without_noice,
        40: generate_complex_pulse_signal_with_noice,
        50: generate_gaussian_pulse_signal_without_noise,
        60: generate_gaussian_pulse_signal_with_noise,
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


def load_dataset__gph(dirpath: str, dataset_size: int | None = None) -> tuple[list[np.ndarray], list[list[float]]]:
    """
    Загружает датасет изображений (в формате png)
    :param dirpath: путь до датасета
    :return: tuple(X, answers), где:
        - X - изображения
        - answers - координаты двух x точек - начала сигнала и отражения
    """
    from PIL import Image

    all_images = []
    all_answers = []

    # Загружаем соответствующие json файлы из raw_data
    raw_dir = dirpath.replace("fig_data", "raw_data")

    filenames = [filename for filename in sorted(os.listdir(dirpath)) if filename.endswith(".png")]
    if dataset_size is not None:
        if dataset_size > len(filenames):
            raise ValueError(f"Размер датасета ({len(filenames)} шт.) меньше желаемого ({dataset_size} шт.)")
        filenames = filenames[:dataset_size]
    for filename in filenames:
        # Загружаем изображение
        img_path = os.path.join(dirpath, filename)
        img = Image.open(img_path).convert("L")  # Конвертируем в grayscale
        img_array = np.array(img) / 255.0  # Нормализуем в [0, 1]
        img_array = np.expand_dims(img_array, axis=-1)  # Добавляем размерность канала
        all_images.append(img_array)

        # Загружаем соответствующие ответы из raw_data
        json_path = os.path.join(raw_dir, filename.replace(".png", ".json"))
        with open(json_path, "r") as f:
            data = json.load(f)
            all_answers.append(data["answers"])

    return all_images, all_answers


class GphLazyDataGenerator(keras.utils.Sequence):
    def __init__(self, dirpath: str, batch_size: int = 32, dataset_size: int | None = None) -> None:
        self._dirpath_gph = dirpath
        self._dirpath_raw = dirpath.replace("fig_data", "raw_data")
        self._dataset_size = dataset_size
        self._batch_size = batch_size

    def __len__(self):
        filepaths = sorted(filepath for filepath in os.listdir(self._dirpath_gph) if filepath.endswith(".png") and filepath.replace(".png", "").isnumeric())
        if self._dataset_size is not None:
            if self._dataset_size > len(filepaths):
                raise ValueError(f"Размер датасета ({len(filepaths)} шт.) меньше желаемого ({self._dataset_size} шт.)")
            filepaths = filepaths[:self._dataset_size]
        return int(np.ceil(len(filepaths) / self._batch_size))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        for i in range(idx * self._batch_size + 1, (idx + 1) * self._batch_size):
            filename = f"{i}.png"
            img_path = os.path.join(self._dirpath_gph, filename)
            img = Image.open(img_path).convert("L")  # Конвертируем в grayscale
            img_array = np.array(img) / 255.0  # Нормализуем в [0, 1]
            img_array = np.expand_dims(img_array, axis=-1)  # Добавляем размерность канала
            batch_x.append(img_array)

            json_path = os.path.join(self._dirpath_raw, filename.replace(".png", ".json"))
            with open(json_path, "r") as f:
                data = json.load(f)
                batch_y.append(data["answers"])
        return np.array(batch_x), np.array(batch_y)


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
            Conv1D(32, 5, activation="relu", padding="same"),
            Conv1D(64, 5, activation="relu", padding="same"),
            MaxPooling1D(pool_size=2),
            Conv1D(128, 5, activation="relu", padding="same"),
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
            Input(shape=(1830, 1350, 1)),
            # Первый блок - агрессивное уменьшение размерности
            Conv2D(8, (7, 7), strides=(2, 2), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            # Второй блок
            Conv2D(16, (5, 5), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            # Третий блок
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            # Четвёртый блок
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            # Пятый блок
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            # Шестой блок
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Flatten(),
            # Полносвязные слои
            Dense(256, activation="relu"),
            # Dropout(0.3),
            Dense(128, activation="relu"),
            # Dropout(0.2),
            Dense(Y_SHAPE, activation="linear"),
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
