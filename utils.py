"""
Модуль вспомогательных функций для работы с акустическими сигналами и нейронными сетями.

Модуль содержит функции для:
- Генерации различных типов акустических сигналов (простые/сложные, с шумом/без)
- Визуализации и сохранения акустических сигналов
- Загрузки и предобработки датасетов
- Нормализации данных
- Создания архитектуры нейронной сети
- Callback-функций для сохранения истории обучения
"""

import json
import os
import random
from random import choice
from typing import Callable

import numpy as np
from keras import Sequential, Input
from keras.src.callbacks import Callback
from keras.src.layers import Dense, Flatten, Conv1D, MaxPooling1D, Reshape
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


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
    Генерация акустического импульса с отражением.

    Args:
        fs: Частота дискретизации (Гц)
        duration: Длительность сигнала (сек)
        frequency: Частота синусоиды (Гц)
        pulse_half_cycles: Количество полупериодов основного импульса
        pulse_start: Время начала импульса (сек)
        pulse_decay: Коэффициент затухания основного импульса
        reflection_delay: Задержка отражения после окончания импульса (сек)
        reflection_amplitude: Амплитуда отражения (0-1)
        reflection_decay: Коэффициент затухания отражения
        noise_std: Стандартное отклонение шума (0 = без шума)

    Returns:
        tuple: Кортеж из:
            - time_ms: Массив времени (мс)
            - signal: Результирующий сигнал
            - pulse_start_ms: Начало импульса (мс)
            - reflection_start_ms: Начало отражения (мс)
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
    """
    Генерация простейшего акустического сигнала (синусоида) с отражением без шума.

    Returns:
        tuple: Кортеж из:
            - time_ms: Массив времени (мс)
            - signal: Результирующий сигнал
            - pulse_start_ms: Начало импульса (мс)
            - reflection_start_ms: Начало отражения (мс)
    """
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
    """
    Генерация простейшего акустического сигнала (синусоида) с отражением и шумом.

    Returns:
        tuple: Кортеж из:
            - time_ms: Массив времени (мс)
            - signal: Результирующий сигнал
            - pulse_start_ms: Начало импульса (мс)
            - reflection_start_ms: Начало отражения (мс)
    """
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
    Генерация сложного акустического импульса с отражением.

    Args:
        fs: Частота дискретизации (Гц)
        duration: Длительность сигнала (сек)
        frequencies: Частоты для композиции сигнала (Гц)
        pulse_start: Время начала импульса (сек)
        pulse_duration: Длительность основного импульса (сек)
        pulse_decay: Коэффициент затухания основного импульса
        reflection_delay: Задержка отражения после окончания импульса (сек)
        reflection_amplitude: Амплитуда отражения (0-1)
        reflection_decay: Коэффициент затухания отражения
        noise_std: Стандартное отклонение общего шума (0 = без шума)

    Returns:
        tuple: Кортеж из:
            - time_ms: Массив времени (мс)
            - signal: Результирующий сигнал
            - pulse_start_ms: Начало импульса (мс)
            - reflection_start_ms: Начало отражения (мс)
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
    """
    Генерация сложного акустического сигнала (сумма затухающих синусоид) с отражением без шума.

    Returns:
        tuple: Кортеж из:
            - time_ms: Массив времени (мс)
            - signal: Результирующий сигнал
            - pulse_start_ms: Начало импульса (мс)
            - reflection_start_ms: Начало отражения (мс)
    """
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
    """
    Генерация сложного акустического сигнала (сумма затухающих синусоид) с отражением и шумом.

    Returns:
        tuple: Кортеж из:
            - time_ms: Массив времени (мс)
            - signal: Результирующий сигнал
            - pulse_start_ms: Начало импульса (мс)
            - reflection_start_ms: Начало отражения (мс)
    """
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


def generate_any_pulse_signal() -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Генерация случайного акустического сигнала (выбирается один из доступных генераторов).

    Returns:
        tuple: Кортеж из:
            - time_ms: Массив времени (мс)
            - signal: Результирующий сигнал
            - pulse_start_ms: Начало импульса (мс)
            - reflection_start_ms: Начало отражения (мс)
    """
    random_pulse_signal_function = choice(
        (
            generate_simple_pulse_signal_without_noice,
            generate_simple_pulse_signal_with_noice,
            generate_complex_pulse_signal_without_noice,
            generate_complex_pulse_signal_with_noice,
        )
    )
    return random_pulse_signal_function()


def load_real_signal_from_6_model() -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Загрузка реального сигнала из модели 6.

    Returns:
        tuple: Кортеж из:
            - time_ms: Массив времени (мс)
            - signal: Результирующий сигнал
            - pulse_start_ms: Начало импульса (мс)
            - reflection_start_ms: Начало отражения (мс)
    """
    filenames = [f for f in os.listdir("datasets/6/raw_data")]
    filename = random.choice(filenames)
    with open(f"datasets/6/raw_data/{filename}", "r") as file:
        content = json.load(file)
        return np.array(content["x"]), np.array(content["y"]), content["answers"][0], content["answers"][1]


def get_generator_function_by_model_number(model_number: int) -> Callable:
    """
    Возвращает функцию генерации сигнала для указанной модели.

    Args:
        model_number: Номер модели (1-6)

    Returns:
        Callable: Функция генерации сигнала для указанной модели

    Raises:
        KeyError: Если номер модели не поддерживается
    """
    generator_function_by_model_number = {
        1: generate_simple_pulse_signal_without_noice,
        2: generate_simple_pulse_signal_with_noice,
        3: generate_complex_pulse_signal_without_noice,
        4: generate_complex_pulse_signal_with_noice,
        5: generate_any_pulse_signal,
        6: load_real_signal_from_6_model,
    }
    return generator_function_by_model_number[model_number]


def set_plt_style() -> None:
    """Устанавливает стиль графиков matplotlib."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.grid"] = True


def draw_acoustic_signal(ax: Axes, x: np.ndarray, y: np.ndarray) -> None:
    """
    Отрисовывает акустический сигнал на указанной оси.

    Args:
        ax: Ось matplotlib для отрисовки
        x: Массив значений по оси X
        y: Массив значений по оси Y
    """
    ax.plot(x, y, "black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)


def save_acoustic_signal_as_image(fig: Figure, filename: str) -> None:
    """
    Сохраняет график акустического сигнала в файл.

    Args:
        fig: Объект Figure matplotlib
        filename: Путь для сохранения файла
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, bbox_inches="tight", dpi=300, pad_inches=0)


def save_acoustic_signal_as_json(
    x: np.ndarray, y: np.ndarray, start_x: float, reflection_x: float, filename: str
) -> None:
    """
    Сохраняет данные акустического сигнала в JSON-файл.

    Args:
        x: Массив значений по оси X
        y: Массив значений по оси Y
        start_x: Координата начала импульса
        reflection_x: Координата отражения
        filename: Путь для сохранения файла
    """
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
    Загружает сырой датасет из JSON-файлов.

    Args:
        dirpath: Путь к директории с JSON-файлами

    Returns:
        tuple: Кортеж из:
            - Список массивов значений X
            - Список массивов значений Y
            - Список массивов с ответами (координаты линий уровней)

    Raises:
        FileNotFoundError: Если директория не существует
        json.JSONDecodeError: Если файлы содержат некорректный JSON
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


def normalize(x: np.ndarray, x_min: float | int | None = None, x_max: float | int | None = None) -> np.ndarray:
    """
    Нормализует массив значений в диапазон [0, 1].

    Args:
        x: Входной массив для нормализации
        x_min: Минимальное значение для нормализации (если None - вычисляется из x)
        x_max: Максимальное значение для нормализации (если None - вычисляется из x)

    Returns:
        np.ndarray: Нормализованный массив

    Raises:
        ValueError: Если x_min или x_max не являются числами (когда указаны)
    """
    if x_min is not None or x_max is not None:
        assert isinstance(x_min, float | int) and isinstance(x_max, float | int)
        return (x - x_min) / (x_max - x_min)
    return (x - x.min()) / (x.max() - x.min())


def denormalize(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """
    Денормализует массив значений из диапазона [0, 1] в исходный диапазон.

    Args:
        x: Нормализованный массив
        x_min: Минимальное значение исходного диапазона
        x_max: Максимальное значение исходного диапазона

    Returns:
        np.ndarray: Денормализованный массив
    """
    return x * (x_max - x_min) + x_min


def generate_model__raw(num_of_points: int) -> Sequential:
    """
    Создает модель сверточной нейронной сети для обработки акустических сигналов.

    Args:
        num_of_points: Количество точек в сигнале (определяет размер входного слоя)

    Returns:
        Sequential: Модель Keras Sequential
    """
    return Sequential(
        [
            Input(shape=(num_of_points * 2,)),  # умножаем на 2, т.к. на вход подаются и `x` и `y` координаты
            Reshape((num_of_points * 2, 1)),
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
            Dense(2, activation="linear"),  # 2 точки - координаты линий уровней
        ]
    )


class HistoryToFile(Callback):
    """Callback для сохранения истории обучения в JSON-файл."""

    def __init__(self, history_file):
        """
        Инициализирует callback.

        Args:
            history_file: Путь к файлу для сохранения истории
        """
        super().__init__()
        self.history_file = history_file

    def on_epoch_end(self, epoch, logs=None):
        """
        Сохраняет историю обучения после каждой эпохи.

        Args:
            epoch: Номер текущей эпохи
            logs: Словарь с метриками обучения
        """
        if not self.model.history.history:
            return
        with open(self.history_file, "w") as f:
            json.dump(self.model.history.history, f, indent=4)


class PlotHistory(Callback):
    """Callback для визуализации истории обучения."""

    def __init__(self, image_file):
        """
        Инициализирует callback.

        Args:
            image_file: Путь к файлу для сохранения графика
        """
        super().__init__()
        self.image_file = image_file

    def on_epoch_end(self, epoch, logs=None):
        """
        Сохраняет график истории обучения после каждой эпохи.

        Args:
            epoch: Номер текущей эпохи
            logs: Словарь с метриками обучения
        """
        # Отсекаем первую эпоху, т.к. там очень большие ошибки
        loss = self.model.history.history.get("loss", [])[1:]
        val_loss = self.model.history.history.get("val_loss", [])[1:]

        if not loss or not val_loss:
            return

        plt.figure()
        plt.plot(loss, label="Ошибка на тестовых данных")
        plt.plot(val_loss, label="Ошибка на валидационных данных")
        plt.title("Ошибка на тестовых и валидационных данных")
        plt.xlabel("Эпохи")
        plt.ylabel("Ошибка (%)")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, max(max(loss), max(val_loss)) * 1.1)  # Установка пределов по оси Y
        plt.savefig(self.image_file, dpi=300)
        plt.close()  # Закрываем фигуру, чтобы не перегружать память
