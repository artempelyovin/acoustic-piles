"""
Модуль для обучения нейронных сетей на датасетах различных моделей.

Этот модуль предоставляет функциональность для обучения сверточных нейронных сетей
с использованием TensorFlow/Keras. Включает в себя настройку гиперпараметров,
обработку данных, callbacks для сохранения истории обучения и весов модели.
"""

import argparse
import os
import uuid
from datetime import datetime

import keras
import numpy as np
import tensorflow as tf
from keras.src.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.src.optimizers import Adam

from utils import load_dataset__raw, generate_model__raw, HistoryToFile, PlotHistory, normalize


@keras.saving.register_keras_serializable()
def absolute_percentage_error(y_true, y_pred):
    """
    Кастомная функция потерь для вычисления абсолютной процентной ошибки.

    Вычисляет среднюю абсолютную ошибку между истинными и предсказанными значениями,
    умноженную на 100 для получения процентного значения.

    Args:
        y_true: Истинные значения целевой переменной.
        y_pred: Предсказанные значения модели.

    Returns:
        Абсолютная процентная ошибка между y_true и y_pred.
    """
    return tf.keras.losses.MAE(y_true, y_pred) * 100


def train(
    model_number: int,
    learning_rate: float,
    reduce_learning_rate: float,
    epochs: int,
    batch_size: int,
    dataset_size: int,
) -> None:
    """
    Обучает нейронную сеть для указанной модели с заданными гиперпараметрами.

    Функция загружает датасет, подготавливает данные (нормализация, разбиение на
    обучающую и валидационную выборки), создает модель и проводит обучение с
    сохранением истории и весов.

    Args:
        model_number: Номер модели (1-6) для обучения.
        learning_rate: Начальная скорость обучения для оптимизатора Adam.
        reduce_learning_rate: Флаг для уменьшения learning rate в процессе обучения.
        epochs: Количество эпох обучения.
        batch_size: Размер батча для обучения.
        dataset_size: Максимальный размер используемого датасета.

    Raises:
        ValueError: Если размер датасета меньше требуемого dataset_size.
        AssertionError: Если размерности векторов X, Y и ANSWERS не совпадают.
    """
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    uuid_ = str(uuid.uuid4())[:4]

    # Шаблоны путей для сохранения результатов
    base_file_template = f"{uuid_}__{now}__dataset_size={dataset_size}__start_lr={learning_rate}__reduce_lr={reduce_learning_rate}__batch_size={batch_size}__epochs={epochs}"
    history_file = f"results/history/{model_number}/conv1d/{base_file_template}.json"
    history_image_file = f"results/history/{model_number}/conv1d/{base_file_template}.png"
    weight_file = f"results/weights/{model_number}/conv1d/{base_file_template}__epoch={{epoch:04d}}__val_loss={{val_loss:.6f}}.keras"
    dataset_dir = f"datasets/{model_number}/raw_data"

    # Создаём директории (если ещё не созданы)
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    os.makedirs(os.path.dirname(history_image_file), exist_ok=True)
    os.makedirs(os.path.dirname(weight_file), exist_ok=True)

    # Загружаем датасет и проверяем размерности
    X, Y, ANSWERS = load_dataset__raw(dataset_dir)
    assert (
        len(X) == len(Y) == len(ANSWERS)
    ), f"Разные длины векторов X ({len(X)}), Y ({len(Y)}) и ANSWERS ({len(ANSWERS)})"
    assert len(X[0]) == len(Y[0]), f"Разная размерность X[0] ({len(X[0])}) и Y[0] ({len(Y[0])})"
    if len(X) < dataset_size:
        raise ValueError(f"Размер датасета ({len(X)} шт.) меньше желаемого ({dataset_size} шт.)")
    num_of_points = len(X[0])

    # Обрезаем до dataset_size
    X = np.array(X[:dataset_size])
    Y = np.array(Y[:dataset_size])
    ANSWERS = np.array(ANSWERS[:dataset_size])

    # Нормализуем всё в диапазон [0;1]
    for i in range(len(X)):
        ANSWERS[i] = normalize(ANSWERS[i], x_min=X[i].min(), x_max=X[i].max())  # нормализуем относительно вектора X[i]!
        X[i] = normalize(X[i])
        Y[i] = normalize(Y[i])

    # Преобразуем [x1, x2, ..., xn], [y1, y2, ..., yn] --> [x1, y1, x2, y2, ..., xn, yn]
    NEW_X = np.empty((X.shape[0], 2 * X.shape[1]), dtype=X.dtype)
    NEW_X[:, 0::2] = X
    NEW_X[:, 1::2] = Y

    X = NEW_X  # вход нейросети
    Y = ANSWERS  # выход нейросети

    # Разбиение на обучающую и тестовую выборки
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]

    # Подготовка модели
    model = generate_model__raw(num_of_points=num_of_points)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=absolute_percentage_error)
    model.summary()

    # Обучение
    callbacks = [
        ModelCheckpoint(filepath=weight_file, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
        HistoryToFile(history_file=history_file),
        PlotHistory(image_file=history_image_file),
    ]
    if reduce_learning_rate:
        callbacks.append(
            ReduceLROnPlateau(monitor="val_loss", mode="min", patience=40, factor=0.5, min_lr=0.0005, verbose=1)
        )
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=callbacks)

    # Проверка на тестовом датасете
    model.evaluate(X_test, Y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение нейронной сети для выбранной модели")
    parser.add_argument("--model-number", type=int, choices=[1, 2, 3, 4, 5, 6], required=True, help="Номер модели.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate для оптимизатора Adam")
    parser.add_argument(
        "--reduce-learning-rate",
        action="store_true",
        help="Уменьшать learning rate для оптимизатора Adam в процессе обучения?",
    )
    parser.add_argument("--epochs", type=int, default=250, help="Количество эпох в обучении")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча в обучении")
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=5000,
        help="Размер датасета для обучения "
        "(если загружаемый датасет большего размера, то он будет уменьшен до данного значения)",
    )

    args = parser.parse_args()
    train(
        model_number=args.model_number,
        learning_rate=args.learning_rate,
        reduce_learning_rate=args.reduce_learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_size=args.dataset_size,
    )
