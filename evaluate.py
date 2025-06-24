"""
Модуль для оценки качества обученных нейронных сетей на тестовых данных.

Данный модуль предназначен для загрузки обученной модели и оценки её производительности
на тестовой выборке. Выполняет предобработку данных, нормализацию и формирование
входных векторов в том же формате, который использовался при обучении.
"""

import argparse

import numpy as np
from keras import models, Sequential

# noinspection PyUnresolvedReferences
from train import absolute_percentage_error
from utils import normalize, load_dataset__raw


def evaluate(model_number: int, weights_path: str, dataset_size: int) -> None:
    """
    Выполняет оценку качества обученной нейронной сети на тестовых данных.

    Функция загружает датасет, выполняет предобработку данных (нормализацию и
    форматирование), разделяет данные на обучающую и тестовую выборки,
    загружает обученную модель и оценивает её производительность.

    Args:
        model_number (int): Номер модели для определения пути к датасету
        weights_path (str): Путь к файлу с весами обученной модели
        dataset_size (int): Максимальный размер используемого датасета

    Raises:
        AssertionError: Если размерности векторов данных не соответствуют ожиданиям
        ValueError: Если размер датасета меньше запрашиваемого размера
    """
    # Загружаем датасет и проверяем размерности
    dataset_dir = f"datasets/{model_number}/raw_data"
    X, Y, ANSWERS = load_dataset__raw(dataset_dir)
    assert (
        len(X) == len(Y) == len(ANSWERS)
    ), f"Разные длины векторов X ({len(X)}), Y ({len(Y)}) и ANSWERS ({len(ANSWERS)})"
    assert len(X[0]) == len(Y[0]), f"Разная размерность X[0] ({len(X[0])}) и Y[0] ({len(Y[0])})"
    if len(X) < dataset_size:
        raise ValueError(f"Размер датасета ({len(X)} шт.) меньше желаемого ({dataset_size} шт.)")

    # Обрезаем до нужного размера
    X = np.array(X[:dataset_size])
    Y = np.array(Y[:dataset_size])
    ANSWERS = np.array(ANSWERS[:dataset_size])

    # Нормализуем все данные в диапазон [0;1]
    for i in range(len(X)):
        # Нормализуем ответы относительно диапазона соответствующего вектора X[i]
        ANSWERS[i] = normalize(ANSWERS[i], x_min=X[i].min(), x_max=X[i].max())
        X[i] = normalize(X[i])
        Y[i] = normalize(Y[i])

    # Преобразование формата данных: [x1, x2, ..., xn], [y1, y2, ..., yn] --> [x1, y1, x2, y2, ..., xn, yn]
    NEW_X = np.empty((X.shape[0], 2 * X.shape[1]), dtype=X.dtype)
    NEW_X[:, 0::2] = X
    NEW_X[:, 1::2] = Y

    X = NEW_X  # Входные данные для нейросети
    Y = ANSWERS  # Целевые значения для нейросети

    # Разделение на обучающую и тестовую выборки (80/20)
    split_index = int(0.8 * len(X))
    _, X_test = X[:split_index], X[split_index:]
    _, Y_test = Y[:split_index], Y[split_index:]

    # Загрузка и оценка модели
    model: Sequential = models.load_model(weights_path)
    model.summary()

    model.evaluate(X_test, Y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка обученной нейронной сети на сырых данных")
    parser.add_argument("--model-number", type=int, choices=[1, 2, 3, 4, 5, 6], required=True, help="Номер модели")
    parser.add_argument("--weights-path", type=str, required=True, help="Путь до весов модели")
    parser.add_argument(
        "--dataset-size", type=int, default=5000, help="Обрезание датасета до нужного размера (по умолчанию: 5000)"
    )
    args = parser.parse_args()
    evaluate(model_number=args.model_number, weights_path=args.weights_path, dataset_size=args.dataset_size)
