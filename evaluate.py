import argparse

import numpy as np
from keras import models, Sequential

from utils import normalize, load_dataset__raw


def evaluate(model_number: int, weights_path: str, dataset_size: int) -> None:
    # грузим датасет и проверяем размерности
    dataset_dir = f"datasets/{model_number}/raw_data"
    X, Y, ANSWERS = load_dataset__raw(dataset_dir)
    assert (
        len(X) == len(Y) == len(ANSWERS)
    ), f"Разные длины векторов X ({len(X)}, Y ({len(Y)}) и ANSWERS ({len(ANSWERS)})"
    assert len(X[0]) == len(Y[0]), f"Разная размерность X[0] ({len(X[0])}) и Y[0] ({len(Y[0])})"
    if len(X) < dataset_size:
        raise ValueError(f"Размер датасета ({len(X)} шт.) меньше желаемого ({dataset_size} шт.)")

    # обрезаем до dataset_size
    X = np.array(X[:dataset_size])
    Y = np.array(Y[:dataset_size])
    ANSWERS = np.array(ANSWERS[:dataset_size])

    # нормализуем всё в диапазон [0;1]
    for i in range(len(X)):
        ANSWERS[i] = normalize(ANSWERS[i], x_min=X[i].min(), x_max=X[i].max())  # нормализуем относительно вектора X[i]!
        X[i] = normalize(X[i])
        Y[i] = normalize(Y[i])

    # [x1, x2, ..., xn], [y1, y2, ..., yn] --> [x1, y1, x2, y2, ..., xn, yn]
    NEW_X = np.empty((X.shape[0], 2 * X.shape[1]), dtype=X.dtype)
    NEW_X[:, 0::2] = X
    NEW_X[:, 1::2] = Y

    X = NEW_X  # вход нейросети
    Y = ANSWERS  # выход нейросети

    # разбитие на обучающую и тестовую выборки
    split_index = int(0.8 * len(X))
    _, X_test = X[:split_index], X[split_index:]
    _, Y_test = Y[:split_index], Y[split_index:]

    model: Sequential = models.load_model(weights_path)
    model.summary()

    model.evaluate(X_test, Y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка обученной нейронной сети на сырых данных")
    parser.add_argument(
        "--model-number", type=int, choices=[10, 20, 30, 40], required=True, help="Номер модели."
    )
    parser.add_argument("--weights-path", type=str, required=True, help="Путь до весов модели.")
    parser.add_argument(
        "--dataset-size", type=int, default=5000, help="Обрезание датасета до нужного размера (по умолчанию: 5000)"
    )
    args = parser.parse_args()
    evaluate(model_number=args.model_number, weights_path=args.weights_path, dataset_size=args.dataset_size)
