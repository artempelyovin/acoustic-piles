import numpy as np
from keras import Sequential
from matplotlib import pyplot as plt

from utils import (
    generate_model__raw,
    load_dataset__raw,
    draw_acoustic_signal,
    draw_points,
    X_SHAPE_RAW,
)

DATASET_DIR = "datasets/raw_data"
WEIGHTS_PATH = "weights/raw/model_epoch_06_val_loss_3.70.h5"


def load_model(weights_path: str) -> Sequential:
    model = generate_model__raw()
    model.load_weights(weights_path)
    return model


def main() -> None:
    X, Y = load_dataset__raw(DATASET_DIR)
    split_index = int(0.8 * len(X))
    _, X_test = X[:split_index], X[split_index:]
    _, Y_test = Y[:split_index], Y[split_index:]
    model = load_model(weights_path=WEIGHTS_PATH)

    for X, Y in zip(X_test, Y_test):
        fig, ax = plt.subplots()

        x = X[0::2]  # координата x - это все чётные элементы
        y = X[1::2]  # координата y - это все нечётные элементы
        real_count_points = len(Y)

        # добили до нужного shape значение -1
        X_extended = np.pad(X, (0, X_SHAPE_RAW - len(X)), mode="constant", constant_values=(0, -1))
        predict_count_points = model.predict(np.array([X_extended]))[0]

        print(
            f"Дано {real_count_points}, получено {predict_count_points}. "
            f"Разница: {abs(real_count_points - predict_count_points)}"
        )
        draw_acoustic_signal(ax=ax, x=x, y=y)
        draw_points(ax=ax, zero_crossings_xs=Y, color="red", linestyle="dashdot", alpha=0.5)

        plt.show()


if __name__ == "__main__":
    main()
