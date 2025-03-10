import numpy as np
from keras import Sequential
from matplotlib import pyplot as plt

from utils import generate_model__raw, load_dataset__raw, draw_acoustic_signal, draw_zero_crossings

DATASET_PATH = "datasets/raw_data"
WEIGHTS_PATH = "weights/raw/model_epoch_47_val_loss_1590.07.h5"


def load_model(weights_path: str) -> Sequential:
    model = generate_model__raw()
    model.load_weights(weights_path)
    return model


def main() -> None:
    X, Y = load_dataset__raw(DATASET_PATH)
    split_index = int(0.8 * len(X))
    _, X_test = X[:split_index], X[split_index:]
    _, Y_test = Y[:split_index], Y[split_index:]
    model = load_model(weights_path=WEIGHTS_PATH)

    for X, Y in zip(X_test, Y_test):
        fig, ax = plt.subplots()

        x = X[0::2]  # координата x - это все чётные элементы
        y = X[1::2]  # координата y - это все нечётные элементы

        # избавляемся от пустых значений (которые у нас равны нулю)
        zero_crossings_xs_real = np.array([v for v in Y if v != 0.0])
        zero_crossings_xs_predict = model.predict(np.array([X]))[0]
        # избавляемся от пустых значений (которые меньше нуля)
        zero_crossings_xs_predict = np.array([v for v in zero_crossings_xs_predict if v > 0.0])

        print(zero_crossings_xs_predict)
        draw_acoustic_signal(ax=ax, x=x, y=y)
        draw_zero_crossings(ax=ax, zero_crossings_xs=zero_crossings_xs_predict, color="blue", alpha=0.3)
        draw_zero_crossings(ax=ax, zero_crossings_xs=zero_crossings_xs_real, color="red", alpha=0.5)

        plt.show()


if __name__ == "__main__":
    main()
