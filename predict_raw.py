import numpy as np
from keras import Sequential
from matplotlib import pyplot as plt

from utils import generate_model__raw, load_dataset__raw, draw_acoustic_signal, draw_zero_crossings, Y_SHAPE

DATASET_PATH = "datasets/raw_data"
WEIGHTS_PATH = "weights/raw/model_epoch_249_val_loss_512.70.h5"


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
        zero_crossings_xs_real = Y
        # избавляемся от пустых значений (которые меньше нуля)
        zero_crossings_xs_real = np.array([v for v in zero_crossings_xs_real if v > 0.0])

        zero_crossings_xs_predict = model.predict(np.array([X]))[0]
        # избавляемся от пустых значений (которые меньше нуля)
        zero_crossings_xs_predict = np.array([v for v in zero_crossings_xs_predict if v > 0.0])

        real_points_less_than_zero = Y_SHAPE - len(zero_crossings_xs_real)
        predict_points_less_than_zero = Y_SHAPE - len(zero_crossings_xs_predict)

        print(
            f"Точек меньше нуля должно быть {real_points_less_than_zero}, получено: {predict_points_less_than_zero}, "
            f"разница: {abs(real_points_less_than_zero-predict_points_less_than_zero)}"
        )
        print("Real:   ", [round(float(v), 2) for v in sorted(zero_crossings_xs_real)])
        print("Predict:", [round(float(v), 2) for v in sorted(zero_crossings_xs_predict)])

        draw_acoustic_signal(ax=ax, x=x, y=y)
        draw_zero_crossings(
            ax=ax, zero_crossings_xs=zero_crossings_xs_predict, color="blue", linestyle="dotted", alpha=0.5
        )
        draw_zero_crossings(
            ax=ax, zero_crossings_xs=zero_crossings_xs_real, color="red", linestyle="dashdot", alpha=0.5
        )

        plt.show()


if __name__ == "__main__":
    main()
