import numpy as np
from keras import models, Sequential
from matplotlib import pyplot as plt

from utils import load_dataset__raw, draw_acoustic_signal, draw_points

MODEL_NUMBER = 4
MODEL_TYPE = "conv1d"
DATASET_DIR = f"datasets/{MODEL_NUMBER}/raw_data"
WEIGHTS_PATH = f"results/weights/{MODEL_NUMBER}/{MODEL_TYPE}/220d__2025-05-01T19:30:46__dataset_size=5000__loss=mae__lr=0.001__batch_size=32__epochs=250__epoch=0148__val_mse=0.00021__val_mae=0.01053.keras"


def main() -> None:
    X, Y = load_dataset__raw(DATASET_DIR)
    split_index = int(0.8 * len(X))
    _, X_test = X[:split_index], X[split_index:]
    _, Y_test = Y[:split_index], Y[split_index:]

    model: Sequential = models.load_model(WEIGHTS_PATH)
    model.summary()

    for X, Y in zip(X_test, Y_test):
        predict = model.predict(np.array([X]))[0]
        start_x_predict, reflection_x_predict = predict

        fig, ax = plt.subplots()

        x = X[0::2]  # координата x - это все чётные элементы
        y = X[1::2]  # координата y - это все нечётные элементы
        start_x, reflection_x = Y

        print(f"MAE точки начала: {abs(start_x - start_x_predict)}")
        print(f"MAE таки отражения: {abs(reflection_x - reflection_x_predict)}")
        draw_acoustic_signal(ax=ax, x=x, y=y)
        draw_points(ax=ax, start_x=start_x, reflection_x=reflection_x, color="blue", linestyle="dashdot", alpha=0.5)
        draw_points(
            ax=ax,
            start_x=start_x_predict,
            reflection_x=reflection_x_predict,
            color="red",
            linestyle="dashdot",
            alpha=0.5,
        )

        plt.show()


if __name__ == "__main__":
    main()
