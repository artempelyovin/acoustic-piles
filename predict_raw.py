import numpy as np
from keras import Sequential
from matplotlib import pyplot as plt

from utils import generate_model__raw, load_dataset__raw, draw_acoustic_signal, draw_points

MODEL_NUMBER = 2
MODEL_TYPE = 'raw'
DATASET_DIR = f"datasets/{MODEL_NUMBER}/{MODEL_TYPE}_data"
WEIGHTS_PATH = f"results/weights/{MODEL_NUMBER}/{MODEL_TYPE}/2025-04-30T13:38:43__loss=MAE__batch_size=32__epochs=250__epoch=0232__mse=0.0002__mae=0.0099.keras"


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
