import argparse

import numpy as np
from keras import models, Sequential
from matplotlib import pyplot as plt

from utils import draw_acoustic_signal, draw_level_lines, get_generator_function_by_model_number


def prediction(model_number: int, weights_path: str) -> None:
    generate_pulse_signal = get_generator_function_by_model_number(model_number)

    model: Sequential = models.load_model(weights_path)
    model.summary()

    while True:
        x, y, start_x, reflection_x = generate_pulse_signal()

        model_input = np.column_stack((x, y)).reshape(-1)  # делаем массив точек формата [x1, y1, x2, y2, ..., xn, yn]
        predict = model.predict(np.array([model_input]))[0]
        start_x_predict, reflection_x_predict = predict

        fig, ax = plt.subplots()

        print(f"MAE точки начала: {abs(start_x - start_x_predict)}")
        print(f"MAE таки отражения: {abs(reflection_x - reflection_x_predict)}")
        draw_acoustic_signal(ax=ax, x=x, y=y)
        draw_level_lines(
            ax=ax, start_x=start_x, reflection_x=reflection_x, color="blue", linestyle="dashdot", alpha=0.5
        )
        draw_level_lines(
            ax=ax,
            start_x=start_x_predict,
            reflection_x=reflection_x_predict,
            color="red",
            linestyle="dashdot",
            alpha=0.5,
        )

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка обученной нейронной сети на сырых данных")
    parser.add_argument("--model-number", type=int, choices=[10, 20, 30, 40], required=True, help="Номер модели.")
    parser.add_argument("--weights-path", type=str, required=True, help="Путь до весов модели.")

    args = parser.parse_args()
    prediction(model_number=args.model_number, weights_path=args.weights_path)
