import argparse

import numpy as np
from keras import models, Sequential
from matplotlib import pyplot as plt

from utils import draw_acoustic_signal, get_generator_function_by_model_number, normalize, denormalize


def prediction(model_number: int, weights_path: str) -> None:
    generate_pulse_signal = get_generator_function_by_model_number(model_number)

    model: Sequential = models.load_model(weights_path)
    model.summary()

    while True:
        x, y, start_x, reflection_x = generate_pulse_signal()
        x = np.array(x)
        y = np.array(y)

        # нормализуем
        x_min, x_max = x.min(), x.max()
        x_normalize = normalize(x)
        y_normalize = normalize(y)

        # [x1, x2, ..., xn], [y1, y2, ..., yn] --> [x1, y1, x2, y2, ..., xn, yn]
        model_input = np.empty((2 * x_normalize.shape[0],), dtype=x_normalize.dtype)
        model_input[0::2] = x_normalize
        model_input[1::2] = y_normalize

        # предсказание
        predict = model.predict(np.array([model_input]), verbose=0)[0]
        predict = denormalize(predict, x_min=x_min, x_max=x_max)
        start_x_predict, reflection_x_predict = predict

        # расчёт ошибок
        mae_1_point = abs(start_x - start_x_predict)
        mae_2_point = abs(reflection_x - reflection_x_predict)

        fig, ax = plt.subplots()
        plt.title(f"1 point mae = {mae_1_point}\n2 point mae = {mae_2_point}")
        draw_acoustic_signal(ax=ax, x=x, y=y)
        ax.axvline(x=start_x, color="c", linestyle="dashdot", alpha=0.5, label="1 point (true label)")
        ax.axvline(x=start_x_predict, color="m", linestyle="dashdot", alpha=0.5, label="1 point (model predict)")
        ax.axvline(x=reflection_x, color="c", linestyle="dashdot", alpha=0.5, label="2 point (true label)")
        ax.axvline(x=reflection_x_predict, color="m", linestyle="dashdot", alpha=0.5, label="2 point (model predict)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка обученной нейронной сети на сырых данных")
    parser.add_argument(
        "--model-number", type=int, choices=[10, 20, 30, 40, 50, 60], required=True, help="Номер модели."
    )
    parser.add_argument("--weights-path", type=str, required=True, help="Путь до весов модели.")

    args = parser.parse_args()
    prediction(model_number=args.model_number, weights_path=args.weights_path)
