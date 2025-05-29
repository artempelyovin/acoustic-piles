import argparse

import numpy as np
from keras import models, Sequential
from matplotlib import pyplot as plt

from utils import draw_acoustic_signal, get_generator_function_by_model_number, normalize, denormalize


def prediction(model_number: int, weights_path: str, num_samples: int, interactive_mode: bool) -> None:
    if interactive_mode:
        print("Нажмите любую клавишу, чтобы перейти к следующему предсказанию")

    generate_pulse_signal = get_generator_function_by_model_number(model_number)

    model: Sequential = models.load_model(weights_path, compile=False)
    model.summary()

    mae_1_points, mae_2_points, mae_commons = [], [], []

    for i in range(1, num_samples + 1):
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

        title = f"Predict {i}/{num_samples}"
        # расчёт ошибок
        mae_1_points.append(abs(start_x - start_x_predict))
        mae_1_point_all = sum(mae_1_points) / i if i != 0 else sum(mae_1_points)
        mae_1_point_str = f"1 point mae (curr/all): {mae_1_points[-1]:.3f}/{mae_1_point_all:.3f}"
        mae_2_points.append(abs(reflection_x - reflection_x_predict))
        mae_2_point_all = sum(mae_2_points) / i if i != 0 else sum(mae_2_points)
        mae_2_point_str = f"2 point mae (curr/all): {mae_2_points[-1]:.3f}/{mae_2_point_all:.3f}"
        mae_commons.append((mae_1_points[-1] + mae_2_points[-1]) / 2)
        mae_common_all = sum(mae_commons) / i if i != 0 else sum(mae_commons)
        mae_common_str = f"common mae (curr/all): {mae_commons[-1]:.3f}/{mae_common_all:.3f}"

        if interactive_mode:

            def on_key(event):
                plt.close()
                return

            fig, ax = plt.subplots(figsize=(17, 7))
            fig.canvas.manager.set_window_title(title)
            plt.title(f"{mae_1_point_str}\n{mae_2_point_str}\n{mae_common_str}", fontsize=10)
            draw_acoustic_signal(ax=ax, x=x, y=y)
            ax.axvline(x=start_x, color="c", linestyle="dashdot", alpha=0.5, label="1 point (true label)")
            ax.axvline(x=start_x_predict, color="m", linestyle="dashdot", alpha=0.5, label="1 point (model predict)")
            ax.axvline(x=reflection_x, color="c", linestyle="dashdot", alpha=0.5, label="2 point (true label)")
            ax.axvline(
                x=reflection_x_predict, color="m", linestyle="dashdot", alpha=0.5, label="2 point (model predict)"
            )
            plt.legend()
            fig.canvas.mpl_connect("key_press_event", on_key)
            plt.show()
        else:
            print(title)
            print(mae_1_point_str)
            print(mae_2_point_str)
            print(mae_common_str)

    print("-" * 40)
    print(f"1 point mae (all): {sum(mae_1_points)/num_samples:.3f}")
    print(f"2 point mae (all): {sum(mae_2_points)/num_samples:.3f}")
    print(f"common mae (all): {sum(mae_commons)/num_samples:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка обученной нейронной сети на сырых данных")
    parser.add_argument("--model-number", type=int, choices=[1, 2, 3, 4], required=True, help="Номер модели.")
    parser.add_argument("--weights-path", type=str, required=True, help="Путь до весов модели.")
    parser.add_argument("--num-samples", type=int, default=100, help="Количество примеров для проверки модели.")
    parser.add_argument("--interactive-mode", action="store_true", help="Включить интерактивный режим?")

    args = parser.parse_args()
    prediction(
        model_number=args.model_number,
        weights_path=args.weights_path,
        num_samples=args.num_samples,
        interactive_mode=args.interactive_mode,
    )
