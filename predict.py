"""
Модуль для тестирования обученных нейронных сетей на новых данных.

Данный модуль предназначен для проверки качества предсказаний обученной модели
на синтетически сгенерированных акустических сигналах. Выполняет визуализацию
результатов предсказания, расчет метрик точности и поддерживает интерактивный
режим для детального анализа каждого предсказания.
"""

import argparse

import numpy as np
from keras import models, Sequential
from matplotlib import pyplot as plt

# noinspection PyUnresolvedReferences
from train import absolute_percentage_error
from utils import draw_acoustic_signal, get_generator_function_by_model_number, normalize, denormalize


def prediction(model_number: int, weights_path: str, num_samples: int, interactive_mode: bool) -> None:
    """
    Выполняет тестирование обученной модели на новых синтетических данных.

    Функция генерирует заданное количество акустических сигналов, выполняет
    предсказания с помощью обученной модели, вычисляет метрики точности и
    визуализирует результаты. В интерактивном режиме показывает детальные
    графики с наложением истинных и предсказанных значений.

    Args:
        model_number (int): Номер модели для определения типа генерируемых сигналов
        weights_path (str): Путь к файлу с весами обученной модели
        num_samples (int): Количество тестовых примеров для генерации
        interactive_mode (bool): Флаг включения интерактивного режима с визуализацией
    """
    if interactive_mode:
        print("Нажмите любую клавишу, чтобы перейти к следующему предсказанию")

    generate_pulse_signal = get_generator_function_by_model_number(model_number)

    # Загрузка обученной модели
    model: Sequential = models.load_model(weights_path)
    model.summary()

    # Списки для накопления ошибок
    maep_1_points, maep_2_points, maep_commons = [], [], []

    for i in range(1, num_samples + 1):
        # Генерация тестового сигнала
        x, y, start_x, reflection_x = generate_pulse_signal()
        x = np.array(x)
        y = np.array(y)

        # Нормализация данных
        x_min, x_max = x.min(), x.max()
        x_normalize = normalize(x)
        y_normalize = normalize(y)
        start_x_normalize = normalize(start_x, x_min, x_max)
        reflection_x_normalize = normalize(reflection_x, x_min, x_max)

        # Преобразование формата входных данных: [x1, x2, ..., xn], [y1, y2, ..., yn] --> [x1, y1, x2, y2, ..., xn, yn]
        model_input = np.empty((2 * x_normalize.shape[0],), dtype=x_normalize.dtype)
        model_input[0::2] = x_normalize
        model_input[1::2] = y_normalize

        # Выполнение предсказания
        predict_normalize = model.predict(np.array([model_input]), verbose=0)[0]
        predict = denormalize(predict_normalize, x_min=x_min, x_max=x_max)
        start_x_predict, reflection_x_predict = predict
        start_x_predict_normalize, reflection_x_predict_normalize = predict_normalize

        title = f"Предсказание {i}/{num_samples}"

        # Расчет метрик ошибок
        maep_1_points.append(abs(start_x_normalize - start_x_predict_normalize) * 100)
        mae_1_point_all = sum(maep_1_points) / i if i != 0 else sum(maep_1_points)
        mae_1_point_str = (
            f"Средняя абсолютная ошибка в % для первой линии уровня (текущая/общая): "
            f"{maep_1_points[-1]:.3f}%/{mae_1_point_all:.3f}%"
        )

        maep_2_points.append(abs(reflection_x_normalize - reflection_x_predict_normalize) * 100)
        mae_2_point_all = sum(maep_2_points) / i if i != 0 else sum(maep_2_points)
        mae_2_point_str = (
            f"Средняя абсолютная ошибка в % для второй линии уровня (текущая/общая): "
            f"{maep_2_points[-1]:.3f}%/{mae_2_point_all:.3f}%"
        )

        maep_commons.append((maep_1_points[-1] + maep_2_points[-1]) / 2)
        mae_common_all = sum(maep_commons) / i if i != 0 else sum(maep_commons)
        mae_common_str = (
            f"Средняя абсолютная ошибка в % для обеих линий уровня (текущая/общая): "
            f"{maep_commons[-1]:.3f}%/{mae_common_all:.3f}%"
        )

        if interactive_mode:

            def on_key(event):
                """
                Обработчик нажатий клавиш в интерактивном режиме.

                Args:
                    event: Событие нажатия клавиши
                """
                plt.close()
                return

            # Создание детального графика с результатами
            fig, ax = plt.subplots(figsize=(17, 7))
            fig.canvas.manager.set_window_title(title)
            plt.title(f"{mae_1_point_str}\n{mae_2_point_str}\n{mae_common_str}", fontsize=10)

            # Отрисовка сигнала и линий уровня
            draw_acoustic_signal(ax=ax, x=x, y=y)
            ax.axvline(x=start_x, color="c", linestyle="dashdot", alpha=0.5, label="1 точка (истинная)")
            ax.axvline(x=start_x_predict, color="m", linestyle="dashdot", alpha=0.5, label="1 точка (предсказание)")
            ax.axvline(x=reflection_x, color="c", linestyle="dashdot", alpha=0.5, label="2 точка (истинная)")
            ax.axvline(
                x=reflection_x_predict, color="m", linestyle="dashdot", alpha=0.5, label="2 точка (предсказание)"
            )

            plt.legend()
            fig.canvas.mpl_connect("key_press_event", on_key)
            plt.show()
        else:
            # Вывод результатов в консоль
            print(title)
            print(mae_1_point_str)
            print(mae_2_point_str)
            print(mae_common_str)

    # Итоговая статистика
    print("-" * 40)
    print(f"Средняя абсолютная ошибка для первой линии уровня (общая): {sum(maep_1_points) / num_samples:.3f}%")
    print(f"Средняя абсолютная ошибка для второй линии уровня (общая): {sum(maep_2_points) / num_samples:.3f}%")
    print(f"Средняя абсолютная ошибка для обеих линий уровня (общая): {sum(maep_commons) / num_samples:.3f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Тестирование обученной нейронной сети на новых данных")
    parser.add_argument("--model-number", type=int, choices=[1, 2, 3, 4, 5, 6], required=True, help="Номер модели")
    parser.add_argument("--weights-path", type=str, required=True, help="Путь до весов модели")
    parser.add_argument("--num-samples", type=int, default=100, help="Количество примеров для проверки модели")
    parser.add_argument("--interactive-mode", action="store_true", help="Включить интерактивный режим")

    args = parser.parse_args()
    prediction(
        model_number=args.model_number,
        weights_path=args.weights_path,
        num_samples=args.num_samples,
        interactive_mode=args.interactive_mode,
    )
