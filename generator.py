"""
Модуль для генерации синтетических акустических сигналов.

Данный модуль предназначен для создания датасетов акустических сигналов различных типов.
Поддерживает как автоматическую генерацию заданного количества сигналов, так и 
интерактивный режим с возможностью ручного отбора генерируемых сигналов.
Сохраняет данные в двух форматах: как изображения графиков и как JSON с сырыми данными.
"""

import argparse

from matplotlib import pyplot as plt

from utils import (
    draw_acoustic_signal,
    get_generator_function_by_model_number,
    save_acoustic_signal_as_image,
    save_acoustic_signal_as_json,
)


def generator(model_number: int, dataset_size: int, interactive_mode: bool) -> None:
    """
    Генерирует датасет акустических сигналов заданного типа и размера.

    Функция создает акустические сигналы в соответствии с выбранной моделью,
    визуализирует их и сохраняет в двух форматах: как PNG-изображения графиков
    и как JSON-файлы с сырыми данными. В интерактивном режиме позволяет
    пользователю решать, сохранять ли каждый конкретный сигнал.

    Args:
        model_number (int): Номер модели, определяющий тип генерируемых сигналов
        dataset_size (int): Количество сигналов для генерации
        interactive_mode (bool): Флаг включения интерактивного режима с ручным отбором

    Returns:
        None
    """
    fig_dataset_dir = f"datasets/{model_number}/fig_data"
    raw_dataset_dir = f"datasets/{model_number}/raw_data"
    generate_pulse_signal = get_generator_function_by_model_number(model_number)

    if interactive_mode:
        print("Нажмите Enter, чтобы сохранить результат или любую другую клавишу, чтобы пропустить!")

    i = 0
    while i < dataset_size:
        x, y, start_x, reflection_x = generate_pulse_signal()

        # Создание и настройка графика
        fig, ax = plt.subplots()
        # Убираем все лишние элементы с графика
        ax.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)

        if interactive_mode:
            print(f"Сигнал №{i + 1} из {dataset_size}. Enter - сохранить, любая другая клавиша - пропустить!")
        else:
            print(f"Сигнал №{i + 1} из {dataset_size}")

        draw_acoustic_signal(ax=ax, x=x, y=y)
        plt.margins(x=0)  # Убираем отступы по оси X
        plt.tight_layout()  # Автоматически подбираем границы

        if interactive_mode:

            def on_key(event):
                """
                Обработчик нажатий клавиш в интерактивном режиме.

                Args:
                    event: Событие нажатия клавиши
                """
                nonlocal i
                if event.key != "enter":
                    plt.close()
                    return

                # Сохраняем график как изображение
                save_acoustic_signal_as_image(fig=fig, filename=f"{fig_dataset_dir}/{i + 1}.png")
                # Сохраняем сырые данные в JSON
                save_acoustic_signal_as_json(
                    x=x, y=y, start_x=start_x, reflection_x=reflection_x, filename=f"{raw_dataset_dir}/{i + 1}.json"
                )
                plt.close()
                i += 1

            fig.canvas.mpl_connect("key_press_event", on_key)
            plt.show()
        else:
            # Автоматическое сохранение в неинтерактивном режиме
            # Сохраняем график как изображение
            save_acoustic_signal_as_image(fig=fig, filename=f"{fig_dataset_dir}/{i + 1}.png")
            # Сохраняем сырые данные в JSON
            save_acoustic_signal_as_json(
                x=x, y=y, start_x=start_x, reflection_x=reflection_x, filename=f"{raw_dataset_dir}/{i + 1}.json"
            )
            plt.close()
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерация акустических сигналов")
    parser.add_argument("--model-number", type=int, choices=[1, 2, 3, 4, 5], required=True, help="Номер модели")
    parser.add_argument("--dataset-size", type=int, default=5000, help="Размер датасета (по умолчанию 5000)")
    parser.add_argument("--interactive-mode", action="store_true", help="Включить интерактивный режим")

    args = parser.parse_args()
    generator(model_number=args.model_number, dataset_size=args.dataset_size, interactive_mode=args.interactive_mode)
