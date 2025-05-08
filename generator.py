import argparse

from matplotlib import pyplot as plt

from utils import (
    draw_acoustic_signal,
    get_generator_function_by_model_number,
    save_acoustic_signal_as_image,
    save_acoustic_signal_as_json,
)


def generator(model_number: int, dataset_size: int, interactive_mode: bool) -> None:
    assert len(str(model_number)) == 2, f"model_number должен состоять из 2-х цифр"
    model_dir = f"{str(model_number)[0]}_"
    fig_dataset_dir = f"datasets/{model_dir}/fig_data"
    raw_dataset_dir = f"datasets/{model_dir}/raw_data"
    generate_pulse_signal = get_generator_function_by_model_number(model_number)

    if interactive_mode:
        print("Нажмите Enter, чтобы сохранить результат или любую другую клавишу, чтобы пропустить!")

    i = 0
    while i < dataset_size:
        x, y, start_x, reflection_x = generate_pulse_signal()

        fig, ax = plt.subplots()
        # убираем всё лишнее с графика
        ax.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)
        if interactive_mode:
            print(f"Сигнал №{i + 1} из {dataset_size}. Enter - сохранить, любая другая клавиша - пропустить!")
        else:
            print(f"Сигнал №{i + 1} из {dataset_size}")
        draw_acoustic_signal(ax=ax, x=x, y=y)
        plt.margins(x=0)  # убрали отступы по оси X слева и справа
        plt.tight_layout()  # Автоматически подобрали границу

        if interactive_mode:

            def on_key(event):
                nonlocal i
                if event.key != "enter":
                    plt.close()
                    return

                # сохраняем график без ответов
                save_acoustic_signal_as_image(fig=fig, filename=f"{fig_dataset_dir}/{i + 1}.png")
                # сохраняем сырые данные
                save_acoustic_signal_as_json(
                    x=x, y=y, start_x=start_x, reflection_x=reflection_x, filename=f"{raw_dataset_dir}/{i + 1}.json"
                )
                plt.close()
                i += 1

            fig.canvas.mpl_connect("key_press_event", on_key)
            plt.show()
        else:
            # сохраняем график без ответов
            save_acoustic_signal_as_image(fig=fig, filename=f"{fig_dataset_dir}/{i + 1}.png")
            # сохраняем сырые данные
            save_acoustic_signal_as_json(
                x=x, y=y, start_x=start_x, reflection_x=reflection_x, filename=f"{raw_dataset_dir}/{i + 1}.json"
            )
            plt.close()
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерация акустических сигналов.")
    parser.add_argument(
        "--model-number", type=int, choices=[10, 20, 30, 40], required=True, help="Номер модели."
    )
    parser.add_argument("--dataset-size", type=int, default=5000, help="Размер датасета (по умолчанию 5000).")
    parser.add_argument("--interactive-mode", action="store_true", help="Включить интерактивный режим?")

    args = parser.parse_args()
    generator(model_number=args.model_number, dataset_size=args.dataset_size, interactive_mode=args.interactive_mode)
