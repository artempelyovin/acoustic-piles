import json
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from utils import generate_acoustic_signal, draw_acoustic_signal, normalize

DATASET_SIZE = 5000
FIG_DATASET_DIR = "datasets/fig_data"
RAW_DATASET_DIR = "datasets/raw_data"
INTERACTIVE_MODE = False


def save_fig(fig: Figure, filename: str) -> None:
    if not os.path.exists(FIG_DATASET_DIR):
        os.makedirs(FIG_DATASET_DIR, exist_ok=True)
    fig.savefig(f"{FIG_DATASET_DIR}/{filename}", bbox_inches="tight", pad_inches=0)


def save_raw_data(x: np.ndarray, y: np.ndarray, zero_crossings_xs: np.ndarray, filename: str) -> None:
    if not os.path.exists(RAW_DATASET_DIR):
        os.makedirs(RAW_DATASET_DIR, exist_ok=True)
    with open(f"{RAW_DATASET_DIR}/{filename}", "w") as f:
        points = np.column_stack((x, y)).reshape(-1)  # делаем массив точек формата [x1, y1, x2, y2, ..., xn, yn]
        points = {
            "points": points.tolist(),
            "answers": zero_crossings_xs.tolist(),
        }
        json.dump(points, f, indent=4)


def main() -> None:
    if INTERACTIVE_MODE:
        print("Нажмите Enter, чтобы сохранить результат или любую другую клавишу, чтобы пропустить!")

    i = 0
    while i < DATASET_SIZE:
        x, y, zero_crossings_xs = generate_acoustic_signal()
        # нормируем x-сы!
        x_max, x_min = x.max(), x.min()
        x = normalize(x)
        zero_crossings_xs = normalize(zero_crossings_xs, max_value=x_max, min_value=x_min)

        fig, ax = plt.subplots()
        # убираем всё лишнее с графика
        ax.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)
        if INTERACTIVE_MODE:
            print(f"Сигнал №{i + 1} из {DATASET_SIZE}. Enter - сохранить, любая другая клавиша - пропустить!")
        else:
            print(f"Сигнал №{i + 1} из {DATASET_SIZE}")
        draw_acoustic_signal(ax=ax, x=x, y=y)
        plt.margins(x=0)  # убрали отступы по оси X слева и справа
        plt.tight_layout()  # Автоматически подобрали границу

        if INTERACTIVE_MODE:

            def on_key(event):
                nonlocal i
                if event.key != "enter":
                    plt.close()
                    return

                # сохраняем график без ответов
                save_fig(fig, filename=f"{i + 1}.png")
                # сохраняем сырые данные
                save_raw_data(x=x, y=y, zero_crossings_xs=zero_crossings_xs, filename=f"{i + 1}.json")
                plt.close()
                i += 1

            fig.canvas.mpl_connect("key_press_event", on_key)
            plt.show()
        else:
            # сохраняем график без ответов
            save_fig(fig, filename=f"{i + 1}.png")
            # сохраняем сырые данные
            save_raw_data(x=x, y=y, zero_crossings_xs=zero_crossings_xs, filename=f"{i + 1}.json")
            plt.close()
            i += 1


if __name__ == "__main__":
    main()
