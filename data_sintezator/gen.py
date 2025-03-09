import json
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from data_sintezator.utils import generate_acoustic_signal, draw_acoustic_signal, draw_zero_crossings

DATASET_SIZE = 100
FIG_DATASET_DIR = "datasets/fig_data"
RAW_DATASET_DIR = "datasets/raw_data"


def save_fig(fig: Figure, filename: str) -> None:
    if not os.path.exists(FIG_DATASET_DIR):
        os.makedirs(FIG_DATASET_DIR, exist_ok=True)
    fig.savefig(f"{FIG_DATASET_DIR}/{filename}", bbox_inches="tight", pad_inches=0)


def save_raw_data(x: np.ndarray, y: np.ndarray, zero_crossings_xs: np.ndarray, filename: str) -> None:
    if not os.path.exists(RAW_DATASET_DIR):
        os.makedirs(RAW_DATASET_DIR, exist_ok=True)
    with open(f"{RAW_DATASET_DIR}/{filename}", "w") as f:
        data = {
            "x": x.tolist(),
            "y": y.tolist(),
            "answers": zero_crossings_xs.tolist(),
        }
        json.dump(data, f, indent=4)


def main() -> None:
    print("Нажмите Enter, чтобы сохранить результат или любую другую клавишу, чтобы пропустить!")

    i = 0
    while i < DATASET_SIZE - 1:
        x, y, zero_crossings_xs = generate_acoustic_signal()

        fig, ax = plt.subplots()
        ax.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)
        print(f"Сигнал №{i + 1} из {DATASET_SIZE + 1}. Enter - сохранить, любая другая клавиша - пропустить!")
        draw_acoustic_signal(ax=ax, x=x, y=y)

        def on_key(event):
            nonlocal i
            if event.key != "enter":
                plt.close()
                return

            save_fig(fig, filename=f"{i + 1}_x.png")  # график без ответов
            draw_zero_crossings(ax=ax, zero_crossings_xs=zero_crossings_xs)
            save_fig(fig, filename=f"{i + 1}_y.png")  # график с ответами
            save_raw_data(x=x, y=y, zero_crossings_xs=zero_crossings_xs, filename=f"{i + 1}.json")
            i += 1
            plt.close()

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()


if __name__ == "__main__":
    main()
