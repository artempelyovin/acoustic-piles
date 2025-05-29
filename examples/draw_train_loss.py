import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

from matplotlib.ticker import AutoMinorLocator

from utils import set_plt_style


def draw_loss(input_path: str, output_path: str, without_first_epoch: bool) -> None:
    with open(input_path, "r") as f:
        data = json.load(f)
    train_loss, val_loss = data["loss"], data["val_loss"]

    fig, ax = plt.subplots(figsize=(10, 6))

    miv_val_loss = min(val_loss)
    if without_first_epoch:
        epochs = np.arange(2, len(train_loss) + 1)  # начинаем со 2-й эпохи
        train_loss = train_loss[1:]
        val_loss = val_loss[1:]
        min_epoch = val_loss.index(miv_val_loss) + 2
    else:
        epochs = np.arange(1, len(train_loss) + 1)
        min_epoch = val_loss.index(miv_val_loss) + 1

    ax.plot(epochs, train_loss, color="tab:blue", label="Обучающая выборка")
    ax.plot(epochs, val_loss, color="tab:orange", label="Валидационная выборка")
    ax.plot(min_epoch, miv_val_loss, "o", markersize=5, markerfacecolor="tab:green", alpha=0.8)
    ax.axvline(
        min_epoch,
        color="tab:green",
        linestyle="--",
        alpha=0.8,
        label=f"Минимум {miv_val_loss:.6f} (эпоха {min_epoch })",
    )

    # Настройка осей
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.3)

    # Подписи
    ax.legend()
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Значение функции потерь, мс")
    ax.set_title("Кривая обучения модели", fontsize=14, pad=15)

    # Разметка осей
    ax.set_xticks(np.arange(0, len(train_loss) + 1, max(1, len(train_loss) // 10)))
    ax.set_xlim(1, len(train_loss) + 2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


if __name__ == "__main__":
    set_plt_style()
    parser = argparse.ArgumentParser(
        description="Построение графика функции потерь во время обучения (на обучаемом и валидационных датасетах)"
    )
    parser.add_argument("--input", type=str, help="Путь к .json файлу с данными о потерях")
    parser.add_argument("--output", type=str, help="Путь для сохранения графика")
    parser.add_argument(
        "--without-first-epoch",
        action="store_true",
        default=False,
        help="Исключить первую эпоху? (по умолчанию: False)",
    )
    args = parser.parse_args()
    draw_loss(input_path=args.input, output_path=args.output, without_first_epoch=args.without_first_epoch)
