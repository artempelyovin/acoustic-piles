import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from utils import set_plt_style, normalize


def draw_example() -> None:
    xs = np.array([2, 5, 6])
    ys = np.array([4, 6, 12])
    xs_norm = normalize(xs)
    ys_norm = normalize(ys)

    # Создаем график
    plt.figure(figsize=(10, 4))

    # До нормализации
    plt.subplot(1, 2, 1)
    plt.scatter(xs, ys, color="tab:blue")
    plt.plot(xs, ys, color="tab:blue")
    plt.title("До нормализации")
    plt.xlim(0, 7)
    plt.ylim(0, 13)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.3)

    # После нормализации
    plt.subplot(1, 2, 2)
    plt.scatter(xs_norm, ys_norm, color="tab:orange")
    plt.plot(xs_norm, ys_norm, color="tab:orange")
    plt.title("После нормализации")
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    set_plt_style()
    draw_example()
