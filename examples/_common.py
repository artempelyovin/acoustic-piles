import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def draw_pulse_signal(
    t: np.ndarray,
    pulse: np.ndarray,
    start_x: float,
    reflection_x: float,
    title: str,
    xlim: tuple[float, float] | None = (0, 1500),
    ylim: tuple[float, float] | None = (-1.1, 1.1),
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t, pulse, color="tab:blue", label="Акустический сигнал")
    ax.axvline(start_x, color="tab:red", linestyle="--", label="Начало импульса")
    ax.axvline(reflection_x, color="tab:green", linestyle="--", label="Отражение")

    # Настройка осей и сетки
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.3)

    # Подписи
    ax.legend()
    ax.set_xlabel("Время, мс", fontsize=12)
    ax.set_ylabel("Амплитуда", fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)

    plt.tight_layout()
    plt.show()
