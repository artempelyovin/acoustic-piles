import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from utils import set_plt_style


def draw_dataset_size_effect_on_loss(
    dataset_sizes: list[int], val_losses: list[float], test_losses: list[float], image_name: str
) -> None:
    fig, ax = plt.subplots()

    ax.scatter(dataset_sizes, val_losses, color="tab:blue")
    ax.plot(dataset_sizes, val_losses, color="tab:blue", label="Ошибка на валидационных данных")
    ax.scatter(dataset_sizes, test_losses, color="tab:orange")
    ax.plot(dataset_sizes, test_losses, color="tab:orange", label="Ошибка на тестовых данных")

    # Настройка осей
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.3)

    # Подписи
    ax.legend()
    ax.set_xlabel("Размер датасета")
    ax.set_ylabel("Ошибка")
    ax.set_title("Зависимость ошибки от размера датасета", fontsize=14, pad=15)

    plt.tight_layout()
    plt.savefig(image_name, dpi=300)


if __name__ == "__main__":
    set_plt_style()

    dataset_sizes = [100, 300, 1000, 5000]
    val_losses = [0.030797, 0.017569, 0.006211, 0.000991]
    test_losses = [0.0477, 0.0175, 0.0064, 0.00073]
    image_name = "10_dataseet_size_effect_on_loss.png"
    draw_dataset_size_effect_on_loss(
        dataset_sizes=dataset_sizes, val_losses=val_losses, test_losses=test_losses, image_name=image_name
    )
