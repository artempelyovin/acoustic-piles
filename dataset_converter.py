import argparse
import json
import os
import shutil

from matplotlib import pyplot as plt

from utils import save_acoustic_signal_as_image, draw_acoustic_signal


def convert(dataset_path: str, model_number: int) -> None:
    filenames = sorted([f for f in os.listdir(dataset_path)])
    non_json_filenames = [f for f in filenames if not f.endswith(".json")]
    if len(non_json_filenames) > 0:
        raise ValueError(f"Найдено {len(non_json_filenames)} не-JSON файлов: {non_json_filenames}")

    new_raw_dataset_path = f"datasets/{model_number}/raw_data"
    new_fig_dataset_path = f"datasets/{model_number}/fig_data"
    os.makedirs(new_raw_dataset_path, exist_ok=True)
    os.makedirs(new_fig_dataset_path, exist_ok=True)
    print(f"Создание директорий: {new_raw_dataset_path}, {new_fig_dataset_path}")

    # Check files and find max dim
    max_dim = 0
    for filename in filenames:
        old_path = f"{dataset_path}/{filename}"

        with open(old_path, "r") as file:
            content = json.load(file)
            assert "x" in content, f'В файле {old_path} отсутствует поле "x"X'
            assert "y" in content, f'В файле {old_path} отсутствует поле "y"'
            assert 'answers" in content, f"В файле {old_path} отсутствует поле "answers"'
            assert len(content["x"]) == len(
                content["y"]
            ), f'Размерности "x" и "y" различны ({len(content["x"]) != {len(content["y"])}})'
            if len(content["x"]) > max_dim:
                max_dim = len(content["x"])

    print(f"Все файлы будут расширены до {max_dim} точек...")

    # save
    for i, filename in enumerate(filenames, start=1):
        old_path = f"{dataset_path}/{filename}"
        new_raw_path = f"{new_raw_dataset_path}/{i}.json"
        new_fig_path = f"{new_fig_dataset_path}/{i}.png"

        with open(old_path, "r") as file:
            content = json.load(file)
            content["x"] = [x * 1000 for x in content["x"]]  # из секунд в мс
            content["answers"][0] *= 1000
            content["answers"][1] *= 1000
            if len(content["x"]) != max_dim:
                x_min = min(content["x"])
                x_max = max(content["x"])
                x_delta = (x_max - x_min) / len(content["x"])
                last_xs = [x_max + x_delta * i for i in range(max_dim - len(content["x"]))]
                content["x"].extend(last_xs)
            if len(content["y"]) != max_dim:
                last_ys = [0 for _ in range(max_dim - len(content["y"]))]
                content["y"].extend(last_ys)
        # raw_data
        with open(new_raw_path, "w") as file:
            json.dump(content, file, indent=4)

        # fig_data
        fig, ax = plt.subplots()
        # убираем всё лишнее с графика
        ax.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)
        plt.margins(x=0)  # убрали отступы по оси X слева и справа
        plt.tight_layout()  # Автоматически подобрали границу
        draw_acoustic_signal(ax=ax, x=content["x"], y=content["y"])
        save_acoustic_signal_as_image(fig=fig, filename=new_fig_path)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Конвертор датасета, сгенерированного в Matlab, в формат, необходимый для данного проекта"
    )
    parser.add_argument("--dataset-path", type=str, required=True, help="Путь до датасета")
    parser.add_argument("--model-number", type=int, choices=[6], required=True, help="Номер модели.")
    args = parser.parse_args()
    convert(dataset_path=args.dataset_path, model_number=args.model_number)
