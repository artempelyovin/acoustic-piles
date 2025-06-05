import os
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Пути к директориям
old_history_dir = Path("old_results/history")
new_history_dir = Path("results/history")

# Создаем новую директорию, если её нет
new_history_dir.mkdir(parents=True, exist_ok=True)


# Функция для перерисовки графиков
def plot_history(history_data, image_path):
    plt.figure()

    # Берем данные начиная со второй эпохи (как в оригинальном callback)
    loss = history_data.get("loss", [])[1:]
    val_loss = history_data.get("val_loss", [])[1:]

    if loss:
        plt.plot(loss, label="Ошибка на тестовых данных")
    if val_loss:
        plt.plot(val_loss, label="Ошибка на валидационных данных")

    plt.title("Ошибка на тестовых и валидационных данных")
    plt.xlabel("Эпохи")
    plt.ylabel("Ошибка (%)")
    plt.legend()
    plt.grid(True)

    # Автоматическое масштабирование
    max_value = max(loss) if loss else 0
    if val_loss:
        max_value = max(max_value, max(val_loss))
    plt.ylim(0, max_value * 1.1)

    plt.savefig(image_path, dpi=300)
    plt.close()


# Проходим по всем файлам в старой директории
for root, dirs, files in os.walk(old_history_dir):
    for file in files:
        if file.endswith(".json"):
            # Полный путь к старому файлу
            old_file_path = Path(root) / file

            # Создаем пути для новых файлов
            rel_path = old_file_path.relative_to(old_history_dir)
            new_json_path = new_history_dir / rel_path
            new_png_path = new_json_path.with_suffix(".png")

            # Создаем поддиректории, если их нет
            new_json_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Processing: {old_file_path}")

            try:
                # Читаем данные
                with open(old_file_path, "r") as f:
                    history_data = json.load(f)

                # Конвертируем значения loss
                for key in ["loss", "val_loss"]:
                    if key in history_data:
                        history_data[key] = [value / 1500 * 100 for value in history_data[key]]

                # Сохраняем обновленный json
                with open(new_json_path, "w") as f:
                    json.dump(history_data, f, indent=2)

                # Перерисовываем график
                plot_history(history_data, new_png_path)

                print(f"Successfully saved: {new_json_path} and {new_png_path}")

            except Exception as e:
                print(f"Error processing {old_file_path}: {e}")

print("All history files processed!")
