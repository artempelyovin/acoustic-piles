import os
import re

from tensorflow import keras
from pathlib import Path

from train import absolute_percentage_error

# Пути к директориям
old_weights_dir = Path("results/weights")
new_weights_dir = Path("results/weights")

# Регулярное выражение для извлечения val_loss
val_loss_pattern = re.compile(r"val_loss=([\d.]+)\.keras$")

# Создаем новую директорию, если её нет
new_weights_dir.mkdir(parents=True, exist_ok=True)

# Проходим по всем файлам в старой директории
for root, dirs, files in os.walk(old_weights_dir):
    for file in files:
        if file.endswith(".keras"):
            # Полный путь к старому файлу
            old_file_path = Path(root) / file

            # Извлекаем val_loss из имени файла
            match = val_loss_pattern.search(file)
            if not match:
                print(f"Warning: Could not extract val_loss from {file}. Skipping...")
                continue

            old_val_loss = float(match.group(1))
            new_val_loss = old_val_loss / 1500 * 100  # Пересчитываем в проценты

            # Формируем новое имя файла с обновлённым val_loss
            new_file_name = val_loss_pattern.sub(f"val_loss={new_val_loss:.6f}.keras", file)
            # Относительный путь для новой директории

            # Относительный путь для новой директории
            rel_path = old_file_path.relative_to(old_weights_dir)
            new_file_path = new_weights_dir / rel_path.parent / new_file_name

            # Создаем поддиректории, если их нет
            new_file_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Processing: {old_file_path} -> {new_file_path}")

            # Загружаем модель без компиляции
            try:
                model = keras.models.load_model(old_file_path, compile=False)

                # Перекомпилируем с новой функцией потерь
                model.compile(optimizer=keras.optimizers.Adam(), loss=absolute_percentage_error)

                # Сохраняем модель с новой функцией потерь
                model.save(new_file_path)
                print(f"Successfully saved: {new_file_path}")

            except Exception as e:
                print(f"Error processing {old_file_path}: {e}")

print("All models processed!")
