import argparse
import os
import uuid
from datetime import datetime

import numpy as np
from keras.src.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.src.optimizers import Adam

from utils import load_dataset__gph, generate_model__gph, HistoryToFile, PlotHistory, normalize, X_MIN, X_MAX, \
    GphLazyDataGenerator


def train(
    model_number: int,
    learning_rate: float,
    reduce_learning_rate: float,
    loss: str,
    epochs: int,
    batch_size: int,
    dataset_size: int,
) -> None:
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    uuid_ = str(uuid.uuid4())[:4]

    # шаблоны путей
    base_file_template = f"{uuid_}__{now}__dataset_size={dataset_size}__loss={loss}__start_lr={learning_rate}__reduce_lr={reduce_learning_rate}__batch_size={batch_size}__epochs={epochs}"
    history_file = f"results/history/{model_number}/conv2d/{base_file_template}.json"
    history_image_file = f"results/history/{model_number}/conv2d/{base_file_template}.png"
    weight_file = f"results/weights/{model_number}/conv2d/{base_file_template}__epoch={{epoch:04d}}__val_loss={{val_loss:.6f}}.keras"
    model_dir = f"{str(model_number)[0]}_"
    dataset_dir = f"datasets/{model_dir}/fig_data"

    # создаём директории (если ещё не созданы)
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    os.makedirs(os.path.dirname(history_image_file), exist_ok=True)
    os.makedirs(os.path.dirname(weight_file), exist_ok=True)

    train_gen = GphLazyDataGenerator(dirpath=dataset_dir, batch_size=batch_size, dataset_size=dataset_size)

    # подготовка модели
    model = generate_model__gph()
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss)
    model.summary()

    # обучение
    callbacks = [
        ModelCheckpoint(filepath=weight_file, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
        HistoryToFile(history_file=history_file),
        PlotHistory(image_file=history_image_file),
    ]

    if reduce_learning_rate:
        callbacks.append(
            ReduceLROnPlateau(monitor="val_loss", mode="min", patience=40, factor=0.5, min_lr=0.0005, verbose=1)
        )

    model.fit(train_gen, epochs=epochs, batch_size=batch_size, callbacks=callbacks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение нейронной сети для выбранной модели на изображениях")
    parser.add_argument("--model-number", type=int, choices=[11, 21, 31, 41], required=True, help="Номер модели.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate для оптимизатора Adam")
    parser.add_argument(
        "--reduce-learning-rate",
        action="store_true",
        help="Уменьшать learning rate для оптимизатора Adam в процессе обучения?",
    )
    parser.add_argument("--loss", type=str, default="mae", choices=["mae", "mse"], help="Функция потерь")
    parser.add_argument("--epochs", type=int, default=250, help="Кол-во эпох в обучении")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча в обучении")
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=5000,
        help="Размер датасета для обучения "
        "(если загружаемый датасет большего размера, то он будет уменьшен до данного значения)",
    )

    args = parser.parse_args()
    train(
        model_number=args.model_number,
        learning_rate=args.learning_rate,
        reduce_learning_rate=args.reduce_learning_rate,
        loss=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_size=args.dataset_size,
    )
