import argparse
import os
import uuid
from datetime import datetime

import numpy as np
from keras.src.callbacks import ModelCheckpoint
from keras.src.optimizers import Adam

from utils import load_dataset__raw, generate_model__raw, HistoryToFile, PlotHistory, X_SHAPE_RAW


def train(model_number: int, learning_rate: float, loss: str, epochs: int, batch_size: int, dataset_size: int) -> None:
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    uuid_ = str(uuid.uuid4())[:4]

    base_file_template = f"{uuid_}__{now}__dataset_size={dataset_size}__loss={loss}__lr={learning_rate}__batch_size={batch_size}__epochs={epochs}"
    history_file = f"results/history/{model_number}/conv1d/{base_file_template}.json"
    history_image_file = f"results/history/{model_number}/conv1d/{base_file_template}.png"
    weight_file = f"results/weights/{model_number}/conv1d/{base_file_template}__epoch={{epoch:04d}}__val_loss={{val_loss:.6f}}.keras"
    dataset_dir = f"datasets/{model_number}/raw_data"

    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    os.makedirs(os.path.dirname(history_image_file), exist_ok=True)
    os.makedirs(os.path.dirname(weight_file), exist_ok=True)

    X, Y = load_dataset__raw(dataset_dir)
    if len(X) < dataset_size:
        raise ValueError(f"Размер датасета ({len(X)} шт.) меньше желаемого ({dataset_dir} шт.)")

    X = np.array(X[:dataset_size])
    Y = np.array(Y[:dataset_size])

    # нормализуем все координаты в диапазон [0;1]
    X[:, 0::2] = (X[:, 0::2] - 0) / (X_SHAPE_RAW / 2 - 0)
    Y[:, ] = (Y[:, ] - 0) / (X_SHAPE_RAW / 2 - 0)

    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]

    model = generate_model__raw()
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss)
    model.summary()

    model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[
            ModelCheckpoint(filepath=weight_file, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
            HistoryToFile(history_file=history_file),
            PlotHistory(image_file=history_image_file),
        ],
    )

    model.evaluate(X_test, Y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение нейронной сети для выбранной модели на сырых данных")
    parser.add_argument(
        "--model-number", type=int, choices=[10, 20, 30, 40, 50, 60], required=True, help="Номер модели."
    )
    parser.add_argument("--learning-rate", type=int, default=0.001, help="...")
    parser.add_argument("--loss", type=str, default="mae", help="...")
    parser.add_argument("--epochs", type=int, default=250, help="...")
    parser.add_argument("--batch-size", type=int, default=32, help="...")
    parser.add_argument("--dataset-size", type=int, default=5000, help="...")

    args = parser.parse_args()
    train(
        model_number=args.model_number,
        learning_rate=args.learning_rate,
        loss=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_size=args.dataset_size,
    )
