import json
import os
from datetime import datetime

import numpy as np
from keras.src.callbacks import ModelCheckpoint
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt

from utils import load_dataset__raw, generate_model__raw

NOW = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

# Настраиваемые константы
MODEL_NUMBER = 2
MODEL_TYPE = "raw"
LOSS = "MAE"
EPOCHS = 250
BATCH_SIZE = 32

DATASET_DIR = f"datasets/{MODEL_NUMBER}/{MODEL_TYPE}_data"

BASE_FILE_TEMPLATE = f"{NOW}__loss={LOSS}__batch_size={BATCH_SIZE}__epochs={EPOCHS}"

HISTORY_FILE = f"results/history/{MODEL_NUMBER}/{MODEL_TYPE}/{BASE_FILE_TEMPLATE}.json"
HISTORY_IMAGE_FILE = f"results/history/{MODEL_NUMBER}/{MODEL_TYPE}/{BASE_FILE_TEMPLATE}.png"
WEIGHT_FILE = f"results/weights/{MODEL_NUMBER}/{MODEL_TYPE}/{BASE_FILE_TEMPLATE}__epoch={{epoch:04d}}__mse={{mse:.4f}}__mae={{mae:.4f}}.keras"


def create_dirs() -> None:
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(HISTORY_IMAGE_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(WEIGHT_FILE), exist_ok=True)


def main() -> None:
    create_dirs()

    X, Y = load_dataset__raw(DATASET_DIR)
    # X = expand_arrays_to_length(X, length=X_SHAPE_RAW, fill_value=-1)  # подгоняем всё под один размер, заполняя -1
    X = np.array(X)
    Y = np.array(Y)

    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]

    model = generate_model__raw()
    model.compile(optimizer=Adam(learning_rate=0.001), loss=LOSS, metrics=["mse", "mae"])
    model.summary()

    history = model.fit(
        X_train,
        Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[
            ModelCheckpoint(filepath=WEIGHT_FILE, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
        ],
    )

    model.evaluate(X_test, Y_test)

    # Записываем результаты history в файл
    with open(HISTORY_FILE, "w") as f:
        json.dump(history.history, f, indent=4)

    # отсекаем первую эпоху, т.к. там очень большие ошибки
    mse_history = history.history["mse"][1:]
    mae_history = history.history["mae"][1:]
    title = f""
    plt.plot(mse_history, "o-", label="mse")
    plt.plot(mae_history, "o-", label="mae")
    plt.title(title)
    plt.xlabel("Эпоха")
    plt.ylabel("Потеря")
    plt.legend()
    plt.savefig(HISTORY_IMAGE_FILE)
    plt.show()


if __name__ == "__main__":
    main()
