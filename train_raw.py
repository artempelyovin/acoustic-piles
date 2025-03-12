from datetime import datetime

import numpy as np
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt

from utils import load_dataset__raw, generate_model__raw, expand_arrays_to_length, X_SHAPE_RAW

now = datetime.now()

TYPE = "1_point"
LOSS = "mae"
BATCH_SIZE = 32
DATASET_DIR = "datasets/raw_data"
HISTORY_IMGS_DIR = "history/"
WEIGHTS_PATH = f"weights/raw/{TYPE}__loss={LOSS}__batch_size={BATCH_SIZE}__epoch={{epoch:04d}}__mse={{mse:.2f}}__mae={{mae:.2f}}__{now:%Y-%m-%dT%H:%M:%S}.h5"


def main() -> None:
    X, Y = load_dataset__raw(DATASET_DIR)
    X = expand_arrays_to_length(X, length=X_SHAPE_RAW, fill_value=-1)  # подгоняем всё под один размер, заполняя -1
    Y = np.array([len(y) for y in Y])  # нам важно только кол-во спец. нулевых точек, а не сами точки!

    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]

    model = generate_model__raw()
    model.compile(optimizer=Adam(learning_rate=0.001), loss=LOSS, metrics=["mse", "mae"])
    model.summary()

    history = model.fit(
        X_train,
        Y_train,
        epochs=250,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[
            ModelCheckpoint(filepath=WEIGHTS_PATH, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
            # EarlyStopping(monitor="val_loss", mode="min", min_delta=0.01, patience=10),
        ],
    )

    model.evaluate(X_test, Y_test)

    # отсекаем первую эпоху, т.к. там очень большие ошибки
    mse_history = history.history["mse"][1:]
    mae_history = history.history["mae"][1:]
    title = f"{TYPE}__loss={LOSS}__batch_size={BATCH_SIZE}__{now:%Y-%m-%dT%H:%M:%S}"
    plt.plot(mse_history, "o-", label="mse")
    plt.plot(mae_history, "o-", label="mae")
    plt.title(title)
    plt.xlabel("Эпоха")
    plt.ylabel("Потеря")
    plt.legend()
    plt.savefig(f"{HISTORY_IMGS_DIR}/{title}.png")
    plt.show()


if __name__ == "__main__":
    main()
