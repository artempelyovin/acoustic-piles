from keras.src.callbacks import ModelCheckpoint
from keras.src.optimizers import Adam

from utils import load_dataset__raw, generate_model__raw

DATASET_PATH = "datasets/raw_data"
WEIGHTS_PATH = "weights/raw/model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5"


def main() -> None:
    X, Y = load_dataset__raw(DATASET_PATH)
    split_index = int(0.8 * len(X))
    X_train, _ = X[:split_index], X[split_index:]
    Y_train, _ = Y[:split_index], Y[split_index:]

    model = generate_model__raw()
    model.compile(optimizer=Adam(learning_rate=0.0005), loss="mean_squared_error", metrics=["mae"])
    model.summary()

    checkpoint_callback = ModelCheckpoint(
        filepath=WEIGHTS_PATH, monitor="val_loss", mode="min", save_best_only=True, verbose=1
    )
    model.fit(X_train, Y_train, epochs=500, batch_size=128, validation_split=0.2, callbacks=[checkpoint_callback])


if __name__ == "__main__":
    main()
