from keras import Sequential
from keras import models
from keras.src.utils import plot_model


def draw_model(weights_path: str, output: str) -> None:
    model: Sequential = models.load_model(weights_path)
    plot_model(model, to_file=output, show_shapes=True, show_layer_names=True, show_layer_activations=True)


if __name__ == "__main__":
    draw_model(
        weights_path="results/weights/1/conv1d/a829__2025-05-03T12:09:07__dataset_size=5000__loss=mae__start_lr=0.001__reduce_lr=False__batch_size=32__epochs=250__epoch=0233__val_loss=0.000991.keras",
        output="examples/conv1d_model.png",
    )
