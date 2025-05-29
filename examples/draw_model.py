from keras import Sequential
from keras.src.utils import plot_model

from utils import generate_model__raw


def draw_model(output: str) -> None:
    model: Sequential = generate_model__raw()
    plot_model(model, to_file=output, show_shapes=True, show_layer_names=True, show_layer_activations=True)


if __name__ == "__main__":
    draw_model(output="examples/conv1d_model.png")
