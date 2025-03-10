import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from utils import generate_model__gph

IMG_HEIGHT = 369
IMG_WIDTH = 496
CHANNELS = 1  # у нас чёрно-белая картинка

Y_SHAPE = 20


def main() -> None:
    model = generate_model__gph()
    model.summary()


if __name__ == "__main__":
    main()


# Если используется sample weighting с маской, можно написать кастомную loss-функцию,
# которая будет учитывать маску для каждой координаты.
#
# Пример кастомной функции потерь, которая игнорирует те координаты, где mask=0:
def masked_mse(y_true, y_pred):
    # Предположим, что y_true содержит реальные координаты, а в недостающих местах стоит -1
    # и маска передаётся отдельно не как часть y_true.
    # Альтернативно, можно объединить y_true и mask в один тензор и разделять их.
    # Здесь для простоты будем считать, что mask доступна через дополнительный вход (но Keras не поддерживает несколько
    # целевых переменных для loss без дополнительной настройки).
    # Поэтому рассмотрим вариант: заменим недостающие значения на y_pred (так ошибки там будут = 0)
    # Если значение отсутствует, мы ожидаем в y_true значение -1
    mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
    # На местах отсутствующих значений, заменяем на y_pred:
    y_true_masked = tf.where(tf.equal(mask, 1.0), y_true, y_pred)
    mse = tf.square(y_pred - y_true_masked)
    mse = tf.reduce_sum(mse) / tf.reduce_sum(mask + 1e-6)
    return mse


# Вместо кастомного masked_mse можно подготовить target так, чтобы недостающие координаты были 0,
# а sample_weight = маска и передавать sample_weight при вызове fit.
# Для простоты, сначала обучим модель используя стандартную MSE и будем учитывать, что данные корректно подготовлены.

model.compile(optimizer=Adam(learning_rate=1e-3), loss=MeanSquaredError())

# Если хотите обучать с sample_weight:
# model.fit(images, coords, batch_size=32, epochs=20, sample_weight=mask)

# Обучение:
history = model.fit(images, coords, batch_size=32, epochs=20, validation_split=0.2)

# После обучения модель будет принимать изображение и выдавать 20 чисел (координаты).
# Если требуется получить только действительные координаты, то необходимо применить обратную обработку:
# например, отфильтровать те, которые меньше порогового значения или использовать дополнительную модель/классификатор
# для определения количества значащих точек.
