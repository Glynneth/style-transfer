from typing import List

import tensorflow as tf
from tensorflow.python.keras import Model

from style_transfer_gs_2023.hyperparameters import LayerAndWeight


def load_vgg_model(img_size: int) -> Model:
    tf.random.set_seed(1)
    vgg = tf.keras.applications.VGG19(
        include_top=False,
        input_shape=(img_size, img_size, 3),
        weights="imagenet",
    )
    vgg.trainable = False
    return vgg


def model_outputs(model: Model, layers: List[LayerAndWeight]) -> Model:
    outputs = [model.get_layer(layer.name).output for layer in layers]
    model = tf.keras.Model([model.input], outputs)
    return model
