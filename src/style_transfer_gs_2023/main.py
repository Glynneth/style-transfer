from typing import List

import tensorflow as tf

from style_transfer_gs_2023 import ROOT_PATH
from style_transfer_gs_2023.costs import total_cost
from style_transfer_gs_2023.hyperparameters import HYPERPARAMS, LayerAndWeight
from style_transfer_gs_2023.model import load_vgg_model, model_outputs
from style_transfer_gs_2023.utils import (
    clip,
    guarenteed_directory,
    load_image,
    to_image,
)


def style_transfer() -> None:
    imgs_path = ROOT_PATH / "data" / "resized"
    content_img = tf.constant(load_image(imgs_path / "content_resized.jpg"))
    style_img = tf.constant(
        load_image(imgs_path / "portrait_arden_resized.jpg")
    )
    generated_img = initialise_generated_img(
        content_img, HYPERPARAMS["initial_noise"]  # type: ignore
    )
    model = load_vgg_model(img_size=200)
    layers: List[LayerAndWeight] = HYPERPARAMS["style_cost_layers"] + HYPERPARAMS["content_cost_layer"]  # type: ignore
    output_layers = model_outputs(model, layers)
    content_out = output_layers(
        tf.Variable(tf.image.convert_image_dtype(content_img, tf.float32))
    )
    style_out = output_layers(
        tf.Variable(tf.image.convert_image_dtype(style_img, tf.float32))
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    @tf.function()
    def train_step(generated_image: tf.Variable) -> tf.Tensor:
        with tf.GradientTape() as tape:
            generated_output = output_layers(generated_image)
            cost = total_cost(generated_output, content_out, style_out)
        grad = tape.gradient(cost, generated_image)
        optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(clip(generated_image))
        return cost

    epochs = 2
    for _ in range(epochs):
        train_step(generated_img)
    image = to_image(generated_img)
    image.save(
        guarenteed_directory(ROOT_PATH / "output")
        / f"output_epochs_{epochs}.jpg"
    )


def initialise_generated_img(
    content: tf.constant, noise: float
) -> tf.Variable:
    content = tf.Variable(tf.image.convert_image_dtype(content, tf.float32))
    noise = tf.random.uniform(tf.shape(content), -noise, noise)
    generated_image = tf.clip_by_value(
        tf.add(content, noise), clip_value_min=0.0, clip_value_max=1.0
    )
    return tf.Variable(generated_image)


if __name__ == "__main__":
    style_transfer()
