import math
from time import time
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
    timestamp,
    to_image,
)


def style_transfer() -> None:
    imgs_path = ROOT_PATH / "data" / "resized"
    content_img = tf.constant(load_image(imgs_path / "content_resized.jpg"))
    style_img = tf.constant(
        load_image(imgs_path / "portrait_self_1_resized.jpg")
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=HYPERPARAMS["learning_rate"], beta_1=0.99, epsilon=1e-1)

    @tf.function()
    def train_step(generated_image: tf.Variable) -> tf.Tensor:
        with tf.GradientTape() as tape:
            generated_output = output_layers(generated_image)
            cost = total_cost(generated_output, content_out, style_out)
            cost += HYPERPARAMS["total_variation_weight"]*tf.image.total_variation(generated_image)
        grad = tape.gradient(cost, generated_image)
        optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(clip(generated_image))
        return cost

    epochs = HYPERPARAMS["epochs"]
    steps_per_epoch = HYPERPARAMS["steps_per_epoch"]
    start = time()
    for e in range(epochs):
        for i in range(steps_per_epoch):
            train_step(generated_img)
        print(f"Step {e*steps_per_epoch} out of {epochs*steps_per_epoch}. {math.floor(time()-start)}s")
        image = to_image(generated_img)
        image.save(
            guarenteed_directory(ROOT_PATH / "output")
            / timestamp(f"self_portrait_steps_{e*steps_per_epoch}.jpg", start)
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
