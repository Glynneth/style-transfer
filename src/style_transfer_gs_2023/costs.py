from typing import List

import tensorflow as tf

from style_transfer_gs_2023.hyperparameters import HYPERPARAMS, LayerAndWeight


def _content_cost(
    generated_img_output: tf.Tensor,
    content_img_output: tf.Tensor,
) -> tf.Tensor:
    """Compute the content cost of layer defined in hyperparams"""
    content_layer_idx = -1
    content_a = content_img_output[content_layer_idx]
    generated_a = generated_img_output[content_layer_idx]
    _, n_H, n_W, n_C = generated_a.get_shape().as_list()

    def unrolled(tensor: tf.Tensor) -> tf.Tensor:
        return tf.reshape(tensor, [-1, n_H * n_W, n_C])

    sum_squares = tf.reduce_sum(
        tf.square(tf.subtract(unrolled(content_a), unrolled(generated_a)))
    )
    content_cost = 1 / (4 * n_H * n_W * n_C) * sum_squares
    return content_cost


def _style_cost(
    generated_img_output: tf.Tensor,
    style_img_output: tf.Tensor,
    layers: List[LayerAndWeight],
) -> tf.Tensor:
    """Compute the style cost of layers and weights defined in hyperparams"""

    def layer_cost(layer_idx: int) -> tf.Tensor:
        generated_layer = generated_img_output[layer_idx]
        _, n_H, n_W, n_C = generated_layer.get_shape().as_list()

        def reshape(tensor: tf.Tensor) -> tf.Tensor:
            return tf.reshape(
                tf.transpose(tensor, perm=[0, 3, 1, 2]), [n_C, n_H * n_W]
            )

        style = _gram_matrix(reshape(style_img_output[layer_idx]))
        generated = _gram_matrix(reshape(generated_layer))
        factor = 1 / (4 * (n_H * n_W * n_C) ** 2)
        return factor * tf.reduce_sum(tf.square(tf.subtract(generated, style)))

    weighted_costs = [
        layer.weight * layer_cost(idx)
        for idx, layer in enumerate(layers)  # type: ignore
    ]

    return tf.add_n(weighted_costs)


def _gram_matrix(input: tf.Tensor) -> tf.Tensor:
    return tf.linalg.matmul(input, tf.transpose(input))


def cost(
    generated_img_output: tf.Tensor,
    content_img_output: tf.Tensor,
    style_img_output: tf.Tensor,
) -> float:
    """
    Compute the total cost of the generated image
    using content and style weights defined in hyperparams
    """
    content_cost = HYPERPARAMS["alpha"] * _content_cost(
        generated_img_output,
        content_img_output,
    )
    style_cost = HYPERPARAMS["beta"] * _style_cost(
        generated_img_output,
        style_img_output,
        HYPERPARAMS["style_cost_layers"],  # type: ignore
    )
    return content_cost + style_cost
