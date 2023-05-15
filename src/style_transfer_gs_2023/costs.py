import tensorflow as tf

from style_transfer_gs_2023.hyperparameters import HYPERPARAMS


def _content_cost(
    generated_output: tf.Tensor, content_output: tf.Tensor
) -> tf.Tensor:
    """Compute the content cost using the activations from the final layer of the model"""
    content, generated = content_output[-1], generated_output[-1]
    _, n_H, n_W, n_C = generated.get_shape().as_list()

    def unrolled(tensor: tf.Tensor) -> tf.Tensor:
        return tf.reshape(tensor, [-1, n_H * n_W, n_C])

    sum_squares = tf.reduce_sum(
        tf.square(tf.subtract(unrolled(content), unrolled(generated)))
    )
    content_cost = 1 / (4 * n_H * n_W * n_C) * sum_squares
    return content_cost


def _style_cost(generated_output: tf.Tensor, style_output: tf.Tensor) -> float:
    return 0


def cost(
    generated_output: tf.Tensor,
    content_output: tf.Tensor,
    style_output: tf.Tensor,
) -> float:
    total_cost = HYPERPARAMS["alpha"] * _content_cost(
        generated_output, content_output
    ) + HYPERPARAMS["beta"] * _style_cost(generated_output, style_output)
    return total_cost
