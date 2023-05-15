import tensorflow as tf
from tensorflow.python.keras import Model

from style_transfer_gs_2023.hyperparameters import HYPERPARAMS


def content_cost(
    generated_output: tf.Tensor, content_output: tf.Tensor
) -> float:
    return 0


def style_cost(generated_output: tf.Tensor, style_output: tf.Tensor) -> float:
    return 0


def cost(
    generated_output: tf.Tensor,
    content_output: tf.Tensor,
    style_output: tf.Tensor,
) -> float:
    total_cost = HYPERPARAMS["alpha"] * content_cost(
        generated_output, content_output
    ) + HYPERPARAMS["beta"] * style_cost(generated_output, style_output)
    return total_cost
