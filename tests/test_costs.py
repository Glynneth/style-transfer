import tensorflow as tf
from pytest import approx

from style_transfer_gs_2023.costs import _content_cost, _style_cost
from style_transfer_gs_2023.hyperparameters import LayerAndWeight


def test_content_cost():
    tf.random.set_seed(2)
    content_activations = tf.random.normal([1, 1, 4, 4, 3], mean=1, stddev=4)
    generated_activations = tf.random.normal([1, 1, 4, 4, 3], mean=1, stddev=4)
    J_content = _content_cost(
        content_activations, generated_activations
    )
    J_content_zero = _content_cost(
        content_activations, content_activations
    )
    assert float(J_content) == approx(7.33876)
    assert float(J_content_zero) == approx(0.0)


def test_style_cost():
    tf.random.set_seed(2)
    style_activations = tf.random.normal([10, 1, 4, 4, 3], mean=1, stddev=4)
    generated_activations = tf.random.normal([3, 1, 4, 4, 3], mean=1, stddev=4)
    layers = [LayerAndWeight(idx=1, weight=1)]
    J_style = _style_cost(style_activations, generated_activations, layers)
    J_style_zero = _style_cost(style_activations, style_activations, layers)
    assert float(J_style) == approx(24.63678)
    assert float(J_style_zero) == approx(0.0)
