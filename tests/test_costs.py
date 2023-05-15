import tensorflow as tf
from pytest import approx

from style_transfer_gs_2023.costs import _content_cost


def test_content_cost():
    tf.random.set_seed(2)
    content_activations = tf.random.normal([1, 1, 4, 4, 3], mean=1, stddev=4)
    generated_activations = tf.random.normal([1, 1, 4, 4, 3], mean=1, stddev=4)
    J_content = _content_cost(content_activations, generated_activations)
    J_content_0 = _content_cost(content_activations, content_activations)
    assert float(J_content_0) == approx(0.0)
    assert float(J_content) == approx(7.33876)
