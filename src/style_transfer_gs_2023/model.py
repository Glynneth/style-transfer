import tensorflow as tf
from tensorflow.python.keras import Model


def load_vgg_model(image_size: int) -> Model:
    tf.random.set_seed(1)
    vgg = tf.keras.applications.VGG19(
        include_top=False,
        input_shape=(image_size, image_size, 3),
        weights="pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",
    )
    vgg.trainable = False
    return vgg
