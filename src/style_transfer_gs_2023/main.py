from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
from style_transfer_gs_2023 import ROOT_PATH
from style_transfer_gs_2023.hyperparameters import HYPERPARAMS


def style_transfer() -> None:
    imgs_path = ROOT_PATH / "data" / "resized"
    content_image = tf.constant(load_image(imgs_path / "content_resized.jpg"))
    style_image = tf.constant(load_image(imgs_path / "portrait_arden_resized.jpg"))
    generated_image = initialise_generated_img(content_image, HYPERPARAMS["initial_noise"])


def load_image(img_path: Path) -> np.array:
    img = np.array(Image.open(img_path))
    return np.reshape(img, ((1,) + img.shape))

def initialise_generated_img(content: tf.constant, noise: float) -> np.array:
    content = tf.Variable(
        tf.image.convert_image_dtype(content, tf.float32)
    )
    noise = tf.random.uniform(tf.shape(content), -noise, noise)
    generated_image = tf.clip_by_value(
        tf.add(content, noise), clip_value_min=0.0, clip_value_max=1.0
    )
    return generated_image

if __name__ == "__main__":
    style_transfer()
