import math
import os
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image


def guarenteed_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_image(img_path: Path) -> Any:
    img = np.array(Image.open(img_path))
    return np.reshape(img, ((1,) + img.shape))


def clip(t: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(t, clip_value_min=0.0, clip_value_max=1.0)


def to_image(tensor: tf.Tensor) -> Image:
    """Convert tensor of shape (1, x, x, 3) to an image"""
    tensor = np.array(tensor * 255, dtype=np.uint8)
    return Image.fromarray(tensor[0])


def timestamp(string: str, start_time: float) -> str:
    """Timestamp a file name with seconds since the start time"""
    base, extension = os.path.splitext(string)
    timestamp = math.floor(time() - start_time)
    return f"{base}_{timestamp}s{extension}"