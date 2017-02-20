import numpy as np


def normalize_image(img):
    """
    Normalize image between 0 and 255 and cast to uint8
    (useful for visualization)
    """
    img = np.float32(img)

    img = img / img.max() * 255

    return np.uint8(img)