import numpy as np


def onnx_preprocessing(image):
    image = (image - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image


get_transform = {
    "ONNXModel": onnx_preprocessing,
}
