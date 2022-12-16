from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.transforms import functional as f


def onnx_preprocessing(image: np.ndarray) -> Dict[str, np.array]:
    image = (image - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return {"image": image}


def ssdlite_preprocessing(img: np.ndarray) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
    """
    Preproc image for model input

    Parameters
    ----------
    img : np.ndarray
        input image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)
    width, height = image.size

    image = ImageOps.pad(image, (max(width, height), max(width, height)))
    padded_width, padded_height = image.size
    image = image.resize((320, 320))

    img_tensor = f.pil_to_tensor(image)
    img_tensor = f.convert_image_dtype(img_tensor)
    img_tensor = img_tensor[None, :, :, :]
    return {"image": img_tensor, "size": (width, height), "padded_size": (padded_width, padded_height)}


def ssdlite_postprocessing(size: Tuple[int, int], padded_size: Tuple[int, int], box: Tensor) -> np.array:
    """
    Postprocess box

    Parameters
    ----------
    size : Tuple[int, int]
        Size of image
    padded_size : Tuple[int, int]
        Size of padding
    box : Tensor
        Coordinate of detected box
    """
    width, height = size
    padded_width, padded_height = padded_size
    scale = max(width, height) / 320
    padding_w = abs(padded_width - width) // (2 * scale)
    padding_h = abs(padded_height - height) // (2 * scale)
    x1 = int((box[0] - padding_w) * scale)
    y1 = int((box[1] - padding_h) * scale)
    x2 = int((box[2] - padding_w) * scale)
    y2 = int((box[3] - padding_h) * scale)

    return np.array([x1, y1, x2, y2])


get_postprocess = {"SSD": ssdlite_postprocessing}

get_transform = {"ONNXModel": onnx_preprocessing, "SSD": ssdlite_preprocessing}
