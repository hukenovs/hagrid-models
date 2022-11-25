import os
from functools import singledispatch
from typing import Any

import requests
from tqdm.auto import tqdm

from .model_zoo import models
from .onnx_model import ONNXModel
from .pytorch_model import pytorch_model

__all__ = ["get_model"]


class URL(str):
    ...


class PATH(str):
    ...


class Name(str):
    ...


@singledispatch
def _get_model(model, **kwargs: Any):
    return model


@_get_model.register
def _(model: URL, progress: bool, **kwargs: Any):
    model_storage = os.path.join(os.path.dirname(__file__), "model_storage")
    if not os.path.exists(model_storage):
        os.mkdir(model_storage)
    response = requests.get(model, stream=True)
    url = response.url
    file_size = int(response.headers.get("content-length", 0))
    filename = os.path.join(model_storage, url.split("/")[-1])
    if os.path.exists(filename):
        return _get_model(PATH(filename))
    progress_bar = tqdm(total=file_size, unit="iB", unit_scale=True, disable=not progress, desc=url.split("/")[-1])
    block_size = 1024
    with open(filename, "wb") as handle:
        for data in response.iter_content(chunk_size=block_size):
            progress_bar.update(len(data))
            handle.write(data)
    progress_bar.close()

    if file_size != 0 and progress_bar.n != file_size:
        raise ValueError("Error, something went wrong")

    return _get_model(PATH(filename), **kwargs)


@_get_model.register
def _(model: PATH, **kwargs: Any):
    if not os.path.exists(model):
        raise ValueError(f"Checkpoint {model} not found.")
    if ".onnx" in model:
        return ONNXModel(model)
    elif ".pth" in model or ".pt" in model:
        return pytorch_model(checkpoint=model, **kwargs)
    else:
        raise ValueError(f"Checkpoint {model} not supported.")


@_get_model.register
def _(model: Name, progress: bool, **kwargs: Any):
    model = models[model]
    model = _get_model(URL(model), progress, **kwargs)

    return model


def get_model(model: str, progress: bool = True, pretrained: bool = True, device: str = None, **kwargs: Any):
    if pretrained:
        if "https" in model and model in models.values():
            model = _get_model(URL(model), progress, **kwargs)
        elif ".onnx" in model or ".pth" in model or ".pt" in model:
            model = _get_model(PATH(model), kwargs, **kwargs)
        elif model in models.keys():
            model = _get_model(Name(model), progress, **kwargs)
        else:
            raise ValueError(f"Model {model} not found.")
    else:
        if model in models.keys():
            return pytorch_model(model_name=model, **kwargs)
        else:
            raise ValueError(f"Model {model} not found.")

    return model


if __name__ == "__main__":
    out = get_model("SwinT_FasterRCNN")
    print(type(out).__name__)
