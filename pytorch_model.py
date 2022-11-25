import os
from typing import Any

import torch

from . import models


def pytorch_model(model_name: str = None, checkpoint: str = None, **kwargs: Any):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name is None:
        if checkpoint is None:
            raise ValueError("Either model_name or checkpoint must be specified.")
        model_name = os.path.basename(checkpoint).split(".")[0]
    try:
        model_class = getattr(
            models, model_name.lower()
        )  # parse model from model_zoo, need to add raise error if model not found
    except AttributeError:
        raise AttributeError(f"Model {model_name} not found.") from None

    model = model_class(**kwargs)  # return model class with or without pretrained weights
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict["state_dict"])
    model.to(device)

    return model


if __name__ == "__main__":
    model = pytorch_model("ResNet18")
