import os
import models
import torch


class PytorchModel:
    def __init__(self, model_name: str = None, checkpoint: str = None, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name is None:
            if checkpoint is None:
                raise ValueError("Either model_name or checkpoint must be specified.")
            model_name = os.path.basename(checkpoint).split('.')[0]
        self.model_name = model_name.lower()
        try:
            model_class = getattr(models,
                                  self.model_name)  # parse model from model_zoo, need to add raise error if model not found
        except AttributeError:
            raise AttributeError(f"Model {model_name} not found.") from None

        self.model = model_class()  # return model class with or without pretrained weights
        if checkpoint is not None:
            self.load_state_dict(checkpoint)
        self.model.to(self.device)

    def load_state_dict(self, checkpoint: str, *args, **kwargs):
        #TODO: ты забыл в код моделей добавить кастомные слои для хагрида, почти каждая модель валится с
        # ошибкой при загрузке весов
        state_dict = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state_dict["state_dict"])

    def __call__(self, *args, **kwargs):  # need to add forward code with preprocessing
        ...

    def __repr__(self):
        return self.model.__repr__()


if __name__ == "__main__":
    model = PytorchModel("ResNet18")
    print(model.model_name)
