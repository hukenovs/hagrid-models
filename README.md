# HaGRID Classification and Detection Vision models
[HaGRID main repo](https://github.com/hukenovs/hagrid)

## Overview
This repository contains the source code of models used in [HaGRID](https://github.com/hukenovs/hagrid) as pretrained classification and detection CNNs for hand detection and classification tasks. Most of the source code is borrowed from [TorchVision](https://github.com/pytorch/vision) repo.

### List of models:
* Classification:
    * MobileNetV3
    * ResNet
    * Vision Transformer
* Detection:
    * SSDLite
    * SwinS_FasterRCNN
    * SwinT_FasterRCNN

List of all available pre-trained models can be found in model_zoo.py

### Project

```
.
├── models/
│   ├── classifiers/ # source code for classification models
│   │   ├── mobilenet.py # MobileNetV3
│   │   ├── resnet.py # ResNet
│   │   ├── vision_transformer.py # Vision Transformer
│   ├── detectors/ # source code for SSD detection model
│   │   ├── ssd.py # Single Shot Detector
│   │   ├── ssdlite.py # SSDLite scoring and regression heads
        ...
├── utils/ # useful utils
│   ├── onnx_utils.py # Utils for ONNX models
│   ├── torch_utils.py # Utils for pytorch models
├── get_model.py # Get model method
├── model_zoo.py # List of pre-trained availeble models
├── onnx_model.py # ONNX model wrapper
├── pytorch_model.py # Pytorch model wrapper
├── processing.py # Pre/post processing methods for images
```
