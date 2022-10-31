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
    * FatserRCNN
    * SSDLite

### Project

```
.
├── detectors/ # source code for detection models
│   ├── ssd.py # Single Shot Detector
│   ├── frcnn.py # Faster Rcnn
├── classifiers/ # source code for classification models
│   ├── mobilenet.py # MobileNetV3
│   ├── resnet.py # ResNet
│   ├── vision_transformer.py # vision transformer
├── util/ # useful utils
```



