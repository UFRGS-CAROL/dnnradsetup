import enum

from torchvision import models, transforms


class DNNType(enum.Enum):
    """Small Enum to define which type is each DNN
    When a DNN is DETECTION_SEGMENTATION or DETECTION_KEYPOINT it is capable of doing more than 1 task
    """
    CLASSIFICATION, DETECTION, SEGMENTATION, KEYPOINT, DETECTION_SEGMENTATION, DETECTION_KEYPOINT = range(6)

    def __str__(self): return str(self.name)

    def __repr__(self): return str(self)


DNN_MODELS = {
    # Check the spreadsheet to more information
    # Inception V3
    # Config: https://pytorch.org/hub/pytorch_vision_inception_v3/
    "InceptionV3": {
        "model": models.inception_v3,
        "type": DNNType.CLASSIFICATION,
        # Channels, height, width
        "input_size": [3, 299, 299],
        "transform": transforms.Compose([
            transforms.Resize(299), transforms.CenterCrop(299), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    },

    # Configurations for the nets are take from
    # github.com/pytorch/vision/tree/1de53bef2d04c34544655b0c1b6c55a25e4739fe/references/classification
    # https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/
    "ResNet-50": {
        "model": models.resnet50,
        "type": DNNType.CLASSIFICATION,
        # Channels, height, width
        "input_size": [3, 256, 256],
        "transform": transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    },

    # https://github.com/lukemelas/EfficientNet-PyTorch/blob/
    # 1039e009545d9329ea026c9f7541341439712b96/efficientnet_pytorch/utils.py#L562-L564
    "EfficientNet-B7": {
        "model": models.efficientnet_b7,
        "type": DNNType.CLASSIFICATION,
        # Channels, height, width
        "input_size": [3, 600, 600],
        "transform": transforms.Compose([
            transforms.Resize(600, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(600), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    },

    # Object detection, segmentation, and keypoint
    "RetinaNetResNet-50FPN": {
        "model": models.detection.ssdlite320_mobilenet_v3_large,
        "type": DNNType.DETECTION
    },
    "SSDMobileNetV2": {
        "model": models.detection.ssd300_vgg16,
        "type": DNNType.DETECTION
    },

    "EfficientDet-Lite3": {
        "model": models.detection.ssd300_vgg16,
        "type": DNNType.DETECTION
    },
    "FasterR-CNNResNet-50FPN": {
        "model": models.detection.fasterrcnn_resnet50_fpn,
        "type": DNNType.DETECTION
    },

    ####################################################################################################################
    # For the future
    "MaskR-CNNResNet-50FPN": {
        "model": models.detection.maskrcnn_resnet50_fpn,
        "type": DNNType.DETECTION_SEGMENTATION
    },
    "KeypointR-CNNResNet-50FPN": {
        "model": models.detection.keypointrcnn_resnet50_fpn,
        "type": DNNType.DETECTION_KEYPOINT
    },
}
