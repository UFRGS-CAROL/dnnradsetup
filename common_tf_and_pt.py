import argparse
import enum
import logging
import time
from typing import Tuple, List
import random
from PIL import Image
import tensorflow as tf
import dnn_log_helper as dnn_log_helper

# Classification
INCEPTION_V3 = "InceptionV3"
RESNET_50 = "ResNet-50"
EFFICIENT_NET_B0 = "EfficientNet-B0"
EFFICIENT_NET_B3 = "EfficientNet-B3"

# Detection
RETINA_NET_RESNET_FPN50 = "RetinaNetResNet-50FPN"
FASTER_RCNN_RESNET_FPN50 = "FasterR-CNNResNet-50FPN"
SSD_MOBILENET_V2 = "SSDMobileNetV2"
EFFICIENT_DET_LITE3 = "EfficientDet-Lite3"

OPTIMAL_INPUT_SIZE = {
    INCEPTION_V3: (299, 299),
    RESNET_50: (224, 224),
    EFFICIENT_NET_B0: (224, 224),
    EFFICIENT_NET_B3: (300, 300),
    # Detection does not need resizing (this values are for tflite)
    RETINA_NET_RESNET_FPN50: None,
    FASTER_RCNN_RESNET_FPN50: None,
    SSD_MOBILENET_V2: (300, 300),
    EFFICIENT_DET_LITE3: (512, 512),

}

ALL_DNNS = list(OPTIMAL_INPUT_SIZE.keys())

BATCH_SIZE_GPU = 5

CLASSIFICATION_ABS_THRESHOLD = 1e-6
DETECTION_BOXES_ABS_THRESHOLD = 1e-6
DETECTION_SCORES_ABS_THRESHOLD = 1e-6

MAXIMUM_ERRORS_PER_ITERATION = 4096


def parse_args():
    """ Parse the args and return an args namespace and the tostring from the args    """
    parser = argparse.ArgumentParser(description='PyTorch DNN radiation setup')
    parser.add_argument('--model', default=ALL_DNNS[0], help=f'Network name. It can be ' + ', '.join(ALL_DNNS))
    parser.add_argument('--imglist', help='Path to the list of images as input')
    parser.add_argument('--precision', default="fp32", help="Precision of the network, can be fp16 and fp32")
    parser.add_argument('--iterations', default=int(1e12), help="Iterations to run forever", type=int)
    parser.add_argument('--batchsize', default=1, help="Batches to process in parallel", type=int)

    parser.add_argument('--generate', default=False, action="store_true", help="Set this flag to generate the gold")
    parser.add_argument('--disableconsolelog', default=False, action="store_true",
                        help="Set this flag disable console logging")

    parser.add_argument('--goldpath', help="Path to the gold file")
    parser.add_argument('--grtruthcsv', help="Path ground truth verification at generate process.", default=None,
                        type=str)
    parser.add_argument('--tflite', default=False, action="store_true", help="Is it necessary to use Tensorflow lite.")
    args = parser.parse_args()
    # Check if the model is correct
    if args.model not in ALL_DNNS:
        parser.print_help()
        raise ValueError("Not the correct model")

    # Check if it is only to generate the gold values
    if args.generate:
        args.iterations = 1
    else:
        args.grtruthcsv = None
    args_text = " ".join([f"{k}={v}" for k, v in vars(args).items()])
    return args, args_text


class Timer:
    time_measure = 0

    def tic(self):
        self.time_measure = time.time()

    def toc(self):
        self.time_measure = time.time() - self.time_measure

    @property
    def diff_time(self):
        return self.time_measure

    def __str__(self):
        return f"{self.time_measure:.4f}s"

    def __repr__(self):
        return str(self)


class DNNType(enum.Enum):
    """Small Enum to define which type is each DNN
    When a DNN is DETECTION_SEGMENTATION or DETECTION_KEYPOINT it is capable of doing more than 1 task
    """
    CLASSIFICATION, DETECTION, SEGMENTATION, KEYPOINT, DETECTION_SEGMENTATION, DETECTION_KEYPOINT = range(6)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self)


def load_image_list(image_list_path: str) -> Tuple[List[Image.Image], List[str]]:
    with open(image_list_path, 'r') as f:
        image_files = f.read().splitlines()
    images = list(map(Image.open, image_files))
    return images, image_files

