#!/usr/bin/python3

import os

from common_tf_and_pt import INCEPTION_V3, RESNET_50, EFFICIENT_NET_B0, EFFICIENT_NET_B3
from common_tf_and_pt import SSD_MOBILENET_V2, EFFICIENT_DET_LITE3, FASTER_RCNN_RESNET_FPN50

TENSORFLOW_MODELS = {
    SSD_MOBILENET_V2: "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2?tf-hub-format=compressed",
    EFFICIENT_DET_LITE3: "https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1?tf-hub-format=compressed",
    FASTER_RCNN_RESNET_FPN50:
        "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1?tf-hub-format=compressed"

}

CORAL_LINK = "https://raw.githubusercontent.com/google-coral"
TENSORFLOW_LITE_MODELS = {
    INCEPTION_V3: f"{CORAL_LINK}/test_data/master/inception_v3_299_quant.tflite",
    RESNET_50: f"{CORAL_LINK}/test_data/master/tfhub_tf2_resnet_50_imagenet_ptq.tflite",
    EFFICIENT_NET_B0: f"{CORAL_LINK}/test_data/master/efficientnet-edgetpu-S_quant.tflite",
    EFFICIENT_NET_B3: f"{CORAL_LINK}/test_data/master/efficientnet-edgetpu-L_quant.tflite",
    SSD_MOBILENET_V2: f"{CORAL_LINK}/test_data/master/ssd_mobilenet_v2_coco_quant_postprocess.tflite",
    EFFICIENT_DET_LITE3: f"{CORAL_LINK}/test_data/master/efficientdet_lite3_512_ptq.tflite",
}

tf_models = "data/tf_models"
if os.path.isdir(tf_models) is False:
    os.mkdir(tf_models)
for m, link in TENSORFLOW_MODELS.items():
    final_path = f"{tf_models}/{m}"
    tar_file = final_path + ".tar.gz"
    if os.path.isfile(tar_file):
        os.remove(tar_file)
    os.system(f"wget {link} -O {tar_file}")
    if os.path.isdir(final_path) is False:
        os.mkdir(final_path)
    os.system(f"tar xzf {final_path}.tar.gz -C {final_path}/")

for m, link in TENSORFLOW_LITE_MODELS.items():
    final_path = f"{tf_models}/{m}.tflite"
    os.system(f"wget {link} -O {final_path}")
