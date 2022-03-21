#!/usr/bin/python3

import os

from common_tf_and_pt import SSD_MOBILENET_V2, EFFICIENT_DET_LITE3, FASTER_RCNN_RESNET_FPN50

TENSORFLOW_MODELS = {
    SSD_MOBILENET_V2: "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2?tf-hub-format=compressed",
    EFFICIENT_DET_LITE3: "https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1?tf-hub-format=compressed",
    FASTER_RCNN_RESNET_FPN50:
        "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1?tf-hub-format=compressed"

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
