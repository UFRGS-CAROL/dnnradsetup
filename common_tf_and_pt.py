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

#def compare_detection(dnn_output_dict: dict, dnn_golden_dict: dict, current_image: str, output_logger: logging.Logger,
                      copy_tensor_to_cpu_caller: callable, equal_caller: callable, detection_keys: dict = None) -> int:
    """ Compare the detections and return the number of errors. Also log on the logfile  """
    # We use for detection batch always equal to one
#    score_errors_count, labels_errors_count, box_errors_count = 0, 0, 0
#    if detection_keys is None:
#        detection_keys = dict(boxes="boxes", scores="scores", labels="labels")
#    boxes_gold = copy_tensor_to_cpu_caller(dnn_golden_dict[detection_keys["boxes"]])
#    labels_gold = copy_tensor_to_cpu_caller(dnn_golden_dict[detection_keys["labels"]])
#    scores_gold = copy_tensor_to_cpu_caller(dnn_golden_dict[detection_keys["scores"]])
#    # Make sure that we are on the CPU
#    boxes_out = copy_tensor_to_cpu_caller(dnn_output_dict[detection_keys["boxes"]])
#    labels_out = copy_tensor_to_cpu_caller(dnn_output_dict[detection_keys["labels"]])
#    scores_out = copy_tensor_to_cpu_caller(dnn_output_dict[detection_keys["scores"]])
    # # Debug injection
    # for i in range(100):
    #     scores_out[34 + i] = i
    #     boxes_out[i][i % 4] = i
    #     labels_out[40 + i] = i
    #  It is better compare to a threshold
#    if all([equal_caller(rhs=scores_gold, lhs=scores_out, threshold=DETECTION_SCORES_ABS_THRESHOLD),
#            equal_caller(rhs=boxes_gold, lhs=boxes_out, threshold=DETECTION_BOXES_ABS_THRESHOLD),
#            equal_caller(labels_gold, labels_out)]) is False:
#        # Logging the score indexes that in fact have errors
#        for s_i, (score_gold, score_out) in enumerate(zip(scores_gold, scores_out)):
#            if abs(score_gold - score_out) > DETECTION_SCORES_ABS_THRESHOLD:
#                score_error = f"img:{current_image} scorei:{s_i} g:{score_gold:.6e} o:{score_out:.6e}"
#                output_logger.error(score_error)
#                dnn_log_helper.log_error_detail(score_error)
#                score_errors_count += 1
#        # Logging the boxes indexes that in fact have errors
#        for b_i, (box_gold, box_out) in enumerate(zip(boxes_gold, boxes_out)):
#            if equal_caller(box_gold, box_out, DETECTION_BOXES_ABS_THRESHOLD) is False:
#                gx1, gx2, gx3, gx4 = box_gold
#                ox1, ox2, ox3, ox4 = box_out
#                box_error = f"img:{current_image} boxi:{b_i:.6e}"
#                box_error += f" gx1:{gx1:.6e} gx2:{gx2:.6e} gx3:{gx3:.6e} gx4:{gx4:.6e}"
#                box_error += f" ox1:{ox1:.6e} ox2:{ox2:.6e} ox3:{ox3:.6e} ox4:{ox4:.6e}"
#                output_logger.error(box_error)
#                dnn_log_helper.log_error_detail(box_error)
#                box_errors_count += 1
#        # Logging the boxes indexes that in fact have errors
#        for l_i, (label_gold, label_out) in enumerate(zip(labels_gold, labels_out)):
#            if label_gold != label_out:
#                label_error = f"img:{current_image} labeli:{l_i} g:{label_gold} o:{label_out}"
#                output_logger.error(label_error)
#                dnn_log_helper.log_error_detail(label_error)
#                labels_errors_count += 1
#
#    return score_errors_count + box_errors_count + labels_errors_count


# else:
#     for i,(output_tensor_elem, gold_tensor_elem) in enumerate(zip(dnn_output_tensor,dnn_golden_tensor)):
#         #if i==3:
#         #    output_tensor_elem[1]+=1
#         for output_elem, gold_elem in zip(output_tensor_elem,gold_tensor_elem):
#             #print(output_elem)
#             #print(gold_elem)
#
#             if output_elem != gold_elem:
#                 an_error = f"img:{i} g:{gold_tensor_elem} o:{output_tensor_elem}"
#                 output_logger.error(an_error)
#                 dnn_log_helper.log_error_detail(an_error)
#                 score_errors_count+=1
#                 break

#def compare_classification(dnn_output_tensor, dnn_golden_tensor, setup_iteration: int,
#                           batch_iteration: int, current_image_names: list, output_logger: logging.Logger,
#                           copy_tensor_to_cpu_caller: callable, equal_caller: callable) -> int:
#    # Make sure that they are on CPU
#    dnn_output_tensor_cpu = copy_tensor_to_cpu_caller(dnn_output_tensor)
#    # # Debug injection
#    # if setup_iteration + batch_iteration == 20:
#    #     for i in range(300, 900):
#    #         dnn_output_tensor_cpu[3][i] = 34.2
#    output_errors = 0
#    # using the same approach as the detection, compare only the positions that differ
#    if equal_caller(rhs=dnn_golden_tensor, lhs=dnn_output_tensor_cpu, threshold=CLASSIFICATION_ABS_THRESHOLD) is False:
#        output_logger.error("Not equal output tensors")
#        if dnn_golden_tensor.shape != dnn_output_tensor_cpu.shape:
#            error_detail = f"DIFF_SIZE g:{dnn_golden_tensor.shape} o:{dnn_output_tensor_cpu.shape}"
#            output_logger.error(error_detail)
#            dnn_log_helper.log_error_detail(error_detail)

    #    for img_name_i, current_gold_tensor, current_output_tensor in zip(current_image_names, dnn_golden_tensor,
    #                                                                      dnn_output_tensor_cpu):
    #        for i, (gold, found) in enumerate(zip(current_gold_tensor, current_output_tensor)):
    #            if abs(gold - found) > CLASSIFICATION_ABS_THRESHOLD:
    #                output_errors += 1
    #                error_detail = f"img:{img_name_i} setupit:{setup_iteration} "
    #                error_detail += f"batchti:{batch_iteration} i:{i} g:{gold:.6e} o:{found:.6e}"
    #                output_logger.error(error_detail)
    #                dnn_log_helper.log_error_detail(error_detail)
    #return output_errors