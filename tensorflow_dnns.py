#!/usr/bin/python3.8

"""
Main file for Tensorflow DNNs setup
"""
import os
from typing import Union

import numpy
import tensorflow
from PIL.Image import BICUBIC, BILINEAR
from keras.applications import efficientnet
from keras.applications import inception_v3
from keras.applications import resnet
from keras.preprocessing.image import img_to_array
from tensorflow import keras

import console_logger
from common_tf_and_pt import *

DNN_MODELS = {
    INCEPTION_V3: {
        "model": inception_v3.InceptionV3,
        "type": DNNType.CLASSIFICATION,
        "transform": inception_v3.preprocess_input,
        "interpolation": BILINEAR
    },
    RESNET_50: {
        "model": resnet.ResNet50,
        "type": DNNType.CLASSIFICATION,
        "transform": resnet.preprocess_input,
        "interpolation": BILINEAR
    },
    EFFICIENT_NET_B0: {
        "model": efficientnet.EfficientNetB0,
        "type": DNNType.CLASSIFICATION,
        "transform": efficientnet.preprocess_input,
        "interpolation": BILINEAR
    },
    EFFICIENT_NET_B3: {
        "model": efficientnet.EfficientNetB3,
        "type": DNNType.CLASSIFICATION,
        "transform": efficientnet.preprocess_input,
        "interpolation": BICUBIC
    },
    # Object detection, segmentation, and keypoint
    SSD_MOBILENET_V2: {
        # https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2
        # Inputs
        # A three-channel image of variable size - the model does NOT support batching.
        # The input tensor is a tf.uint8 tensor with shape [1, height, width, 3] with values in [0, 255].
        # Outputs
        # The output dictionary contains:
        #     num_detections: a tf.int tensor with only one value, the number of detections [N].
        #     detection_boxes: a tf.float32 tensor of shape [N, 4] containing bounding box coordinates in the following
        #     order: [ymin, xmin, ymax, xmax].
        #     detection_classes: a tf.int tensor of shape [N] containing detection class index from the label file.
        #     detection_scores: a tf.float32 tensor of shape [N] containing detection scores.
        #     raw_detection_boxes: a tf.float32 tensor of shape [1, M, 4] containing decoded detection boxes without
        #     Non-Max suppression. M is the number of raw detections.
        #     raw_detection_scores: a tf.float32 tensor of shape [1, M, 90] and contains class score logits for raw
        #     detection boxes. M is the number of raw detections.
        #     detection_anchor_indices: a tf.float32 tensor of shape [N] and contains the anchor indices of the
        #     detections after NMS.
        #     detection_multiclass_scores: a tf.float32 tensor of shape [1, N, 91] and contains class
        #     score distribution (including background) for detection boxes in the image including background class.
        "model": None, "interpolation": None, "transform": None,
        "type": DNNType.DETECTION,
    },
    EFFICIENT_DET_LITE3: {
        # https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1
        # Inputs
        # A batch of three-channel images of variable size. The input tensor is a
        # tf.uint8 tensor with shape [None, height, width, 3] with values in [0, 255].
        # Outputs
        # The output dictionary contains:
        #     detection_boxes: a tf.float32 tensor of shape [N, 4]
        #     containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax].
        #     detection_scores: a tf.float32 tensor of shape [N] containing detection scores.
        #     detection_classes: a tf.int tensor of shape [N] containing detection class index from the label file.
        #     num_detections: a tf.int tensor with only one value, the number of detections [N].
        "model": None, "interpolation": None, "transform": None,
        "type": DNNType.DETECTION,
    },
    FASTER_RCNN_RESNET_FPN50: {
        # https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1
        # Inputs
        # A three-channel image of variable size - the model does NOT support batching. The input tensor is a tf.uint8
        # tensor with shape [1, height, width, 3] with values in [0, 255].
        # Outputs
        # The output dictionary contains:
        #     num_detections: a tf.int tensor with only one value, the number of detections [N].
        #     detection_boxes: a tf.float32 tensor of shape [N, 4] containing bounding box coordinates in the following
        #     order: [ymin, xmin, ymax, xmax].
        #     detection_classes: a tf.int tensor of shape [N] containing detection class index from the label file.
        #     detection_scores: a tf.float32 tensor of shape [N] containing detection scores.
        #     raw_detection_boxes: a tf.float32 tensor of shape [1, M, 4] containing decoded detection boxes without
        #     Non-Max suppression. M is the number of raw detections.
        #     raw_detection_scores: a tf.float32 tensor of shape [1, M, 90] and contains class score logits for raw
        #     detection boxes. M is the number of raw detections.
        #     detection_anchor_indices: a tf.float32 tensor of shape [N] and contains the anchor indices of the
        #     detections after NMS.
        #     detection_multiclass_scores: a tf.float32 tensor of shape [1, N, 90] and contains class score
        #     distribution (including background) for detection boxes in the image including background class.
        "model": None, "interpolation": None, "transform": None,
        "type": DNNType.DETECTION,
    },
    # Not available for tensorflow_hub yet
    # RETINA_NET_RESNET_FPN50: NotImplementedError
}


def is_not_close(rhs: tensorflow.Tensor, lhs: tensorflow.Tensor, threshold: float) -> tensorflow.Tensor:
    """ Function to be equivalent to PyTorch """
    return tensorflow.greater(tensorflow.abs(tensorflow.subtract(rhs, lhs)), threshold)


def equal(rhs: tensorflow.Tensor, lhs: tensorflow.Tensor, threshold: float = None) -> bool:
    if threshold:
        return bool(
            tensorflow.reduce_all(tensorflow.less_equal(tensorflow.abs(tensorflow.subtract(rhs, lhs)), threshold)))
    else:
        return bool(tensorflow.equal(rhs, lhs))


def copy_tensor_to_cpu(x_tensor):
    return x_tensor


def compare_output_with_gold(dnn_output_tensor: tensorflow.Tensor, dnn_golden_tensor: tensorflow.Tensor,
                             dnn_type: DNNType, setup_iteration: int, batch_iteration: int, current_image_names: list,
                             output_logger: logging.Logger) -> int:
    output_errors = 0
    # Make sure that they are on CPU
    with tensorflow.device('/CPU'):
        if dnn_type == DNNType.CLASSIFICATION:
            output_errors = compare_classification(dnn_output_tensor=dnn_output_tensor,
                                                   dnn_golden_tensor=dnn_golden_tensor,
                                                   setup_iteration=setup_iteration, batch_iteration=batch_iteration,
                                                   current_image_names=current_image_names, output_logger=output_logger,
                                                   copy_tensor_to_cpu_caller=copy_tensor_to_cpu,
                                                   equal_caller=equal)
        elif dnn_type == DNNType.DETECTION:
            output_errors = compare_detection(dnn_output_tensor=dnn_output_tensor, dnn_golden_tensor=dnn_golden_tensor,
                                              current_image_names=current_image_names,
                                              output_logger=output_logger,
                                              copy_tensor_to_cpu_caller=copy_tensor_to_cpu,
                                              equal_caller=equal)
    dnn_log_helper.log_error_count(output_errors)
    return output_errors


def load_dataset(transforms: callable, interpolation: int, image_list_path: str, logger: logging.Logger,
                 batch_size: int, device: str, dnn_type: DNNType,
                 dnn_input_size: tuple) -> Tuple[Union[tensorflow.Tensor, list], list]:
    timer = Timer()
    timer.tic()
    images, image_list = load_image_list(image_list_path)
    image_list = list(map(os.path.basename, image_list))
    assert batch_size <= len(image_list), "Batch size must be equal or smaller than img list"
    with tensorflow.device(device):
        if dnn_type == DNNType.CLASSIFICATION:
            # Equivalent to pytorch resize + to_tensor
            input_tensor = tensorflow.stack(
                [transforms(img_to_array(img.resize(dnn_input_size, resample=interpolation))) for img in images]
            )
            # Split here is different
            num_of_splits = int(input_tensor.shape[0] / batch_size)
            input_tensor = tensorflow.split(input_tensor, num_or_size_splits=num_of_splits)
        else:
            input_tensor = [tensorflow.expand_dims(img_to_array(img, dtype=numpy.uint8), axis=0) for img in images]

    timer.toc()
    logger.debug(f"Input images loaded and resized successfully: {timer}")
    return input_tensor, image_list


def load_model(precision: str, model_loader: callable, device: str, dnn_type: DNNType, model_name: str) -> keras.Model:
    with tensorflow.device(device):
        if dnn_type == DNNType.CLASSIFICATION:
            weights = 'imagenet'
            dnn_model = model_loader(weights=weights)
        elif dnn_type == DNNType.DETECTION:
            # FIXME: find a way to speedup the model load
            model_path = f"data/tf_models/{model_name}"
            dnn_model = tensorflow.saved_model.load(model_path)

        # It means that I want to convert the model into FP16, but make sure that it is not quantized
        if precision == "fp16":
            converter = tensorflow.lite.TFLiteConverter.from_keras_model(dnn_model)
            converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tensorflow.float16]
            dnn_model = converter.convert()
    return dnn_model


def get_predictions(batched_output: tensorflow.Tensor, dnn_type: DNNType, img_names: list) -> list:
    pred = list()
    if dnn_type == DNNType.CLASSIFICATION:
        for img, x in zip(img_names, batched_output):
            label = tensorflow.argmax(x, 1)
            pred.append({"img_name": img, "class_id_predicted": int(label[0])})
    return pred


def main():
    # tensorflow.debugging.set_log_device_placement(True)
    is_in_eager_mode = tensorflow.executing_eagerly()
    # Check the available device
    device = "/GPU" if tensorflow.config.list_physical_devices('GPU') else "/CPU"
    timer = Timer()
    timer.tic()
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    output_logger = console_logger.ColoredLogger(main_logger_name)
    args, args_conf = parse_args()
    for k, v in vars(args).items():
        output_logger.debug(f"{k}:{v}")
    generate = args.generate
    iterations = args.iterations
    image_list_path = args.imglist
    gold_path = args.goldpath
    precision = args.precision
    model_name = args.model
    disable_console_logger = args.disableconsolelog
    batch_size = args.batchsize

    if disable_console_logger:
        output_logger.level = logging.ERROR

    # Set the parameters for the DNN
    model_parameters = DNN_MODELS[model_name]
    input_size = OPTIMAL_INPUT_SIZE[model_name]
    interpolation = model_parameters["interpolation"]
    dnn_type = model_parameters["type"]
    transform = model_parameters["transform"]
    dnn_model = load_model(precision=precision, model_loader=model_parameters["model"],
                           device=device, dnn_type=dnn_type, model_name=model_name)

    timer.toc()
    output_logger.debug(f"Time necessary to load the model and config it: {timer}")

    # First step is to load the inputs in the memory
    timer.tic()
    input_list, image_names = load_dataset(transforms=transform, image_list_path=image_list_path, logger=output_logger,
                                           batch_size=batch_size, device=device, dnn_type=dnn_type,
                                           dnn_input_size=input_size, interpolation=interpolation)
    timer.toc()
    output_logger.debug(f"Time necessary to load the inputs: {timer}")

    dnn_gold_tensors = list()
    # Load if it is not a gold generating op
    timer.tic()
    if generate is False:
        with tensorflow.device("/CPU"):
            dnn_gold_tensors = numpy.load(gold_path)

    timer.toc()
    output_logger.debug(f"Time necessary to load the golden outputs: {timer}")

    # Start the setup
    args_conf += f" eager_mode:{is_in_eager_mode}"
    dnn_log_helper.start_setup_log_file(framework_name="TensorFlow", args_conf=args_conf, model_name=model_name,
                                        max_errors_per_iteration=MAXIMUM_ERRORS_PER_ITERATION, generate=generate)

    # Main setup loop
    setup_iteration = 0
    while setup_iteration < iterations:
        total_errors = 0
        # Loop over the input list
        for batch_i, batched_input in enumerate(input_list):
            batch_iteration = batch_i * batch_size
            current_image_names = image_names[batch_iteration:batch_iteration + batch_size]
            timer.tic()
            dnn_log_helper.start_iteration()
            with tensorflow.device(device):
                current_output = dnn_model(batched_input)
            dnn_log_helper.end_iteration()
            # show_classification_result(output=current_output, batch_size=batch_size, image_list=current_image_names)

            timer.toc()
            kernel_time = timer.diff_time
            # Then compare the golden with the output
            timer.tic()
            errors = 0
            if generate is False:
                current_gold = dnn_gold_tensors[batch_i]
                errors = compare_output_with_gold(dnn_output_tensor=current_output, dnn_golden_tensor=current_gold,
                                                  dnn_type=dnn_type, setup_iteration=setup_iteration,
                                                  batch_iteration=batch_iteration, output_logger=output_logger,
                                                  current_image_names=current_image_names)
            else:
                dnn_gold_tensors.append(current_output)

            total_errors += errors
            timer.toc()
            comparison_time = timer.diff_time

            iteration_out = f"It:{setup_iteration:<3} imgit:{batch_i:<3}"
            iteration_out += f" {batch_size:<2} batches inference time:{kernel_time:.5f}"
            time_pct = (comparison_time / (comparison_time + kernel_time)) * 100.0
            iteration_out += f", gold compare time:{comparison_time:.5f} ({time_pct:.1f}%) errors:{errors}"
            output_logger.debug(iteration_out)

        # Reload after error
        if total_errors != 0:
            del input_list
            del dnn_model
            dnn_model = load_model(precision=precision, model_loader=model_parameters["model"], device=device,
                                   dnn_type=dnn_type, model_name=model_name)
            input_list, image_names = load_dataset(transforms=transform, image_list_path=image_list_path,
                                                   logger=output_logger, batch_size=batch_size, device=device,
                                                   dnn_type=dnn_type, dnn_input_size=input_size,
                                                   interpolation=interpolation)

        setup_iteration += 1
    timer.tic()
    if generate:
        with tensorflow.device("/CPU"):
            dnn_gold_tensors = numpy.array(dnn_gold_tensors)
            numpy.save(gold_path, dnn_gold_tensors)
            timer.toc()
            output_logger.debug(f"Time necessary to save the golden outputs: {timer}")
            output_logger.debug(f"Accuracy measure")
            # verify_network_accuracy(predictions=get_predictions(dnn_gold_tensors, dnn_type=dnn_type,
            #                                                     img_names=image_names),
            #                         ground_truth_csv=args.grtruthcsv, dnn_type=dnn_type)

    # finish the logfile
    dnn_log_helper.end_log_file()


if __name__ == '__main__':
    main()

# def compare_classification(dnn_output_tensor: tensorflow.Tensor, dnn_golden_tensor: tensorflow.Tensor,
#                            setup_iteration: int, batch_iteration: int, current_image_names: list,
#                            output_logger: logging.Logger) -> int:
#     # # Debug injection
#     # if setup_iteration + batch_iteration == 20:
#     #     for i in range(300, 900):
#     #         dnn_output_tensor[3][i] = 34.2
#
#     output_errors = 0
#     # using the same approach as the detection, compare only the positions that differ
#     if equal(rhs=dnn_golden_tensor, lhs=dnn_output_tensor, threshold=CLASSIFICATION_ABS_THRESHOLD) is False:
#         output_logger.error("Not equal output tensors")
#         if dnn_golden_tensor.shape != dnn_output_tensor.shape:
#             error_detail = f"DIFF_SIZE g:{dnn_golden_tensor.shape} o:{dnn_output_tensor.shape}"
#             output_logger.error(error_detail)
#             dnn_log_helper.log_error_detail(error_detail)
#
#         for img_name_i, current_gold_tensor, current_output_tensor in zip(current_image_names, dnn_golden_tensor,
#                                                                           dnn_output_tensor):
#             for i, (gold, found) in enumerate(zip(current_gold_tensor, current_output_tensor)):
#                 if abs(gold - found) > CLASSIFICATION_ABS_THRESHOLD:
#                     output_errors += 1
#                     error_detail = f"img:{img_name_i} sit:{setup_iteration} "
#                     error_detail += f"bti:{batch_iteration} pos:{i} g:{gold:.6e} o:{found:.6e}"
#                     output_logger.error(error_detail)
#                     dnn_log_helper.log_error_detail(error_detail)
#     return output_errors
#
#
# def compare_detection(dnn_output_tensor: tensorflow.Tensor, dnn_golden_tensor:
# tensorflow.Tensor, batch_iteration: int,
#                       current_image_names: list, output_logger: logging.Logger) -> int:
#     score_errors_count, labels_errors_count, box_errors_count = 0, 0, 0
#     for img_name_i, gold_batch_i, out_batch_i in zip(current_image_names, dnn_golden_tensor, dnn_output_tensor):
#         boxes_gold, labels_gold, scores_gold = gold_batch_i["boxes"], gold_batch_i["labels"], gold_batch_i["scores"]
#         # Make sure that we are on the CPU
#         boxes_out = out_batch_i["boxes"]
#         labels_out = out_batch_i["labels"]
#         scores_out = out_batch_i["scores"]
#         # for i in range(10):
#         #     scores_out[34 + i] = i
#         #     boxes_out[i][i % 4] = i
#         #     labels_out[40 + i] = i
#         #  It is better compare to a threshold
#         if all([equal(rhs=scores_gold, lhs=scores_out, threshold=DETECTION_SCORES_ABS_THRESHOLD),
#                 equal(rhs=boxes_gold, lhs=boxes_out, threshold=DETECTION_BOXES_ABS_THRESHOLD),
#                 equal(labels_gold, labels_out)]):
#
#             # Logging the score indexes that in fact have errors
#             for s_i, (score_gold, score_out) in enumerate(zip(scores_gold, scores_out)):
#                 if abs(score_gold - score_out) > DETECTION_SCORES_ABS_THRESHOLD:
#                     score_error = f"img:{img_name_i} scorei:{s_i} g:{score_gold} o:{score_out}"
#                     output_logger.error(score_error)
#                     dnn_log_helper.log_error_detail(score_error)
#                     score_errors_count += 1
#
#             # Logging the boxes indexes that in fact have errors
#             for b_i, (box_gold, box_out) in enumerate(zip(boxes_gold, boxes_out)):
#                 if equal(box_gold, box_out, DETECTION_BOXES_ABS_THRESHOLD) is False:
#                     gx1, gx2, gx3, gx4 = box_gold
#                     ox1, ox2, ox3, ox4 = box_out
#                     box_error = f"img:{img_name_i} boxi:{b_i} gx1:{gx1:.6e} gx2:{gx2:.6e} gx3:{gx3:.6e} gx4:{gx4:.6e}"
#                     box_error += f" ox1:{ox1:.6e} ox2:{ox2:.6e} ox3:{ox3:.6e} ox4:{ox4:.6e}"
#                     output_logger.error(box_error)
#                     dnn_log_helper.log_error_detail(box_error)
#                     box_errors_count += 1
#             # Logging the boxes indexes that in fact have errors
#             for l_i, (label_gold, label_out) in enumerate(zip(labels_gold, labels_out)):
#                 if label_gold != label_out:
#                     label_error = f"img:{img_name_i} labeli:{l_i} g:{label_gold} o:{label_out}"
#                     output_logger.error(label_error)
#                     dnn_log_helper.log_error_detail(label_error)
#                     labels_errors_count += 1
#
#     return score_errors_count + box_errors_count + labels_errors_count
