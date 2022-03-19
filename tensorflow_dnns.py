#!/usr/bin/python3.8

"""
Main file for Tensorflow DNNs setup
"""
import logging
import os
from typing import Union

import tensorflow
from tensorflow import keras

import console_logger
import dnn_log_helper
from common_tf_and_pt import *

DNN_MODELS = {
    INCEPTION_V3: {
        "model": keras.InceptionV3,
        "type": DNNType.CLASSIFICATION,
        "transform": None
    },
    RESNET_50: {
        "model": keras.ResNet50,
        "type": DNNType.CLASSIFICATION,
        "transform": None
    },
    EFFICIENT_NET_B7: {
        "model": keras.EfficientNetB7,
        "type": DNNType.CLASSIFICATION,
        "transform": None
    },
    # Object detection, segmentation, and keypoint
    SSD_MOBILENET_V2: {
        "model": None,
        "type": DNNType.DETECTION,
        "transform": None
    },

    EFFICIENT_DET_LITE3: {
        "model": None,
        "transform": NotImplementedError(),
        "type": DNNType.DETECTION
    },

    # ONLY FOR GPUs
    RETINA_NET_RESNET_FPN50: {
        "model": None,
        "type": DNNType.DETECTION,
        "transform": None
    },

    FASTER_RCNN_RESNET_FPN50: {
        "model": None,
        "transform": NotImplementedError(),
        "type": DNNType.DETECTION
    },
}


def is_not_close(rhs: tensorflow.Tensor, lhs: tensorflow.Tensor, threshold: float) -> tensorflow.Tensor:
    """ Function to be equivalent to PyTorch """
    return tensorflow.greater(tensorflow.abs(tensorflow.subtract(rhs, lhs)), threshold)


def equal(rhs: tensorflow.Tensor, lhs: tensorflow.Tensor, threshold: float = None) -> bool:
    if threshold:
        return tensorflow.reduce_all(tensorflow.less_equal(tensorflow.abs(tensorflow.subtract(rhs, lhs)), threshold))
    else:
        return tensorflow.equal(rhs, lhs)


def compare_classification(dnn_output_tensor: tensorflow.Tensor, dnn_golden_tensor: tensorflow.Tensor, batch_size: int,
                           setup_iteration: int, batch_iteration: int, current_image_names: list,
                           output_logger: logging.Logger) -> int:
    # # Debug injection
    # if setup_iteration + batch_iteration == 20:
    #     for i in range(300, 900):
    #         dnn_output_tensor[3][i] = 34.2
    output_errors = 0
    # using the same approach as the detection, compare only the positions that differ
    if equal(rhs=dnn_golden_tensor, lhs=dnn_output_tensor, threshold=CLASSIFICATION_ABS_THRESHOLD) is False:
        output_logger.error("Not equal output tensors")
        if dnn_golden_tensor.shape != dnn_output_tensor.shape:
            info_detail = f"Shapes differ on size {dnn_golden_tensor.shape} {dnn_output_tensor.shape}"
            output_logger.error(info_detail)
            dnn_log_helper.log_info_detail(info_detail)

        # Loop through the images
        # for batch_i in range(0, batch_size):
        #     img_name_i = current_image_names[batch_i]
        for img_name_i, current_gold_tensor, current_output_tensor in zip(current_image_names,
                                                                          dnn_golden_tensor,
                                                                          dnn_output_tensor):
            diff_tensor_index = is_not_close(rhs=current_gold_tensor, lhs=current_output_tensor,
                                             threshold=CLASSIFICATION_ABS_THRESHOLD)

            output_errors += diff_tensor_index.sum()
            diff_detail = f"diff img:{img_name_i} scores:{diff_tensor_index.sum()}"
            output_logger.error(diff_detail)
            dnn_log_helper.log_error_detail(diff_detail)
            # Only the elements that differ are compared here
            for i, (gold, found) in enumerate(zip(current_gold_tensor[diff_tensor_index],
                                                  current_output_tensor[diff_tensor_index])):
                error_detail = f"img:{img_name_i} s_it:{setup_iteration} "
                error_detail += f"bti:{batch_iteration} pos:{i} g:{gold:.6e} f:{found:.6e}"
                output_logger.error(error_detail)
                dnn_log_helper.log_error_detail(error_detail)
    return output_errors


def compare_detection(dnn_output_tensor: tensorflow.Tensor, dnn_golden_tensor: tensorflow.Tensor, batch_iteration: int,
                      current_image_names: list, output_logger: logging.Logger) -> int:
    total_errors = 0
    for img_name_i, gold_batch_i, out_batch_i in zip(current_image_names,
                                                     dnn_golden_tensor,
                                                     dnn_output_tensor):
        boxes_gold, labels_gold, scores_gold = gold_batch_i["boxes"], gold_batch_i["labels"], gold_batch_i["scores"]
        # Make sure that we are on the CPU
        boxes_out = out_batch_i["boxes"]
        labels_out = out_batch_i["labels"]
        scores_out = out_batch_i["scores"]
        # for i in range(10):
        #     scores_out[34 + i] = i
        #     boxes_out[i][i % 4] = i
        #     labels_out[40 + i] = i
        #  It is better compare to a threshold
        diff_scores_index = is_not_close(rhs=scores_gold, lhs=scores_out, threshold=DETECTION_SCORES_ABS_THRESHOLD)
        diff_boxes_index = is_not_close(rhs=boxes_gold, lhs=boxes_out, threshold=DETECTION_BOXES_ABS_THRESHOLD)
        # Labels are integers
        diff_labels_index = tensorflow.math.not_equal(labels_gold, labels_out)

        if any([tensorflow.reduce_any(diff_boxes_index), tensorflow.reduce_any(diff_labels_index),
                tensorflow.reduce_any(diff_scores_index)]):
            diff_scores = scores_out[diff_scores_index]
            # For boxes, we have to work with the indexes
            diff_boxes = boxes_out[diff_boxes_index.any(dim=1)]
            diff_labels = labels_out[diff_labels_index]
            total_errors += diff_scores.numel() + diff_labels.numel() + diff_boxes.numel()
            error_detail = f"diff img:{img_name_i} scores:{diff_scores.numel()} bti:{batch_iteration} "
            error_detail += f"labels:{diff_labels.numel()} boxes:{diff_boxes.numel()}"
            output_logger.error(error_detail)
            dnn_log_helper.log_error_detail(error_detail)

            # Logging the score indexes that in fact have errors
            for s_i, (score_gold, score_out) in enumerate(zip(scores_gold[diff_scores_index], diff_scores)):
                score_error = f"si:{s_i} g:{score_gold} o:{score_out}"
                output_logger.error(score_error)
                dnn_log_helper.log_error_detail(score_error)

            # Logging the boxes indexes that in fact have errors
            for b_i, (box_gold, box_out) in enumerate(zip(boxes_gold[diff_boxes_index.any(dim=1)], diff_boxes)):
                gx1, gx2, gx3, gx4 = box_gold
                ox1, ox2, ox3, ox4 = box_out
                box_error = f"img:{img_name_i} bi:{b_i} gx1:{gx1:.6e} gx2:{gx2:.6e} gx3:{gx3:.6e} gx4:{gx4:.6e}"
                box_error += f" ox1:{ox1:.6e} ox2:{ox2:.6e} ox3:{ox3:.6e} ox4:{ox4:.6e}"
                output_logger.error(box_error)
                dnn_log_helper.log_error_detail(box_error)
            # Logging the boxes indexes that in fact have errors
            for l_i, (label_gold, label_out) in enumerate(zip(labels_gold[diff_labels_index], diff_labels)):
                label_error = f"img:{img_name_i} li:{l_i} g:{label_gold} o:{label_out}"
                output_logger.error(label_error)
                dnn_log_helper.log_error_detail(label_error)

    return total_errors


def compare_output_with_gold(dnn_output_tensor: tensorflow.Tensor, dnn_golden_tensor: tensorflow.Tensor,
                             dnn_type: DNNType,
                             batch_size: int, setup_iteration: int, batch_iteration: int, current_image_names: list,
                             output_logger: logging.Logger) -> int:
    output_errors = 0
    # Make sure that they are on CPU
    with tensorflow.device('/CPU'):
        if dnn_type == DNNType.CLASSIFICATION:
            output_errors = compare_classification(dnn_output_tensor=dnn_output_tensor,
                                                   dnn_golden_tensor=dnn_golden_tensor,
                                                   batch_size=batch_size, setup_iteration=setup_iteration,
                                                   batch_iteration=batch_iteration,
                                                   current_image_names=current_image_names,
                                                   output_logger=output_logger)
        elif dnn_type == DNNType.DETECTION:
            output_errors = compare_detection(dnn_output_tensor=dnn_output_tensor, dnn_golden_tensor=dnn_golden_tensor,
                                              batch_iteration=batch_iteration, current_image_names=current_image_names,
                                              output_logger=output_logger)
    dnn_log_helper.log_error_count(output_errors)
    return output_errors


def load_dataset(transforms: callable, image_list_path: str, logger: logging.Logger,
                 batch_size: int, device: str, dnn_type: DNNType) -> Tuple[Union[tensorflow.Tensor, list], list]:
    timer = Timer()
    timer.tic()
    images, image_list = load_image_list(image_list_path)
    # Remove the base path
    image_list = list(map(os.path.basename, image_list))
    input_tensor = tensorflow.Tensor()
    timer.toc()
    logger.debug(f"Input images loaded and resized successfully: {timer}")
    return input_tensor, image_list


def load_model(precision: str, model_loader: callable, device: str, dnn_type: DNNType) -> keras.Model:
    with tensorflow.device(device):
        weights = 'imagenet' if DNNType.CLASSIFICATION else "coco2017"
        dnn_model = model_loader(weights=weights)
        # It means that I want to convert the model into FP16, but make sure that it is not quantized
        if precision == "fp16":
            converter = tensorflow.lite.TFLiteConverter.from_keras_model(dnn_model)
            converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tensorflow.float16]
            dnn_model = converter.convert()
    return dnn_model


def main():
    tensorflow.debugging.set_log_device_placement(True)
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
    dnn_type = model_parameters["type"]
    transform = model_parameters["transform"]
    dnn_model = load_model(precision=precision, model_loader=model_parameters["model"],
                           device=device, dnn_type=dnn_type)

    timer.toc()
    output_logger.debug(f"Time necessary to load the model and config it: {timer}")

    # First step is to load the inputs in the memory
    timer.tic()
    input_list, image_names = load_dataset(transforms=transform, image_list_path=image_list_path, logger=output_logger,
                                           batch_size=batch_size, device=device, dnn_type=dnn_type)
    timer.toc()
    output_logger.debug(f"Time necessary to load the inputs: {timer}")

    dnn_gold_tensors = list()
    # Load if it is not a gold generating op
    timer.tic()
    if generate is False:
        with tensorflow.device("/CPU"):
            raise NotImplementedError
            # dnn_gold_tensors = tensorflow.tensorflow.io.read_file(gold_path)

    timer.toc()
    output_logger.debug(f"Time necessary to load the golden outputs: {timer}")

    # Start the setup
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
            # show_classification_result(batch_input=current_output, image_list=current_image_names)

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
                                                  batch_size=batch_size, current_image_names=current_image_names)
            else:
                assert len(current_output) == batch_size, str(current_output)
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
                                   dnn_type=dnn_type)
            input_list, image_names = load_dataset(transforms=transform, image_list_path=image_list_path,
                                                   logger=output_logger, batch_size=batch_size, device=device,
                                                   dnn_type=dnn_type)

        setup_iteration += 1
    timer.tic()
    if generate:
        raise NotImplementedError
    timer.toc()
    output_logger.debug(f"Time necessary to save the golden outputs: {timer}")

    # finish the logfile
    dnn_log_helper.end_log_file()


if __name__ == '__main__':
    main()
