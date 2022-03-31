#!/usr/bin/python3.8

"""
Main file for Tensorflow DNNs setup
"""
import logging
import os
import pickle
from typing import Union

import keras
import numpy
import tensorflow
from PIL.Image import BICUBIC, BILINEAR, ANTIALIAS
from keras.applications import efficientnet
from keras.applications import inception_v3
from keras.applications import resnet
from keras.preprocessing.image import img_to_array

import console_logger
import dnn_log_helper
import tensorflow_lite_utils
from common_tf_and_pt import *

FORCE_GPU = True
NUM_THREAD = 6
tensorflow.config.threading.set_inter_op_parallelism_threads(num_threads=NUM_THREAD)
tensorflow.config.threading.set_intra_op_parallelism_threads(num_threads=NUM_THREAD)

DNN_MODELS = {
    INCEPTION_V3: {
        "model": inception_v3.InceptionV3,
        "type": DNNType.CLASSIFICATION,
        "interpolation": BILINEAR
    },
    RESNET_50: {
        "model": resnet.ResNet50,
        "type": DNNType.CLASSIFICATION,
        "interpolation": BILINEAR
    },
    EFFICIENT_NET_B0: {
        "model": efficientnet.EfficientNetB0,
        "type": DNNType.CLASSIFICATION,
        "interpolation": BILINEAR
    },
    EFFICIENT_NET_B3: {
        "model": efficientnet.EfficientNetB3,
        "type": DNNType.CLASSIFICATION,
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
        "model": None, "interpolation": None, "type": DNNType.DETECTION,
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
        "model": None, "interpolation": None, "type": DNNType.DETECTION,
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
        "model": None, "interpolation": None, "type": DNNType.DETECTION,
    },
    # Not available for tensorflow_hub yet
    # RETINA_NET_RESNET_FPN50: NotImplementedError
}


def compare_detection(dnn_output_dict: dict, dnn_golden_dict: dict, current_image: str, output_logger: logging.Logger,
                      # copy_tensor_to_cpu_caller: callable, detection_keys: dict = None,
                      use_tflite: bool = False) -> int:
    """ Compare the detections and return the number of errors. Also log on the logfile  """
    with tensorflow.device('/CPU'):
        # We use for detection batch always equal to one
        score_errors_count, labels_errors_count, box_errors_count = 0, 0, 0
        # if detection_keys is None:
        #     detection_keys = dict(boxes="boxes", scores="scores", labels="labels")
        if use_tflite:
            boxes_gold = dnn_golden_dict["boxes"]
            labels_gold = dnn_golden_dict["labels"]
            scores_gold = dnn_golden_dict["scores"]
            boxes_out = dnn_output_dict["boxes"]
            labels_out = dnn_output_dict["labels"]
            scores_out = dnn_output_dict["scores"]
            # if (random.randint(0, 4) == 0):
            #     scores_out[0] = 1
            # if (random.randint(0, 4) == 0):
            #     boxes_out[0][0] = 2
            # if (random.randint(0, 4) == 0):
            #     labels_out[0] = 3
        else:
            boxes_gold = dnn_golden_dict["detection_boxes"]
            labels_gold = dnn_golden_dict["detection_classes"]
            scores_gold = dnn_golden_dict["detection_scores"]
            boxes_out = dnn_output_dict["detection_boxes"]
            labels_out = dnn_output_dict["detection_classes"]
            scores_out = dnn_output_dict["detection_scores"]
            # print(boxes_gold)
            # print(boxes_out)
            # print(boxes_gold.device)
            # print(boxes_out.device)
            # if random.randint(0, 4) == 0:
            #     temp = scores_out.numpy()
            #     temp[0] += 1
            #     scores_out = tf.convert_to_tensor(temp, dtype=tf.float32)
            # if random.randint(0, 4) == 0:
            #     temp = boxes_out.numpy()
            #     temp[random.randrange(0, len(temp))] += 2
            #     boxes_out = tf.convert_to_tensor(temp, dtype=tf.float32)
            #
            # if random.randint(0, 4) == 0:
            #     temp = labels_out.numpy()
            #     temp[random.randrange(0, len(temp))] += 2
            #     labels_out = tf.convert_to_tensor(temp, dtype=tf.float32)

            # # Debug
        # print(boxes_out)
        # print(labels_out)
        # print(scores_out)

        # print(scores_out)
        # if(random.randint(0,4)==0):
        #    scores_out[0] = 1
        #    boxes_out[0][0] = 2
        #    labels_out[0] = 3
        # It is better compare to a threshold

        # Logging the score indexes that in fact have errors
        if all([equal(rhs=scores_gold, lhs=scores_out, threshold=DETECTION_SCORES_ABS_THRESHOLD),
                equal(rhs=boxes_gold, lhs=boxes_out, threshold=DETECTION_BOXES_ABS_THRESHOLD),
                equal(labels_gold, labels_out)]) is False:
            for s_i, (score_gold, score_out) in enumerate(zip(scores_gold, scores_out)):
                if abs(score_gold - score_out) > DETECTION_SCORES_ABS_THRESHOLD:
                    score_error = f"img:{current_image} scorei:{s_i} g:{score_gold:.6e} o:{score_out:.6e}"
                    output_logger.error(score_error)
                    dnn_log_helper.log_error_detail(score_error)
                    score_errors_count += 1
            # Logging the boxes indexes that in fact have errors
            for b_i, (box_gold, box_out) in enumerate(zip(boxes_gold, boxes_out)):
                if equal(box_gold, box_out, DETECTION_BOXES_ABS_THRESHOLD) is False:
                    gx1, gx2, gx3, gx4 = box_gold
                    ox1, ox2, ox3, ox4 = box_out
                    box_error = f"img:{current_image} boxi:{b_i:.6e}"
                    box_error += f" gx1:{gx1:.6e} gx2:{gx2:.6e} gx3:{gx3:.6e} gx4:{gx4:.6e}"
                    box_error += f" ox1:{ox1:.6e} ox2:{ox2:.6e} ox3:{ox3:.6e} ox4:{ox4:.6e}"
                    output_logger.error(box_error)
                    dnn_log_helper.log_error_detail(box_error)
                    box_errors_count += 1
            # Logging the boxes indexes that in fact have errors
            for l_i, (label_gold, label_out) in enumerate(zip(labels_gold, labels_out)):
                if label_gold != label_out:
                    label_error = f"img:{current_image} labeli:{l_i} g:{label_gold} o:{label_out}"
                    output_logger.error(label_error)
                    dnn_log_helper.log_error_detail(label_error)
                    labels_errors_count += 1

    return score_errors_count + box_errors_count + labels_errors_count


def compare_classification(dnn_output_tensor, dnn_golden_tensor, setup_iteration: int, current_image: str,
                           output_logger: logging.Logger, use_tflite: bool) -> int:
    # Make sure that they are on CPU
    with tensorflow.device('/CPU'):
        if use_tflite:
            dnn_output_tensor_cpu = dnn_output_tensor
            # # Debug injection

            # if random.randint(0,4)==0:
            #    dnn_output_tensor_cpu[0] = 34.2
        else:
            dnn_golden_tensor = dnn_golden_tensor[0].numpy()
            dnn_output_tensor_cpu = dnn_output_tensor[0].numpy()
            # print(dnn_golden_tensor)
            # print(dnn_output_tensor_cpu)
            # print(dnn_golden_tensor.device)
            # print(dnn_output_tensor_cpu.device)
            # if random.randint(0,4)==0:
            #    temp=dnn_output_tensor_cpu.numpy()
            #    temp[0] = 34.2
            #    dnn_output_tensor_cpu=tf.convert_to_tensor(temp,dtype=tf.float32)
        # print(dnn_output_tensor_cpu)

        output_errors = 0
        # using the same approach as the detection, compare only the positions that differ
        if equal_numpy(rhs=dnn_golden_tensor, lhs=dnn_output_tensor_cpu,
                       threshold=CLASSIFICATION_ABS_THRESHOLD) is False:
            output_logger.error("Not equal output tensors")
            if dnn_golden_tensor.shape != dnn_output_tensor_cpu.shape:
                error_detail = f"DIFF_SIZE g:{dnn_golden_tensor.shape} o:{dnn_output_tensor_cpu.shape}"
                output_logger.error(error_detail)
                dnn_log_helper.log_error_detail(error_detail)

            # for img_name_i, current_gold_tensor, current_output_tensor in zip(current_image_names, dnn_golden_tensor,
            #                                                                  dnn_output_tensor_cpu):
            for i, (gold, found) in enumerate(zip(dnn_golden_tensor, dnn_output_tensor_cpu)):
                if abs(gold - found) > CLASSIFICATION_ABS_THRESHOLD:
                    # print(gold)
                    # print(found)
                    output_errors += 1
                    error_detail = f"img:{current_image} setupit:{setup_iteration} i:{i} g:{gold:.6e} o:{found:.6e}"
                    output_logger.error(error_detail)
                    dnn_log_helper.log_error_detail(error_detail)
            # else:
            #    for img_name_i, current_gold_tensor, current_output_tensor in zip(current_image_names, dnn_golden_tensor,
            #                                                                  dnn_output_tensor_cpu):
            #        for i, (gold, found) in enumerate(zip(current_gold_tensor, current_output_tensor)):
            #            if abs(gold - found) > CLASSIFICATION_ABS_THRESHOLD:
            #                output_errors += 1
            #                error_detail = f"img:{img_name_i} setupit:{setup_iteration} "
            #                error_detail += f"batchti:{batch_iteration} i:{i} g:{gold:.6e} o:{found:.6e}"
            #                output_logger.error(error_detail)
            #                dnn_log_helper.log_error_detail(error_detail)
    return output_errors


def equal(rhs: tensorflow.Tensor, lhs: tensorflow.Tensor, threshold: float = None) -> bool:
    if threshold:
        return bool(
            tensorflow.reduce_all(tensorflow.less_equal(tensorflow.abs(tensorflow.subtract(rhs, lhs)), threshold)))
    else:
        return bool(tensorflow.reduce_all(tensorflow.equal(rhs, lhs)))


def equal_numpy(rhs: numpy.array, lhs: numpy.array, threshold: float = None) -> bool:
    if threshold:
        return bool(
            numpy.all(numpy.less_equal(numpy.abs(numpy.subtract(rhs, lhs)), threshold)))
    else:
        return bool(numpy.all(numpy.equal(rhs, lhs)))


def compare_output_with_gold(dnn_output_tensor: tensorflow.Tensor, dnn_golden_tensor: tensorflow.Tensor,
                             dnn_type: DNNType, setup_iteration: int, current_image: str, output_logger: logging.Logger,
                             use_tflite: bool) -> int:
    output_errors = 0
    # Make sure that they are on CPU
    if dnn_type == DNNType.CLASSIFICATION:
        output_errors = compare_classification(dnn_output_tensor=dnn_output_tensor,
                                               dnn_golden_tensor=dnn_golden_tensor,
                                               setup_iteration=setup_iteration, current_image=current_image,
                                               output_logger=output_logger, use_tflite=use_tflite)
    elif dnn_type == DNNType.DETECTION:
        output_errors = compare_detection(dnn_output_dict=dnn_output_tensor, dnn_golden_dict=dnn_golden_tensor,
                                          current_image=current_image, output_logger=output_logger,
                                          use_tflite=use_tflite)
    dnn_log_helper.log_error_count(output_errors)

    return output_errors


def load_dataset(interpolation: int, image_list_path: str, logger: logging.Logger, device: str, dnn_type: DNNType,
                 dnn_input_size: tuple, use_tflite: bool) -> Tuple[Union[tensorflow.Tensor, list], list]:
    timer = Timer()
    timer.tic()
    images, image_list = load_image_list(image_list_path)
    image_list = list(map(os.path.basename, image_list))
    # assert batch_size <= len(image_list), "Batch size must be equal or smaller than img list"
    with tensorflow.device(device):
        if use_tflite is False and dnn_type == DNNType.CLASSIFICATION:
            input_tensor = [tensorflow.expand_dims(
                img_to_array(img.resize(dnn_input_size, resample=interpolation), dtype=numpy.uint8), axis=0) for img in
                images]
        elif use_tflite is False and dnn_type == DNNType.DETECTION:
            input_tensor = [tensorflow.expand_dims(img_to_array(img, dtype=numpy.uint8), axis=0) for img in images]
        else:
            input_tensor = [img.resize(dnn_input_size, resample=ANTIALIAS) for img in images]

    timer.toc()
    logger.debug(f"Input images loaded and resized successfully: {timer}")
    return input_tensor, image_list


def load_model(model_loader: callable, device: str, dnn_type: DNNType, model_name: str,
               use_tflite: bool) -> Union[keras.Model, tensorflow.lite.Interpreter]:
    with tensorflow.device(device):
        model_path = f"{os.path.dirname(os.path.realpath(__file__))}/data/tf_models/{model_name}"
        if use_tflite is False:
            if dnn_type == DNNType.CLASSIFICATION:
                dnn_model = model_loader(weights='imagenet')
            elif dnn_type == DNNType.DETECTION:
                dnn_model = tensorflow.saved_model.load(model_path)
        else:
            dnn_model = tensorflow_lite_utils.create_interpreter(model_file=model_path + ".tflite")
            dnn_model.allocate_tensors()
    return dnn_model


def verify_network_accuracy(batched_output: Union[numpy.array, list], dnn_type: DNNType, img_names: list,
                            ground_truth_csv: str, use_tflite: bool):
    from verify_accuracy import verify_classification_accuracy, verify_detection_accuracy
    if use_tflite is False:
        if dnn_type == DNNType.CLASSIFICATION:
            pred = list()
            for img, x in zip(img_names, batched_output):
                label = tensorflow.argmax(x, 1)
                pred.append({"img_name": img, "class_id_predicted": int(label[0])})
            verify_classification_accuracy(pred, ground_truth_csv)
        else:
            pred = list()
            verify_detection_accuracy(pred, ground_truth_csv)
    else:
        pass


def pickle_save_file(file_path, data):
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle)


def pickle_load_file(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def main():
    # tensorflow.debugging.set_log_device_placement(True)
    is_in_eager_mode = tensorflow.executing_eagerly()
    # Check the available device
    device = "/CPU"
    if tensorflow.config.list_physical_devices('GPU'):
        dnn_log_helper.set_iter_interval_print(30)
        device = "/GPU"
    if FORCE_GPU:
        assert tensorflow.config.list_physical_devices('GPU'), f"GPU NOT PRESENT AND FORCE_GPU SET TO {FORCE_GPU}"
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
    model_name = args.model
    disable_console_logger = args.disableconsolelog
    # batch_size = args.batchsize
    use_tf_lite = args.tflite

    if disable_console_logger:
        output_logger.level = logging.FATAL

    # Set the parameters for the DNN
    model_parameters = DNN_MODELS[model_name]
    input_size = OPTIMAL_INPUT_SIZE[model_name]
    interpolation = model_parameters["interpolation"]
    dnn_type = model_parameters["type"]
    dnn_model = load_model(model_loader=model_parameters["model"], device=device, dnn_type=dnn_type,
                           model_name=model_name, use_tflite=use_tf_lite)

    timer.toc()
    output_logger.debug(f"Time necessary to load the model and config it: {timer}")

    # First step is to load the inputs in the memory
    timer.tic()

    input_list, image_names = load_dataset(image_list_path=image_list_path, logger=output_logger,
                                           device=device, dnn_type=dnn_type,
                                           dnn_input_size=input_size, interpolation=interpolation,
                                           use_tflite=use_tf_lite)
    timer.toc()
    output_logger.debug(f"Time necessary to load the inputs: {timer}")

    dnn_gold_tensors = list()
    # Load if it is not a gold generating op
    timer.tic()
    if generate is False:
        # with tensorflow.device("/CPU"):
        #     dnn_gold_tensors = numpy.load(gold_path, allow_pickle=True)
        dnn_gold_tensors = pickle_load_file(file_path=gold_path)

    timer.toc()
    output_logger.debug(f"Time necessary to load the golden outputs: {timer}")

    # Start the setup
    args_conf += f" eager_mode:{is_in_eager_mode}"
    dnn_log_helper.start_setup_log_file(framework_name="TensorFlow", args_conf=args_conf, model_name=model_name,
                                        max_errors_per_iteration=MAXIMUM_ERRORS_PER_ITERATION, generate=generate)

    # Main setup loop
    setup_iteration = 0
    # Avoid doing multiple comparisons
    if dnn_type == DNNType.CLASSIFICATION:
        output_parser = tensorflow_lite_utils.get_classification_scores
    elif dnn_type == DNNType.DETECTION:
        output_parser = tensorflow_lite_utils.get_all_objects
    else:
        raise ValueError("Incorrect DNN type")

    while setup_iteration < iterations:
        total_errors = 0
        # Loop over the input list
        for img_i, batched_input in enumerate(input_list):
            # batch_iteration = batch_i * batch_size
            # current_image_names = image_names[batch_iteration:batch_iteration + batch_size]
            timer.tic()
            dnn_log_helper.start_iteration()
            with tensorflow.device(device):
                if use_tf_lite:
                    tensorflow_lite_utils.set_input(interpreter=dnn_model, data=batched_input)
                    dnn_model.invoke()
                    current_output = output_parser(interpreter=dnn_model)
                else:
                    current_output = dnn_model(batched_input)
            dnn_log_helper.end_iteration()

            timer.toc()
            kernel_time = timer.diff_time
            # Then compare the golden with the output
            timer.tic()
            errors = 0
            if generate is False:
                current_gold = dnn_gold_tensors[img_i]
                errors = compare_output_with_gold(dnn_output_tensor=current_output, dnn_golden_tensor=current_gold,
                                                  dnn_type=dnn_type, setup_iteration=setup_iteration,
                                                  output_logger=output_logger, current_image=image_names[img_i],
                                                  use_tflite=use_tf_lite)
            else:
                dnn_gold_tensors.append(current_output)

            total_errors += errors
            timer.toc()
            comparison_time = timer.diff_time

            iteration_out = f"It:{setup_iteration:<3} imgit:{img_i:<3} inference time:{kernel_time:.5f}"
            time_pct = (comparison_time / (comparison_time + kernel_time)) * 100.0
            iteration_out += f", gold compare time:{comparison_time:.5f} ({time_pct:.1f}%) errors:{errors}"
            output_logger.debug(iteration_out)
        # Reload after error
        if total_errors != 0:
            del input_list
            del dnn_model
            dnn_model = load_model(model_loader=model_parameters["model"], device=device, dnn_type=dnn_type,
                                   model_name=model_name, use_tflite=use_tf_lite)
            if use_tf_lite:
                dnn_model, input_details, output_details = dnn_model

            input_list, image_names = load_dataset(image_list_path=image_list_path, logger=output_logger, device=device,
                                                   dnn_type=dnn_type, dnn_input_size=input_size,
                                                   interpolation=interpolation, use_tflite=use_tf_lite)

        setup_iteration += 1
    timer.tic()
    if generate:
        with tensorflow.device("/CPU"):
            # dnn_gold_tensors = numpy.array(dnn_gold_tensors)
            # numpy.save(gold_path, dnn_gold_tensors)
            if dnn_type == DNNType.CLASSIFICATION:
                class_list = [i.numpy() for i in dnn_gold_tensors]
                pickle_save_file(file_path=gold_path, data=class_list)
            else:
                det_list = list()
                for det in dnn_gold_tensors:
                    det_dict = {k: det[k].numpy() for k in det}
                    det_list.append(det_dict)
                pickle_save_file(file_path=gold_path, data=det_list)

            timer.toc()
            output_logger.debug(f"Time necessary to save the golden outputs: {timer}")
            output_logger.debug(f"Accuracy measure")
            verify_network_accuracy(batched_output=dnn_gold_tensors, dnn_type=dnn_type, img_names=image_names,
                                    ground_truth_csv=args.grtruthcsv, use_tflite=use_tf_lite)

    # finish the logfile
    dnn_log_helper.end_log_file()


if __name__ == '__main__':
    main()
