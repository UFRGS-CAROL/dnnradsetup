#!/usr/bin/python3.8

"""
Main file for Pytorch DNNs setup
"""
import os
from typing import Union

import torch
import torchvision

import console_logger
from common_tf_and_pt import *
from common_tf_and_pt import compare_detection, compare_classification

COMMON_NORMALIZATION = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
DNN_MODELS = {
    # Check the spreadsheet to more information
    # Inception V3 - Config: https://pytorch.org/hub/pytorch_vision_inception_v3/
    INCEPTION_V3: {
        "model": torchvision.models.inception_v3,
        "type": DNNType.CLASSIFICATION,
        "transform": torchvision.transforms.Compose([
            torchvision.transforms.Resize(OPTIMAL_INPUT_SIZE[INCEPTION_V3]),
            torchvision.transforms.ToTensor(), COMMON_NORMALIZATION])
    },
    # Resnet50 - github.com/pytorch/vision/tree/1de53bef2d04c34544655b0c1b6c55a25e4739fe/references/classification
    # https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/
    RESNET_50: {
        "model": torchvision.models.resnet50,
        "type": DNNType.CLASSIFICATION,
        "transform": torchvision.transforms.Compose([
            torchvision.transforms.Resize(OPTIMAL_INPUT_SIZE[RESNET_50]),
            torchvision.transforms.ToTensor(), COMMON_NORMALIZATION])
    },

    # https://github.com/lukemelas/EfficientNet-PyTorch/blob/
    # 1039e009545d9329ea026c9f7541341439712b96/efficientnet_pytorch/utils.py#L562-L564
    EFFICIENT_NET_B0: {
        "model": torchvision.models.efficientnet_b0,
        "type": DNNType.CLASSIFICATION,
        # Channels, height, width
        "transform": torchvision.transforms.Compose([
            torchvision.transforms.Resize(OPTIMAL_INPUT_SIZE[EFFICIENT_NET_B0],
                                          interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(), COMMON_NORMALIZATION])
    },
    EFFICIENT_NET_B3: {
        "model": torchvision.models.efficientnet_b3,
        "type": DNNType.CLASSIFICATION,
        # Channels, height, width
        "transform": torchvision.transforms.Compose([
            torchvision.transforms.Resize(OPTIMAL_INPUT_SIZE[EFFICIENT_NET_B3],
                                          interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(), COMMON_NORMALIZATION])
    },
    # Object detection, segmentation, and keypoint
    RETINA_NET_RESNET_FPN50: {
        "model": torchvision.models.detection.retinanet_resnet50_fpn,
        "transform": torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
        "type": DNNType.DETECTION
    },

    FASTER_RCNN_RESNET_FPN50: {
        "model": torchvision.models.detection.fasterrcnn_resnet50_fpn,
        "transform": torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
        "type": DNNType.DETECTION
    },
}


def is_not_close(rhs: torch.Tensor, lhs: torch.Tensor, threshold: float) -> torch.Tensor:
    """ Function to be equivalent to TensorFlow ApproximateEqual """
    return torch.greater(torch.abs(torch.subtract(rhs, lhs)), threshold)


def equal(rhs: torch.Tensor, lhs: torch.Tensor, threshold: float = None) -> bool:
    if threshold:
        return bool(torch.all(torch.le(torch.abs(torch.subtract(rhs, lhs)), threshold)))
    else:
        return bool(torch.equal(rhs, lhs))


def copy_tensor_to_cpu(x_tensor):
    return x_tensor.to("cpu")


def compare_output_with_gold(dnn_output_tensor: torch.tensor, dnn_golden_tensor: torch.tensor, dnn_type: DNNType,
                             setup_iteration: int, batch_iteration: int, current_image_names: list,
                             output_logger: logging.Logger) -> int:
    output_errors = 0
    if dnn_type == DNNType.CLASSIFICATION:
        output_errors = compare_classification(dnn_output_tensor=dnn_output_tensor, dnn_golden_tensor=dnn_golden_tensor,
                                               setup_iteration=setup_iteration, batch_iteration=batch_iteration,
                                               current_image_names=current_image_names, output_logger=output_logger,
                                               copy_tensor_to_cpu_caller=copy_tensor_to_cpu,
                                               equal_caller=equal)
    elif dnn_type == DNNType.DETECTION:
        # assert len(dnn_output_tensor) == 1 and len(dnn_golden_tensor) != 0
        output_errors = compare_detection(dnn_output_dict=dnn_output_tensor[0], dnn_golden_dict=dnn_golden_tensor[0],
                                          current_image=current_image_names,
                                          output_logger=output_logger, copy_tensor_to_cpu_caller=copy_tensor_to_cpu,
                                          equal_caller=equal)
    dnn_log_helper.log_error_count(output_errors)
    return output_errors


def load_dataset(transforms: torchvision.transforms, image_list_path: str, logger: logging.Logger,
                 batch_size: int, device: str, dnn_type: DNNType) -> Tuple[Union[torch.tensor, list], list]:
    timer = Timer()
    timer.tic()
    images, image_list = load_image_list(image_list_path)
    # Remove the base path
    image_list = list(map(os.path.basename, image_list))
    assert batch_size <= len(image_list), "Batch size must be equal or smaller than img list"
    # THIS IS Necessary as Classification models expect a tensor, and detection expect a list of tensors
    input_tensor = list()
    if dnn_type == DNNType.CLASSIFICATION:
        resized_images = list()
        for image in images:
            resized_images.append(transforms(image))
        input_tensor = torch.stack(resized_images).to(device)
        input_tensor = torch.split(input_tensor, batch_size)
    elif dnn_type == DNNType.DETECTION:
        for im_i in range(0, len(images), batch_size):
            chunk = [transforms(im_to).to(device) for im_to in images[im_i: im_i + batch_size]]
            input_tensor.append(chunk)
    timer.toc()
    logger.debug(f"Input images loaded and resized successfully: {timer}")
    return input_tensor, image_list


def load_model(precision: str, model_loader: callable, device: str) -> torch.nn.Module:
    dnn_model = model_loader(pretrained=True)
    dnn_model = dnn_model.eval()
    # It means that I want to convert the model into FP16, but make sure that it is not quantized
    if precision == "fp16":
        dnn_model = dnn_model.half()
    return dnn_model.to(device)


def verify_network_accuracy(batched_output: torch.tensor, dnn_type: DNNType, img_names: list, ground_truth_csv: str):
    from verify_accuracy import verify_classification_accuracy, verify_detection_accuracy
    if dnn_type == DNNType.CLASSIFICATION:
        if dnn_type == DNNType.CLASSIFICATION:
            pred = list()
            for img, x in zip(img_names, batched_output):
                prob, label = torch.max(x, 1)
                pred.append({"img_name": img, "class_id_predicted": int(label[0])})

            verify_classification_accuracy(pred, ground_truth_csv)
    else:
        pred = list()
        verify_detection_accuracy(pred, ground_truth_csv)


def main():
    # Check the available device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
        output_logger.level = logging.FATAL

    # Set the parameters for the DNN
    model_parameters = DNN_MODELS[model_name]
    dnn_model = load_model(precision=precision, model_loader=model_parameters["model"], device=device)
    dnn_type = model_parameters["type"]
    transform = model_parameters["transform"]

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
    if generate is False:
        timer.tic()
        dnn_gold_tensors = torch.load(gold_path)
        timer.toc()
        output_logger.debug(f"Time necessary to load the golden outputs: {timer}")

    # Start the setup
    dnn_log_helper.start_setup_log_file(framework_name="PyTorch", args_conf=args_conf, model_name=model_name,
                                        max_errors_per_iteration=MAXIMUM_ERRORS_PER_ITERATION, generate=generate)
    # Main setup loop
    setup_iteration = 0
    with torch.no_grad():
        while setup_iteration < iterations:
            total_errors = 0
            # Loop over the input list
            for batch_i, batched_input in enumerate(input_list):
                batch_iteration = batch_i * batch_size
                current_image_names = image_names[batch_iteration:batch_iteration + batch_size]
                timer.tic()
                dnn_log_helper.start_iteration()
                current_output = dnn_model(batched_input)
                dnn_log_helper.end_iteration()
                # show_classification_result(batch_input=current_output, image_list=current_image_names)

                timer.toc()
                kernel_time = timer.diff_time
                # Then compare the golden with the output
                timer.tic()
                errors = 0
                if generate is False:
                    errors = compare_output_with_gold(dnn_output_tensor=current_output,
                                                      dnn_golden_tensor=dnn_gold_tensors[batch_i],
                                                      dnn_type=dnn_type, setup_iteration=setup_iteration,
                                                      batch_iteration=batch_iteration, output_logger=output_logger,
                                                      current_image_names=current_image_names)
                else:
                    assert len(current_output) == batch_size, str(current_output)
                    dnn_gold_tensors.append(current_output.to(device))

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
                dnn_model = load_model(precision=precision, model_loader=model_parameters["model"], device=device)
                input_list, image_names = load_dataset(transforms=transform, image_list_path=image_list_path,
                                                       logger=output_logger, batch_size=batch_size, device=device,
                                                       dnn_type=dnn_type)

            setup_iteration += 1
    if generate:
        timer.tic()
        # dnn_gold_tensors = torch.stack(dnn_gold_tensors).to("cpu")
        torch.save(dnn_gold_tensors, gold_path)
        timer.toc()
        output_logger.debug(f"Time necessary to save the golden outputs: {timer}")
        output_logger.debug(f"Accuracy measure")
        verify_network_accuracy(batched_output=dnn_gold_tensors, ground_truth_csv=args.grtruthcsv, dnn_type=dnn_type,
                                img_names=image_names)

    # finish the logfile
    dnn_log_helper.end_log_file()


if __name__ == '__main__':
    main()

####################################################################################################################
# For the future
# "MaskR-CNNResNet-50FPN": {
#     "model": torchvision.models.detection.maskrcnn_resnet50_fpn,
#     "type": DNNType.DETECTION_SEGMENTATION
# },
# "KeypointR-CNNResNet-50FPN": {
#     "model": torchvision.models.detection.keypointrcnn_resnet50_fpn,
#     "type": DNNType.DETECTION_KEYPOINT
# },
#        SSD_MOBILENET_V2 : {
#         "model": torchvision.models.detection.ssd300_vgg16,
#         "type": DNNType.DETECTION
#     },
#
#     EFFICIENT_DET_LITE3: {
#         "model": torchvision.models.detection.ssd300_vgg16,
#         "type": DNNType.DETECTION
#     },
