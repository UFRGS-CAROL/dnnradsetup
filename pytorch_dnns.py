#!/usr/bin/python3.8

"""
Main file for Pytorch DNNs setup
"""
import logging
import os

import torch
import torchvision

import console_logger
import dnn_log_helper as dnn_log_helper
from common_tf_and_pt import DNNType, INCEPTION_V3, RESNET_50, load_image_list
from common_tf_and_pt import INCEPTION_B7, RETINA_NET_RESNET_FPN50, FASTER_RCNN_RESNET_FPN50
from common_tf_and_pt import parse_args, Timer

DNN_MODELS = {
    # Check the spreadsheet to more information
    # Inception V3 - Config: https://pytorch.org/hub/pytorch_vision_inception_v3/
    INCEPTION_V3: {
        "model": torchvision.models.inception_v3,
        "type": DNNType.CLASSIFICATION,
        "transform": torchvision.transforms.Compose([
            torchvision.transforms.Resize(299), torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    },
    # Resnet50 - github.com/pytorch/vision/tree/1de53bef2d04c34544655b0c1b6c55a25e4739fe/references/classification
    # https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/
    RESNET_50: {
        "model": torchvision.models.resnet50,
        "type": DNNType.CLASSIFICATION,
        "transform": torchvision.transforms.Compose([
            torchvision.transforms.Resize(256), torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    },

    # https://github.com/lukemelas/EfficientNet-PyTorch/blob/
    # 1039e009545d9329ea026c9f7541341439712b96/efficientnet_pytorch/utils.py#L562-L564
    INCEPTION_B7: {
        "model": torchvision.models.efficientnet_b7,
        "type": DNNType.CLASSIFICATION,
        # Channels, height, width
        "transform": torchvision.transforms.Compose([
            torchvision.transforms.Resize(600, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    },
    # Object detection, segmentation, and keypoint
    RETINA_NET_RESNET_FPN50: {
        "model": torchvision.models.detection.retinanet_resnet50_fpn,
        "transform": NotImplementedError(),
        "type": DNNType.DETECTION
    },

    FASTER_RCNN_RESNET_FPN50: {
        "model": torchvision.models.detection.fasterrcnn_resnet50_fpn,
        "transform": NotImplementedError(),
        "type": DNNType.DETECTION
    },
}


def compare_output_with_gold(dnn_output_tensors: torch.tensor,
                             dnn_golden_tensors: torch.tensor,
                             dnn_type: DNNType, batch_size: int, batch_iteration: int) -> int:
    output_errors = 0
    print(dnn_output_tensors)
    print(dnn_golden_tensors)
    if dnn_type == DNNType.CLASSIFICATION:
        pass
    elif dnn_type == DNNType.DETECTION:
        pass
    else:
        pass
    return output_errors


def load_input_images_to_tensor(transforms: torchvision.transforms, image_list_path: str, logger: logging.Logger,
                                batch_size: int, device: str) -> torch.tensor:
    timer = Timer()
    timer.tic()
    images = load_image_list(image_list_path)
    resized_images = list()
    for image in images:
        resized_images.append(transforms(image))
    print(resized_images[0])
    input_tensor = torch.stack(resized_images)

    input_tensor = torch.split(input_tensor, batch_size).to(device)
    timer.toc()
    logger.info(f"Input images loaded and resized successfully: {timer}")
    return input_tensor


def load_model(precision: str, model_loader: callable, device: str) -> torch.nn.Module:
    dnn_model = model_loader(pretrained=True)
    dnn_model.eval()
    # It means that I want to convert the model into FP16, but make sure that it is not quantized
    if precision == "fp16":
        dnn_model = dnn_model.half()
    return dnn_model.to(device)


def main():
    # Check the available device
    device, batch_size = "cpu", 1
    if torch.cuda.is_available():
        device = "cuda:0"
        batch_size = 1
    timer = Timer()
    timer.tic()
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    output_logger = console_logger.ColoredLogger(main_logger_name)
    args, args_conf = parse_args()
    generate = args.generate
    iterations = args.iterations
    image_list_path = args.imglist
    gold_path = args.goldpath
    precision = args.precision
    model_name = args.model

    # Set the parameters for the DNN
    model_parameters = DNN_MODELS[model_name]
    dnn_model = load_model(precision=precision, model_loader=model_parameters["model"], device=device)
    dnn_type = model_parameters["type"]
    transform = model_parameters["transform"]

    timer.toc()
    output_logger.debug(f"Time necessary to load the model and config it: {timer}")

    # First step is to load the inputs in the memory
    timer.tic()
    input_list = load_input_images_to_tensor(transforms=transform, image_list_path=image_list_path,
                                             logger=output_logger, batch_size=batch_size, device=device)
    input_list = input_list.to(device)
    timer.toc()
    output_logger.debug(f"Time necessary to load the inputs: {timer}")

    dnn_gold_tensors = torch.empty(0)
    # Load if it is not a gold generating op
    timer.tic()
    if generate is False:
        dnn_gold_tensors = torch.load(gold_path)
    timer.toc()
    output_logger.debug(f"Time necessary to load the golden outputs: {timer}")

    # Start the setup if it is not generate
    if generate is False:
        dnn_log_helper.start_log_file(bench_name=model_name, header=args_conf)

    # Main setup loop
    setup_iteration = 0
    while setup_iteration <= iterations:
        total_errors = 0
        # Loop over the input list
        for batch_iteration, batched_input in enumerate(input_list):
            timer.tic()
            if generate is False:
                dnn_log_helper.start_iteration()
            current_output = dnn_model(batched_input)
            if generate is False:
                dnn_log_helper.end_iteration()
            timer.toc()
            output_logger.debug(
                f"It:{setup_iteration} - input it: {batch_iteration} time necessary process an inference: {timer}")

            # Then compare the golden with the output
            timer.tic()
            errors = compare_output_with_gold(dnn_output_tensors=current_output, dnn_golden_tensors=dnn_gold_tensors,
                                              dnn_type=dnn_type, batch_size=batch_size, batch_iteration=batch_iteration)
            total_errors += errors
            timer.toc()
            output_logger.debug(
                f"Iteration:{setup_iteration} time necessary compare the gold: {timer} errors: {errors}")
        # Reload after error
        if total_errors != 0:
            del input_list
            del dnn_model
            dnn_model = load_model(precision=precision, model_loader=model_parameters["model"], device=device)
            input_list = load_input_images_to_tensor(transforms=transform, image_list_path=image_list_path,
                                                     logger=output_logger, batch_size=batch_size, device=device)

        setup_iteration += 1
    timer.tic()
    if generate:
        torch.save(dnn_gold_tensors, gold_path)
    timer.toc()
    output_logger.debug(f"Time necessary to save the golden outputs: {timer}")

    # finish the logfile
    if generate is False:
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
