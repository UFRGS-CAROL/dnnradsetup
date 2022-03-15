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
from common_tf_and_pt import DNNType, INCEPTION_V3, RESNET_50, BATCH_SIZE_GPU
from common_tf_and_pt import INCEPTION_B7, RETINA_NET_RESNET_FPN50, FASTER_RCNN_RESNET_FPN50
from common_tf_and_pt import parse_args, Timer, load_image_list

DNN_MODELS = {
    # Check the spreadsheet to more information
    # Inception V3 - Config: https://pytorch.org/hub/pytorch_vision_inception_v3/
    INCEPTION_V3: {
        "model": torchvision.models.inception_v3,
        "type": DNNType.CLASSIFICATION,
        "transform": torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)), torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    },
    # Resnet50 - github.com/pytorch/vision/tree/1de53bef2d04c34544655b0c1b6c55a25e4739fe/references/classification
    # https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/
    RESNET_50: {
        "model": torchvision.models.resnet50,
        "type": DNNType.CLASSIFICATION,
        "transform": torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)), torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    },

    # https://github.com/lukemelas/EfficientNet-PyTorch/blob/
    # 1039e009545d9329ea026c9f7541341439712b96/efficientnet_pytorch/utils.py#L562-L564
    INCEPTION_B7: {
        "model": torchvision.models.efficientnet_b7,
        "type": DNNType.CLASSIFICATION,
        # Channels, height, width
        "transform": torchvision.transforms.Compose([
            torchvision.transforms.Resize((600, 600), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
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


def compare_output_with_gold(dnn_output_tensor: torch.tensor, dnn_golden_tensor: torch.tensor, dnn_type: DNNType,
                             batch_size: int, setup_iteration: int, batch_iteration: int) -> int:
    output_errors = 0
    if dnn_type == DNNType.CLASSIFICATION:
        print(dnn_golden_tensor)
        print(dnn_output_tensor)
    elif dnn_type == DNNType.DETECTION:
        print(dnn_golden_tensor)
        print(dnn_output_tensor)
    else:
        raise NotImplementedError("Only CLASSIFICATION AND DETECTION SUPPORTED FOR NOW")
    return output_errors


def load_dataset(transforms: torchvision.transforms, image_list_path: str, logger: logging.Logger,
                 batch_size: int, device: str) -> torch.tensor:
    timer = Timer()
    timer.tic()
    images = load_image_list(image_list_path)
    resized_images = list()
    for image in images:
        resized_images.append(transforms(image))
    input_tensor = torch.stack(resized_images).to(device)
    input_tensor = torch.split(input_tensor, batch_size)
    timer.toc()
    logger.debug(f"Input images loaded and resized successfully: {timer}")
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
    device, batch_size = "cpu", 5
    if torch.cuda.is_available():
        device = "cuda:0"
        batch_size = BATCH_SIZE_GPU
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

    # Set the parameters for the DNN
    model_parameters = DNN_MODELS[model_name]
    dnn_model = load_model(precision=precision, model_loader=model_parameters["model"], device=device)
    dnn_type = model_parameters["type"]
    transform = model_parameters["transform"]

    timer.toc()
    output_logger.debug(f"Time necessary to load the model and config it: {timer}")

    # First step is to load the inputs in the memory
    timer.tic()
    input_list = load_dataset(transforms=transform, image_list_path=image_list_path,
                              logger=output_logger, batch_size=batch_size, device=device)
    timer.toc()
    output_logger.debug(f"Time necessary to load the inputs: {timer}")

    dnn_gold_tensors = list()
    # Load if it is not a gold generating op
    timer.tic()
    if generate is False:
        dnn_gold_tensors = torch.load(gold_path)

    timer.toc()
    output_logger.debug(f"Time necessary to load the golden outputs: {timer}")

    # Start the setup if it is not generate
    if generate:
        dnn_log_helper.disable_logging()
    dnn_log_helper.start_log_file(bench_name=model_name, header=args_conf)

    # Main setup loop
    setup_iteration = 0
    with torch.no_grad():
        while setup_iteration < iterations:
            total_errors = 0
            # Loop over the input list
            for batch_iteration, batched_input in enumerate(input_list):
                timer.tic()
                dnn_log_helper.start_iteration()
                current_output = dnn_model(batched_input)
                dnn_log_helper.end_iteration()
                timer.toc()
                iteration_out = f"It:{setup_iteration} - input it:{batch_iteration}"
                iteration_out += f" inference time:{timer}"

                # Then compare the golden with the output
                timer.tic()
                errors = 0
                if generate is False:
                    current_gold = dnn_gold_tensors[batch_iteration]
                    errors = compare_output_with_gold(dnn_output_tensor=current_output, dnn_golden_tensor=current_gold,
                                                      dnn_type=dnn_type, batch_size=batch_size,
                                                      setup_iteration=setup_iteration, batch_iteration=batch_iteration)
                else:
                    assert len(current_output) == batch_size, str(current_output)
                    dnn_gold_tensors.append(current_output)

                total_errors += errors
                timer.toc()
                iteration_out += f", gold compare time:{timer} errors:{errors}"
                output_logger.debug(iteration_out)

            # Reload after error
            if total_errors != 0:
                del input_list
                del dnn_model
                dnn_model = load_model(precision=precision, model_loader=model_parameters["model"], device=device)
                input_list = load_dataset(transforms=transform, image_list_path=image_list_path,
                                          logger=output_logger, batch_size=batch_size, device=device)

            setup_iteration += 1
    timer.tic()
    if generate:
        # dnn_gold_tensors = torch.stack(dnn_gold_tensors)
        torch.save(dnn_gold_tensors, gold_path)
    timer.toc()
    output_logger.debug(f"Time necessary to save the golden outputs: {timer}")

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
