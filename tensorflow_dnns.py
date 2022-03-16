#!/usr/bin/python3.8



import os
import tensorflow
import console_logger
from common_tf_and_pt import DNNType, INCEPTION_V3, RESNET_50, SSD_MOBILENET_V2, EFFICIENT_DET_LITE3, BATCH_SIZE_GPU
from common_tf_and_pt import INCEPTION_B7, RETINA_NET_RESNET_FPN50, FASTER_RCNN_RESNET_FPN50
from common_tf_and_pt import parse_args, Timer, load_image_list
from tensorflow.keras.applications import ResNet50, InceptionV3, efficientnet
from tensorflow.keras.applications import resnet50.preprocess_input, inception_v3.preprocess_input
from tensorflow.keras.applications.efficientnet import EfficientNetB7
import numpy as np


def load_dataset(transforms, image_list_path: str, logger: logging.Logger,
                 batch_size: int, device: str) -> torch.tensor:
    timer = Timer()
    timer.tic()
    images = load_image_list(image_list_path)
    resized_images = list()
    for image in images:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = transforms(x)
        resized_images.append(x)
    timer.toc()
    logger.debug(f"Input images loaded and resized successfully: {timer}")
    return input_tensor




DNN_MODELS = {
    INCEPTION_V3: {
        "model": InceptionV3,
        "type": DNNType.CLASSIFICATION,
        "transform": inception_v3.preprocess_input
    },
    RESNET_50: {
        "model": ResNet50,
        "type": DNNType.CLASSIFICATION,
        "transform": resnet50.preprocess_input
    },
    INCEPTION_B7: {
        "model": None,
        "type": DNNType.CLASSIFICATION,
        "transform": none
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
    # Object detection, segmentation, and keypoint
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


def main():
    # Check the available device
    device, batch_size = "/device:cpu:0", 1
    if tensorflow.test.is_gpu_available(cuda_only=True):
        device = "/device:gpu:0"
        batch_size = BATCH_SIZE_GPU

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
    model_parameters=DNN_MODELS[model_name]
    model=model_parameters["model"](weights='imagenet')
    dnn_type = model_parameters["type"]
    transform = model_parameters["transform"]
    input_list=load_dataset(transforms=transform, image_list_path=image_list_path,
                              logger=output_logger, batch_size=batch_size, device=device)
    # Main setup loop
    setup_iteration = 0
    while setup_iteration < iterations:
        total_errors = 0
        # Loop over the input list
        for batch_iteration, batched_input in enumerate(input_list):
            timer.tic()
            dnn_log_helper.start_iteration()
            current_output = model(batched_input)
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
            del model
            model = model=model_parameters["model"](weights='imagenet')
            input_list = load_dataset(transforms=transform, image_list_path=image_list_path,
                                      logger=output_logger, batch_size=batch_size, device=device)

        setup_iteration += 1
    with tensorflow.device(device_name=device):
        pass


if __name__ == '__main__':
    main()
