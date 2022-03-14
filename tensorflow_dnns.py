import os

import console_logger
from common_tf_and_pt import DNNType, INCEPTION_V3, RESNET_50
from common_tf_and_pt import INCEPTION_B7, RETINA_NET_RESNET_FPN50, FASTER_RCNN_RESNET_FPN50
from common_tf_and_pt import parse_args, Timer

DNN_MODELS = {
    INCEPTION_V3: {
        "model": None,
        "type": DNNType.CLASSIFICATION,
        "transform": None
    },
    RESNET_50: {
        "model": None,
        "type": DNNType.CLASSIFICATION,
        "transform": None
    },
    INCEPTION_B7: {
        "model": None,
        "type": DNNType.CLASSIFICATION,
        "transform": None
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
    device, batch_size = "cpu", 1

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


if __name__ == '__main__':
    main()
