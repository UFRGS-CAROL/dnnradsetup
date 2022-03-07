#!/usr/bin/python3.8

"""
Main file for Pytorch DNNs setup
"""
import argparse
import os
import time

import torch
import torchvision

import console_logger
from dnns_configs import DNN_MODELS, DNNType
from libLogHelper.build import log_helper


def parse_args():
    """ Parse the args and return a args.Namespace and the tostring from the args    """
    parser = argparse.ArgumentParser(description='PyTorch DNN radiation setup')
    parser.add_argument('--model', default=list(DNN_MODELS.keys())[0],
                        help=f'Network name. It can be ' + ', '.join(DNN_MODELS.keys()))
    parser.add_argument('--dataset', default='imagenet', help='Dataset that will serve as input')
    parser.add_argument('--datadir', default='./data', help='Path to dataset.')
    parser.add_argument('--subsetsize', default=10,
                        help="To avoid load the whole dataset from the memory set the number of images to process by"
                             " --subsetsize <num>. Default is 10", type=int)
    parser.add_argument('--precision', default="fp32", help="Precision of the network, can be fp16 and fp32")
    parser.add_argument('--iterations', default=int(1e12), help="Iterations to run forever", type=int)
    parser.add_argument('--generate', default=False, help="Set this flag to generate the gold", type=bool)
    args = parser.parse_args()
    # Check if the model is correct
    if args.model not in DNN_MODELS:
        parser.print_help()
        raise ValueError("Not the correct model")

    # Check if it is only to generate the gold values
    if args.generate:
        args.iterations = 1

    args_text = " ".join([f"{k}={v}" for k, v in vars(args).items()])
    return args, args_text


def compare_output_with_gold(dnn_output_tensors: torch.tensor,
                             dnn_gold_list: torch.tensor, dnn_type: DNNType):
    if dnn_type == DNNType.CLASSIFICATION:
        raise NotImplementedError
    elif dnn_type == DNNType.DETECTION:
        raise NotImplementedError
    else:
        raise NotImplementedError


def load_input_images_to_tensor(data_path: str, device: str,
                                transforms: torchvision.transforms) -> torch.tensor:
    input_tensor_list = list()

    input_tensor_list = torch.tensor(input_tensor_list)
    return input_tensor_list


def inference_on_a_batched_input(batch_input: torch.tensor, dnn_type: DNNType) -> torch.tensor:
    time.sleep(1)
    if dnn_type == DNNType.CLASSIFICATION:
        raise NotImplementedError
    elif dnn_type == DNNType.DETECTION:
        raise NotImplementedError
    else:
        raise NotImplementedError


class Timer:
    time_measure = 0

    def tic(self): self.time_measure = time.time()

    def toc(self): self.time_measure = time.time() - self.time_measure

    def __str__(self): return f"{self.time_measure:.4f}s"


def main():
    timer = Timer()
    timer.tic()
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    output_logger = console_logger.ColoredLogger(main_logger_name)
    args, args_conf = parse_args()
    generate = args.generate
    iterations = args.iterations
    # Set the parameters for the DNN
    model_parameters = DNN_MODELS[args.model]
    model_loader_function = model_parameters["model"]
    dnn_model = model_loader_function(pretrained=True)
    dnn_model.eval()
    dnn_type = model_parameters["type"]
    transform = model_parameters["transform"]
    input_size = model_parameters["input_size"]
    # It means that I want to convert the model into FP16, but make sure that it is not quantized
    if args.precision == "fp16":
        dnn_model = dnn_model.half()
    timer.toc()
    output_logger.debug(f"Time necessary to load the model and config it: {timer}")

    # Check the available device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # First step is to load the inputs in the memory
    timer.tic()
    input_list = load_input_images_to_tensor(data_path=args.datapath, device=device, transforms=transform)
    timer.toc()
    output_logger.debug(f"Time necessary to load the inputs: {timer}")

    dnn_gold_tensors = torch.empty(0)
    gold_path = f"{args.model}_{args.dataset}_{dnn_type}_{args.precision}.pt"

    # Load if it is not a gold generating op
    timer.tic()
    if generate is False:
        dnn_gold_tensors = torch.load(gold_path)
    timer.toc()
    output_logger.debug(f"Time necessary to load the golden outputs: {timer}")

    # Start the setup if it is not generate
    if generate is False:
        log_helper.start_log_file(benchmark_name=args.model, test_info=args_conf)

    # Main setup loop
    for setup_iteration in range(iterations):

        # Loop over the input list
        # TODO: Modify it to load the batch according to the architecture capabilities
        for batched_input in input_list:
            timer.tic()
            if generate is False:
                log_helper.start_iteration()
            current_output = inference_on_a_batched_input(batch_input=batched_input, dnn_type=dnn_type)
            if generate is False:
                log_helper.end_iteration()
            timer.toc()
            output_logger.debug(f"Iteration:{setup_iteration} time necessary process an inference: {timer}")

            # Then compare the golden with the output
            timer.tic()
            compare_output_with_gold(dnn_output_tensors=current_output,
                                     dnn_gold_list=dnn_gold_tensors,
                                     dnn_type=dnn_type)
            timer.toc()
            output_logger.debug(f"Iteration:{setup_iteration} time necessary compare the gold: {timer}")

    timer.tic()
    if args.generate:
        torch.save(dnn_gold_tensors, gold_path)
    timer.toc()
    output_logger.debug(f"Time necessary to save the golden outputs: {timer}")

    # finish the logfile
    if generate is False:
        log_helper.end_log_file()


if __name__ == '__main__':
    main()
