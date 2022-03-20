#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
from enum import Enum
from time import time
from PIL import Image
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
import dnn_log_helper as lh
from tensorflow import convert_to_tensor
from tensorflow.keras.initializers import Constant

def set_input_n_op(input_image):

    input = np.load(input_image)

    if len(input.shape) == 2:
        input = input[..., np.newaxis]
    return input

def save_output_golden(output, model_file):
    np.save(model_file, output)
    print(f"Golden output saved to `{model_file}`")

def check_output_against_golden(output, golden_file):    
    if os.path.isfile(golden_file):
        golden = np.load(golden_file)
        errors = np.sum(output != golden)
        return errors
    else:
        raise FileNotFoundError

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-I', '--input', required=True,
                        help='Path to input (..npy)')
    parser.add_argument('-G', '--golden', required=True, default=None,
                        help='Path to golden output file (.npy)')
    parser.add_argument('--iterations', required=False, default=10,
                        help='Number of iterations')
    parser.add_argument('--save-golden', default=False, action='store_true',
                        help='Whether the output should be saved to a binary file in .npy format or not')
    parser.add_argument('-O', "--operation" ,required=True,
                        help='Specify type of Operation: (Conv2D | DepthConv2d)')
    parser.add_argument('-k', "--kernel" ,required=True,
                    help='Specify size of kernel: (H,W))')
    parser.add_argument('-kt', "--kernel_type" ,required=True,
                    help='Specify type of kernel: AVG | ONES')
    args = parser.parse_args()

    input_image_file = args.input
    golden_file = args.golden
    iterations = int(args.iterations)

    operation = args.operation
    kernel_size = tuple(map(int, args.kernel.split(",")))
    kernel_type = args.kernel_type
    # Setup log helper
    benchmarkName = "Arm"+operation
    benchmarkInfo = f"operation: {operation} input_file: {input_image_file} golden_file: {golden_file} iterations: {iterations}"
    lh.start_setup_log_file("tensorflow", benchmarkInfo,benchmarkName,500,args.save_golden)
    lh.set_max_errors_iter(500)
    lh.set_max_infos_iter(1)

    t1 = time()
    input = set_input_n_op(input_image_file)
    input = convert_to_tensor(input,np.float32)
    t2 = time()
    print(f"Load input: {t2 - t1}s")
    input_shape=input.shape
    print(input_shape)
    if args.save_golden:
        if operation == "Conv2D":
            if kernel_type == "ONES":
                output = Conv2D(2, kernel_size, kernel_initializer="Ones")(input)
            elif kernel_type == "AVG":
                output = Conv2D(2, kernel_size, kernel_initializer=Constant(value=1/(kernel_size[0]*kernel_size[1])))(input)
            else:
                raise Exception("invalid kernel_type")
        elif operation == "DepthwiseConv2D":
            if kernel_type == "ONES":
                y=DepthwiseConv2D(kernel_size,depthwise_initializer='Ones')(input)
            elif kernel_type == "AVG":
                 y=DepthwiseConv2D(kernel_size,depthwise_initializer=Constant(value=1/(kernel_size[0]*kernel_size[1])) )(input)
            else:
                raise Exception("invalid kernel_type")
        save_output_golden(output, golden_file)
        exit(0)
    

    for i in range(iterations):        
        t2 = time()        
        lh.start_iteration()
        if operation == "Conv2D":
            if kernel_type == "ONES":
                output = Conv2D(2, kernel_size, kernel_initializer="Ones")(input)
            elif kernel_type == "AVG":
                output = Conv2D(2, kernel_size, kernel_initializer=Constant(value=1/(kernel_size[0]*kernel_size[1])) )(input)
            else:
                raise Exception("invalid kernel_type")
        elif operation == "DepthwiseConv2D":
            if kernel_type == "ONES":
                y=DepthwiseConv2D(kernel_size,depthwise_initializer='Ones')(input)
            elif kernel_type == "AVG":
                 y=DepthwiseConv2D(kernel_size,depthwise_initializer=Constant(value=1/(kernel_size[0]*kernel_size[1])))(input)
            else:
                raise Exception("invalid kernel_type")
        else:
            raise Exception("invalid kernel_type")
        lh.end_iteration()
        t3 = time()
        print(f"Run interpreter: {t3 - t2}s")     
        t4 = time()
        if golden_file is None:
            try:
                errors = check_output_against_golden(output, golden_file)
                t5 = time()
                print(f"Check output: {t5 - t4}s - {errors} error(s)")
                lh.log_error_count(int(errors))
            except: pass
        else:
            try:
                errors = check_output_against_golden(output, golden_file)
                t5 = time()
                print(f"Check output: {t5 - t4}s - {errors} error(s)")
                lh.log_error_count(int(errors))
            except FileNotFoundError:
                print(f"Could not open golden file `{golden_file}`")
                lh.log_error_count(f"Could not open golden file `{golden_file}`")

if __name__ == "__main__":
    main()