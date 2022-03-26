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
from tensorflow import convert_to_tensor, float32,equal,reduce_all,less_equal,subtract, Tensor, device
from tensorflow.keras.initializers import Constant
import logging
import console_logger
import random


ABS_THRESHOLD = 1e-5

def compare_fast(rhs: Tensor, lhs: Tensor, threshold: float = None) -> bool:
    if threshold:
        return bool(
            reduce_all(less_equal(abs(subtract(rhs, lhs)), threshold)))
    else:
        return bool(reduce_all(equal(rhs, lhs)))

def set_input_n_op_n_gold(input_image,gold_file):

    input = np.load(input_image)

    if len(input.shape) == 2:
        input = input[..., np.newaxis]

    golden = np.load(gold_file)
    return input, golden

def save_output_golden(output, model_file,output_logger):
    np.save(model_file, output)
    output_logger.debug(f"Golden output saved to `{model_file}`")

def check_output_against_golden(output, golden,output_logger):    

    #print(output)
    #if random.randint(0,4)==0:
    #    temp=output.numpy()
    #    temp[0][0][0] += 34.2
    #    output=convert_to_tensor(temp,dtype=float32)
    errors=0
    if(compare_fast(output,golden,ABS_THRESHOLD) == False):
        for i, (out,gold) in enumerate(zip(output[0][0],golden[0][0])):
            #print(out)
            if(out != gold):
                score_error = "i:"+str(i)+f" score:{out[0]:.6e} g:{gold[0]:.6e}"
                output_logger.error(score_error)
                lh.log_error_detail(score_error)
                errors+=1
    return errors


def generate_random_input(input_size,op,output_logger):

    rand_input = np.random.uniform(low=-10,high=10,size=input_size)
    input_file = "input_"+op+"_"+str(input_size[0])+"_"+str(input_size[1])+"_"+str(input_size[2])+"_"+str(input_size[3])
    output_logger.debug("saved input: "+ input_file + ".npy")
    np.save(input_file, rand_input)

    return input_file, rand_input



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-I', '--input', required=True,
                        help='Path to input (..npy)')
    parser.add_argument('-G', '--golden', required=True, default=None,
                        help='Path to golden output file (.npy)')
    parser.add_argument('--iterations', required=False, default=10,
                        help='Number of iterations')   
    parser.add_argument('-O', "--operation" ,required=True,
                        help='Specify type of Operation: (Conv2D | DepthConv2d)')
    parser.add_argument('-k', "--kernel" ,required=True,
                    help='Specify size of kernel: (H,W))')
    parser.add_argument('-kt', "--kernel_type" ,required=True,
                    help='Specify type of kernel: AVG | ONES')
    parser.add_argument('--generate', default=False, action="store_true", help="Set this flag to generate the gold")
    parser.add_argument('-S', '--input-size', required=False,
                    help='Input size when generating (format: H,W)')
    parser.add_argument('--disableconsolelog', default=False, action="store_true",
                    help="Set this flag disable console logging")
    args = parser.parse_args()

    input_image_file = args.input
    golden_file = args.golden
    iterations = int(args.iterations)
    generate=args.generate
    operation = args.operation
    kernel_size = tuple(map(int, args.kernel.split(",")))
    kernel_type = args.kernel_type
    disable_console_logger = args.disableconsolelog
    # Setup log helper
    benchmarkName = "Arm_"+operation
    benchmarkInfo = f"operation: {operation} input_file: {input_image_file} iterations: {iterations}"
    lh.start_setup_log_file("tensorflow", benchmarkInfo,benchmarkName,500,generate)
    lh.set_max_errors_iter(500)
    lh.set_max_infos_iter(1)
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    output_logger = console_logger.ColoredLogger(main_logger_name)
    if disable_console_logger:
        output_logger.level = logging.FATAL
    t1 = time()
    if generate:
        input_size = [1]
        input_size = np.append(input_size,tuple(map(int, args.input_size.split(","))))
        if operation == "DepthConv2d":
            input_size = np.append(input_size, 3)
        elif operation == "Conv2D": 
            input_size = np.append(input_size, 1)
        else:
            raise Exception("invalid op")    
        input_name,input=generate_random_input(input_size,operation,output_logger)

    else:
        input,golden = set_input_n_op_n_gold(input_image_file,golden_file)
    #print(input)
    #print("printed input")
    input = convert_to_tensor(input,np.float32)

    t2 = time()
    #print(f"Load input: {t2 - t1}s")
    #input_shape=input.shape
    #print(input_shape)
    if generate:
        with device('/cpu:0'):
            if operation == "Conv2D":
                if kernel_type == "ONES":
                    output = Conv2D(1, kernel_size, kernel_initializer="Ones")(input)
                elif kernel_type == "AVG":
                    output = Conv2D(1, kernel_size, kernel_initializer=Constant(value=1/(kernel_size[0]*kernel_size[1])))(input)
                else:
                    raise Exception("invalid kernel_type")
            elif operation == "DepthwiseConv2D":
                if kernel_type == "ONES":
                    output=DepthwiseConv2D(kernel_size,depthwise_initializer='Ones')(input)
                elif kernel_type == "AVG":
                     output=DepthwiseConv2D(kernel_size,depthwise_initializer=Constant(value=1/(kernel_size[0]*kernel_size[1])) )(input)
                else:
                    raise Exception("invalid kernel_type")
            golden_file = "gold_"+operation+"_"+str(input_size[0])+"_"+str(input_size[1])+"_"+str(input_size[2])+"_"+str(input_size[3])+"_"+str(kernel_size[0])+"_"+str(kernel_size[1])
            save_output_golden(output, golden_file,output_logger)
            exit(0)


    for i in range(iterations):        
        t2 = time()        
        lh.start_iteration()
        with device('/cpu:0'):
            if operation == "Conv2D":
                if kernel_type == "ONES":
                    output = Conv2D(1, kernel_size, kernel_initializer="Ones")(input)
                elif kernel_type == "AVG":
                    output = Conv2D(1, kernel_size, kernel_initializer=Constant(value=1/(kernel_size[0]*kernel_size[1])) )(input)
                else:
                    raise Exception("invalid kernel_type")
            elif operation == "DepthwiseConv2D":
                if kernel_type == "ONES":
                    output=DepthwiseConv2D(kernel_size,depthwise_initializer='Ones')(input)
                elif kernel_type == "AVG":
                     output=DepthwiseConv2D(kernel_size,depthwise_initializer=Constant(value=1/(kernel_size[0]*kernel_size[1])))(input)
                else:
                    raise Exception("invalid kernel_type")
            else:
                raise Exception("invalid kernel_type")
            #print(output.shape)
            lh.end_iteration()
        t3 = time()
        output_logger.debug(f"Run interpreter: {t3 - t2}s")     
        t4 = time()
        errors = check_output_against_golden(output, golden,output_logger)
        t5 = time()
        output_logger.debug(f"Check output: {t5 - t4}s - {errors} error(s)")
        lh.log_error_count(int(errors))


if __name__ == "__main__":
    main()