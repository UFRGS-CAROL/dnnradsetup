#!/usr/bin/env python3

import os
import json
import logging
import argparse
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tensorflow as tf



np.random.seed(0)

log = logging.getLogger("OpModelCreator")
log.setLevel(logging.INFO)
np.random.seed(0)
    
def generate_random_input(output_size,op):

    rand_input = np.random.random(output_size)
    output_file = "input_"+op+"_"+str(output_size[0])+"_"+str(output_size[1])+"_"+str(output_size[2])+"_"+str(output_size[3])

    np.save(output_file, rand_input)

    return output_file, rand_input


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-O', '--operation', required=True,
                        help='Operation: DEPTHWISE_CONV_2D | CONV_2D')
    parser.add_argument('-I', '--input-size', required=True,
                        help='Input size (format: H,W)')


    args = parser.parse_args()

    op = args.operation.lower()
    input_size = [1]

    input_size = np.append(input_size,tuple(map(int, args.input_size.split(","))))

    print("")
    print(f"Input size: {input_size}")


    if op == "depthwise_conv_2d":

        input_size = np.append(input_size, 3)
    elif op == "conv_2d": 
        input_size = np.append(input_size, 1)
    else:
        raise Exception("invalid op")    
    out_file, out_arr = generate_random_input(input_size,op)

    zero_count = np.sum(out_arr == 0)
    print(f'Generated input saved to `{out_file}` with dimensions {out_arr.shape} and {zero_count}')

if __name__ == "__main__":
    main()