#!/usr/bin/python3

import configparser
import glob
import json
import os.path
import time
from socket import gethostname

from common_tf_and_pt import DNNType
from common_tf_and_pt import INCEPTION_V3, RESNET_50, EFFICIENT_NET_B3, EFFICIENT_NET_B0
from common_tf_and_pt import SSD_MOBILENET_V2, EFFICIENT_DET_LITE3, FASTER_RCNN_RESNET_FPN50

CONFIG_FILE = "/etc/radiation-benchmarks.conf"
DISABLE_CONSOLE_LOGGING = False
BATCH_SIZE = 1
ITERATIONS = int(1e12)
USE_TF_LITE = False
DOWNLOAD_MODELS = False

FRAMEWORKS = {
    "tensorflow": [INCEPTION_V3, RESNET_50, EFFICIENT_NET_B3, EFFICIENT_NET_B0, SSD_MOBILENET_V2],
    "pytorch": [INCEPTION_V3, RESNET_50, EFFICIENT_NET_B3, EFFICIENT_NET_B0, FASTER_RCNN_RESNET_FPN50]
}

DNN_MODELS = {
    INCEPTION_V3: dict(type=DNNType.CLASSIFICATION, support_tflite=True, dataset="imagenet2012"),
    RESNET_50: dict(type=DNNType.CLASSIFICATION, support_tflite=True, dataset="imagenet2012"),
    EFFICIENT_NET_B0: dict(type=DNNType.CLASSIFICATION, support_tflite=True, dataset="imagenet2012"),
    EFFICIENT_NET_B3: dict(type=DNNType.CLASSIFICATION, support_tflite=True, dataset="imagenet2012"),
    # Object detection, segmentation, and keypoint
    SSD_MOBILENET_V2: dict(type=DNNType.DETECTION, support_tflite=True, dataset="coco2017"),
    EFFICIENT_DET_LITE3: dict(type=DNNType.DETECTION, support_tflite=True, dataset="coco2017"),
    FASTER_RCNN_RESNET_FPN50: dict(type=DNNType.DETECTION, support_tflite=False, dataset="coco2017"),
}

TEST_JSONS = False


def main():
    try:
        config = configparser.RawConfigParser()
        config.read(CONFIG_FILE)
        server_ip = config.get('DEFAULT', 'serverip')
    except IOError as e:
        raise IOError("Configuration setup error: " + str(e))

    hostname = gethostname()
    jsons_path = f"data/{hostname}_jsons"
    if os.path.isdir(jsons_path) is False:
        os.mkdir(jsons_path)

    if DOWNLOAD_MODELS:
        print("Download all the models")
        os.system("./download_models.py")
    current_directory = os.getcwd()
    for dnn_model, config_vals in DNN_MODELS.items():
        for framework in FRAMEWORKS:
            if dnn_model in FRAMEWORKS[framework]:
                dnn_type = config_vals["type"]
                dataset = config_vals["dataset"]
                # Default filename will build the other names
                default_file_name = f"{hostname}_config_{framework}_{dnn_model}_"
                default_file_name += f"{dnn_type}_{dataset}_batch_size_{BATCH_SIZE}"
                json_file_name = f"{jsons_path}/{default_file_name}.json"
                file_ext = ".npy"
                if framework == "pytorch":
                    file_ext = ".pt"
                gold_path = f"{current_directory}/data/{default_file_name}" + file_ext

                script_name = f"{framework}_dnns.py"
                dataset_img_list = f"{current_directory}/data/{dataset}_img_list.txt"

                parameters = [
                    f"{current_directory}/{script_name} ",
                    f"--model {dnn_model}",
                    f"--imglist {dataset_img_list}",
                    "--precision fp32",
                    f"--iterations {ITERATIONS}",
                    f"--batchsize {BATCH_SIZE}",
                    "--disableconsolelog",
                    f"--goldpath {gold_path}",
                ]

                if USE_TF_LITE and framework == "tensorflow":
                    parameters += ["--tflite"]

                generate_parameters = parameters + ["--generate"]
                generate_parameters.remove("--disableconsolelog")
                generate_cmd = " ".join(generate_parameters)

                exec_cmd = " ".join(parameters)
                command_list = [{
                    "killcmd": f"pkill -9 -f {script_name}",
                    "exec": exec_cmd,
                    "codename": script_name.replace(".py", ""),
                    "header": " ".join(parameters)
                }]
                # dump json
                with open(json_file_name, "w") as json_fp:
                    json.dump(obj=command_list, fp=json_fp, indent=4)

                print(f"Executing generate for {generate_cmd}")
                if os.system(generate_cmd) != 0:
                    raise OSError(f"Could not execute command {generate_cmd}")

    print("Json creation and golden generation finished")
    print(f"You may run: scp -r {jsons_path} carol@{server_ip}:"
          f"/home/carol/radiation-setup/radiation-setup/machines_cfgs/")


def test_all_jsons(timeout=30):
    hostname = gethostname()
    jsons_path = f"data/{hostname}_jsons"

    for file in glob.glob(rf"{jsons_path}/*.json", recursive=True):
        with open(file, "r") as fp:
            json_data = json.load(fp)

        for v in json_data:
            print("EXECUTING", v["exec"])
            os.system(v['exec'] + "&")
            time.sleep(timeout)
            os.system(v["killcmd"])


if __name__ == "__main__":
    if TEST_JSONS:
        test_all_jsons()
    else:
        main()
