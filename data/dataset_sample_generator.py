import os
import random
import shutil
from PIL import Image
DATA_SET_SIZE = 100


def generate_sample(root_dir, output_dir, img_list_path):
    files_list = list()
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            files_list.append(os.path.join(subdir, file))

    reduced_dataset = list()
    if os.path.exists(output_dir) is True:
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    i = 0
    while i < DATA_SET_SIZE:
        chosen = random.choice(files_list)
        im = Image.open(chosen)
        if "RGB" in im.mode:
            reduced_dataset.append(chosen)
            shutil.copy(chosen, output_dir)
            i += 1
    with open(output_dir + img_list_path, "w") as fp:
        for line in reduced_dataset:
            fp.write(line + "\n")


def main():
    imagenet_path = "/home/fernando/git_research/dnnradsetup/data/imagenet/validation"
    imagenet_output = "/home/fernando/git_research/dnnradsetup/data/imagenet/subset/"
    generate_sample(imagenet_path, imagenet_output, "../../imagenet2012_img_list.txt")
    coco_path = "/home/fernando/git_research/dnnradsetup/data/coco2017/val2017/"
    coco_output = "/home/fernando/git_research/dnnradsetup/data/coco2017/subset/"
    generate_sample(coco_path, coco_output, "../../coco2017_img_list.txt")


if __name__ == '__main__':
    main()
