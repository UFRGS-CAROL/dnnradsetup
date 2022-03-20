# DNNRadSetup

Deep Neural network Radiation experiment setup

# Requirements

- numpy>=1.21.5
- torch>=1.10.1
- torchvision>=0.11.2
- Pillow~=9.0.0
- [liLogHelper](https://github.com/UFRGS-CAROL/libLogHelper)

# Example of gold generation for Resnet50

```shell
./pytorch_dnns.py --model ResNet-50 --precision fp32 --imglist data/imagenet2012_img_list.txt --goldpath ./gold_ResNet-50_fp32_imagenet2012_btsz_1.pt --batchsize 1 --generate
```

# Evaluating for 10 iterations

```shell
./pytorch_dnns.py --model ResNet-50 --precision fp32  --imglist data/imagenet2012_img_list.txt --goldpath ./gold_ResNet-50_fp32_imagenet2012_btsz_1.pt --batchsize 1 --iterations 10
```