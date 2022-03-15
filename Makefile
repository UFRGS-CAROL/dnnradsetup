
IMGLIST = data/coco2017_img_list.txt
MODEL = ResNet-50
PRECISION = fp32
ITERATIONS = 10
GOLDPATH = ./gold.pt

all: generate_pytorch test_pytorch

generate_pytorch:
	./pytorch_dnns.py --model $(MODEL) --precision $(PRECISION) \
					  --imglist $(IMGLIST) --goldpath $(GOLDPATH) --generate

test_pytorch:
	./pytorch_dnns.py --model $(MODEL) --precision $(PRECISION) \
					  --imglist $(IMGLIST) --goldpath $(GOLDPATH)
