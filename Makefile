
DATASET = imagenet2012
IMGLIST = data/$(DATASET)_img_list.txt
MODEL = ResNet-50
PRECISION = fp32
ITERATIONS = 10
GOLDPATH = ./gold_$(MODEL)_$(PRECISION)_$(DATASET).pt

all: generate_pytorch test_pytorch

generate_pytorch:
	./pytorch_dnns.py --model $(MODEL) --precision $(PRECISION) \
					  --imglist $(IMGLIST) --goldpath $(GOLDPATH) --generate

test_pytorch:
	./pytorch_dnns.py --model $(MODEL) --precision $(PRECISION) \
					  --imglist $(IMGLIST) --goldpath $(GOLDPATH)
