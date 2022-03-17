CLASSIFY = 1
DISABLE_CONSOLE_LOGGING = 0

ifeq ($(CLASSIFY), 1)
DATASET = imagenet2012
MODEL = ResNet-50
else
DATASET = coco2017
MODEL = RetinaNetResNet-50FPN
endif

ifeq ($(DISABLE_CONSOLE_LOGGING), 1)
CONSOLE_LOGGING = --disableconsolelog
endif

IMGLIST = data/$(DATASET)_img_list.txt

PRECISION = fp32
ITERATIONS = 10
GOLDPATH = ./gold_$(MODEL)_$(PRECISION)_$(DATASET).pt

all: generate_pytorch test_pytorch

generate_pytorch:
	./pytorch_dnns.py --model $(MODEL) --precision $(PRECISION) $(CONSOLE_LOGGING) \
					  --imglist $(IMGLIST) --goldpath $(GOLDPATH) --generate

test_pytorch:
	./pytorch_dnns.py --model $(MODEL) --precision $(PRECISION) --iterations $(ITERATIONS) $(CONSOLE_LOGGING) \
					  --imglist $(IMGLIST) --goldpath $(GOLDPATH)
generate_tensorflow:
	./tensorflow_dnns.py --model $(MODEL) --precision $(PRECISION) $(CONSOLE_LOGGING) \
                                          --imglist $(IMGLIST) --goldpath $(GOLDPATH) --generate
