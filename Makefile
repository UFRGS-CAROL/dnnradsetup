# Just for debug purposes
CLASSIFY = 1
DISABLE_CONSOLE_LOGGING = 0
BATCH_SIZE = 1
# tensorflow or pytorch
FRAMEWORK = pytorch

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
GOLDPATH = ./gold_$(MODEL)_$(PRECISION)_$(DATASET)_btsz_$(BATCH_SIZE).pt

EXEC = ./$(FRAMEWORK)_dnns.py
all: generate test

generate:
	$(EXEC) --model $(MODEL) --precision $(PRECISION) $(CONSOLE_LOGGING) \
					  --imglist $(IMGLIST) --goldpath $(GOLDPATH) --batchsize $(BATCH_SIZE) --generate

test:
	$(EXEC) --model $(MODEL) --precision $(PRECISION) $(CONSOLE_LOGGING) \
					  --imglist $(IMGLIST) --goldpath $(GOLDPATH) --batchsize $(BATCH_SIZE) --iterations $(ITERATIONS)

