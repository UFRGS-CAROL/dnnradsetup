# Just for debug purposes
CLASSIFY = 1
DISABLE_CONSOLE_LOGGING = 0
BATCH_SIZE = 1
# tensorflow or pytorch
FRAMEWORK=pytorch
USE_TFLITE = 0

ifeq ($(CLASSIFY), 1)
DATASET = imagenet2012
MODEL = ResNet-50
GRT_CSV = data/imagenet/imagenet-subset-ground-truths.csv
else
DATASET = coco2017
MODEL = RetinaNetResNet-50FPN
GRT_CSV = data/coco2017/coco-subset-ground-truths.csv
endif

ifeq ($(USE_TFLITE), 1)
USE_TFLITE_FLAG=--tflite
endif

ifeq ($(DISABLE_CONSOLE_LOGGING), 1)
CONSOLE_LOGGING = --disableconsolelog
endif

ifeq ($(FRAMEWORK), tensorflow)
FILE_EXT +=npy
else
FILE_EXT +=pt
endif

IMGLIST = data/$(DATASET)_img_list.txt

PRECISION = fp32
ITERATIONS = 10
GOLD_PATH = ./gold_$(MODEL)_$(PRECISION)_$(DATASET)_btsz_$(BATCH_SIZE)_$(FRAMEWORK)_tflite_$(USE_TFLITE).$(FILE_EXT)



EXEC = ./$(FRAMEWORK)_dnns.py
all: generate test

generate:
	$(EXEC) --model $(MODEL) --precision $(PRECISION) $(CONSOLE_LOGGING) --grtruthcsv $(GRT_CSV) $(USE_TFLITE_FLAG) \
					  --imglist $(IMGLIST) --goldpath $(GOLD_PATH) --batchsize $(BATCH_SIZE) --generate

test:
	$(EXEC) --model $(MODEL) --precision $(PRECISION) $(CONSOLE_LOGGING)  $(USE_TFLITE_FLAG) \
					  --imglist $(IMGLIST) --goldpath $(GOLD_PATH) --batchsize $(BATCH_SIZE) --iterations $(ITERATIONS)

