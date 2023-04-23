#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 05:45:38 2023
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi

"""
import string
############################## Dependencies ##############################
## Native
import time
import math
import os
import glob
import random
import subprocess
import string
import sys
import uuid
import logging

##Fix for protobuf version incompat.
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

print("Imported built-in stuff.")

import tensorflow
print(f"Imported tensorflow=={tensorflow.__version__}")

import cv2
print(f"Imported cv2=={cv2.__version__}")

import numpy as np
print(f"Imported numpy=={np.__version__}")

import PIL
from PIL import Image, ImageDraw
print(f"Imported PIL=={PIL.__version__}")

import arabic_reshaper
print(f"Imported arabic_reshaper=={arabic_reshaper.__version__}")

from bidi.algorithm import get_display
print(f"Imported bidi==0.4.2")

import matplotlib
from matplotlib import pyplot
print(f"Imported matplotlib=={matplotlib.__version__}")

from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools_deepsort import generate_detections as gdet
print(f"Imported deep_sort==latest")

import paddleocr
from paddleocr import PaddleOCR
print(f"Imported paddleocr=={paddleocr.__version__}")

DARKNET_PATH = os.getcwd() + "/darknet/"
print("DARKNET_PATH=" + DARKNET_PATH)
sys.path.insert(1, DARKNET_PATH)

import darknet
from darknet_images import load_images
from darknet_images import image_detection
print(f"Imported darknet==0.2.5.4")


############################## Custom Modules ##############################
import config
import helpers

############################## Main Flow ##############################
#Disable paddle's excessive logging.
logging.getLogger("paddle").setLevel(logging.WARN)

#Fetch all images in the image directory.
files = os.listdir(config.IMAGE_PATH)
images = []
image_names = []
# Load images.
for f in files:
    if f.endswith('.jpg') or f.endswith('.jpeg'):
        images.append(cv2.imread(config.IMAGE_PATH + f))
        image_names.append(f)

# Load NN.
network, class_names, class_colors = darknet.load_network(config.CONFIG_FILE,
        config.DATA_FILE,
        config.WEIGHTS_FILE,
        batch_size=config.BATCH_SIZE)

# Setup both OCRs.
english_ocr = PaddleOCR(use_angle_cls=True, lang="en", rec_algorithm="CRNN", show_log=config.SHOW_ENGLISH_LOG)
arabic_ocr = PaddleOCR(use_angle_cls=True, lang="ar", rec_algorithm="SVTR_LCNet", show_log=config.SHOW_ARABIC_LOG)

#Keeps track of complete images.
processed_images = []

# Plate recognition and OCR.
for i in range(0, len(images)):
    image = images[i]
    # Extract (and mark) plate locations.
    print(f"Extracting plate from image: {image_names[i]}\t\tat: {i}")
    image_with_plate, bboxes, scores, det_time = yolo_det(image,
            config.CONFIG_FILE,
            config.DATA_FILE, 
            config.BATCH_SIZE,
            config.WEIGHTS_FILE,
            config.CONFIDENCE_THRESHOLD,
            network,
            class_names,
            class_colors)
    
    # Loop over all found plates per-image.
    for bbox in bboxes:
        # Crop plate out of original image.
        bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
        plate_image = helpers.crop(image, bbox)
        if plate_image.shape[0] <= 0 or plate_image.shape[1] <= 0:
            print(f"Plate image too small. Plate shape: {plate_image.shape}")
            continue
        ocr_text, is_arabic = ocr.performOcr(plate_image)

        #Draw the bounding-box of the license plate.
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), config.BBOX_COLOR, 2)

        if is_arabic:
            helpers.drawArabicText(image, ocr_text, bbox)
        else:
            helpers.drawEnglishText(image, ocr_text, bbox)
    #Once all looping is done, the final state of the image is stored separately.
    h, w, _ = image.shape
    cv2.putText(image, f"ID:{i}", (0, h - 20), config.FONT, config.FONT_SCALE, config.TEXT_COLOR, config.FONT_THICKNESS)
    processed_images.append(image)

#Prepare plot and process images.
_, axes = pyplot.subplots(config.DATA_DISPLAY_W, config.DATA_DISPLAY_H, figsize=(9, 9))
if config.DATA_DISPLAY_W * config.DATA_DISPLAY_H > 1:
    axes = axes.flatten()

index = 0
print(f"Processed a total of {len(processed_images)} images.")
for img, ax in zip(processed_images, axes):
    print(f"Drawing processed image: {index}")
    index += 1
    ax.imshow(img)
pyplot.show()




