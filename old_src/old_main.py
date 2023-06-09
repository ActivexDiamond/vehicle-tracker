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
import old_config as config

############################## Helpers ##############################
def yolo_det(frame, config_file, data_file, batch_size, weights, 
             threshold, network, class_names, class_colors):
    #Used to track execution time.
    prev_time = time.time()

    #Get some stats about the ML network.
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    #Prepare buffer.
    darknet_image = darknet.make_image(width, height, 3)

    #Minimal preprocessing.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))

    #Copy buffer.
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    #Detect license plates.
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=threshold)
    #Free up used memory.
    darknet.free_image(darknet_image)

    #Mark the license plate on the image with an outline.
    image = darknet.draw_boxes(detections, image_resized, class_colors)

    #Compute current execution time.
    det_time = time.time() - prev_time
    fps = int(1/(time.time() - prev_time))

    #Prepare output image.
    out_size = frame.shape[:2]
    in_size = image_resized.shape[:2]
    coord, scores = resize_bbox(detections, out_size, in_size)
    image_with_plate = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image_with_plate, coord, scores, det_time

#Converts bounding-boxes into image-coords.
def resize_bbox(detections, out_size, in_size):

    coord = []
    scores = []

    for det in detections:
        points = list(det[2])
        conf = det[1]
         
        xmin, ymin, xmax, ymax = darknet.bbox2points(points)
        y_scale = float(out_size[0]) / in_size[0]
        x_scale = float(out_size[1]) / in_size[1]
        ymin = int(y_scale * ymin)
        ymax = int(y_scale * ymax)
        xmin = int(x_scale * xmin) if int(x_scale * xmin) > 0 else 0
        xmax = int(x_scale * xmax)
           
        final_points = [xmin, ymin, xmax-xmin, ymax-ymin]
        scores.append(conf)
        coord.append(final_points)
    
    return coord, scores

#Simple cropping wrapper.
def crop(image, coord):
    cr_img = image[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]
    return cr_img

     
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
        plate_image = crop(image, bbox)
        if plate_image.shape[0] <= 0 or plate_image.shape[1] <= 0:
            print(f"Plate image too small. Plate shape: {plate_image.shape}")
            continue

        # Perform OCR.
        english_texts = []
        english_scores = []
        english_confidence = 0
        english_result = english_ocr.ocr(plate_image, cls=True)
        #Concat all text into a single string.
        for j in range(len(english_result)):
            r = english_result[j]
            for line in r:
                english_texts.append(line[1][0])
                english_scores.append(line[1][1])

        #Identical to the above, but for Arabic.
        arabic_texts = []
        arabic_scores = []
        arabic_confidence = 0
        arabic_result = arabic_ocr.ocr(plate_image, cls=True)
        #Concat all text into a single string.
        for k in range(len(arabic_result)):
            r = arabic_result[k]
            for line in r:
                arabic_texts.append(line[1][0])
                arabic_scores.append(line[1][1])

        #If any OCRs occurred; compute their mean.
        if len(english_scores) > 0:
            english_confidence = sum(english_scores) / len(english_scores)
        if len(arabic_scores) > 0:
            arabic_confidence = sum(arabic_scores) / len(arabic_scores)

        #Fallbakc in case OCR failed.
        ocr_text = "Unknown"
        is_arabic = False

        # Check scores to deduce text language.
        if english_confidence >= arabic_confidence and english_confidence >= config.MIN_CONFIDENCE:
            ocr_text = " ".join(english_texts)
        elif arabic_confidence >= config.MIN_CONFIDENCE:
            ocr_text = " ".join(arabic_texts)
            is_arabic = True

        
        #Draw the bounding-box of the license plate.
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), config.BBOX_COLOR, 2)

        if is_arabic:
            #Remove all white space.
            ar_text = ocr_text.translate({ord(c): None for c in string.whitespace})

            #Reverse direction (improves visibility in images.)
            ar_text = "".join(reversed(ar_text))

            #Seperate all letters (improves visibility in images.)
            ar_text = " ".join(ar_text)

            #Convert into bidi text.
            print(f"Will draw Arabic text {ar_text}")
            reshaped_text = arabic_reshaper.reshape(ar_text)
            bidi_text = get_display(reshaped_text)

            #Convert image to PIL.
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)

            #Draw rectangle behind text.
            (_, baseline), _ = config.ARABIC_FONT.font.getsize(ar_text)
            (x, y) = tuple(map(int, [int(bbox[0]), int(bbox[1])]))
            text_bbox = draw.textbbox((x, y), ar_text, font=config.ARABIC_FONT)
            draw.rectangle(text_bbox, fill=config.TEXT_BACKGROUND_COLOR)

            #Draw text.
            draw.text((x, y), bidi_text, font=config.ARABIC_FONT, fill="red")

            #Convert back numpy-style image.
            image = np.array(pil_image)
        else:
            #Draw rectangle behind text.
            (text_w, text_h), baseline = cv2.getTextSize(ocr_text, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS)
            (x, y) = tuple(map(int, [int(bbox[0]), int(bbox[1])]))
            cv2.rectangle(image, (x, y), (x + text_w, y - text_h), config.TEXT_BACKGROUND_COLOR, -1)

            #Draw text.
            print(f"Will draw English text {ocr_text}")
            cv2.putText(image, ocr_text, (x, y), config.FONT, config.FONT_SCALE, config.TEXT_COLOR, config.FONT_THICKNESS)
        #Label images.
        h, w, _ = image.shape
        cv2.putText(image, f"ID:{i}", (0, h - 20), config.FONT, config.FONT_SCALE, config.TEXT_COLOR, config.FONT_THICKNESS)
    #Once all looping is done, the final state of the image is stored separately.
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




