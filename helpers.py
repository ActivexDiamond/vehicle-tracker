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


############################## YOLO Detection ##############################
def yolo_det(frame, config_file, data_file, batch_size, weights,
             threshold, network, class_names, class_colors):
    # Used to track execution time.
    prev_time = time.time()

    # Get some stats about the ML network.
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    # Prepare buffer.
    darknet_image = darknet.make_image(width, height, 3)

    # Minimal preprocessing.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))

    # Copy buffer.
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    # Detect license plates.
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=threshold)
    # Free up used memory.
    darknet.free_image(darknet_image)

    # Mark the license plate on the image with an outline.
    image = darknet.draw_boxes(detections, image_resized, class_colors)

    # Compute current execution time.
    det_time = time.time() - prev_time
    fps = int(1 / (time.time() - prev_time))

    # Prepare output image.
    out_size = frame.shape[:2]
    in_size = image_resized.shape[:2]
    coord, scores = resize_bbox(detections, out_size, in_size)
    image_with_plate = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image_with_plate, coord, scores, det_time

############################## Geometrey ##############################
# Converts bounding-boxes into image-coords.
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

        final_points = [xmin, ymin, xmax - xmin, ymax - ymin]
        scores.append(conf)
        coord.append(final_points)

    return coord, scores


# Simple cropping wrapper.
def crop(image, coord):
    cr_img = image[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]
    return cr_img

############################## Text Helpers##############################
def drawArabicText(image, ocr_text, bbox):
    # Remove all white space.
    ar_text = ocr_text.translate({ord(c): None for c in string.whitespace})

    # Reverse direction (improves visibility in images.)
    ar_text = "".join(reversed(ar_text))

    # Seperate all letters (improves visibility in images.)
    ar_text = " ".join(ar_text)

    # Convert into bidi text.
    print(f"Will draw Arabic text {ar_text}")
    reshaped_text = arabic_reshaper.reshape(ar_text)
    bidi_text = get_display(reshaped_text)

    # Convert image to PIL.
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # Draw rectangle behind text.
    (_, baseline), _ = config.ARABIC_FONT.font.getsize(ar_text)
    (x, y) = tuple(map(int, [int(bbox[0]), int(bbox[1])]))
    text_bbox = draw.textbbox((x, y), ar_text, font=config.ARABIC_FONT)
    draw.rectangle(text_bbox, fill=config.TEXT_BACKGROUND_COLOR)

    # Draw text.
    draw.text((x, y), bidi_text, font=config.ARABIC_FONT, fill="red")

    # Convert back numpy-style image.
    image = np.array(pil_image)

def drawEnglishText(image, ocr_text, bbox):
    # Draw rectangle behind text.
    (text_w, text_h), baseline = cv2.getTextSize(ocr_text, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS)
    (x, y) = tuple(map(int, [int(bbox[0]), int(bbox[1])]))
    cv2.rectangle(image, (x, y), (x + text_w, y - text_h), config.TEXT_BACKGROUND_COLOR, -1)

    # Draw text.
    print(f"Will draw English text {ocr_text}")
    cv2.putText(image, ocr_text, (x, y), config.FONT, config.FONT_SCALE, config.TEXT_COLOR, config.FONT_THICKNESS)
    # Label images.

