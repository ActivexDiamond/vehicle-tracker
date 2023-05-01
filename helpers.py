#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:54:56 2023
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Dependencies ##############################
## Native
import string, sys, os
print("Imported built-in stuff.")

import cv2
print(f"Imported cv2=={cv2.__version__}")

import numpy as np
print(f"Imported numpy=={np.__version__}")

import PIL
print(f"Imported PIL=={PIL.__version__}")

import arabic_reshaper
print(f"Imported arabic_reshaper=={arabic_reshaper.__version__}")

from bidi.algorithm import get_display
print("Imported bidi==unkown")

DARKNET_PATH = os.getcwd() + "/darknet/"
print("DARKNET_PATH=" + DARKNET_PATH)
sys.path.insert(1, DARKNET_PATH)

import darknet
from darknet_images import load_images
from darknet_images import image_detection
print("Imported darknet==unkown")

############################## Custom Modules ##############################
from preprocessing import binary_otsus, deskew
import config

############################## Constants ##############################
ARABIC_LETTERS = list("ابجدهوزحطيكلمنسعفصقرشتثخذضظغء")

############################## Image ##############################
def preprocess(image):

    # Maybe we end up using only gray level image.
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.bitwise_not(gray_img)

    binary_img = binary_otsus(gray_img, 0)
    # cv.imwrite('origin.png', gray_img)

    # deskewed_img = deskew(binary_img)
    deskewed_img = deskew(binary_img)
    # cv.imwrite('output.png', deskewed_img)

    # binary_img = binary_otsus(deskewed_img, 0)
    # breakpoint()

    # Visualize

    # breakpoint()
    return deskewed_img

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

############################## Text Helpers ##############################
def drawArabicText(image, ocr_text, bbox):
    # Remove all white space.
    ar_text = ocr_text.translate({ord(c): None for c in string.whitespace})

    # Reverse direction (improves visibility in images.)
    ar_text = "".join(reversed(ar_text))

    # Seperate all letters (improves visibility in images.)
    ar_text = " ".join(ar_text)

    # Convert into bidi text.
    print(f"Will draw Arabic text {ocr_text}")
    reshaped_text = arabic_reshaper.reshape(ar_text)
    bidi_text = get_display(reshaped_text)

    # Convert image to PIL.
    pil_image = PIL.Image.fromarray(image)
    draw = PIL.ImageDraw.Draw(pil_image)

    # Draw rectangle behind text.
    (_, baseline), _ = config.ARABIC_FONT.font.getsize(ar_text)
    (x, y) = tuple(map(int, [int(bbox[0]), int(bbox[1])]))
    text_bbox = draw.textbbox((x, y), ar_text, font=config.ARABIC_FONT)
    draw.rectangle(text_bbox)# fill=config.TEXT_BACKGROUND_COLOR)

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

