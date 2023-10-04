#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 09:15:33 2023
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Dependencies ##############################
## Native
import time
import os
import sys
print("Imported built-in stuff.")

import cv2
print(f"Imported cv2=={cv2.__version__}")

import darknet
from darknet_images import load_images
from darknet_images import image_detection
print("Imported darknet==unkown")

############################## Custom Modules ##############################
import config
import helpers

############################## Load Darknet NN ##############################
if config.PERFORM_OCR:
    network, classNames, classColors = darknet.load_network(config.CONFIG_FILE,
            config.DATA_FILE,
            config.WEIGHTS_FILE,
            batch_size=config.BATCH_SIZE)
else:
    network = ""
    classNames = ""
    classColors = ""
    
############################## YOLO Detection ##############################
def yoloDet(frame):
    if not config.PERFORM_OCR: return frame, [], [], 0
    
    # Used to track execution time.
    prevTime = time.time()

    # Get some stats about the ML network.
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    # Prepare buffer.
    darknetImage = darknet.make_image(width, height, 3)

    # Minimal preprocessing.
    imageRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imageResized = cv2.resize(imageRgb, (width, height))

    # Copy buffer.
    darknet.copy_image_from_bytes(darknetImage, imageResized.tobytes())
    # Detect license plates.
    detections = darknet.detect_image(network, classNames, darknetImage,
                                      thresh=config.CONFIDENCE_THRESHOLD)
    # Free up used memory.
    darknet.free_image(darknetImage)

    # Mark the license plate on the image with an outline.
    image = darknet.draw_boxes(detections, imageResized, classColors)

    # Compute current execution time.
    detTime = time.time() - prevTime

    # Prepare output image.
    outSize = frame.shape[:2]
    inSize = imageResized.shape[:2]
    coord, scores = helpers.resizeBbox(detections, outSize, inSize)
    imageWithPlate = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print(f"Darknet found plates: {coord}")
    return imageWithPlate, coord, scores, detTime