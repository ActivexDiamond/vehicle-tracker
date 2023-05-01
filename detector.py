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
network, class_names, class_colors = darknet.load_network(config.CONFIG_FILE,
        config.DATA_FILE,
        config.WEIGHTS_FILE,
        batch_size=config.BATCH_SIZE)

#config.CONFIG_FILE,
#config.DATA_FILE,
#config.BATCH_SIZE,
#config.WEIGHTS_FILE,

############################## YOLO Detection ##############################
def yolo_det(frame):
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
    detections = darknet.detect_image(network, class_names, darknet_image,
                                      thresh=config.CONFIDENCE_THRESHOLD)
    # Free up used memory.
    darknet.free_image(darknet_image)

    # Mark the license plate on the image with an outline.
    image = darknet.draw_boxes(detections, image_resized, class_colors)

    # Compute current execution time.
    det_time = time.time() - prev_time

    # Prepare output image.
    out_size = frame.shape[:2]
    in_size = image_resized.shape[:2]
    coord, scores = helpers.resize_bbox(detections, out_size, in_size)
    image_with_plate = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image_with_plate, coord, scores, det_time