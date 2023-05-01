#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 05:01:38 2023
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Dependencies ##############################
import cv2
import os
from PIL import ImageFont

############################## Debugging ##############################
DEBUG_PREPROCESSING = False

DATA_DISPLAY_W = 3
DATA_DISPLAY_H = 4

############################## Testing ##############################
IMAGE_PATH = "./test/images2/"

############################## YOLO/Darknet ##############################
CONFIG_FILE = "yolov4-obj.cfg"
DATA_FILE = "data/obj.data"
WEIGHTS_FILE = "yolov4_obj_best.weights"

BATCH_SIZE = 1
CONFIDENCE_THRESHOLD = 0.6

############################## OCR ##############################
MIN_CONFIDENCE = 0.2

SHOW_ENGLISH_LOG = False
SHOW_ARABIC_LOG = False
############################## Visual ##############################
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 1
FONT_THICKNESS = 1

ARABIC_FONT = ImageFont.truetype("arial.ttf", 16)
BBOX_COLOR = (255, 0, 255)
TEXT_COLOR = (255, 0, 0)

TEXT_BACKGROUND_COLOR = (0, 0, 0, 128)

############################## Speed ##############################
MAX_UNSEEN = 10         #frames
MAX_DISTANCE = 175      #pxs
TRACKING_FRAMES = 4
SPEED_CONFIDENCE_THRESHOLD = 0.4

FRAME_WIDTH = 400       #pxs
SPEED_ZONES = {"A": 120, "B": 160, "C": 200, "D": 240}
WORLD_DISTANCE = 16     #meters
SPEED_LIMIT = 15        #mph

SPEED_CSV_FILE = "speed/log.csv"

############################## Metadata ##############################
_METADATA = {
    "TITLE": "Traffic Watcher",
    "DESCRIPTION": "An M.L. model capable of detecting traffic violations and multi-lingually identifying and recognising license plates.",
    "TYPE": "CLI",
    "VERSION": "0.1.0-alpha",
    "LICENSE": "MIT",
    "AUTHOR": "Dulfiqar 'activexdiamond' H. Al-Safi"
}

def echo_metadata():
    print(f"{'---~~~ INFO ~~~---':=^50}=")
    print(f"{'=> Echoing project metadata!': <50}=")
    print(f"{'=> Title: ' + _METADATA['TITLE']: <50}=")
    print(f"{'=> Type: ' + _METADATA['TYPE']: <50}=")
    print(f"{'=> Version: ' + _METADATA['VERSION']: <50}=")
    print(f"{'=> License: ' + _METADATA['LICENSE']: <50}=")
    print(f"{'=> Author: ' + _METADATA['AUTHOR']: <50}=")
    print(f"{'---~~~ ~~~~ ~~~---':=^50}=")
echo_metadata()

