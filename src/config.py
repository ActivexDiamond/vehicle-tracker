#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 05:01:38 2023
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Env Fixes/Mods ##############################
#IMPORTANT: paddleocr must be imported before darknet otherwise it complains 
# about protobuf-version issues. I hope this is fixed once an update rolls
# out for either of those libs.

##Fix for protobuf version incompat.
import os, sys, logging
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

##Darkent fix
DARKNET_PATH = os.getcwd() + "/darknet/"
print("DARKNET_PATH=" + DARKNET_PATH)
sys.path.insert(1, DARKNET_PATH)

##Change out of src/ dir to make asset paths more natural.
#os.chdir(os.path.dirname(os.getcwd()))

#Disable paddle's excessive logging.
logging.getLogger("paddle").setLevel(logging.WARN)

############################## Dependencies ##############################
import cv2
import os
from PIL import ImageFont

############################## Debugging ##############################
DEBUG_PREPROCESSING = False

DATA_DISPLAY_W = 3
DATA_DISPLAY_H = 4

############################## Image Paths ##############################
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

ARABIC_FONT = ImageFont.truetype("./arial.ttf", 16)
BBOX_COLOR = (255, 0, 255)
TEXT_COLOR = (255, 0, 0)

TEXT_BACKGROUND_COLOR = (0, 0, 0, 128)

############################## Speed - General ##############################
MAX_SPEED = 70                  #km/h
VIOLATION_IMAGE_PATH = "./output/"
VIOLATION_PLATE_PATH = "./output/"

MIN_CAR_SIZE = 1000

if not os.path.exists(VIOLATION_IMAGE_PATH):
    os.makedirs(VIOLATION_IMAGE_PATH)
if not os.path.exists(VIOLATION_PLATE_PATH):
    os.makedirs(VIOLATION_PLATE_PATH)
    
############################## Speed - Video Specific ##############################
VIDEO_PATH = "./test/videos1/traffic4.mp4"
#Values for traffic4.mp4:
ENTRY_LINE_Y = 410          
EXIT_LINE_Y = 235
LINE_Y_OFFSET = 20

#To calculate this, you need a vehicle with a known speed (km/h).
#Compute its time passing (seconds) between lines ENTRY and EXIT,
#   this value is then equal to speed * time.
LINES_DISTANCE = 214.15

#Displacement, between 2 frames, below which to consider an object is the same 
#   as the one detected last frame.
MAX_DISPLACEMENT = 65

#If the video has extra noise/objects beyond the car path,
#   this can be used to crop the region of interest.
#Comment it out to use the full frame.
FRAME_ROI = [50, 540, 200, 960]
    
############################## Metadata ##############################
_METADATA = {
    "TITLE": "Traffic Watcher",
    "DESCRIPTION": "An M.L. model capable of detecting traffic violations and multi-lingually identifying and recognising license plates.",
    "TYPE": "CLI",
    "VERSION": "UNSTABLE-DEV",
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

