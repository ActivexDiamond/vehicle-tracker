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

from datetime import datetime
RUN_START_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

############################## Dependencies ##############################
import cv2
import os
from PIL import ImageFont

############################## Custom Modules ##############################
#Should contain Twilio credentials!
import secret

############################## Debugging ##############################
SEND_SMS = False

DEBUG_VIDEO = True

DATA_DISPLAY_W = 3
DATA_DISPLAY_H = 4

SHOW_BBOX = False
SHOW_BBOX_SIZE = True

############################## SMS ##############################
TWILIO_SID = secret.TWILIO_SID
TWILIO_AUTH_TOKEN = secret.TWILIO_AUTH_TOKEN

SENDER_PHONE_NUMBER = "++12545276516"

#TARGET_PHONE_NUMBERS = ["+9647709206760", "+9647738057710"]
TARGET_PHONE_NUMBERS = ["+9647726914819"]

VIOLATION_LOCATION = "Baghdad"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

############################## Image Paths ##############################
IMAGE_PATH = "./test/images2/"
IMAGE_TEST_RESULT_PATH = "./output/image-test-result.txt"
IMAGE_TEST_RESULT_PATH_EXTENDED = "./output/image-test-result-extended.txt"

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

#Maximum degree of difference between an Arabic string from our list of valid
# texts to consider it the same word. Inclusive.
MAX_LEVENSHTEIN_DISTANCE = 3

############################## Visual ##############################
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 1
FONT_THICKNESS = 1

ARABIC_FONT = ImageFont.truetype("./arial.ttf", 16)
BBOX_COLOR = (255, 0, 255)
TEXT_COLOR = (255, 0, 0)

TEXT_BACKGROUND_COLOR = (0, 0, 0, 128)

############################## Speed - General ##############################
MAX_SPEED = 64                  #km/h
VIOLATION_IMAGE_PATH = "./output/"
VIOLATION_PLATE_PATH = "./output/"

MIN_CAR_SIZE = 1000

#Displacement, between 2 frames, below which to consider an object is the same 
#   as the one detected last frame.
MAX_DISPLACEMENT = 100

#Displacement, between 2 frames, below which to consider an object is the still.
MAX_PARKED_DISPLACEMENT = 10

#Time, in seconds, before a still-car is considered parked.
MAX_PARK_TIME = 5

if not os.path.exists(VIOLATION_IMAGE_PATH):
    os.makedirs(VIOLATION_IMAGE_PATH)
if not os.path.exists(VIOLATION_PLATE_PATH):
    os.makedirs(VIOLATION_PLATE_PATH)

if not os.path.exists(VIOLATION_IMAGE_PATH + RUN_START_TIME + "/"):
    os.makedirs(VIOLATION_IMAGE_PATH + RUN_START_TIME + "/")
if not os.path.exists(VIOLATION_PLATE_PATH + RUN_START_TIME + "/"):
    os.makedirs(VIOLATION_PLATE_PATH + RUN_START_TIME + "/")    
    
############################## Speed - Video Specific ##############################
VIDEOS = [
    {   # 0
        "VIDEO_PATH": "./test/videos1/traffic4.mp4",
        "ENTRY_LINE_Y": 820,
        "EXIT_LINE_Y": 470,
        "LINE_Y_OFFSET": 20,
        #To calculate this, you need a vehicle with a known speed (km/h).
        #Compute its time passing (seconds) between lines ENTRY and EXIT,
        #   this value is then equal to speed * time.
        "LINES_DISTANCE": 59.1,
        #If the video has extra noise/objects beyond the car path,
        #   this can be used to crop the region of interest.
        #Comment it out to use the full frame.
        "FRAME_ROI": [100, 1080, 400, 960*2],
    },{ # 1
        "VIDEO_PATH": "./test/videos1/IMG_0102.mp4",
        "ENTRY_LINE_Y": 620,
        "EXIT_LINE_Y": 270,
        "LINE_Y_OFFSET": 15,
        "LINES_DISTANCE": 45,
    },{ # 2
        "VIDEO_PATH": "./test/videos1/top-day.MP4",
        "ENTRY_LINE_Y": 1700,
        "EXIT_LINE_Y": 1400 - 150,
        "LINE_Y_OFFSET": 60,
        "LINES_DISTANCE": 80 * 0.32,
        "MIN_CAR_SIZE": 30e3,
        "MAX_DISPLACEMENT": 250,
        "MAX_PARKED_DISPLACEMENT": 30,
    }
]

#Note: 4k is 3840 x 2160

SELECTED_VIDEO = 2
_v = VIDEOS[SELECTED_VIDEO]
VIDEO_PATH = _v["VIDEO_PATH"]
ENTRY_LINE_Y = _v["ENTRY_LINE_Y"]
EXIT_LINE_Y = _v["EXIT_LINE_Y"]
LINE_Y_OFFSET = _v["LINE_Y_OFFSET"]
LINES_DISTANCE = _v["LINES_DISTANCE"]

print(f"Selected video  = [{SELECTED_VIDEO}]")
print(f"VIDEO_PATH      = {VIDEO_PATH}")
print(f"ENTRY_LINE_Y    = {ENTRY_LINE_Y}")
print(f"EXIT_LINE_Y     = {EXIT_LINE_Y}")
print(f"LINE_Y_OFFSET   = {LINE_Y_OFFSET}")
print(f"LINES_DISTANCE  = {LINES_DISTANCE}")
if "FRAME_ROI" in _v:
    FRAME_ROI = _v["FRAME_ROI"]
    print(f"FRAME_ROI = {FRAME_ROI}")

#A particular video can also override the defaults of any general video configs.
#This is useful for videos with higher/lower resolutions.
if "MIN_CAR_SIZE" in _v:
    MIN_CAR_SIZE = _v["MIN_CAR_SIZE"]
    print(f"MIN_CAR_SIZE    = {MIN_CAR_SIZE}")
if "MAX_DISPLACEMENT" in _v:
    MAX_DISPLACEMENT = _v["MAX_DISPLACEMENT"]
    print(f"MAX_DISPLACEMENT= {MAX_DISPLACEMENT}")
if "MAX_PARKED_DISPLACEMENT" in _v:
    MAX_PARKED_DISPLACEMENT = _v["MAX_PARKED_DISPLACEMENT"]
    print(f"MAX_PARKED_DISPLACEMENT = {MAX_PARKED_DISPLACEMENT}")


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

