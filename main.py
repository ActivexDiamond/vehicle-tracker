#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 05:45:38 2023
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi

"""
############################## Dependencies ##############################
#IMPORTANT: paddleocr must be imported before darknet otherwise it complains 
# about protobuf-version issues. I hope this is fixed once an update rolls
# out for either of those libs.

##Fix for protobuf version incompat.
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import paddleocr

## Native
import os
import logging
print("Imported built-in stuff.")

import cv2
print(f"Imported cv2=={cv2.__version__}")

import matplotlib
from matplotlib import pyplot
print(f"Imported matplotlib=={matplotlib.__version__}")

############################## Custom Modules ##############################
import config
import helpers

import detector
import ocr

############################## Main Functions ##############################

############################## Main Flow ##############################
#Disable paddle's excessive logging.
logging.getLogger("paddle").setLevel(logging.WARN)

#Fetch all images in the image directory.
files = os.listdir(config.IMAGE_PATH)
images = []
image_names = []
image_paths = []

# Load images.
for f in files:
    if f.endswith('.jpg') or f.endswith('.jpeg'):
        images.append(cv2.imread(config.IMAGE_PATH + f))
        image_names.append(f)
        image_paths.append(config.IMAGE_PATH + f)

#Keeps track of complete images.
processed_images = []
            
# Plate recognition and OCR.
if config.DEBUG_PREPROCESSING:
    for i in range(0, len(images)):        
        processed_images.append(helpers.preprocess(images[i]))
else:
    for i in range(0, len(images)):
        #image = helpers.preprocess(images[i])
        image = images[i]
        # Extract (and mark) plate locations.
        print(f"Extracting plate from image: {image_names[i]}\t\tat: {i}")
        image_with_plate, bboxes, scores, det_time = detector.yolo_det(image)
        
        # Loop over all found plates per-image.
        for bbox in bboxes:
            # Crop plate out of original image.
            bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
            plate_image = helpers.crop(image, bbox)
            if plate_image.shape[0] <= 0 or plate_image.shape[1] <= 0:
                print(f"Plate image too small. Plate shape: {plate_image.shape}")
                continue
            ocr_text, is_arabic = ocr.performBilingualOcr(plate_image, image_paths[i])
    
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




