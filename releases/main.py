#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 05:45:38 2023
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi

"""
############################## Dependencies ##############################
## Native
import os, platform
print("Imported built-in stuff.")

import cv2
print(f"Imported cv2=={cv2.__version__}")

# Config has to be imported here for workaround.
import config
if config.DEBUG_VIDEO:
    if platform.system() == "Linux":
        print("Performing cv2 SEGFAULT  work around since running on Linux.")
        cv2.namedWindow('roi', 0)
        cv2.resizeWindow("roi", 900, 900)

import matplotlib
from matplotlib import pyplot
print(f"Imported matplotlib=={matplotlib.__version__}")

## Required so paddle doesn't complain about protobuf-version incompat.
import paddleocr

############################## Custom Modules ##############################
import helpers
from CarTracker import CarTracker

############################## Main Functions ##############################
def testImagesDir():
    #Fetch all images in the image directory.
    files = os.listdir(config.IMAGE_PATH)
    images = []
    imageNames = []
    imagePaths  = []
    
    # Load images.
    for f in files:
        if f.endswith('.jpg') or f.endswith('.jpeg'):
            images.append(cv2.imread(config.IMAGE_PATH + f))
            imageNames.append(f)
            imagePaths.append(config.IMAGE_PATH + f)
    
    #Keeps track of complete images.
    processedImages = []
                
    # Plate recognition and OCR.
    file = open(config.IMAGE_TEST_RESULT_PATH, "w", encoding="utf-8")
    extFile = open(config.IMAGE_TEST_RESULT_PATH_EXTENDED, "w", encoding="utf-8")
    for i in range(0, len(images)):
        text, img, isArabic, ext = helpers.extractPlate(images[i])
        
        string = f"img={imageNames[i]}, text={text}, isArabic={isArabic}\n"
        #print(string)
        extString = string + ext + "\n=========\n"
        
        file.write(string)
        extFile.write(extString)
        
        processedImages.append(img)
    file.close()    
    extFile.close()
    
    #Prepare plot and process images.
    _, axes = pyplot.subplots(config.DATA_DISPLAY_W, config.DATA_DISPLAY_H, figsize=(9, 9))
    if config.DATA_DISPLAY_W * config.DATA_DISPLAY_H > 1:
        axes = axes.flatten()
    
    index = 0
    print(f"Processed a total of {len(processedImages)} images.")
    for img, ax in zip(processedImages, axes):
        print(f"Drawing processed image: {index}")
        index += 1
        ax.imshow(img)
    pyplot.show()
    
    
def testVideo():
    carTracker = CarTracker()
    carTracker.processVideo(config.VIDEO_PATH)
    
############################## Main Flow ##############################

if config.DEBUG_VIDEO:
    testVideo()
else:
    testImagesDir()



