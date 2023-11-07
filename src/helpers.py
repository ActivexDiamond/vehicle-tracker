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
import difflib
from datetime import datetime

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

import darknet
from darknet_images import load_images
from darknet_images import image_detection
print("Imported darknet==unkown")

import Levenshtein

############################## Custom Modules ##############################
import config

import detector
import ocr

if hasattr(config, "DEBUG_OCR") and config.DEBUG_OCR:
    from debugOcr import debugOcr

############################## Constants ##############################
ARABIC_LETTERS = list("ابجدهوزحطيكلمنسعفصقرشتثخذضظغء")
ENGLISH_LETTERS = list("abcdefghijklmnopqrstuvwxyz")

ARABIC_DIGITS = list("٠١٢٣٤٥٦٧٨٩")
ENGLISH_DIGITS = list("0123456789")

TRANSLATION_MAP = [
    list("abcdefghijklmnopqrstuvwxyzgad0123456789"),
    list("ابجدهوزحطيكلمنسعفصسرشتثخذضظغء٠١٢٣٤٥٦٧٨٩"),
]

ARABIC_WORD_LIST = [
"بغداد",
"البصرة",
"ذي قار",
"المثنى",
"نينوى ",
"بابل",
"ديالى",
"الانبار",
"القادسية",
"ميسان",
"صلاح الدين",
"كربلاء",
"النجف",
"كركوك",
"دهوك",
"اربيل",
"السليمانية",
"واسط",
"حلبجة",
"اجرة",
"خصوصي",
"حمل",
"حكومي",
]

print("==============================")
print(f"Using the following as valid Arabic words: \n{ARABIC_WORD_LIST}\n")
print("==============================")

############################## Geometrey ##############################
# Converts bounding-boxes into image-coords.
def resizeBbox(detections, outSize, inSize):
    coord = []
    scores = []
    for det in detections:
        points = list(det[2])
        conf = det[1]

        xmin, ymin, xmax, ymax = darknet.bbox2points(points)
        yScale = float(outSize[0]) / inSize[0]
        xScale = float(outSize[1]) / inSize[1]
        ymin = int(yScale * ymin)
        ymax = int(yScale * ymax)
        xmin = int(xScale * xmin) if int(xScale * xmin) > 0 else 0
        xmax = int(xScale * xmax)

        finalPoints = [xmin, ymin, xmax - xmin, ymax - ymin]
        scores.append(conf)
        coord.append(finalPoints)

    return coord, scores

# Simple cropping wrapper.
def crop(image, coord):
    return image[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]

def bindRectToRect(s, b):
    x = bind(s[0], b[0], b[0] + b[2])
    y = bind(s[1], b[1], b[1] + b[3])
    if x + s[2] >= b[0] + b[2]:
        w = (b[2] - x) - 1
    else:
        w = s[2]
    if y + s[3] >= b[1] + b[3]:
        h = (b[3] - y) - 1
    else:
        h = s[3]
    print("s", s)
    print("b", b)
    print("n", x, y, w, h)
    return [x, y, w, h]

############################## Math Helpers ##############################
def bind(x, lower, upper):
    return max(x, min(x, upper))

############################## Human Readable Numbers ##############################
def isNumberSafe(num):
    if isinstance(num, str):
        num = float(num)
    return float('{:.3g}'.format(abs(num)))

def formatNumber(num):
    num = isNumberSafe(num)
    sign = ''

    metric = {'T': 1e12, 'B': 1e9, 'M': 1e6, 'K': 1e3, '': 1}
    for index in metric:
        check = num / metric[index]
        if(check>= 1):
            num = check
            sign = index
            break

    return f"{str(num).rstrip('0').rstrip('.')}{sign}"

############################## Text Helpers ##############################
def cleanArabicText(text):
    # Remove all white space.
    arText = text.translate({ord(c): None for c in string.whitespace})

    # Reverse direction (improves visibility in images.)
    arText = "".join(reversed(arText))

    # Seperate all letters (improves visibility in images.)
    arText = " ".join(arText)
    return arText    

def drawArabicText(image, ocrText, bbox):
    arText = cleanArabicText(ocrText)

    # Convert into bidi text.
    #print(f"Will draw Arabic text {ocrText}")
    reshapedText = arabic_reshaper.reshape(arText)
    bidiText = get_display(reshapedText)

    # Convert image to PIL.
    pilImage = PIL.Image.fromarray(image)
    draw = PIL.ImageDraw.Draw(pilImage)

    # Draw rectangle behind text.
    (_, baseline), _ = config.ARABIC_FONT.font.getsize(arText)
    (x, y) = tuple(map(int, [int(bbox[0]), int(bbox[1])]))
    textBbox = draw.textbbox((x, y), arText, font=config.ARABIC_FONT)
    draw.rectangle(textBbox)# fill=config.TEXT_BACKGROUND_COLOR)

    # Draw text.
    draw.text((x, y), bidiText, font=config.ARABIC_FONT, fill="red")

    # Convert back numpy-style image.
    image = np.array(pilImage)

def drawEnglishText(image, ocr_text, bbox):
    # Draw rectangle behind text.
    (text_w, text_h), baseline = cv2.getTextSize(ocr_text, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS)
    (x, y) = tuple(map(int, [int(bbox[0]), int(bbox[1])]))
    cv2.rectangle(image, (x, y), (x + text_w, y - text_h), config.TEXT_BACKGROUND_COLOR, -1)

    # Draw text.
    #print(f"Will draw English text {ocr_text}")
    cv2.putText(image, ocr_text, (x, y), config.FONT, config.FONT_SCALE, config.TEXT_COLOR, config.FONT_THICKNESS)
    # Label images.

def strictifyArabicTexts(texts):
    res = []
    for text in texts:
        closest = 1000
        closestText = ""
        for valid in ARABIC_WORD_LIST:
            distance = Levenshtein.distance(text, valid)
            if distance < closest:
                closest = distance 
                closestText = valid
        s = "Unknown" if closest > config.MAX_LEVENSHTEIN_DISTANCE else closestText
        res.append(s)
    return res

############################## OCR-specific Helpers ##############################
def formatPaddleResult(result):
    texts, numbers, scores = [], [], []
    print (result)
    if result[0] is None:
        print("blank list. returning.")
        return texts, numbers, scores
    for i in range(len(result)):
        for line in result[i]:
            string = line[1][0]
            # If string contains MORE than one letter, consider it
            # the plate text. 
            letterCount = 0
            for char in string:
                if (any(char == letter for letter in ARABIC_LETTERS)
                        or any(char == letter for letter in ENGLISH_LETTERS)):
                    letterCount += 1
            if letterCount > 1:
                texts.append(string)    
            else:
                numbers.append(string)
            scores.append(line[1][1])
    return texts, numbers, scores

#loop over str
#if char one of map then new[char]=Map[0]->Map[1]
def translateNumbersToArabic(english):
    translated = []
    print(english, translated)
    for enIndex, enStr in enumerate(english):
        translated.append([""] * len(enStr))
        enStr = enStr.lower()
        print(english, translated)
        print(enIndex, enStr)
        for i, mapChar in enumerate(TRANSLATION_MAP[0]):
            for j, char in enumerate(enStr):
                if mapChar == char:
                    translated[enIndex][j] = TRANSLATION_MAP[1][i]
    print(translated)
    print("=====")
    return [b for a in translated for b in a]
        
#print(translateNumbersToArabic(["A123Hello1234"]))
############################## Plate Extraction Helpers ##############################
_imageNameCounter = 0
def extractPlate(image, drawOnImage=True, imageName="default"):
    global _imageNameCounter
    
    if not config.PERFORM_OCR: 
        return "OCR disabled.", image, False, "OCR disabled."
    if hasattr(config, "DEBUG_OCR") and config.DEBUG_OCR:
        print("Performing debug OCR.")
        ocrText,isArabic, ext = debugOcr(image)
        return ocrText, image, isArabic, ext
    
    # Extract (and mark) plate locations.
    imageWithPlate, bboxes, scores, detTime = detector.yoloDet(image)
    ocrText = "Unknown"
    isArabic = False
    ext = "Unknown"
    
    # Loop over all found plates per-image.
    # Note: Currently only returns the last found text, drawing howver can
    # handle multiple plates.
    for bbox in bboxes:
        # Crop plate out of original image.
        bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
        plateImage = crop(image, bbox)
        if plateImage.shape[0] <= 0 or plateImage.shape[1] <= 0:
            print(f"Plate image too small. Plate shape: {plateImage.shape}")
            continue
        #Debug save OCR plate.
        if config.SAVE_OCR_INPUT_PLATE: 
            if imageName == "default":
                imageName = str(_imageNameCounter) + ".png"
                _imageNameCounter += 1
                
            path = config.OCR_PLATE_IMAGE_PATH +\
                    config.RUN_START_TIME + "/" + imageName
            cv2.imwrite(path, plateImage)    
        #Perform OCR.
        ocrText, isArabic, ext = ocr.performBilingualOcr(plateImage)

        #Draw the bounding-box of the license plate.
        if drawOnImage:
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), config.BBOX_COLOR, 2)
            if isArabic:
                drawArabicText(image, ocrText, bbox)
            else:
                drawEnglishText(image, ocrText, bbox)
    #Once all looping is done, the final state of the image is returned alongside the plate text.
    return ocrText, image, isArabic, ext

