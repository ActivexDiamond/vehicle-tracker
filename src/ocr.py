#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 17:15:25 2023
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Custom Modules ##############################
import config
import helpers
  
if hasattr(config, "DEBUG_OCR") and config.DEBUG_OCR:
    from debugOcr import debugOcr

############################## Dependencies ##############################
import math

import numpy

if config.PERFORM_OCR:
    import paddleocr
    print(f"Imported paddleocr=={paddleocr.__version__}")
    
    import easyocr 
    from easyocr import Reader
    print(f"Imported easyocr=={easyocr.__version__}")

#import cv2

############################## Init ##############################
# Setup both OCRs.
if config.PERFORM_OCR:
    englishOcr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en", rec_algorithm="CRNN", show_log=config.SHOW_ENGLISH_LOG)
    arabicOcr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ar", rec_algorithm="SVTR_LCNet", show_log=config.SHOW_ARABIC_LOG)
    arabicReader = Reader(["ar"], gpu=-1 > 0)
else:
    print("OCR disabled. Only dummy text will be provided.")
    englishOcr = ""
    arabicOcr = ""
    arabicReader = ""

############################## Color Detection ##############################
def colorDistance(color1, color2):
    #Normalize ranges
    n1 = numpy.divide(color1, 255)
    n2 = numpy.divide(color2, 255)
            #d= math.sqrt((x1−x2)^2 + (y1−y2)^2 + (z1−z2)^2)
    rm = 0.5 * (n1[0] + n2[0])
    d = sum((2 + rm, 4, 3 - rm) * (n1 - n2) ** 2) ** 0.5
    return d

import cv2
def typeFromColor(image):
    h, w, _ = image.shape
    colorWidth = math.floor(w * config.COLOR_AREA_PERCENTAGE)
    colorRegion = image[0:h, 0:colorWidth]
    cv2.imwrite("im.png", image)
    cv2.imwrite("col.png", colorRegion)
    averageColor = numpy.mean(colorRegion, axis=(0,1))
    minDistance = 1e9
    plateType = "unkown"
    for name, color in config.COLOR_TYPES.items():
        dist = colorDistance(averageColor, color)
        print(averageColor, color, dist)
        if dist < minDistance:
            minDistance = dist
            plateType = name
            print("new min. Name is now", name)
    return plateType
            
    
############################## OCR Helpers ##############################
def performBilingualOcr(image):
    if not config.PERFORM_OCR: 
        print("OCR disabled, returning.")
        return "OCR disabled.", False, "OCR disabled."
    if hasattr(config, "DEBUG_OCR") and config.DEBUG_OCR:
        print("Performing debug OCR.")
        return debugOcr(image)
    print("Performing bilingual OCR")
    
    # Perform OCR.
    print("Performing EN OCR.")
    enResult = englishOcr.ocr(image, cls=True)
    print("Performing AR OCR.")
    arResult = arabicOcr.ocr(image, cls=True)
    
    # Format paddle string into friendlier format.
    print("Formatting EN OCR result.")
    enTexts, enNumbers, enScores = helpers.formatPaddleResult(enResult)
    print("Formatting AR OCR result.")
    arTexts, arNumbers, arScores = helpers.formatPaddleResult(arResult)
    print("Strictifying AR.")
    arStrictTexts = helpers.strictifyArabicTexts(arTexts)
    print("==============")
    # Check if OCRs occured between computing their means.
    enConfidence = 0
    if len(enScores) > 0:
        enConfidence = sum(enScores) / len(enScores)
    arConfidence = 0
    if len(arScores) > 0:
        arConfidence = sum(arScores) / len(arScores)
        
        #If two texts of length 2+ chars exist, then they will be strictified,
        #   so add %25 bonus for that.
        if len(arTexts) > 1 and len(arTexts[0]) > 2 and len(arTexts[1]) > 2:
            arConfidence += .25

    # Fallback in case OCR failed.
    ocrText = "Unknown"
    isArabic = False
    
    #This is only needed so that debugInfo doesn't throw an UnboundLocalError
    #   if no modding happened.
    arNumbersModded = []
    
    # Check scores to deduce text language.
    if enConfidence >= arConfidence and enConfidence >= config.MIN_CONFIDENCE:
        #Is English.
        ocrText = " ".join(enTexts) + " " + " ".join(enNumbers)
    elif arConfidence >= config.MIN_CONFIDENCE:
        if len(enNumbers) > 1:
            #Is english with translated numbers.
            translatedNumbers = helpers.translateNumbersToArabic(enNumbers)
            ocrText = " ".join(arStrictTexts) + " " + " ".join(translatedNumbers)
        else:
            #Is arabic with native numbers.
            #Remember, the OCR returns a list that could contain multiple strings.
            if len(arNumbers) > 0:
                arNumbersModded = [None] * len(arNumbers)
                for num in arNumbers:
                    if num[0] == "١":
                        arNumbersModded = "أ" + num[1:]
                    else:
                        arNumbersModded = num
            ocrText = " ".join(arStrictTexts) + " " + " ".join(arNumbersModded)
        isArabic = True
    
    if not isArabic:
        ocrText = ocrText + " (" + typeFromColor(image) + ")"
    debugInfo = (f"enConfidence={enConfidence}\tenTexts={enTexts}\tenNumbers={enNumbers}\n" +
    f"arConfidence={arConfidence}\tarTexts={arTexts}\tarNumbers={arNumbers}\n" +
    f"arNumbersModded={arNumbersModded}\n" +
    f"arStrictTexts={arStrictTexts}\n" +
    f"translatedNumbers={helpers.translateNumbersToArabic(enNumbers)}\n\n" +
    f"enResult={enResult}\n\narResult={arResult}")
    
    return ocrText, isArabic, debugInfo