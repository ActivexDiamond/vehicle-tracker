#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 18:50:37 2023
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Custom Modules ##############################
import helpers
import ocr
import config

############################## Constants ##############################
RECORDED_VIOLATIONS_RAW = [
    [], #0
    [], #1
    [], #2
    [   #3
        False,
        False,
        False,    
        "A123",
        "C123",
        "S4567",
    ],[ #4
       
    ],[ #5
       
    ]
]

RECORDED_MAX_SPEEDS = [
    0,  #0    
    0,  #1
    0,  #2
    75, #3
    75, #4
    75, #5
]

VIOLATION_LIST = RECORDED_VIOLATIONS_RAW[config.SELECTED_VIDEO]

if config.MAX_SPEED != RECORDED_VIOLATIONS_RAW[config.SELECTED_VIDEO]:
    print("[WARN] Max speed in `config.py` and in recorded list do not match. Overriding!")
    config.MAX_SPEED = RECORDED_VIOLATIONS_RAW[config.SELECTED_VIDEO]

############################## State ##############################
violationCount = -1

############################## Core API ##############################
def debugOcr(image):
    global violationCount
    violationCount += 1
    if violationCount >= len(VIOLATION_LIST): 
        print("end of list")
        return "Unkown", False, ""
    
    # Perform OCR.
    enResult = ocr.englishOcr.ocr(image, cls=True)
    arResult = ocr.arabicOcr.ocr(image, cls=True)
    
    # Format paddle string into friendlier format.
    enTexts, enNumbers, enScores = helpers.formatPaddleResult(enResult)
    arTexts, arNumbers, arScores = helpers.formatPaddleResult(arResult)
    arStrictTexts = helpers.strictifyArabicTexts(arTexts)
    
    enConfidence = 0
    if len(enScores) > 0:
        enConfidence = sum(enScores) / len(enScores)
    arConfidence = 0
    if len(arScores) > 0:
        arConfidence = sum(arScores) / len(arScores)

    # Fallback in case OCR failed.
    ocrText = "Unknown"
    isArabic = False

    #This is only needed so that debugInfo doesn't throw an UnboundLocalError
    #   if no modding happened.
    arNumbersModded = []
    translatedNumbers = []
    
    violation = VIOLATION_LIST[violationCount]
    if not violation:       #For English.
        ocrText = " ".join(enTexts) + " " + " ".join(enNumbers)
    else:                   #For Arabic.
        isArabic = True
        translatedNumbers = helpers.translateNumbersToArabic([violation])
        ocrText = " ".join(arStrictTexts) + " " + " ".join(translatedNumbers)

    debugInfo = (f"enConfidence={enConfidence}\tenTexts={enTexts}\tenNumbers={enNumbers}\n" +
    f"arConfidence={arConfidence}\tarTexts={arTexts}\tarNumbers={arNumbers}\n" +
    f"arNumbersModded={arNumbersModded}\n" +
    f"arStrictTexts={arStrictTexts}\n" +
    f"translatedNumbers={helpers.translateNumbersToArabic(enNumbers)}\n\n" +
    f"enResult={enResult}\n\narResult={arResult}")
    print(ocrText, isArabic, debugInfo)
    return ocrText, isArabic, debugInfo
    
    