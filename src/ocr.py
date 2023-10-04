#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 17:15:25 2023
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""


############################## Dependencies ##############################
import paddleocr
print(f"Imported paddleocr=={paddleocr.__version__}")

import easyocr 
from easyocr import Reader
print(f"Imported easyocr=={easyocr.__version__}")

#import cv2

############################## Custom Modules ##############################
import config
import helpers
  
if hasattr(config, "DEBUG_OCR") and config.DEBUG_OCR:
    from debugOcr import debugOcr

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
    
############################## OCR Helpers ##############################
def performBilingualOcr(image):
    if not config.PERFORM_OCR: 
        return "OCR disabled.", False, "OCR disabled."
    if hasattr(config, "DEBUG_OCR") and config.DEBUG_OCR:
        return altOcr(image)
    
    # Perform OCR.
    enResult = englishOcr.ocr(image, cls=True)
    arResult = arabicOcr.ocr(image, cls=True)
    
    # Format paddle string into friendlier format.
    enTexts, enNumbers, enScores = helpers.formatPaddleResult(enResult)
    arTexts, arNumbers, arScores = helpers.formatPaddleResult(arResult)
    arStrictTexts = helpers.strictifyArabicTexts(arTexts)
    
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
    
    debugInfo = (f"enConfidence={enConfidence}\tenTexts={enTexts}\tenNumbers={enNumbers}\n" +
    f"arConfidence={arConfidence}\tarTexts={arTexts}\tarNumbers={arNumbers}\n" +
    f"arNumbersModded={arNumbersModded}\n" +
    f"arStrictTexts={arStrictTexts}\n" +
    f"translatedNumbers={helpers.translateNumbersToArabic(enNumbers)}\n\n" +
    f"enResult={enResult}\n\narResult={arResult}")
    
    return ocrText, isArabic, debugInfo