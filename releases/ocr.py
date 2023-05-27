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

############################## Custom Modules ##############################
import config
import helpers

############################## Init ##############################
# Setup both OCRs.
englishOcr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en", rec_algorithm="CRNN", show_log=config.SHOW_ENGLISH_LOG)
arabicOcr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ar", rec_algorithm="SVTR_LCNet", show_log=config.SHOW_ARABIC_LOG)
arabicReader = Reader(["ar"], gpu=-1 > 0)

############################## OCR Helpers ##############################
def performBilingualOcr(image):
    # Perform OCR.
    enResult = englishOcr.ocr(image, cls=True)
    arResult = arabicOcr.ocr(image, cls=True)
    
    # Format paddle string into friendlier format.
    enTexts, enNumbers, enScores = helpers.formatPaddleResult(enResult)
    arTexts, arNumbers, arScores = helpers.formatPaddleResult(arResult)
    
    # Check if OCRs occured between computing their means.
    enConfidence = 0
    if len(enScores) > 0:
        enConfidence = sum(enScores) / len(enScores)
    arConfidence = 0
    if len(arScores) > 0:
        arConfidence = sum(arScores) / len(arScores)

    # Fallback in case OCR failed.
    ocrText = "Unknown"
    isArabic = False
    
    #This is only needed so that debugInfo doesn't through an UnboundLocalError.
    arNumbersModded = []
    
    # Check scores to deduce text language.
    if enConfidence >= arConfidence and enConfidence >= config.MIN_CONFIDENCE:
        ocrText = " ".join(enTexts) + " ".join(enNumbers)
    elif arConfidence >= config.MIN_CONFIDENCE:
        if len(enNumbers) > 1:
            translatedNumbers = helpers.translateNumbersToArabic(enNumbers)
            ocrText = " ".join(arTexts) + " ".join(translatedNumbers)
        else:
            #Remember, the OCR returns a list that could contain multiple strings.
            if len(arNumbers) > 0:
                arNumbersModded = [None] * len(arNumbers)
                for num in arNumbers:
                    if num[0] == "١":
                        arNumbersModded = "أ" + num[1:]
                    else:
                        arNumbersModded = num
            ocrText = " ".join(arTexts) + " ".join(arNumbersModded)
        isArabic = True
    
    debugInfo = (f"enConfidence={enConfidence}\tenTexts={enTexts}\tenNumbers={enNumbers}\n" +
    f"arConfidence={arConfidence}\tarTexts={arTexts}\tarNumbers={arNumbers}\n" +
    f"arNumbersModded={arNumbersModded}\n" +
    f"translatedNumbers={helpers.translateNumbersToArabic(enNumbers)}\n\n" +
    f"enResult={enResult}\n\narResult={arResult}")
    
    return ocrText, isArabic, debugInfo