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
    englishTexts = []
    englishScores = []
    englishConfidence = 0
    englishResult = englishOcr.ocr(image, cls=True)
    # Concat all text into a single string.
    for j in range(len(englishResult)):
        r = englishResult[j]
        for line in r:
            englishTexts.append(line[1][0])
            englishScores.append(line[1][1])

    # Identical to the above, but for Arabic.
    arabicTexts = []
    arabicNumbers = []
    arabicScores = []
    arabicConfidence = 0
    arabicPaddleResult = arabicOcr.ocr(image, cls=True)
    arabicReader = Reader(["ar"], gpu=-1 > 0)
    arabicReaderResult = arabicReader.readtext(image)
    
    # Concat all text into a single string.
    # For PaddleOCR, only grab plate text.
    for k in range(len(arabicPaddleResult)):
        r = arabicPaddleResult[k]
        for line in r:
            string = line[1][0]
            # If string contains MORE than one Arabic letter, consider it
            # the plate text. 
            arabicLetterCount = 0
            for char in string:
                if any(char == letter for letter in helpers.ARABIC_LETTERS):
                    arabicLetterCount += 1
            if arabicLetterCount > 1:
                arabicTexts.append(string)    
                arabicScores.append(line[1][1])
    
    # For Reader, only grab plate number.
    for k in range(len(arabicReaderResult)):
        r = arabicReaderResult[k]
        string = r[1]
        # If string contains LESS than one Arabic letter, consider it
        # the plate number. 
        arabicLetterCount = 0
        for char in string:
            if any(char == letter for letter in helpers.ARABIC_LETTERS):
                arabicLetterCount += 1
        if arabicLetterCount <= 1:
            arabicNumbers.append(string)    
            arabicScores.append(r[2])
                
    # If any OCRs occurred; compute their mean.
    if len(englishScores) > 0:
        englishConfidence = sum(englishScores) / len(englishScores)
    if len(arabicScores) > 0:
        arabicConfidence = sum(arabicScores) / len(arabicScores)

    # Fallbakc in case OCR failed.
    ocrText = "Unknown"
    isArabic = False

    # Check scores to deduce text language.
    if englishConfidence >= arabicConfidence and englishConfidence >= config.MIN_CONFIDENCE:
        ocrText = " ".join(englishTexts)
    elif arabicConfidence >= config.MIN_CONFIDENCE:
        ocrText = " ".join(arabicTexts) + "=====" + " ".join(arabicNumbers)
        isArabic = True
    
    return ocrText, isArabic