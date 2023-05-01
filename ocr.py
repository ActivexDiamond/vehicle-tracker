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
english_ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en", rec_algorithm="CRNN", show_log=config.SHOW_ENGLISH_LOG)
arabic_ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="ar", rec_algorithm="SVTR_LCNet", show_log=config.SHOW_ARABIC_LOG)
arabic_reader = Reader(["ar"], gpu=-1 > 0)

############################## OCR Helpers ##############################
def performBilingualOcr(image, image_path):
    # Perform OCR.
    english_texts = []
    english_scores = []
    english_confidence = 0
    english_result = english_ocr.ocr(image, cls=True)
    # Concat all text into a single string.
    for j in range(len(english_result)):
        r = english_result[j]
        for line in r:
            english_texts.append(line[1][0])
            english_scores.append(line[1][1])

    # Identical to the above, but for Arabic.
    arabic_texts = []
    arabic_numbers = []
    arabic_scores = []
    arabic_confidence = 0
    arabic_paddle_result = arabic_ocr.ocr(image, cls=True)
    arabic_reader = Reader(["ar"], gpu=-1 > 0)
    arabic_reader_result = arabic_reader.readtext(image)
    
    # Concat all text into a single string.
    # For PaddleOCR, only grab plate text.
    for k in range(len(arabic_paddle_result)):
        r = arabic_paddle_result[k]
        for line in r:
            string = line[1][0]
            # If string contains MORE than one Arabic letter, consider it
            # the plate text. 
            arabic_letter_count = 0
            for char in string:
                if any(char == letter for letter in helpers.ARABIC_LETTERS):
                    arabic_letter_count += 1
            if arabic_letter_count > 1:
                arabic_texts.append(string)    
                arabic_scores.append(line[1][1])
    
    # For Reader, only grab plate number.
    for k in range(len(arabic_reader_result)):
        r = arabic_reader_result[k]
        string = r[1]
        # If string contains LESS than one Arabic letter, consider it
        # the plate number. 
        arabic_letter_count = 0
        for char in string:
            if any(char == letter for letter in helpers.ARABIC_LETTERS):
                arabic_letter_count += 1
        if arabic_letter_count <= 1:
            arabic_numbers.append(string)    
            arabic_scores.append(r[2])
                
    # If any OCRs occurred; compute their mean.
    if len(english_scores) > 0:
        english_confidence = sum(english_scores) / len(english_scores)
    if len(arabic_scores) > 0:
        arabic_confidence = sum(arabic_scores) / len(arabic_scores)

    # Fallbakc in case OCR failed.
    ocr_text = "Unknown"
    is_arabic = False

    # Check scores to deduce text language.
    if english_confidence >= arabic_confidence and english_confidence >= config.MIN_CONFIDENCE:
        ocr_text = " ".join(english_texts)
    elif arabic_confidence >= config.MIN_CONFIDENCE:
        ocr_text = " ".join(arabic_texts) + "=====" + " ".join(arabic_numbers)
        is_arabic = True

    return ocr_text, is_arabic