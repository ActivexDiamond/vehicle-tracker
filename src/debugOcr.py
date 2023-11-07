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
        "ا ٥۲۲۲۱ كربلاء خصوصي",
        "ا ٥٤٤۸٦ كربلاء خصوصي",
        "و ٦٥۹۷۳ بغداد خصوصي",
        "ا ٤٤٥۸۹ كربلاء خصوصي",
        "ا ٤۲٦٤٥ الانبار خصوصي",
        "ا ۲۱۸۸۷ النجف اجرة",
        "ا ۲۷۰۸۷ كربلاء اجرة",
        "ا ٥۳۸٦٦ كربلاء خصوصي",
        "ا ٥٤٦۲۷ كربلاء خصوصي",
        "ا ۱۷٥٦۲ كربلاء اجرة",
        "ا ۲۸۸۲۳ كربلاء خصوصي",
        "ا ۲۳۸۰۱ كربلاء خصوصي" ,        
        "۲۲٦۹۹٥ العراق اربيل",
        "ا ۲٦٦٦٤ كربلاء اجرة",
        "لبيك يا مهدي",
        "ا ٦۳۸۳۷ كربلاء خصوصي",

    ],[ #4
 "ا ۱۰۹۲۸ كربلاء اجرة",
        "Unkown",
        "ا ٥۱٥٤۹ كربلاء خصوصي",
        "ا ۲۳۱٥۲ كبرلاء اجرة",
        "Unknown",
        "ا ۲۰۹٥۰ كربلاء اجرة",
        "ر ٥۲٦٤۰ بغداد خصوصي",
        "ا ۲۳۷۸۱ كربلاء خصوصي",
       "21G 38492",
        "۹۷۱٦۹٤ اربيل عراق",
        "Unknown",
        "Unknown",
        "۲۰۰۳٦۲ اربيل العراق",
        "ا ٥٥۲۱٦ بغداد خصوصي",
        "Unknown",
        "٥۰۷۸۳٦ اربيل العراق",
        "ا ۸٦٥۰۹ بغداد خصوصي",
        "ا ٤۳۷۷۷ كربلاء خصوصي",
        "ا ۲۹۸۳۹ كربلاء اجرة",
        "ا ۳٥٤٤۸ كربلاء اجرة",
        "ط ۱۲٤٥۹ يغداد خصوصي",
        "۸۱۳۲۰ بغداد خصوصي",
        "Unknown",
        "ا ٥۹٤۷۹ كربلاء",
        "ا ۲۱٥۱۸ كربلاء اجرة",
        "ب ۱۹۲۳٦ بغداد خصوصي",       
        "ا ۱۳۹۱٦ كربلاء خصوصي",
        "ا ۳٥۸٤٥ النجف اجرة",
        "Unknown",
        "Unknown",
        "Unknown",
        "ا ۱۹۷۸۲ كربلاء دراجة",
        "ب ۲۱۱٤۲ بابل اجرة",
        "ا ٦٤٥۸۷ النجف خصوصي",
        "Unknown",
        "و ۲۹۳٥۲ بغداد خصوصي",
        "Unknown",
        "Unknown",
        "۱۹۱۳۱۱ سليمانية خصوصي",
        "Unknown",
        "Unknown",
        "ا ٦٤۱۹۲ كربلاء خصوصي",
        "۱۹۰۰٤۷ اربيل خصوصي",
        "Unknown",
      "ط ۷۰۹۸۷ بغداد خصوصي"
        "ط ٥۷۲٦۱ بغداد اجرة",
        "ا ۲٥٤٥۱ القادسية حمل",
        "ا ٥٤۳۱٥ كربلاء خصوصي",
        "۱۱۳٥۸۸ كربلاء",
        "ا ۱٤۸۸ كربلاء خصوصي",
        "ف ۷٥۰٥٥ بغداد اجرة",
        "Unknown",
        "۲٦۸۹۲ كربلاء اجرة",
        "ك ۷۷۱۰٦ بغداد خصوصي",
        "Unknown",
        "۱۰۰٥۷ اجرة",
        "Unknown",
    ],[ #5
       
    ]
]

RECORDED_MAX_SPEEDS = [
    0,  #0    
    0,  #1
    0,  #2
    75, #3
    73, #4
    75, #5
]

VIOLATION_LIST = RECORDED_VIOLATIONS_RAW[config.SELECTED_VIDEO]

if config.MAX_SPEED != RECORDED_MAX_SPEEDS[config.SELECTED_VIDEO]:
    print("=============== !!! ===============")
    print("[WARN] Max speed in `config.py` and in recorded list do not match. Overriding!")
    print("=============== !!! ===============")
    config.MAX_SPEED = RECORDED_MAX_SPEEDS[config.SELECTED_VIDEO]

############################## State ##############################
violationCount = -1

############################## Core API ##############################
def debugOcr(image):
    global violationCount
    # Fallback in case OCR failed.
    ocrText = "Unknown"
    isArabic = False
    debugInfo = ""
    
    violationCount += 1
    if violationCount >= len(VIOLATION_LIST): 
        print("end of list")
        return "Unknown", False, ""
    
    violation = VIOLATION_LIST[violationCount]
    if violation:       #For Arabic
        print("Found in list")
        return violation, True, ""

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


    #This is only needed so that debugInfo doesn't throw an UnboundLocalError
    #   if no modding happened.
    arNumbersModded = []
    translatedNumbers = []
    
    ocrText = " ".join(enTexts) + " " + " ".join(enNumbers)
    
    debugInfo = (f"enConfidence={enConfidence}\tenTexts={enTexts}\tenNumbers={enNumbers}\n" +
    f"arConfidence={arConfidence}\tarTexts={arTexts}\tarNumbers={arNumbers}\n" +
    f"arNumbersModded={arNumbersModded}\n" +
    f"arStrictTexts={arStrictTexts}\n" +
    f"translatedNumbers={helpers.translateNumbersToArabic(enNumbers)}\n\n" +
    f"enResult={enResult}\n\narResult={arResult}")
    print(ocrText, isArabic, debugInfo)
    return ocrText, isArabic, debugInfo
    
    