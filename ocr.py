############################## Text Helpers##############################
def performOcr(image):
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
    arabic_scores = []
    arabic_confidence = 0
    arabic_result = arabic_ocr.ocr(image, cls=True)
    # Concat all text into a single string.
    for k in range(len(arabic_result)):
        r = arabic_result[k]
        for line in r:
            arabic_texts.append(line[1][0])
            arabic_scores.append(line[1][1])

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
        ocr_text = " ".join(arabic_texts)
        is_arabic = True

    return ocr_text, is_arabic