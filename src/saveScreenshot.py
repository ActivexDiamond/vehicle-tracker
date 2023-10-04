#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 00:13:46 2023
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

import os
import cv2

VIDEO_NAME = "side-day3"

VIDEO_PATH =  "./test/videos1/" + VIDEO_NAME + ".MP4"
OUTPUT_PATH = "test/screens_" + VIDEO_NAME + "/"
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    
print(f"Processing video: {VIDEO_PATH}")
video = cv2.VideoCapture(VIDEO_PATH)
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = 0

cv2.namedWindow('Saver', 0)
cv2.resizeWindow("Saver", 900, 900)
while True:
    nxt, frame = video.read()
    if not nxt: break
    frameCount += 1
    cv2.imshow("Saver", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('s'):
        print("Saving screenshot...")
        path = OUTPUT_PATH + str(frameCount) + ".jpg"
        cv2.imwrite(path, frame)
video.release()
cv2.destroyAllWindows()
print("Done!")