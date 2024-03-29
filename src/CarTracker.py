#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 22:03:49 2023
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Dependencies ##############################
#Native
import math
from datetime import datetime

#Image Manipulation
import cv2

#Math Libs
import numpy 

############################## Custom Modules ##############################
from Car import Car
import trafficLight
import config
import helpers
import messager

############################## CarTracker Class ##############################
class CarTracker:
    #Violation types.
    VTYPE_SPEEDING = "Driving over the speed limit!"
    VTYPE_REVERSING = "Driving in the wrong direction!"
    VTYPE_PARKING = "Illegally parked!"
    VTYPE_RED_LIGHT = "Passed red light!"
    
    def __init__(self):
        #List of registered cars.
        self.cars = []
        
        #Only used for file names.
        self.violationCount = 0
        self.screenshotCount = 0
        self.mog2ScreenshotCount = 0

        #A number of kernels used for object detection.
        self.kernalOp = numpy.ones((3,3), numpy.uint8)
        self.kernalOp2 = numpy.ones((5,5), numpy.uint8)
        self.kernalCl = numpy.ones((11,11), numpy.uint8)
        self.kernal_e = numpy.ones((5,5), numpy.uint8)
        
        self.detector = cv2.createBackgroundSubtractorMOG2(
                varThreshold=config.VAR_THRESHOLD,
                detectShadows=False)
        #print(self.detector.getShadowThreshold())   #0.5
        #self.detector.setShadowThreshold(0.03)
        
        #print(self.detector.getShadowValue())  #127
        #self.detector.setShadowValue(5)
        
        #print(self.detector.getNMixtures())    #5
        #self.detector.setNMixtures(3)
        
        

############################## Core API ##############################
    #Called once per frame with a list of objs in that frame.
    def processVideo(self, path):
        print(f"Processing video: {path}")
        video = cv2.VideoCapture(path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frameCount = 0
        
        if config.SKIP_FRAMES > 0:
            for i in range(0, config.SKIP_FRAMES):
                video.read()
            frameCount = config.SKIP_FRAMES
            
        while True:
            #Read the next frame.
            nxt, frame = video.read()
            if not nxt: break
            frameCount += 1
            #print("Processing next frame.")
            #Scale down a bit, and get dims.
            #frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            
            #If an ROI is specified, crop to it
            roi = frame
            if hasattr(config, "FRAME_ROI"):
                r = config.FRAME_ROI
                roi = frame[r[0]:r[1],r[2]:r[3]]
            else:
                roi = frame
            #Apply object detection.
            '''
            rgb_planes = cv2.split(roi)

            result_planes = []
            result_norm_planes = []
            for plane in rgb_planes:
                dilated_img = cv2.dilate(plane, numpy.ones((7,7), numpy.uint8))
                bg_img = cv2.medianBlur(dilated_img, 21)
                diff_img = 255 - cv2.absdiff(plane, bg_img)
                norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                result_planes.append(diff_img)
                result_norm_planes.append(norm_img)
            result = cv2.merge(result_planes)
            result_norm = cv2.merge(result_norm_planes)
            '''          
            #th = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
            #img_dilation = cv2.dilate(img, kernel, iterations=1)
            _, thresholdRoi = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY)
            detectionsMask = self.detector.apply(thresholdRoi)
            
            #Apply a threshold layer, and 2 morphology layers to extract objects.
            _, binary = cv2.threshold(detectionsMask, 200, 255, cv2.THRESH_BINARY)
            mask1 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernalOp)
            mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, self.kernalCl)
            #Final image with object highlights.
            eroded = cv2.erode(mask2, self.kernal_e)
            contours,_ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if config.TAKE_MOG2_SCREENSHOTS and  (frameCount / fps) in config.MOG2_SCREENSHOT_TIMES:
                print("Taking screenshot of MOG2 masks!")
                self._takeMog2Screenshot(roi, thresholdRoi, detectionsMask,
                                         mask1, mask2, eroded)
            if (config.EXIT_AFTER_SCREENSHOTS_COMPLETION and
                    all(frameCount >= x * fps for x in config.MOG2_SCREENSHOT_TIMES) and
                    all(frameCount >= x * fps for x in config.SCREENSHOT_TIMES)):
                print("Took all specified MOG2 screenshots, exitting quickly.")
                break
            
            objects = []
            for c in contours:
                if config.SHOW_BBOX:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 3)
                #If large enough, consider to be a car.
                if cv2.contourArea(c) > config.MIN_CAR_SIZE:
                    x, y, w, h = cv2.boundingRect(c)
                    objects.append([x, y, w, h])
                    #Used for drawing.                    
                    if config.SHOW_BBOX_SIZE:
                        bs = helpers.formatNumber(cv2.contourArea(c))
                        cv2.putText(roi, f"bs={bs}",(x, y + 30), 
                                    cv2.FONT_HERSHEY_PLAIN,
                                    7,
                                    (255, 128, 0),
                                    10)                    
            #Compute speeds and check for violations.
            self.processFrame(roi, objects, fps, frameCount)
            for car in self.cars:
                if config.SHOW_CAR_BBOX:
                    x, y, w, h = car.getBbox()
                    cv2.rectangle(roi, (x, y), (x + w, y + h), car.color, 4)
                speed = car.getSpeed()
                if config.SHOW_CAR_SPEED and speed != car.INVALID_STATE:
                    x, y, w, h = car.getBbox()
                    cv2.putText(roi, f"{speed}km/h",(x, y - 15), 
                                cv2.FONT_HERSHEY_PLAIN,
                                7,
                                (0, 0, 255),
                                10)
                    
            #DAhmed has already been waiting for ages an exraw distance lines.
            h, w, _ = frame.shape
            a = config.ENTRY_LINE_Y
            b = config.EXIT_LINE_Y
            stroke = 4
            
            color = (255, 0, 0)
            cv2.line(roi, (0, a), (w, a), color, stroke)
    
            color = (0, 255, 0)
            cv2.line(roi, (0, b), (w, b), color, stroke)
            
            #Draw fps
            cv2.putText(roi, f"FPS: {fps:.0f}",(0, 100), 
                        cv2.FONT_HERSHEY_PLAIN,
                        7,
                        (0, 0, 255),
                        10)
            #Draw traffic light
            trafficLightColor = (0, 64, 0) if trafficLight.allowPassage() else (0, 0, 64)
            cv2.rectangle(roi, (550, 40), (550 + 50, 40 + 50), trafficLightColor, -1)
            
            #Save screenshots
            if config.TAKE_SCREENSHOTS and (frameCount / fps) in config.SCREENSHOT_TIMES:
                print("Taking screenshot of current frame!")
                self._takeScreenshot(roi)
                if (config.EXIT_AFTER_SCREENSHOTS_COMPLETION and
                        all(frameCount >= x * fps for x in config.MOG2_SCREENSHOT_TIMES) and
                        all(frameCount >= x * fps for x in config.SCREENSHOT_TIMES)):
                    print("Took all specified screenshots, exitting quickly.")
                    break
            
            #Display to user.
            #cv2.namedWindow("roi", cv2.WINDOW_KEEPRATIO)
            cv2.imshow("roi", roi)
            #cv2.resizeWindow("roi", 800, 800)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            if key == ord('r'): trafficLight.onPress('r')
        video.release()
        cv2.destroyAllWindows()
    def processFrame(self, frame, objs, fps, frameCount):
        #Loop over all objs in the current frame.
        #print("Looping over all objects an checking for speeds")
        for obj in objs:
            cx, cy = self._getCenter(obj)
            newObj = True
            #For every obj in frame, loop over all previous objs and look for matches to update.
            for car in self.cars:
                carCx, carCy = car.getCenter()
                dist = math.hypot(cx - carCx, cy - carCy)

                if dist < config.MAX_DISPLACEMENT:
                    #Then `car` is the same object as `obj`, which has been processed before.
                    newObj = False
                    
                    #Update its bbox.
                    car.setBbox(obj)
                    
                    #Update its timers.
                    x, y, w, h = car.getBbox()
                    
                    if y <= config.ENTRY_LINE_Y and y + h >= config.ENTRY_LINE_Y:
                        #print("Setting start time")
                        car.setStartTime(frameCount)
                            
                    if y <= config.EXIT_LINE_Y and y + h >= config.EXIT_LINE_Y:
                        #print("Setting end time")
                        car.setEndTime(frameCount)
                        #print(f"[DEBUG] Total time: {car.getTime()} Speed {car.getSpeed()}")
                        
                    #if dist < config.MAX_PARKED_DISPLACEMENT and not car.isProcessed():
                    #    car.setProcessed(True)
                    #    self.onViolation(car, frame, self.VTYPE_PARKING)
                
            #If newly detected object, create a car instance for it.
            if newObj:
                print("New object entered frame.")
                self.cars.append(Car(obj, fps))
        
        #Remove any duplicate objects.
        removals = []
        for i, c in enumerate(self.cars):
            for j in range(i + 1, len(self.cars)):
                c2 = self.cars[j]
                if c == c2: continue
                if any(j == item for item in removals): continue
                cx, cy = c.getCenter()
                cx2, cy2 = c2.getCenter()
                dist = math.hypot(cx - cx2, cy - cy2)
                if dist < config.MAX_DISPLACEMENT:
                    removals.append(j)
        removals.sort()
        for i in range(len(removals) - 1, -1, -1):
            #print("Removed duplicate!")
            del self.cars[removals[i]]
        
        #Once all processing is done, loop over all cars, check for any violations,
        #    if yes; issue onViolation callback.
        for i in range(len(self.cars) - 1, -1, -1):
            car = self.cars[i]
            if car.isProcessed(): continue
        
            if car.isComplete():
                print(f"Car complete. Checking violations. t={car.getTime():.3}")
                
                #print(f"[DEBUG] Complete. Speed {car.getSpeed()}")
                car.setProcessed(True)
                self.checkViolation(car, frame)
                
        
        #If car has no matching objects, remove it from the list.
        for i in range(len(self.cars) - 1, -1, -1):
            car = self.cars[i]
            carCx, carCy = car.getCenter()
            foundMatch = False
            for obj in objs:
                cx, cy = self._getCenter(obj)
                dist = math.hypot(cx - carCx, cy - carCy)
                if dist < config.MAX_DISPLACEMENT:
                    car.resetStaleFrames()
                    foundMatch = True
                    break
            if not foundMatch:
                car.incrementStaleFrames()
                if car.isStale():
                    del self.cars[i]        
        
############################## Violation Methods ##############################
    def checkViolation(self, car, frame):
        speed = car.getSpeed()
        if not trafficLight.allowPassage() and speed != 0:
            self.onViolation(car, frame, self.VTYPE_RED_LIGHT)
            
        if speed > config.MAX_SPEED:
            self.onViolation(car, frame, self.VTYPE_SPEEDING)
        elif speed < 0:
            self.onViolation(car, frame, self.VTYPE_REVERSING)
    
    def onViolation(self, car, frame, vtype):
        print(f"Got violation of type \"{vtype}\". speed={car.getSpeed()}")
        
        #Inc violation counter, only used for file names.
        self.violationCount += 1
        
        #Crop car out of the image.
        x, y, w, h = car.getBbox()
        image = frame[y:y + h, x:x + w]
        
        #Fetch plate info.
        plateText, img, isArabic, _ = helpers.extractPlate(image, False)
        
        #Prepare and write image file.
        imagePath = config.VIOLATION_IMAGE_PATH +\
                config.RUN_START_TIME + "/" +\
                str(self.violationCount) + ".png"
                
        cv2.imwrite(imagePath, image)
        
        #Prepare text.
        text = "Violation: " + vtype + "\n"
        if vtype != self.VTYPE_PARKING:
            text += "Speed: " + str(car.getSpeed()) + "km/h\n"
        text += "Plate Number: " + plateText + "\n"
        text += "Location: " + config.VIOLATION_LOCATION + "\n"
        text += "Time: " + datetime.now().strftime(config.TIME_FORMAT) + "\n"
        #Send SMS.
        messager.sendViolation(text)
        
        #Prepare text file; write violation type and plate name to it.
        textPath = config.VIOLATION_PLATE_PATH +\
            config.RUN_START_TIME + "/" +\
            str(self.violationCount) + ".txt"
        file = open(textPath, "w", encoding="utf-8")
        file.write(text)
        file.close()

############################## Internals ##############################    
    def _getCenter(self, obj):
        x, y, w, h = obj
        return (x * 2 + w) // 2, (y * 2 + h) // 2
    
    def _takeScreenshot(self, image):
        self.screenshotCount += 1
        imagePath = config.SCREENSHOT_PATH +\
                config.SCREENSHOT_NAME_PREFIX + config.RUN_START_TIME + "/" +\
                str(self.screenshotCount) + ".png"
                
        cv2.imwrite(imagePath, image)
    
    def _takeMog2Screenshot(self, roi, thresholdRoi, detectionsMask, mask1, mask2, eroded):
        self.mog2ScreenshotCount += 1
        path = config.MOG2_SCREENSHOT_PATH + config.RUN_START_TIME + "/" +\
                str(self.mog2ScreenshotCount)
        cv2.imwrite(path + " roi.png",    roi) 
        cv2.imwrite(path + " thresholdRoi.png",    thresholdRoi) 
        cv2.imwrite(path + " detectionsMask.png",  detectionsMask)
        cv2.imwrite(path + " mask1.png",           mask1)
        cv2.imwrite(path + " mask2.png",           mask2)
        cv2.imwrite(path + " eroded.png",          eroded)
        
############################## Getters ##############################
