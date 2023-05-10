#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 22:03:49 2023
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Dependencies ##############################
#Native
import math
import time

#Image Manipulation
import cv2

#Math Libs
import numpy 

############################## Custom Modules ##############################
from Car import Car
import config
import helpers

############################## CarTracker Class ##############################
class CarTracker:
    _FPS = 25
    _DELAY = int(1e3 / (_FPS - 1))
    
    #Violation types.
    VTYPE_SPEEDING = "Driving over the speed limit!"
    VTYPE_REVERSING = "Driving in the wrong direction!"
    VTYPE_PARKING = "Illegally parked!"
    def __init__(self):
        #List of registered cars.
        self.cars = []
        #Only used for file names.
        self.violationCount = 0

        #A number of kernels used for object detection.
        self.kernalOp = numpy.ones((3,3), numpy.uint8)
        self.kernalOp2 = numpy.ones((5,5), numpy.uint8)
        self.kernalCl = numpy.ones((11,11), numpy.uint8)
        self.detector = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.kernal_e = numpy.ones((5,5), numpy.uint8)

############################## Core API ##############################
    #Called once per frame with a list of objs in that frame.
    def processVideo(self, path):
        print(f"Processing video: {path}")
        video = cv2.VideoCapture(path)
        
        while True:
            #Read the next frame.
            nxt, frame = video.read()
            if not nxt: break
            #print("Processing next frame.")
            
            #Scale down a bit, and get dims.
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            #If an ROI is specified, crop to it
            roi = frame
            if config.FRAME_ROI:
                r = config.FRAME_ROI
                roi = frame[r[0]:r[1],r[2]:r[3]]
            else:
                roi = frame
            #Apply object detection.
            detectionsMask = self.detector.apply(roi)
            #Apply a threshold layer, and 2 morphology layers to extract objects.
            _, binary = cv2.threshold(detectionsMask, 200, 255, cv2.THRESH_BINARY)
            mask1 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernalOp)
            mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, self.kernalCl)
            #Final image with object highlights.
            image = cv2.erode(mask2, self.kernal_e)
    
            contours,_ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            objects = []
            for c in contours:
                #If large enough, consider to be a car.
                if cv2.contourArea(c) > config.MIN_CAR_SIZE:
                    x, y, w, h = cv2.boundingRect(c)
                    objects.append([x, y, w, h])
                    #Used for drawing.                    
                    cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    
            #Compute speeds and check for violations.
            self.processFrame(roi, objects)
            
            for car in self.cars:
                x, y, w, h = car.getBbox()
                cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 255, 0), 2)
                
                speed = car.getSpeed()
                if speed != car.INVALID_STATE:
                    cv2.putText(roi, f"{speed}km/h",(x, y - 15), 
                                cv2.FONT_HERSHEY_PLAIN,
                                1,
                                (0, 0, 255),
                                2)
                    
            #Draw distance lines.
            h, w, _ = frame.shape
            a = config.ENTRY_LINE_Y
            b = config.ENTRY_LINE_Y + config.LINE_Y_OFFSET
            c = config.EXIT_LINE_Y
            d = config.EXIT_LINE_Y + + config.LINE_Y_OFFSET
            
            color = (255, 0, 0)
            stroke = 1
            
            cv2.line(roi, (0, a), (w, a), color, stroke)
            cv2.line(roi, (0, b), (w, b), color, stroke)
    
            cv2.line(roi, (0, c), (w, c), color, stroke)
            cv2.line(roi, (0, d), (w, d), color, stroke)
            
            #Display to user.
            cv2.imshow("roi", roi)
            key = cv2.waitKey(self._DELAY - 10)
            
            if key == 27: break
        video.release()
        cv2.destroyAllWindows()

    def processFrame(self, frame, objs):
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
                    
                    #Update its bbox and timers.
                    car.setBbox(obj)
                    if cy >= config.ENTRY_LINE_Y and cy <= config.ENTRY_LINE_Y + config.LINE_Y_OFFSET:
                        #print("Setting start time")
                        car.setStartTime(time.time())
                            
                    if cy >= config.EXIT_LINE_Y and cy <= config.EXIT_LINE_Y + config.LINE_Y_OFFSET:
                        #print("Setting end time")
                        car.setEndTime(time.time())
                        #print(f"[DEBUG] Total time: {car.getTime()} Speed {car.getSpeed()}")
            #If newly detected object, create a car instance for it.
            if newObj:
                print("New object entered frame.")
                self.cars.append(Car(obj))
        
        #Once all processing is done, loop over all cars, check for any violations,
        #    if yes; issue onViolation callback.
        for i in range(len(self.cars) - 1, -1, -1):
            car = self.cars[i]
            if car.isProcessed(): continue
        
            if car.isComplete():
                print("Car complete. Checking violations.")
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
                    foundMatch = True
                    break
            if not foundMatch:
                del self.cars[i]        
        
############################## Violation Methods ##############################
    def checkViolation(self, car, frame):
        speed = car.getSpeed()
        if speed > config.MAX_SPEED:
            self.onViolation(car, frame, self.VTYPE_SPEEDING)
        if speed < 0:
            self.onViolation(car, frame, self.VTYPE_REVERSING)
        if speed < 1 and speed > -1:
            self.onViolation(car, frame, self.VTYPE_PARKING)
    
    def onViolation(self, car, frame, vtype):
        print(f"Got violation of type \"{vtype}\". speed={car.getSpeed()}")
        #Inc violation counter, only used for file names.
        self.violationCount += 1
        #Crop car out of the image.
        x, y, w, h = car.getBbox()
        image = frame[y - 5:y + h + 5, x - 5:x + w + 5]
        
        #Prepare and write image file.
        imagePath = config.VIOLATION_IMAGE_PATH + str(self.violationCount) + ".png"
        cv2.imwrite(imagePath, image)
        
        #Prepare text file; write violation type and plate name to it.
        textPath = config.VIOLATION_PLATE_PATH + str(self.violationCount) + ".txt"
        file = open(textPath, "w")
        file.write("Violation: " + vtype + "\n")
        file.write("Speed: " + str(car.getSpeed()) + "km/h\n")
        plateText, img, isArabic = helpers.extractPlate(image, False)
        file.write("Plate Number: " + plateText + "\n")
        file.close()

############################## Internals ##############################    
    def _getCenter(self, obj):
        x, y, w, h = obj
        return (x * 2 + w) // 2, (y * 2 + h) // 2
############################## Getters ##############################