#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 22:03:49 2023
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Dependencies ##############################

############################## Custom Modules ##############################
import config

############################## Car Class ##############################
class Car:
    INVALID_STATE = {}
    def __init__(self, bbox, fps):
        self.startTime = -1
        self.endTime = -1
        self.bbox = bbox
        self.fps = fps
        self.processed = False
        
############################## Setters ##############################        
    def setProcessed(self, b):
        self.processed = b
    
    def setStartTime(self, t):
        if self.startTime != -1: return
        self.startTime = t

    def setEndTime(self, t):
        if self.endTime != -1: return
        self.endTime = t
        
    def setBbox(self, bbox):
        self.bbox = bbox
        
############################## Getters ##############################        
    def isProcessed(self):
        return self.processed
    
    def isComplete(self):
        return self.startTime != -1 and self.endTime != -1
    
    def getTime(self):
        if not self.isComplete():
            return self.INVALID_STATE
        frameTime = 1 / self.fps
        return (self.endTime - self.startTime) * frameTime
    
    def getSpeed(self):
        if not self.isComplete():
            return self.INVALID_STATE
        
        t = self.getTime()
        if t == 0: return 0              #Div-by-zero guard.
        return int(config.LINES_DISTANCE / t)
            
    def getBbox(self):
        return self.bbox
    
    def getCenter(self):
        x, y, w, h = self.bbox
        return (x * 2 + w) // 2, (y * 2 + h) // 2