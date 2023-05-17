#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 03 05:10:57 2023
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## State ##############################
mayPass = True

############################## Callbacks ##############################
def onPress(key):
    global mayPass
    if key == 'r':
        mayPass = not mayPass        
    state = "Green" if mayPass else "Red"
    print(f"Traffic-Control is now set to: {state}")
    
############################## Traffic Control ##############################
#Should return True if traffic-light control is green.
#False if red!
def allowPassage():
    return mayPass