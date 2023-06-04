#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 06:00:04 2023
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Dependencies ##############################
from twilio.rest import Client

############################## Custom Modules ##############################
import config

############################## Internals ##############################
client = Client(config.TWILIO_SID, config.TWILIO_AUTH_TOKEN)

############################## API ##############################
def sendViolation(body):
    if not config.SEND_SMS: 
        print("===== Messaging disabled. Fake text=====")    
        print(body)
        print("========================================")
        return False
    for num in config.TARGET_PHONE_NUMBERS:
        msg = client.messages.create(body=body, to=num,
                from_=config.SENDER_PHONE_NUMBER)
        print(f"Sent SMS! receiver={num}\tSID={msg.sid}\tText={body}")
    return True
