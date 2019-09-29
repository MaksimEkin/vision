#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: maksim
"""
# Libraries needed
import os
import speak
from time import ctime
from recognize_object_lite import image_process
import readtext
from time import strftime

def jarvis(data):
    #statement = image_process("/home/pi/Desktop/jarvis-master/image/current_view.jpg")
    #speak.speak(statement)
    if("time" in data):
        speak.speak("It is" + strftime("%H:%M"))
        print("")
    elif ("what is this" in data or ("what is that" in data) or "what is" in data):
        speak.speak("Let me see!")
        statement = image_process("/home/pi/Desktop/jarvis-master/image/current_view.jpg")
        speak.speak(statement)
        print("")
    elif("guide me" in data):
        speak.speak("Let me see!")
        statement = image_process("/home/pi/Desktop/jarvis-master/image/current_view.jpg")
        speak.speak(statement)
        print("")
    elif("where is" in data):
        speak.speak("Let me see!")
        statement = image_process("/home/pi/Desktop/jarvis-master/image/current_view.jpg")
        speak.speak(statement)
        print("")
    elif("read" in data):
        print("hi")
        speak.speak("Hold on, let me read that!")
        response = readtext.driver("/home/pi/Desktop/jarvis-master/image/current_view.jpg")
        print(response)
   
        if (response):
            speak.speak(response)
        else:
            speak.speak("Hmmmm, I don't recognize the text")
        
               