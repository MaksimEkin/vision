#!/usr/bin/env python

from picamera import PiCamera
from time import sleep
import os

camera = PiCamera()
camera.rotation = 90
while(True):
	os.rename('/home/pi/Desktop/jarvis-master/image/buffer.jpg',\
              '/home/pi/Desktop/jarvis-master/image/current_view.jpg')
    
	camera.capture('/home/pi/Desktop/jarvis-master/image/buffer.jpg')
	sleep(1)
