#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: maksim
"""
# Libraries needed
import os
from gtts import gTTS

def speak(audioString):
    print(audioString)
    tts = gTTS(text=audioString, lang='en')
    tts.save("audio.mp3")
    os.system("mpg321 audio.mp3")
