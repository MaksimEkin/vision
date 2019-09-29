#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: maksim
"""
# Libraries needed
import time
import recordAudio
import speak
import jarvis

# Main Function
def main():
         
    # initialization
    time.sleep(2)
    #speak.speak("Hi, what can I do for you?")
    speak.speak("Hello, how can I help you?")
    while 1:
        data = recordAudio.recordAudio()
        jarvis.jarvis(data)
    
if __name__ == '__main__':
    main()
