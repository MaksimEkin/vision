#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: maksim
"""
# Libraries needed
import speech_recognition as sr

def recordAudio():
    sample_rate = 48000
    chunk_size = 2048

    # Record Audio
    r = sr.Recognizer()
    with sr.Microphone(device_index = 0, sample_rate = 48000) as source:
        audio = r.record(source, duration = 2)
    with open("x.wav", "wb") as f:
        f.write(audio.get_wav_data())
        
    with sr.AudioFile("x.wav") as source:
        audio = r.listen(source)
         
    # Speech recognition using Google Speech Recognition
    data = ""
    try:
        # Uses the default API key
        # To use another API key: `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        data = r.recognize_google(audio)
        print("You said: " + data)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
         
    return data