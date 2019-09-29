#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
 @author Maksim Eren
"""


# LIBRARIES NEEDED
# OpenCV2
import cv2


# Numpy for math operations
import numpy as np

# Image processing to extract text
import pytesseract

# library written to hold regex filters
import filter_re

# regex
import re

# Path of working folder on Disk


    
#==============================================================================
# DRIVER
#==============================================================================
def driver(img_path):
    # HOLD EXTRACTED TEXT
    extracted_text = dict()
    
    # Read image with opencv
    img = cv2.imread(img_path)

    # DEFAULT
    extracted_text['DEFAULT'] = get_string(img)
    
    # removen noise
    img = remove_noise(img)
    # GET NO-NOISE TEXT
    extracted_text['NO_NOISE'] = get_string(img)
    # GET ADAPTIVE THRESH TEXT
    #extracted_text['ADAPTIVE_THRESH'] = get_string(adaptive_thresh(img))
    
    
    # get various thresh
    threshs = get_thresh(img)
    
    # binary
    extracted_text['BINARY'] = get_string(threshs[0])
    # inverse binary
    extracted_text['BINARY_INV'] = get_string(threshs[1])
    # trunc
    extracted_text['TRUNC'] = get_string(threshs[2])
    # to-zero
    extracted_text['TOZERO'] = get_string(threshs[3])
    # to-zero inverse
    extracted_text['TOZERO_INV'] = get_string(threshs[4])
    
    data = ""
    # get information from each extracted text
    for key, value in extracted_text.items():
        data = data + value
        print(key, " ", value)
    print(str(data))
    return str(data)  
#==============================================================================
# FORMAT TO PHONE NUMBER
#==============================================================================
def phone_format(n):                                                                                                                                  
    return format(int(n[:-1]), ",").replace(",", "-") + n[-1]      

    
    
#==============================================================================
# EXTRACT PHONE FROM THE TEXT
#==============================================================================
def find_phone(text): 
    return re.findall(filter_re.PHONE_REGEX,text)



#==============================================================================
# EXTRACT EMAIL FROM THE TEXT
#==============================================================================
def find_email(text): 
    return re.findall(filter_re.EMAIL_REGEX,text)



#==============================================================================
# EXTRACT URL FROM THE TEXT
#==============================================================================
def find_url(text): 
    return re.findall(filter_re.WEB_URL_REGEX,text)



#==============================================================================
# GENERATE THRESH
#==============================================================================
def get_thresh(img):
    # binary
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # inverse binary
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    # trunc
    ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    # to-zero
    ret,thresh4 = cv2.threshold(img,167,255,cv2.THRESH_TOZERO)
    # to-zero inverse
    ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
    
    return [thresh1, thresh2, thresh3, thresh4, thresh5]



#==============================================================================
# APPLY ADAPTIVE THRESH TO THE IMAGE - BLACK AND WHITE
#==============================================================================
def adaptive_thresh(img):
    return cv2.adaptiveThreshold(img, \
                                 255, \
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                 cv2.THRESH_BINARY, \
                                 31, \
                                 2)
    
    
    
#==============================================================================
# EXTRACT TEXT USING pytesseract
#==============================================================================
def get_string(img):
    return pytesseract.image_to_string(img)



#==============================================================================
# REMOVE NOISE
#==============================================================================
def remove_noise(img):
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    
    return img      

# execute the code after setup