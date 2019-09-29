# image processing libraries
import os
import random
import math
import cv2
from collections import Counter
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from PIL import Image

def distanceFormula(x1, y1, x2, y2):
  """
  distance formula finds the diagonal distance between the two points
  """
  is_negative = False

  dif1 = (x1 - x2)
  if dif1 > 0:
    is_negative = True
  dif1 = math.pow(dif1, 2)
  dif2 = (y1 - y2)
  dif2 = math.pow(dif2, 2)
  return math.sqrt(dif1 + dif2), is_negative


def checkDistance(left, top, right, bottom, imgShape):
  
  d, __ = distanceFormula(left,top,right,bottom)
  m1 = (bottom-top) / (left-right) 
  m2 = - 1 / m1
  bbox_area = d * m2
  
  image_size = imgShape[0] * imgShape[1]
  proximity = bbox_area / image_size

  return proximity


def pathCheck(left, top, right, bottom, imgShape):

  x_center = imgShape[0] / 2
  y_center = (imgShape[1] / 2) * (4/3)

  leftDist, leftNeg = distanceFormula(left, top, x_center, y_center)
  if leftNeg:
    leftDist = -leftDist

  rightDist, rightNeg = distanceFormula(right, bottom, x_center, y_center)
  if rightNeg:
    rightDist = -rightDist


  loc = abs(leftDist + rightDist)

  return(loc)

def form_speech_string(obj_list):

  class_list = []
  if obj_list:
    closest = obj_list[0]
  else:
    return "There are no known objects in your immediate vicinity"
  for obj in obj_list:

    if len(obj) != 3:
      raise Exception("Expected 3 attributes")

    class_list.append(obj[0])
    if obj[1] > closest[1] and obj[2] < closest[2]:
      closest = obj

  close_phrase = "You are looking at a " + closest[0] + "."
  if len(objs) > 1:
    close_phrase += "There are "+ str(len(obj_list) - 1) + " other identified objects."
    for i, value in enumerate(Counter(class_list).values()):
      if str(list(Counter(class_list))[i]) == 'person' and value > 1:
        close_phrase += " " + str(value) + " people,"
      else:
        close_phrase += " " + str(value) + " " + str(list(Counter(class_list))[i]) + ","

  return close_phrase

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def image_process(image):
    
  model_file = "/tmp/mobilenet_v1_1.0_224.tflite"
  label_file = "/tmp/labels.txt"
  
  interpreter = Interpreter(model_path=model_file)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]  
  img = Image.open(image).resize((width, height))

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - 127.5) / 127.5

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = np.squeeze(output_data)

    

  top_k = results.argsort()[-3:][::-1]
  labels = load_labels(label_file)
  
  phrase = "There is a " + str(labels[top_k[0]]).split(":")[1] + " in front of you"
  return phrase
  
  for i in top_k:
    if floating_model:
      print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
    else:
      print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
  