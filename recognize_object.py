# image processing libraries
import argparse
#import colorsys
#import imghdr
import os
import random
import math

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import tensorflow as tf
import h5py


############# Model Wrapping Functions #################
def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])

def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """Filter YOLO boxes based on object and class confidence."""
    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    prediction_mask = box_class_scores >= threshold

    # TODO: Expose tf.boolean_mask to Keras backend?
    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)
    return boxes, scores, classes

def yolo_eval(yolo_outputs,
              image_shape,
              max_boxes=10,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input batch and return filtered boxes."""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    boxes, scores, classes = yolo_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=score_threshold)

    # Scale boxes back to original image shape.
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_index = tf.image.non_max_suppression(
        boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, nms_index)
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)
    return boxes, scores, classes

def yolo_head(feats, anchors, num_classes):
    """Convert final layer features to bounding box parameters.
    """
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])

    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.softmax(feats[..., 5:])

    # Adjust preditions to each spatial grid point and anchor size.
    # Note: YOLO iterates over height index before width index.
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs

############################################################

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


def evaluate_img(image_file):
  image = Image.open(image_file)
  image = image.resize((int(image.size[0]/4), int(image.size[1]/4)), Image.BICUBIC)

  if is_fixed_size:  # TODO: When resizing we can use minibatch input.
      resized_image = image.resize(
          tuple(reversed(model_image_size)), Image.BICUBIC)
      image_data = np.array(resized_image, dtype='float32')
  else:
      # Due to skip connection + max pooling in YOLO_v2, inputs must have
      # width and height as multiples of 32.
      new_image_size = (image.width - (image.width % 32),
                        image.height - (image.height % 32))
      resized_image = image.resize(new_image_size, Image.BICUBIC)
      image_data = np.array(resized_image, dtype='float32')
      print(image_data.shape)

  image_data /= 255.
  image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

  out_boxes, out_scores, out_classes = sess.run(
      [boxes, scores, classes],
      feed_dict={
          yolo_model.input: image_data,
          input_image_shape: [image.size[1], image.size[0]],
          K.learning_phase(): 0
      })
  print('Found {} boxes for {}'.format(len(out_boxes), image_file))
  objects = []

  for i, c in reversed(list(enumerate(out_classes))):
      predicted_class = class_names[c]
      box = out_boxes[i]
      score = out_scores[i]


      top, left, bottom, right = box
      top = max(0, np.floor(top + 0.5).astype('int32'))
      left = max(0, np.floor(left + 0.5).astype('int32'))
      bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
      right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
      
      distance = checkDistance(left,top,right,bottom,image.size)
      proximity = pathCheck(left,top,right,bottom,image.size)
      obj = [predicted_class, distance, proximity]
      objects.append(obj)

  return objects


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

def image_process(img_name):
    classes = '/home/pi/Desktop/jarvis-master/coco_classes.txt'
    anchors = '/home/pi/Desktop/jarvis-master/yolo_anchors.txt' 
    model = '/home/pi/Desktop/jarvis-master/yolo.h5'
    print("processing 1")
    with open(classes) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    print("processing 2")

    converter = tf.lite.TFLiteConverter.from_keras_model_file(model)
    yolo_model = converter.convert()
    #yolo_model = load_model(model)
    print("processing 3")

    score_threshold = .3
    iou_threshold = .5

    sess = K.get_session()
    print("processing 4")

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)
    print("processing 5")

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    print("processing 6")
    input_image_shape = K.placeholder(shape=(2, ))
    print("processing 7")
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold)
    print("processing 8")

    objs = evaluate_img(img_name)
    return(form_speech_string(objs))

