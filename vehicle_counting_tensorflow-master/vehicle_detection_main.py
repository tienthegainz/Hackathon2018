#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------

# Imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import csv
import time
import requests
import json

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util


# input video
cap = cv2.VideoCapture('video_giao_thong.mp4')
idee = 'PC1927'
# Variables
NUMBER_OF_FRAME = 300  # standard number frames before refreshing 5 min
LINE_DEVIATION = 5
ROAD_LENGTH = 12
# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = \
    'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Download Model
# uncomment if you have not download the model yet
# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
            3)).astype(np.uint8)


# Sent 2 server
def sentserver(ide, flow, den):
    spd = flow/den
    flow = str(flow)
    den = str(den)
    spd = str(spd)
    data = {'ID': ide,
            'trafic_flow': flow,
            'traffic_density': den,
            'speed': spd}
    url = 'http://13.250.21.177/api/maytram'
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    r = requests.post(url, data=json.dumps(data), headers=headers)
    # r = requests.post(adr, json = data)
    # print(data_json)

# Detection
def object_detection_function():
    # var init
    # var for counting vehicle
    total_passed_bus = 0  # using it to count bus and truck
    total_passed_car = 0  # using it to count car
    total_passed_bike = 0  # using it to count bike
    density = [0, 0, 0]  # use it for density
    tmp_density = [0, 0, 0]  # to compare with density

    obj_name = []
    frame_count = 0

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while cap.isOpened():
                (ret, frame) = cap.read()

                if not ret:
                    print ('end of the video file...')
                    break

                input_frame = frame
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = \
                    sess.run([detection_boxes, detection_scores,
                             detection_classes, num_detections],
                             feed_dict={image_tensor: image_np_expanded}
                             )
                # Visualization of the results of a detection.
                (obj_name, density) = \
                    vis_util.visualize_boxes_and_labels_on_image_array(
                    cap.get(1),
                    input_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4,
                    )
                # print("cap.get: ", cap.get(1))
                if (tmp_density[0]*0.3+tmp_density[1]+tmp_density[2]*2) > (density[0]*0.3+density[1]+density[2]*2):
                    density = tmp_density
                tmp_density = [0, 0, 0]
                frame_count += 1
                for obj in obj_name:
                    if "person" in obj or "bicycle" in obj or "motorcycle" in obj:
                        total_passed_bike += 1
                    elif "car" in obj:
                        total_passed_car += 1
                    elif "truck" in obj_name or "bus" in obj:
                        total_passed_bus += 1
                if frame_count == NUMBER_OF_FRAME:
                    # cong thuc tinh travel flow
                    trf = (total_passed_bike*0.3+total_passed_car+total_passed_bus*2)*3600/(NUMBER_OF_FRAME/25)
                    trd = (density[0]*0.3+density[1]+density[2]*2)*1000/ROAD_LENGTH
                    print("Bike: ", total_passed_bike)
                    print("Car: ", total_passed_car)
                    print("Truck and Bus: ", total_passed_bus)
                    print(
                    "Traffic Flow: ",
                    trf,
                    "vch/h"
                    )
                    print("Traffic density: ",
                    trd,
                    "vch/km")
                    print("Travel speed: ",trf/trd," km/h\n")
                    # HAY DAT FUNCTION GUI LEN SEVER VAO DAY !!!!!
                    # sentserver(idee, trf,trd)
                    total_passed_bus = 0
                    total_passed_car = 0
                    total_passed_bike = 0
                    frame_count = 0
                    density = [0, 0, 0]

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected Car: ' + str(total_passed_car),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )
                cv2.putText(
                    input_frame,
                    'Detected Bike: ' + str(total_passed_bike),
                    (10, 55),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )
                cv2.putText(
                    input_frame,
                    'Detected Truck or Bus: ' + str(total_passed_bus),
                    (10, 75),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )
                # when the vehicle passed over line and counted, make the color of ROI line green
                cv2.line(input_frame, (0, 245), (3000, 245), (0, 0xFF, 0), 2)
                cv2.line(input_frame, (0, 255), (3000, 255), (0, 0, 0xFF), 2)

                # insert information text to video frame
                cv2.putText(
                    input_frame,
                    'ROI',
                    (545, 244),
                    font,
                    0.6,
                    (0xFF, 0, 0),
                    2,
                    cv2.LINE_AA,
                    )
                cv2.imshow('vehicle detection', input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


object_detection_function()
