#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------
from utils.image_utils import image_saver

is_vehicle_detected = [0]
# current_frame_number_list = [0]
bottom_position_of_detected_vehicle = [0]
top_position_of_detected_vehicle = [0]
LINE_DEVIATION = 5

def predict_speed(
    top,
    bottom,
    right,
    left,
    current_frame_number,
    crop_img,
    roi_position,
    ):
    center_pos = (top+bottom)/2
    # print(center_pos)
    if center_pos >= (roi_position - LINE_DEVIATION) \
        and center_pos <= (roi_position + LINE_DEVIATION) \
        and len(bottom_position_of_detected_vehicle) != 0 \
        and len(top_position_of_detected_vehicle) != 0 :
        if (bottom - top) > ((bottom_position_of_detected_vehicle[0] - top_position_of_detected_vehicle[0]) + LINE_DEVIATION) \
            or (bottom - top) < ((bottom_position_of_detected_vehicle[0] - top_position_of_detected_vehicle[0]) + LINE_DEVIATION) :
            is_vehicle_detected.insert(0, 1)
            # print("Confirm")
            image_saver.save_image(crop_img)  # save detected vehicle image
    # save these var for not committing repeated vehicle
    bottom_position_of_detected_vehicle.insert(0, bottom)
    top_position_of_detected_vehicle.insert(0, top)
    return (is_vehicle_detected)
