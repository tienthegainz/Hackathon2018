#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------
from utils.image_utils import image_saver

# current_frame_number_list = [0]

LINE_DEVIATION = 4
VCH_DEVIATION = 5
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
    is_vehicle_detected = 0
    # print(center_pos)
    if center_pos >= (roi_position - LINE_DEVIATION) \
        and center_pos <= (roi_position + LINE_DEVIATION) \
        and current_frame_number % 2 == 0:
            is_vehicle_detected = 1
            # print("Confirm")
            image_saver.save_image(crop_img)  # save detected vehicle image

    return (is_vehicle_detected)
