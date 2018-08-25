#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------
from utils.image_utils import image_saver

# current_frame_number_list = [0]
NUMBER_OF_FRAME = 300  # standard number frames before refreshing 12s
LINE_DEVIATION = 5
VCH_DEVIATION = 15
vch = {}
def predict_speed(
    position,
    left, right, top, bottom,
    obj_name,
    color,
    current_frame_number,
    crop_img,
    roi_position,
    ):
    is_vehicle_detected = 0
    i = 1
    # print(center_pos)
    if(current_frame_number%NUMBER_OF_FRAME == 0):
        vch.clear()
    if position[1] >= (roi_position - LINE_DEVIATION) and position[1] <= (roi_position + LINE_DEVIATION) and current_frame_number % 2 == 0:
        if vch.get(obj_name) == None:
            is_vehicle_detected = 1
            vch[obj_name] = {1: {'x': int(position[0]), 'y': int(position[1])}}
            image_saver.save_image(crop_img)  # save detected vehicle image
        else:
            length = len(vch[obj_name]) + 1
            for i in range(1, length):
                x = vch[obj_name][i].get('x')
                y = vch[obj_name][i].get('y')
                if(x != None) and (y != None):
                    if (position[0] < x + VCH_DEVIATION) and (position[0] > x - VCH_DEVIATION) and (position[1] < y + VCH_DEVIATION) and (position[1] > y - VCH_DEVIATION):
                        is_vehicle_detected = 0
                        break
                    else:
                        is_vehicle_detected = 1
                vch[obj_name][i]['x'] = int(position[0])
                vch[obj_name][i]['y'] = int(position[1])

    return (is_vehicle_detected)
