# Hackathon 2018
****
This project is used in the SmartCity Hackathon Summer 2018.
## Overview
This is the traffic recognition system.<br \>
Inspired by (Ahmetozlu)[https://github.com/ahmetozlu/vehicle_counting_tensorflow]<br \>
The model I use in this project is MobileNetV1 and using the weight from the Imagenet.<br \>
The reason I choose this model is because of the light-weight property since it'd fit the real-time propose.<br \>
## Code explain
**Note that I already have a lot of comment in the project itself but since it's half Eng half Vie so I will explain the core here**
```
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

```
This is where the model got **imported**. We don't have time to train since you know, Hackathon :(. 
But why not use ImageNet weight since it'd already cover a lot of object labels.

__Load labels__
```
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
```
For parsing video frame and draw some line, i use `OpenCV` since it's so easy to use.<br \>
Where you input your video, this is also in `vehicle_detection_main.py`
```
cap = cv2.VideoCapture('sub-1504614469486.mp4')
```
In the [utils](vehicle_counting_tensorflow-master/utils), you would find a lot of files, but mostly use for drawing boxes.<br \>
Go to `visualization_utils.py` and take a look in function `visualize_boxes_and_labels_on_image_array`:
```  
    display_str_list=box_to_display_str_map[box]
    # we are interested just vehicles (i.e. cars and trucks)
    if (("car" in display_str_list[0]) or ("truck" in display_str_list[0]) or ("bus" in display_str_list[0])):
            is_vehicle_detected, csv_line, update_csv = draw_bounding_box_on_image_array(current_frame_number,
                image,
                ymin,
                xmin,
                ymax,
                xmax,
                color=color,
                thickness=line_thickness,
                display_str_list=box_to_display_str_map[box],
                use_normalized_coordinates=use_normalized_coordinates) 
      
            if keypoints is not None:
              draw_keypoints_on_image_array(
                  image,
                  box_to_keypoints_map[box],
                  color=color,
                  radius=line_thickness / 2,
                  use_normalized_coordinates=use_normalized_coordinates)
```
This is the place where you can change the labels as you wish but make sure them is one of the labels of `ImageNet`
## Running
Easy like a cake:
```
python3  vehicle_detection_main.py
```
And that is it.
