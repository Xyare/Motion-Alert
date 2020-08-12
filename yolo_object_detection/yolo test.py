import numpy as np
import cv2


baseDir = "D:\yolo_object_detection"
#loading yolov
net = cv2.dnn.readNet("yolov3.weights", "yolov.cfg")
classes = []
with open( "coco.names", "r" ) as f:
    classes = [line.strip() for line in f.readlines()]

print(classes)