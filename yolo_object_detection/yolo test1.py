import numpy as np
import cv2


baseDir = "D:\yolo_object_detection"
#loading yolov
net = cv2.dnn.readNet(baseDir + "\yolov3.weights", baseDir + "\yolov3.cfg")
classes = []
with open(baseDir +  "\coco.names", "r" ) as f:
    classes = [line.strip() for line in f.readlines()]
#test laod names
#print(classes)

layerNames = net.getLayerNames()
outputLayers = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
#cv2.VideoCapture(0)
webCam = cv2.VideoCapture('http://192.168.2.147:8080/stream/video/mjpeg?resolution=HD&&Username=admin&&Password=NTZUYWZ0U3RyZWV0&&tempid=0.024356687350152617')
while(True):
    #capture image

    ret, frame = webCam.read()
    height, width, channels = frame.shape

    #detecting object
    blob = cv2.dnn.blobFromImage(frame, .00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outputs = net.forward(outputLayers)
    #print(outputs)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #remove noise
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    #add result boxes
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

    #Display result
    cv2.imshow("Webcam", frame)
    #cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()