import os
import sched
import time
import numpy as np
import cv2


class MotionDetector:
    s = sched.scheduler(time.time, time.sleep)
    cameraList = []
    emailAlert = False
    net = None
    outputLayers = None
    certainLevel = 1

    def __init__(self):
        certainLevel = .7
        # Read in list of video sources from local txt file
        fileLocation = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        cameraFile = open(os.path.join(fileLocation, 'Camera List.txt'), "r")

        # Read video sources line by line
        line = str(cameraFile.readline())
        while line:
            self.cameraList.append(line)
            # print(line)
            line = str(cameraFile.readline())

        # init yolov3 related parts
        baseDir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), r"\yolo_object_detection"))
        # loading yolov3
        net = cv2.dnn.readNet(baseDir + r"\yolov3.weights", baseDir + r"\yolov3.cfg")
        classes = []
        with open(baseDir + r"\coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layerNames = net.getLayerNames()
        self.outputLayers = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Grabs a frame from each video source and calls detectFrames to see what is detected in each.
    # Schedules itself again at specified time interval
    def runFrames(self):
        detected = False

        # Read frame from each source at specified time interval, OpenCV
        for source in self.cameraList:
            vidSource = cv2.VideoCapture(source)
            ret, frame = vidSource.read()
            # Run analysis on captured frames, set detected to true if person found
            detected = self.detectFrames(frame)

        # Send email alert
        if self.emailAlert and detected:
            self.sendNotification()

    # Given a frame, return what was detected. Use of yolov3, returns boolean if detected something
    def detectFrames(self, frame):
        detectedThing = False

        blob = cv2.dnn.blobFromImage(frame, .00392, (416, 416), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outputs = self.net.forward(self.outputLayers)

        # Look for detecting a person with greater than level of certainty
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                if (classId == "person") and (scores[classId] > self.certainLevel):
                    detectedThing = True

        return detectedThing

    # send email notification with screenshot
    def sendNotification(self, image):
        print("send email")

    # Turn on email alerts
    def activateAlerts(self):
        self.emailAlert = True

    def deactivateAlerts(self):
        self.emailAlert = False

    def alertStatus(self):
        return self.emailAlert


if __name__ == "__main__":
    timeInterval = .25
    # create instance of detector
    alertBot = MotionDetector()

    # Timed framegrab/analysis
    while True:
        alertBot.runFrames()
        time.sleep(timeInterval)
