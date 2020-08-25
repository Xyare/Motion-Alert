import os
import sched
import time
import numpy as np
import cv2
import smtplib
import ssl

from abc import ABC, abstractmethod

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class MotionDetector:

    def __init__(self):

        self.state = detectionState

        self.s = sched.scheduler(time.time, time.sleep)
        self.cameraList = []
        self.emailAlert = True
        self.net = None
        self.outputLayers = None
        # confidence level for detection
        self.certainLevel = .7

        # email parameters
        self.subject = "test"
        self.body = "test body"
        self.originEmail = "gsuner51test@gmail.com"
        self.destinationEmail = "gsuner51test@gmail.com"
        self.password = ""
        self.smtpServer = "smtp.gmail.com"
        self.context = ssl.create_default_context()

        # Read in list of video sources from local txt file
        fileLocation = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        cameraFile = open(os.path.join(fileLocation, 'Camera List.txt'), "r")

        # Read video sources line by line
        sourceLine = str(cameraFile.readline())
        while sourceLine:
            if sourceLine == "0":
                self.cameraList.append(0)
            else:
                self.cameraList.append(sourceLine)
            # print(line)
            sourceLine = str(cameraFile.readline())

        # init yolov3 related parts
        baseDir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))) + r"\yolo_object_detection"
        # loading yolov3
        self.net = cv2.dnn.readNet(baseDir + r"\yolov3.weights", baseDir + r"\yolov3.cfg")
        with open(baseDir + r"\coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        layerNames = self.net.getLayerNames()
        self.outputLayers = [layerNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    # Grabs a frame from each video source and calls detectFrames to see what is detected in each.
    # Schedules itself again at specified time interval
    def runFrames(self):
        frameList = []
        # Read frame from each source
        for source in self.cameraList:
            vidSource = cv2.VideoCapture(source)
            ret, frame = vidSource.read()
            frameList.append(frame)

        # detected = self.detectFrames(frame)
        # Run analysis on captured frames depending on current state. Passes list of frames captured
        self.state.runFrames(self, frameList)

    # Given a frame, return what was detected. Use of yolov3, returns boolean if detected something
    def detectFrames(self, frame):

        blob = cv2.dnn.blobFromImage(frame, .00392, (416, 416), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outputs = self.net.forward(self.outputLayers)
        # Look for detecting a person with greater than level of certainty
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)

                if (self.classes[classId] == "person") and (scores[classId] > self.certainLevel):
                    return True  # include delayed screenshot

        return False

    # send email notification with screenshot
    def sendNotification(self, image):
        # save frame as image file for attachment
        file = "temp image.jpg"
        cv2.imwrite(file, image)

        # send email with screenshot
        message = MIMEMultipart()
        message["From"] = self.originEmail
        message["To"] = self.destinationEmail
        message["Subject"] = self.subject

        # body text for email
        message.attach(MIMEText(self.body, "plain"))

        with open("temp image.jpg", "rb") as f:
            mime = MIMEBase('image', 'jpg', fileName='temp image.jpg')
            mime.add_header('Content-Disposition', 'attachment', fileName='temp image.jpg')
            mime.add_header('X-Attachment-Id', '0')
            mime.add_header('Content-ID', '<0>')

            mime.set_payload(f.read())
            # encode attachment
            encoders.encode_base64(mime)
            # add attachment to email
            message.attach(mime)

            # sending the email
            server = smtplib.SMTP_SSL(self.smtpServer, 465, context=self.context)
            server.login(self.originEmail, self.password)
            server.sendmail(self.originEmail, self.destinationEmail, message.as_string())
            server.quit()

    # Turn on email alerts
    def activateAlerts(self):
        self.emailAlert = True

    def deactivateAlerts(self):
        self.emailAlert = False

    def alertStatus(self):
        return self.emailAlert

    # Transition to next state
    def transition(self, state):
        self.state = state


# Abstract base state
class State(ABC):

    @abstractmethod
    def runFrames(self, frame) -> None:
        pass


# Checks to see if there are people in any frame
# If detected, transitions to detection state and returns frames where detection occurred
class detectionState(State):
    def runFrames(self, frames):
        detectedFrames = []
        for i in frames:
            if self.detectFrames(i):
                detectedFrames.append(i)

        self.transition(alertedState)
        return detectedFrames


# Checks to see if subject is still in frame, transitions back to detectionState
# Record frames to video until subject leaves
class alertedState(State):
    def runFrames(self, frames):
        print()


# Logging state when alerts/recordings arent needed
# Saves to sql server
class loggingState(State):
    print()


if __name__ == "__main__":
    timeInterval = .1
    # create instance of detector
    alertBot = MotionDetector()

    # Timed framegrab/analysis
    # while True:
    alertBot.runFrames()
    # time.sleep(timeInterval)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    # break
