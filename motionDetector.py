import os
import sched
import time

cameraList = []
s = sched.scheduler(time.time, time.sleep)
cameraList = []
emailAlert = False


def initVideo():
    # Read in list of video sources from local txt file
    fileLocation = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    cameraFile = open(os.path.join(fileLocation, 'Camera List.txt'), "r")

    # Read video sources line by line
    line = str(cameraFile.readline())

    while line:
        cameraList.append(line)
        print(line)
        line = str(cameraFile.readline())

    # Connect to video sources with open cv <- maybe move to runFrames


def runFrames():
    detected = False
    frames = []
    # Read frame from each source at specified time interval, OpenCV
    for source in cameraList:
        print('Grabbin frames')
        # Run analysis on captured frames, set detected to true if person found
        detectFrames()

    # Send email alert
    if emailAlert and detected:
        print('Send email')

    # Reschedule task
    s.enter(60, 1, runFrames())


#Given list of frames, return what was detected. Use of yolov3
def detectFrames(frame):
    print("Analyzing some frames")


if __name__ == "__main__":
    # run things
    print("Buster crit memes")

    # Initialize camera list
    initVideo()

    # Timed framegrab/analysis. Use of sched module
    s.enter(60, 1, runFrames())
    s.run()
