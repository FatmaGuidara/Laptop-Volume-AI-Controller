import cv2 as cv
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


from HandTrackingModule import *

# pycaw mediapipe opencv
#####################################
wCam, hCam = 640, 480
#####################################

# Open the camera
cap = cv.VideoCapture(1)
if not cap.isOpened():
    cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open Webcam")
cap.set(3, wCam)
cap.set(4, hCam)

# Detector Object
detector = HandDetector()

# Initialization
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate( IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# (-65.25, 0.0, 0.03125)
minVolume, maxVolume = volume.GetVolumeRange()[0], volume.GetVolumeRange()[1]
# volume.SetMasterVolumeLevel(0.0, None)
vol = 0
volBar = 400
volPer = 0


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if(len(lmList)!=0):
        x_thumb,y_thumb = lmList[4][1], lmList[4][2]
        x_index,y_index = lmList[8][1], lmList[8][2]
        x_middle,y_middle = (x_thumb+x_index)//2, (y_thumb+y_index)//2
                
        cv.circle(img, (x_thumb,y_thumb), 10, (105, 106, 180), cv.FILLED)
        cv.circle(img, (x_index,y_index), 10, (105, 106, 180), cv.FILLED)
        cv.circle(img, (x_middle,y_middle), 10, (105, 106, 180), cv.FILLED)
        cv.line(img, (x_thumb,y_thumb), (x_index,y_index), (105, 106, 180), 3)

        length = math.hypot(x_index-x_thumb, y_index-y_thumb)
        # Hand Range 25 - 180
        # Volume Range (-65) - 0
        vol = np.interp(length, [40, 180], [minVolume, maxVolume])
        volBar = np.interp(length, [40, 180], [400, 150])
        volPer = np.interp(length, [40, 180], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)        
        if (length<35):
            cv.circle(img, (x_middle,y_middle), 10, (132, 52, 181), cv.FILLED)
            
    cv.rectangle(img, (50,150), (80,400), (132, 52, 181), 3)
    cv.rectangle(img, (50,int(volBar)), (80,400), (132, 52, 181), cv.FILLED)
    cv.putText(img, f'{int(volPer)} %', (40,450), cv.FONT_HERSHEY_COMPLEX, 1, (132, 52, 181), 3)
        
    cv.imshow("Image", img)
    cv.waitKey(1)