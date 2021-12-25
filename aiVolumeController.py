import cv2 as cv
import numpy as np 
from math import hypot
import mediapipe as mp
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv.VideoCapture(0)

mainHands = mp.solutions.hands 
hands = mainHands.Hands()
mainDraw = mp.solutions.drawing_utils
mainDevices = AudioUtilities.GetSpeakers()
mainInterface = mainDevices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
mainVolume = cast(mainInterface, POINTER(IAudioEndpointVolume))
minVol,maxVol = mainVolume.GetVolumeRange()[:2]

while True:
    success,pic = cap.read()
    imgRGB = cv.cvtColor(pic,cv.COLOR_BGR2RGB)
    final = hands.process(imgRGB)

    lmList = []
    if final.multi_hand_landmarks:
        for handlandmark in final.multi_hand_landmarks:
            for id,lm in enumerate(handlandmark.landmark):
                h,w,_ = pic.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy]) 
            mainDraw.draw_landmarks(pic,handlandmark,mainHands.HAND_CONNECTIONS)
    
    if lmList != []:
        x1,y1 = lmList[4][1],lmList[4][2]
        x2,y2 = lmList[8][1],lmList[8][2]

        cv.circle(pic,(x1,y1),4,(255,0,0),cv.FILLED)
        cv.circle(pic,(x2,y2),4,(255,0,0),cv.FILLED)
        cv.line(pic,(x1,y1),(x2,y2),(255,0,0),3)

        length = hypot(x2-x1,y2-y1)

        volume = np.interp(length,[15,220],[minVol,maxVol])
        print(volume,length)
        mainVolume.SetMasterVolumeLevel(volume, None)

        # Hand range 15 - 220
        # Volume range -63.5 - 0.0
        
    cv.imshow('Image',pic)
    if cv.waitKey(1) & 0xff==ord('q'):
        break
