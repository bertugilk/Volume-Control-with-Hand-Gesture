import cv2
import numpy as np
import time
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


wCam,hCam=648,480
camera=cv2.VideoCapture(0)
camera.set(3, wCam)
camera.set(4, hCam)
pTime=0

detector=htm.HandDetector(detectionCon=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange=volume.GetVolumeRange()

minVol=volRange[0]
maxVol=volRange[1]

vol=0
volBar=400

while True:
    success,img=camera.read()
    
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    
    if len(lmList) != 0:
        #print(lmList[4],lmList[8])
        
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx,cy =  (x1+x2) // 2, (y1+y2) // 2
        
        cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),10,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)    
        
        length = math.hypot(x2-x1, y2-y1)
        print("Length  = ",length)
        
        # Hand range: 50 - 300
        # Volume range: -65 - 0
        
        vol=np.interp(length,[50,300],[minVol,maxVol])
        volBar=np.interp(length,[50,300],[400,150])
        
        print("\nVolume: ",vol)
        
        volume.SetMasterVolumeLevel(vol, None)
        
        
        if length < 50:
            cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)   
        
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
   
    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),cv2.FILLED)
    
    cv2.putText(img,f'Fps: {int(fps)}',(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow("Img",img)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

camera.release()
cv2.destroyAllWindows()