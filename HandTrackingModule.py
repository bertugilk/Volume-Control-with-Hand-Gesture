import cv2
import numpy as np
import mediapipe as np
import time

class HandDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.modelComplexity=modelComplexity
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.npHands=np.solutions.hands
        self.hands=self.npHands.Hands(self.mode,self.maxHands,self.modelComplexity,
                                 self.detectionCon,self.trackCon)
        
        self.npDraw=np.solutions.drawing_utils
    
    def findHands(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLns in self.results.multi_hand_landmarks:
                if draw:
                    self.npDraw.draw_landmarks(img, handLns,self.npHands.HAND_CONNECTIONS)     
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        lnList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id, ln in enumerate(myHand.landmark):
                #print(id,ln)
                h, w, c = img.shape
                cx,cy=int(ln.x * w),int(ln.y * h)
                # print(id,cx,cy)
                lnList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
        return lnList
    
def main():
    pTime=0
    cTime=0
    cap=cv2.VideoCapture(0)
    detector=HandDetector()
    while True:
        success,img=cap.read()
        
        img=detector.findHands(img)
        lnList=detector.findPosition(img)
        
        if len(lnList) != 0:
            print(lnList[4])
        
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,f'Fps: {int(fps)}',(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        
        cv2.imshow("img",img)
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

if __name__ == '__main__':
    main()