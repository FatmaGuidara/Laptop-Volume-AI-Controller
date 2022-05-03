import cv2 as cv
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, 
                 mode = False,
                 maxHands = 2,
                 modelComplex = 1,
                 detectionConfidence = 0.5,
                 trackConfidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplex
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        # Hand Detection
        self.mpHands = mp.solutions.hands
        # only uses rgb images
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw = True):
        img = cv.flip(img, 1)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)

        return img

 
    def findPosition(self, img, handNbr = 0, draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNbr]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w) , int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx,cy), 5, (255,0,255), cv.FILLED)
        return lmList
                   
    
    
def main():
    
    # Time
    pTime = 0
    cTime = 0 
    # Detection
    detector = HandDetector()
    
    # Open the camera
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open Webcam")

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            id, x, y = lmList[8]
            cv.circle(img, (x,y), 15, (0,0,0), cv.FILLED)
        # fsp
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 2)
        
        
        
        cv.imshow("Image", img)
        cv.waitKey(1)
        
if __name__ == "__main__":
    main()