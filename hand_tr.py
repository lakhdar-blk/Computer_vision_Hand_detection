import cv2
import mediapipe as mp
import time

from pip import main


class HandDetector():

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5 ):
        
        self.mode = mode
        self.maxHands = maxHands,
        self.detectionCon = detectionCon,
        self.trackCon = trackCon

        """static_image_mode = False,
        max_num_hands = 2,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5"""

        #======for hand detection=========#
        self.mpHands =mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        #======for hand detection=========#

    def findHands(self, img, draw=True):

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgRGB)

            if self.results.multi_hand_landmarks:
                
                for handLms in self.results.multi_hand_landmarks:
                    
                    if draw:
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

            return img

    def findPosition(self, img, handNo=0, draw=  True):
        
        lmList = []

        if draw:
            
            if self.results.multi_hand_landmarks:

                myHand = self.results.multi_hand_landmarks[handNo]
                
                for id, lm in enumerate(myHand.landmark):

                                    h, w, c = img.shape
                                    cx, cy = int(lm.x*w), int(lm.y*h)

                                    #if id == 8:
                                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return  lmList

def main():

    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0

    detector = HandDetector()

    while True:

        success, img = cap.read()

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == '__main__':

    main()