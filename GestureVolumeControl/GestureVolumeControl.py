import cv2
import time
import numpy as np
from HandTrackingModule import handDetector
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# --------------------------
# Camera setup
# --------------------------
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
if not cap.isOpened():
    print("Cannot access camera")
    exit()

# --------------------------
# Hand Detector
# --------------------------
detector = handDetector(detectionCon=0.75, trackCon=0.75)

# --------------------------
# Audio setup via Pycaw
# --------------------------
speakers = AudioUtilities.GetSpeakers()
volume = speakers.EndpointVolume
volRange = volume.GetVolumeRange()  # (minVol, maxVol)
minVol, maxVol = volRange[0], volRange[1]

volBar = 400
volPer = 0
pTime = 0

# --------------------------
# Smoothing & Debounce
# --------------------------
prevVol = 0
muteState = None   # None, True, False
smoothness = 5     # Higher = smoother

# --------------------------
# Main loop
# --------------------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        # --------------------------
        # Thumb-index distance for volume
        # --------------------------
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

        length = np.hypot(x2 - x1, y2 - y1)

        # Smooth volume interpolation
        vol = np.interp(length, [20, 250], [minVol, maxVol])
        volPer = np.interp(length, [20, 250], [0, 100])
        vol = prevVol + (vol - prevVol) / smoothness
        prevVol = vol

        volBar = np.interp(length, [20, 250], [400, 150])

        volume.SetMasterVolumeLevel(vol, None)

        if length < 25:
            cv2.circle(img, (cx, cy), 12, (0, 255, 0), cv2.FILLED)

        # --------------------------
        # Fist / Open-hand detection for mute/unmute
        # --------------------------
        tipIds = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        distances = []
        for tipId in tipIds:
            x, y = lmList[tipId][1], lmList[tipId][2]
            wx, wy = lmList[0][1], lmList[0][2]  # Wrist
            distances.append(np.hypot(x - wx, y - wy))
        avgDist = np.mean(distances)

        fist_threshold = 60
        open_hand_threshold = 150

        if avgDist < fist_threshold and muteState != True:
            volume.SetMute(1, None)
            muteState = True
        elif avgDist > open_hand_threshold and muteState != False:
            volume.SetMute(0, None)
            muteState = False

        # Status text
        if muteState:
            cv2.putText(img, 'MUTED', (200, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
        elif muteState == False:
            cv2.putText(img, 'UNMUTED', (200, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

    # --------------------------
    # Volume bar
    # --------------------------
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 430),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    # --------------------------
    # FPS
    # --------------------------
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (450, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
