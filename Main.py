import cv2
from cvzone.HandTrackingModule import HandDetector

video = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    ret, frame = video.read()
    hands, frame = detector.findHands(frame)

    if hands:
        # Access hand information if hands are detected
        for hand in hands:
            # Access hand attributes like hand type, bounding box, landmarks, etc.
            handType = hand["type"]  # "Left" or "Right"
            lmList = hand["lmList"]  # List of 21 landmarks
            bbox = hand["bbox"]      # Bounding box coordinates (x, y, w, h)

            # Example: Draw a rectangle around the hand
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Example: Draw landmarks on the hand
            for point in lmList:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
