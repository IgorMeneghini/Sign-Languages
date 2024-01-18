import cv2
import mediapipe as mp


def initialize_mediapipe():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    return mp_drawing, mp_hands, hands


def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera")
        exit()
    return cap


def count_raised_fingers(mp_hands, hand_landmarks, handedness):
    finger_points = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    
    raised_fingers = 0

    for idx, hand_handedness in enumerate(handedness):
        print(hand_handedness.classification[0].label)

    for point in finger_points:
        if (
            hand_landmarks.landmark[point].y
            < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
        ):
            raised_fingers += 1    
    return raised_fingers


def Main():
    mp_drawing, mp_hands, hands = initialize_mediapipe()
    cap = open_camera()

    window_name = "Hand Tracking"

    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        if not ret:
            print("Don't have frames available")
            break

        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(color)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    height, width, _ = frame.shape
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    print(f"point {idx}: ({cx}, {cy})")
                raised_fingers_count = count_raised_fingers(mp_hands, hand_landmarks, results.multi_handedness)
                print(f"fingers up: {raised_fingers_count}")

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    cap.release()
    print("Closed")


if __name__ == "__main__":
    Main()
