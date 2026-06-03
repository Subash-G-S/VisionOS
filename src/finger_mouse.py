import cv2
import mediapipe as mp
import pyautogui

# ==========================
# SCREEN SIZE
# ==========================

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# ==========================
# HAND TRACKING
# ==========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ==========================
# CAMERA
# ==========================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==========================
# SMOOTHING
# ==========================

smooth_x = 0
smooth_y = 0

# ==========================
# LOOP
# ==========================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            # INDEX FINGER TIP
            landmark = hand.landmark[8]

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

            # MAP CAMERA TO SCREEN
            screen_x = int((x / w) * SCREEN_WIDTH)
            screen_y = int((y / h) * SCREEN_HEIGHT)

            # SMOOTHING
            smooth_x = int(smooth_x * 0.9 + screen_x * 0.1)
            smooth_y = int(smooth_y * 0.9 + screen_y * 0.1)

            pyautogui.moveTo(smooth_x, smooth_y)

    cv2.imshow("VisionOS Finger Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()