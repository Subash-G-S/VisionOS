import cv2
import mediapipe as mp
import pyautogui
import time

# =====================================
# SETTINGS
# =====================================

MAX_SPEED = 300
CLICK_COOLDOWN = 0.7

# =====================================
# HAND TRACKING
# =====================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# =====================================
# CAMERA
# =====================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 520)

# =====================================
# FPS
# =====================================

prev_time = time.time()

# =====================================
# CURSOR TRACKING
# =====================================

prev_x = None
prev_y = None

smooth_finger_x = None
smooth_finger_y = None

last_click_time = 0

# =====================================
# MAIN LOOP
# =====================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    current_time = time.time()

    fps = 1 / (current_time - prev_time)
    prev_time = current_time

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

            # =====================================
            # INDEX FINGER
            # =====================================

            index_tip = hand.landmark[8]

            raw_x = int(index_tip.x * w)
            raw_y = int(index_tip.y * h)

            if smooth_finger_x is None:
                smooth_finger_x = raw_x
                smooth_finger_y = raw_y

            smooth_finger_x = smooth_finger_x * 0.3 + raw_x * 0.7
            smooth_finger_y = smooth_finger_y * 0.3 + raw_y * 0.7

            x = int(smooth_finger_x)
            y = int(smooth_finger_y)

            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

            # =====================================
            # THUMB
            # =====================================

            thumb = hand.landmark[4]

            thumb_x = int(thumb.x * w)
            thumb_y = int(thumb.y * h)

            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), -1)

            # =====================================
            # DISTANCE
            # =====================================

            distance = (
                (thumb_x - x) ** 2 +
                (thumb_y - y) ** 2
            ) ** 0.5

            # =====================================
            # FINGER STATES
            # =====================================

            index_extended = (
                hand.landmark[8].y <
                hand.landmark[6].y
            )

            middle_extended = (
                hand.landmark[12].y <
                hand.landmark[10].y
            )

            # =====================================
            # MODE DETECTION
            # =====================================

            pinching = distance < 40

            if index_extended and middle_extended:
                mode = "LOCK"

            elif pinching:
                mode = "CLICK"

            else:
                mode = "MOVE"

            # =====================================
            # DISPLAY
            # =====================================

            cv2.putText(
                frame,
                f"MODE: {mode}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"DIST: {int(distance)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # =====================================
            # MOVE MODE
            # =====================================

            if mode == "MOVE":

                if prev_x is not None:

                    dx = x - prev_x
                    dy = y - prev_y

                    if abs(dx) < 2:
                        dx = 0

                    if abs(dy) < 2:
                        dy = 0

                    move_x = int(dx * 8)
                    move_y = int(dy * 8)

                    move_x = max(
                        min(move_x, MAX_SPEED),
                        -MAX_SPEED
                    )

                    move_y = max(
                        min(move_y, MAX_SPEED),
                        -MAX_SPEED
                    )

                    pyautogui.move(move_x, move_y)

            # =====================================
            # LOCK MODE
            # =====================================

            elif mode == "LOCK":

                pass

            # =====================================
            # CLICK MODE
            # =====================================

            elif mode == "CLICK":

                if (
                    current_time - last_click_time
                    > CLICK_COOLDOWN
                ):

                    pyautogui.click()

                    last_click_time = current_time

            prev_x = x
            prev_y = y

    # =====================================
    # FPS DISPLAY
    # =====================================

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("VisionOS", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()