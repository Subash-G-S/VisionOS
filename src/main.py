import cv2
import mediapipe as mp
import pyautogui
import time
import json
from calibration import save_ratio, draw_instruction
with open("src/calibration.json", "r") as f:
    calibration = json.load(f)
# =====================================
# SCREEN INFO
# =====================================
CAL_LEFT = calibration["left"]
CAL_CENTER = calibration["center"]
CAL_RIGHT = calibration["right"]
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
DEADZONE_MIN = 0.50
DEADZONE_MAX = 0.54
# =====================================
# MEDIAPIPE
# =====================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================
# LANDMARKS
# =====================================

RIGHT_IRIS = [474, 475, 476, 477]

RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263
calibration_mode = True
# =====================================
# CAMERA
# =====================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
prev_time = time.time()

while True:

    success, frame = cap.read()
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            iris_points = []

            for idx in RIGHT_IRIS:

                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                iris_points.append((x, y))

                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

            center_x = sum(p[0] for p in iris_points) // len(iris_points)
            center_y = sum(p[1] for p in iris_points) // len(iris_points)

            cv2.circle(frame, (center_x, center_y), 6, (255, 0, 0), -1)

            eye_left = face_landmarks.landmark[RIGHT_EYE_LEFT]
            eye_right = face_landmarks.landmark[RIGHT_EYE_RIGHT]

            eye_left_x = int(eye_left.x * w)
            eye_right_x = int(eye_right.x * w)

            eye_width = abs(eye_right_x - eye_left_x)

            iris_ratio = abs(center_x - eye_left_x) / eye_width
            if calibration_mode:

                draw_instruction(frame)

                key = cv2.waitKey(1)

                if key == 32:  # SPACE

                    finished = save_ratio(iris_ratio)

                    if finished:
                        calibration_mode = False

            else:

                pass

            # =====================================
            # MAP RATIO TO SCREEN
            # =====================================

            calibrated_left = 0.40
            calibrated_right = 0.65
            if DEADZONE_MIN <= iris_ratio <= DEADZONE_MAX:
                normalized = 0.5
            else :
                normalized = (
                    (iris_ratio - CAL_LEFT)
                    / (CAL_RIGHT - CAL_LEFT)
                )

            normalized = max(0, min(normalized, 1))

            screen_x = int(normalized * SCREEN_WIDTH)

            if "smooth_x" not in globals():
                smooth_x = screen_x

            smooth_x = int(
                smooth_x * 0.92 +
                screen_x * 0.08
            )

            # =====================================
            # MOVE MOUSE
            # =====================================

            pyautogui.moveTo(
                smooth_x,
                SCREEN_HEIGHT // 2,
                duration=0
            )

            cv2.putText(
                frame,
                f"Ratio: {iris_ratio:.2f}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Mouse X: {screen_x}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,0),
                2
            )

    cv2.imshow("VisionOS Mouse Control", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()