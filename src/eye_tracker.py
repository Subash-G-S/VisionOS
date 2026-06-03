import cv2
import mediapipe as mp

# ==========================================
# MEDIAPIPE FACE MESH SETUP
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# IRIS LANDMARKS
# ==========================================

RIGHT_IRIS = [474, 475, 476, 477]

RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# ==========================================
# START CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # ==========================================
            # STORE LEFT IRIS POINTS
            # ==========================================

            iris_points = []

            for idx in RIGHT_IRIS:

                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                iris_points.append((x, y))

                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            # ==========================================
            # DRAW RIGHT IRIS
            # ==========================================

            for idx in RIGHT_IRIS:

                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

            # ==========================================
            # CALCULATE LEFT IRIS CENTER
            # ==========================================

            center_x = sum(p[0] for p in iris_points) // len(iris_points)
            center_y = sum(p[1] for p in iris_points) // len(iris_points)

            cv2.circle(frame, (center_x, center_y), 6, (255, 0, 0), -1)

            # ==========================================
            # GET LEFT EYE BOUNDARIES
            # ==========================================

            eye_left = face_landmarks.landmark[RIGHT_EYE_LEFT]
            eye_right = face_landmarks.landmark[RIGHT_EYE_RIGHT]

            eye_top = face_landmarks.landmark[RIGHT_EYE_TOP]
            eye_bottom = face_landmarks.landmark[RIGHT_EYE_BOTTOM]

            eye_left_x = int(eye_left.x * w)
            eye_right_x = int(eye_right.x * w)
            eye_top_y = int(eye_top.y * h)
            eye_bottom_y = int(eye_bottom.y * h)

            cv2.circle(frame, (eye_left_x, center_y), 8, (0, 255, 0), -1)
            cv2.circle(frame, (eye_right_x, center_y), 8, (255, 255, 0), -1)

            cv2.putText(
                frame,
                f"L:{eye_left_x} R:{eye_right_x} C:{center_x}",
                (20,130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )

            # ==========================================
            # CALCULATE GAZE RATIO
            # ==========================================

            eye_width = abs(eye_right_x - eye_left_x)

            iris_ratio = abs(center_x - eye_left_x) / eye_width
            eye_height = abs(eye_bottom_y - eye_top_y)

            vertical_ratio = abs(center_y - eye_top_y) / eye_height

            # ==========================================
            # DETERMINE GAZE DIRECTION
            # ==========================================

            direction = "CENTER"

            if iris_ratio < 0.48:
                direction = "LEFT"

            elif iris_ratio > 0.58:
                direction = "RIGHT"
            if vertical_ratio < 0.40:
                direction = "UP"

            elif vertical_ratio > 0.65:
                direction = "DOWN"

            # ==========================================
            # DISPLAY RESULTS
            # ==========================================

            cv2.putText(
                frame,
                f"LOOKING: {direction}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"H:{iris_ratio:.2f} V:{vertical_ratio:.2f}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    cv2.imshow("VisionOS Eye Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()