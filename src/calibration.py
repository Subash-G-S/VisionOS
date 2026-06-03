import cv2
import json

# Calibration points
steps = [
    "LOOK LEFT AND PRESS SPACE",
    "LOOK CENTER AND PRESS SPACE",
    "LOOK RIGHT AND PRESS SPACE"
]

ratios = []

current_step = 0


def save_ratio(value):
    global current_step

    ratios.append(value)

    current_step += 1

    if current_step >= len(steps):

        data = {
            "left": ratios[0],
            "center": ratios[1],
            "right": ratios[2]
        }

        with open("src/calibration.json", "w") as f:
            json.dump(data, f, indent=4)

        print("Calibration Saved")

        return True

    return False


def draw_instruction(frame):

    cv2.putText(
        frame,
        steps[current_step],
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )