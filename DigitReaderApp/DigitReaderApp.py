import numpy as nm
import tensorflow as tf
from pathlib import Path
import time

APP_DIR = Path(__file__).resolve().parent

print(tf.__version__)
print(nm.__version__)

import sys
print(sys.executable)

import cv2

print("CV2 version " + cv2.__version__)


def prepare_digit(grayscale_frame):
    _, binary_frame = cv2.threshold(
        grayscale_frame, 100, 255, cv2.THRESH_BINARY_INV
    )
    foreground = cv2.findNonZero(binary_frame)
    if foreground is None or cv2.countNonZero(binary_frame) < 20:
        return None

    x, y, width, height = cv2.boundingRect(foreground)
    digit_frame = binary_frame[y:y + height, x:x + width]
    square_size = max(width, height)
    square_frame = nm.zeros((square_size, square_size), dtype=nm.uint8)
    offset_x = (square_size - width) // 2
    offset_y = (square_size - height) // 2
    square_frame[offset_y:offset_y + height, offset_x:offset_x + width] = digit_frame

    resized_digit = cv2.resize(
        square_frame, (20, 20), interpolation=cv2.INTER_AREA
    )
    model_frame = nm.zeros((28, 28), dtype=nm.uint8)
    model_frame[4:24, 4:24] = resized_digit
    return model_frame



digit_model = tf.keras.models.load_model(APP_DIR / "digit-recognizer.keras")

X_testReload = nm.loadtxt(APP_DIR / "X_testreshaped.csv")
Y_testReload = nm.loadtxt(APP_DIR / "Y_testreshaped.csv")
X_testFrames = nm.reshape(X_testReload,(10000, 28, 28));
Y_testFrames = nm.reshape(Y_testReload,(10000));

print(X_testFrames[0].shape)
print(Y_testFrames.shape)

print("expected result")  
print(Y_testFrames[0])

input_frame = nm.reshape(X_testFrames[0], (1, 28, 28))

result = digit_model.predict(input_frame)

print("actual result") 
print(result)



cv2.namedWindow("Camera")
vc = cv2.VideoCapture(0)


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

frame_height, frame_width = frame.shape[:2] if rval else (0, 0)
roi_size = int(min(frame_height, frame_width) * 0.7)
roi_left = (frame_width - roi_size) // 2
roi_top = (frame_height - roi_size) // 2
last_prediction_time = 0.0
predicted_digit = None

while rval:
    current_time = time.monotonic()
    if current_time - last_prediction_time >= 1.0:
        digit_region = frame[
            roi_top:roi_top + roi_size,
            roi_left:roi_left + roi_size,
        ]
        gray_region = cv2.cvtColor(digit_region, cv2.COLOR_BGR2GRAY)
        model_frame = prepare_digit(gray_region)
        if model_frame is None:
            predicted_digit = None
        else:
            input_frame = model_frame.reshape((1, 28, 28, 1))
            result = digit_model.predict(input_frame, verbose=0)
            predicted_digit = int(nm.argmax(result[0]))
        last_prediction_time = time.monotonic()

    display_frame = frame.copy()
    cv2.rectangle(
        display_frame,
        (roi_left, roi_top),
        (roi_left + roi_size, roi_top + roi_size),
        (0, 255, 255),
        2,
    )
    label = "Digit: --" if predicted_digit is None else f"Digit: {predicted_digit}"
    cv2.rectangle(display_frame, (10, 10), (230, 62), (0, 0, 0), -1)
    cv2.putText(
        display_frame,
        label,
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Camera", display_frame)
    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break

    rval, frame = vc.read()
        
vc.release()
cv2.destroyWindow("Camera")