"""
Reads speed limit signs from a webcam using the trained model.

Shows the camera feed with boxes around the signs it finds, and a LIMIT box in
the top left with the current speed limit. Press q to quit.

"""

import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"

CAMERA = 0                    # 0 = default, but my laptop camera isn't working, so using an external camera
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CONFIDENCE = 0.65             # ignore detections the model isn't this sure about
SMOOTHING_FRAMES = 15         # most common reading in the last 15 detections (moving average)
LIMIT_SECONDS = 2.0           # how long the limit stays on screen after the sign is gone

model = YOLO(str(MODEL_PATH))

camera = cv2.VideoCapture(CAMERA)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
if not camera.isOpened():
    raise RuntimeError("Could not open the camera, try a different CAMERA number")

recent_readings = []     # speed read from the most confident sign in each recent frame
current_limit = None     # the speed shown in the LIMIT box
last_seen = 0            # time of the last detection

frame_count = 0
fps = 0
fps_start = time.time()

while True:
    success, frame = camera.read()
    if not success:
        print("Could not read a frame from the camera")
        break

    results = model.predict(frame, conf=CONFIDENCE, verbose=False)

    # find the most confident sign in this frame
    best_conf = 0
    best_speed = None
    for box in results[0].boxes:
        if float(box.conf[0]) > best_conf:
            best_conf = float(box.conf[0])
            class_name = model.names[int(box.cls[0])]       # e.g. "speed_80"
            best_speed = class_name.replace("speed_", "")   # -> "80"

    if best_speed is not None:
        recent_readings.append(best_speed)
        if len(recent_readings) > SMOOTHING_FRAMES:
            recent_readings.pop(0)  # forget the oldest reading

        # the limit to show is whichever speed shows up most in the recent readings,
        # so one wrong frame doesn't change it
        most_count = 0
        for speed in recent_readings:
            count = recent_readings.count(speed)
            if count > most_count:
                most_count = count
                current_limit = speed
        last_seen = time.time()

    # take the limit off the screen once the sign has been gone for a while
    seconds_since_seen = time.time() - last_seen
    if current_limit is not None and seconds_since_seen > LIMIT_SECONDS:
        current_limit = None
        recent_readings = []

    # frames per second, updated once a second
    frame_count += 1
    if time.time() - fps_start >= 1:
        fps = frame_count / (time.time() - fps_start)
        frame_count = 0
        fps_start = time.time()

    # draw the boxes the model found, then the LIMIT box and the fps on top
    image = results[0].plot()
    width = image.shape[1]
    cv2.putText(image, f"{fps:.1f} FPS", (width - 110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if current_limit is not None:
        # the number gets darker the longer it's been since the sign was seen
        brightness = 1 - seconds_since_seen / LIMIT_SECONDS
        if brightness < 0.3:
            brightness = 0.3
        green = int(255 * brightness)

        cv2.rectangle(image, (10, 10), (260, 90), (0, 0, 0), -1)
        cv2.putText(image, "LIMIT", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(image, current_limit, (25, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, green, 0), 3)

    cv2.imshow("UAE Speed Limit Recognition", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
