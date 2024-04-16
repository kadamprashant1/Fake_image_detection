import cv2
import pafy
import time
from ultralytics import YOLO
import cvzone
import math


# Setup
model = YOLO("l_version_1_300.pt")
classNames = ["fake", "real"]
confidence = 0.6
prev_frame_time = 0

# Video from YouTube
url = "https://www.youtube.com/watch?v=vNePhmCMnbU&list=PLnjNbtXRaPpeCMsQ9bhQDajKrTWVrKjfB"  # Update this
video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv2.VideoCapture(best.url)

# Process Video
while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if success:
        results = model(img, stream=True, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Process each box...
                pass  # Your existing code here

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(fps)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Failed to grab frame")
        break

cv2.destroyAllWindows()
cap.release()
