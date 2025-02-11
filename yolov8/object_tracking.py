import datetime
from ultralytics import YOLO
import cv2
from helper import create_video_writer

# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)

# initialize the video capture object
video_cap = cv2.VideoCapture("2.mp4")
# initialize the video writer object
writer = create_video_writer(video_cap, "output.mp4")

# load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")

# Start processing the video frame by frame
while True:
    # start time to compute the fps
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    # if there are no more frames to process, break out of the loop
    if not ret:
        break

    # run the YOLO model on the frame
    detections = model(frame)[0]
    
    # (Optional) Process detections, draw bounding boxes, and write to output
    # TODO: Add logic to draw bounding boxes, filter detections, and display/save frames

# Release resources
video_cap.release()
writer.release()
cv2.destroyAllWindows()
