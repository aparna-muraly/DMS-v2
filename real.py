import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("/path/to/your/custom_model.pt")  # Change this to your trained model

# GStreamer pipeline for Raspberry Pi Camera
gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

# Open camera with GStreamer pipeline
video_cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not video_cap.isOpened():
    print("Error: Could not open Raspberry Pi Camera.")
    exit()

while True:
    ret, frame = video_cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Run YOLOv8 inference
    results = model(frame)

    # Draw detections
    for result in results:
        for box in result.boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])

            label = f"{model.names[class_id]}: {confidence:.2f}"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show output
    cv2.imshow("Real-Time Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
video_cap.release()
cv2.destroyAllWindows()
