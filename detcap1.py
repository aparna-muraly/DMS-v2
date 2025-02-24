from ultralytics import YOLO
import cv2
import json

# Load YOLOv8 model
model = YOLO("/home/nvidia/yolo_model/best.pt")  # Replace with your trained model

# GStreamer pipeline for camera stream (modify as per your camera)
gst_pipeline = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
)

# Open video stream
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Run inference
    results = model(frame)
    print(results)

    # Extract detections
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])  # Get class ID
            conf = float(box.conf[0])  # Confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            detections.append({
                "class": model.names[cls_id],  # Convert class ID to label
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

    # Save detections to JSON
    with open("detections.json", "w") as f:
        json.dump(detections, f, indent=4)

    print("Detections saved:", detections)

    # Display the frame with bounding boxes
    for det in detections:
        label = f"{det['class']} ({det['confidence']:.2f})"
        cv2.rectangle(frame, (det['bbox'][0], det['bbox'][1]), (det['bbox'][2], det['bbox'][3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (det['bbox'][0], det['bbox'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Cabin Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit loop on 'q' key press

cap.release()
cv2.destroyAllWindows()
