from ultralytics import YOLO
import cv2
import json
import os

# Load YOLOv8 model
model = YOLO("/home/nvidia/yolo_model/best.pt")  # Replace with your trained model
print("Model loaded successfully!")  # Debugging

# Open webcam
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize an empty list to store detections across frames
all_detections = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture frame or frame is empty")
        break

    # Run inference
    results = model(frame)
    print(results)  # Debugging: Check if YOLO detects anything

    # Extract detections
    detections = []
    for r in results:
        if hasattr(r, "boxes"):  # Ensure 'boxes' attribute exists
            for box in r.boxes:
                cls_id = int(box.cls[0])  # Get class ID
                conf = float(box.conf[0])  # Confidence score
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

                detections.append({
                    "class": model.names[cls_id],  # Convert class ID to label
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

    # Debugging: Check if detections list is empty
    if not detections:
        print("No objects detected.")
    else:
        # Append detections to the master list
        all_detections.extend(detections)

        # Save detections to JSON (Append Mode)
        if os.path.exists("new.json"):
            with open("new.json", "r") as f:
                try:
                    existing_data = json.load(f)  # Load existing JSON data
                except json.JSONDecodeError:
                    existing_data = []  # Handle case where JSON is empty or corrupted
        else:
            existing_data = []

        # Combine previous data with new detections
        existing_data.extend(detections)

        # Save the updated JSON
        with open("new.json", "w") as f:
            json.dump(existing_data, f, indent=4)

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

