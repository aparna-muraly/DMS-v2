import cv2
import torch
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("/home/nvidia/yolo_model/best.pt")  # Replace "best.pt" with your actual trained model file

# Load the image
image_path = "/home/nvidia/Downloads/tes.jpeg"  # Replace with your image path
frame = cv2.imread(image_path)

# Perform inference
results = model(frame)

# Define the class names (update based on your trained model's labels)
class_names = {0: 'Open Eye', 1: 'Closed Eye', 2: 'Cigarette', 3: 'Phone', 4: 'Seatbelt'}  # Update based on training

violation_detected = False

# Loop through detections
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0].item())  # Class index
        label = class_names[cls]

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Print detected violations
        if label in class_names:
            print(f"Violation Detected: {label} (Confidence: {conf:.2f})")
            violation_detected = True

# Save or display the output image
output_path = "output.jpg"
cv2.imwrite(output_path, frame)
cv2.imshow("Detected Violations", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

if not violation_detected:
    print("No violations detected.")

