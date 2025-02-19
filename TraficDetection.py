import torch
import cv2
from ultralytics import YOLO

# Set the device to CUDA if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the YOLO model
model = YOLO("yolov8n.pt").to(device)

# Open video file
cap = cv2.VideoCapture(r"27260-362770008_small.mp4")

# Define the frame width and height for resizing
frame_width = 640
frame_height = 480

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to the desired dimensions
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Run the model on the frame
    results = model(frame)  # The model already runs on the assigned device (CPU or CUDA)

    # Filter results to count only vehicles (class ID for vehicles is typically 2 or 3)
    vehicle_count = 0
    for box in results[0].boxes:
        class_id = int(box.cls)  # Get class ID
        if class_id in [2, 3]:  # ID 2 and 3 typically correspond to 'car' and 'truck' in YOLO
            vehicle_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the bounding box coordinates
            # Draw bounding box around the detected vehicle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the count of detected vehicles
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show the frame with detections
    cv2.imshow("Traffic Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
