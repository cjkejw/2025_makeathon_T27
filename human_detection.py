import cv2
import os
import serial
import torch
from ultralytics import YOLO
from torchvision import transforms
import numpy as np

# Set up serial communication with ESP32
try:
    ser = serial.Serial('COM7', 9600)  # Replace 'COM7' with your correct port
    print("Serial communication established.")
except Exception as e:
    print(f"Error: Unable to establish serial communication. {e}")
    exit()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8n.pt' for the Nano model (smallest and fastest)

# Load MiDaS model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
midas.eval()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Webcam initialized. Press 'q' to quit.")

# Initialize variables for motion detection
ret, previous_frame = cap.read()
if not ret:
    print("Error: Failed to read initial frame.")
    exit()

previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
previous_gray = cv2.GaussianBlur(previous_gray, (31, 31), 0)  # Larger kernel to reduce noise

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Exiting...")
        break

    # Motion Detection
    gray_blur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray_blur, (31, 31), 0)

    frame_delta = cv2.absdiff(previous_gray, gray_blur)
    thresh = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]  # Adjust threshold for sensitivity
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if motion is detected
    motion_detected = any(cv2.contourArea(contour) > 1000 for contour in contours)  # Adjust sensitivity

    previous_gray = gray_blur  # Update the previous frame

    # YOLOv8 Detection
    results = model(frame)  # Pass the frame to the YOLOv8 model
    human_detected = False
    nearest_distance = None

    # Preprocess frame for MiDaS depth estimation
    input_batch = midas_transforms(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)

    # Perform depth estimation
    with torch.no_grad():
        depth_map = midas(input_batch).squeeze().cpu().numpy()

    # Normalize depth map for visualization
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    # Iterate through YOLOv8 detections
    for result in results:
        for box in result.boxes:  # Each detected box
            cls = int(box.cls)  # Class ID
            confidence = float(box.conf)  # Confidence score

            if cls == 0 and confidence > 0.5:  # Class ID 0 = 'person'
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for humans
                human_detected = True

                # Estimate distance using the depth map
                depth_roi = depth_map_normalized[y1:y2, x1:x2]
                if depth_roi.size > 0:
                    avg_depth = depth_roi.mean()  # Average depth within the bounding box
                    distance = 1 / (avg_depth + 1e-6)  # Inverse to approximate distance (scaled)

                    # Check for the nearest human
                    if nearest_distance is None or distance < nearest_distance:
                        nearest_distance = distance

                    # Display the distance on the bounding box
                    cv2.putText(frame, f"Dist: {distance:.2f} m", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Send signals to Arduino
    try:
        if human_detected:
            ser.write(b'1')  # Send '1' to trigger the buzzer
            print(f"[Python] Human detected: Signal '1' sent to Arduino. Nearest distance: {nearest_distance:.2f} m")
        elif motion_detected:
            ser.write(b'2')  # Send '2' to indicate motion detected
            print("[Python] Motion detected: Signal '2' sent to Arduino")
        else:
            ser.write(b'0')  # Send '0' to indicate no detection
            print("[Python] No detection: Signal '0' sent to Arduino")
    except Exception as e:
        print(f"Error: Failed to send signal to Arduino. {e}")
        break

    # Display the resulting frame with YOLOv8 detections and depth map
    cv2.imshow("Webcam Feed - Human Detection", frame)
    cv2.imshow("Depth Map", depth_map_normalized)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
ser.close()
cv2.destroyAllWindows()
print("Webcam and windows successfully released.")
