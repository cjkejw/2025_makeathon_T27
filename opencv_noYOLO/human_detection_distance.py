import cv2
import serial
from ultralytics import YOLO

# Set up serial communication with ESP32
try:
    ser = serial.Serial('COM7', 9600)  # Replace 'COM7' with your correct port
    print("Serial communication established.")
except Exception as e:
    print(f"Error: Unable to establish serial communication. {e}")
    exit()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 model

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Webcam initialized. Press 'q' to quit.")

# Constants
KNOWN_HEIGHT = 1.7  # Average height of a human (meters)
CALIBRATED_FOCAL_LENGTH = 700  # Calibrated focal length in pixels (adjust after calibration)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Exiting...")
        break

    # YOLOv8 Detection
    results = model(frame)
    human_detected = False

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            confidence = float(box.conf)

            if cls == 0 and confidence > 0.5:  # If class is "person" and confidence is high
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                pixel_height = y2 - y1  # Bounding box height in pixels

                # Distance estimation
                if pixel_height > 0:
                    distance = (KNOWN_HEIGHT * CALIBRATED_FOCAL_LENGTH) / pixel_height
                    print(f"Estimated Distance: {distance:.2f} meters")

                    # Draw bounding box and display distance
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Dist: {distance:.2f}m", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                human_detected = True

    # Send signal to Arduino
    try:
        if human_detected:
            ser.write(b'1')  # Signal for human detection
            print("[Python] Human detected: Signal '1' sent to Arduino")
        else:
            ser.write(b'0')  # Signal for no detection
            print("[Python] No detection: Signal '0' sent to Arduino")
    except Exception as e:
        print(f"Error: Failed to send signal to Arduino. {e}")
        break

    # Display the resulting frame
    cv2.imshow("Webcam Feed - Human Detection with Distance Estimation", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
ser.close()
cv2.destroyAllWindows()
print("Webcam and windows successfully released.")
