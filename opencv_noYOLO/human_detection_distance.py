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

# Load Haar Cascade for face detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Constants for distance estimation
KNOWN_DISTANCE = 76.2  # Known distance to the reference object in cm
KNOWN_WIDTH = 14.3     # Known width of the reference object (face) in cm
FOCAL_LENGTH = None    # Focal length (to be calculated)

# Focal length calculation using a reference image
def calculate_focal_length(measured_distance, real_width, width_in_pixels):
    return (width_in_pixels * measured_distance) / real_width

# Distance estimation function
def estimate_distance(focal_length, real_width, width_in_pixels):
    if width_in_pixels > 0:
        return (real_width * focal_length) / width_in_pixels
    return -1

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Webcam initialized. Press 'q' to quit.")

# Capture a reference frame to calculate focal length
ret, ref_frame = cap.read()
if not ret:
    print("Error: Failed to capture reference frame.")
    exit()

# Detect face in the reference frame
gray_ref = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
faces_ref = haar_cascade.detectMultiScale(gray_ref, scaleFactor=1.1, minNeighbors=5)

if len(faces_ref) > 0:
    x, y, w, h = faces_ref[0]
    FOCAL_LENGTH = calculate_focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, w)
    print(f"Focal Length Calculated: {FOCAL_LENGTH:.2f}")
else:
    print("No face detected in the reference frame. Ensure proper setup.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Exiting...")
        break

    # YOLOv8 Human Detection
    results = model(frame)
    human_detected = False

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            confidence = float(box.conf)

            if cls == 0 and confidence > 0.5:  # If class is "person" and confidence is high
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                human_detected = True

                # Detect face within YOLO bounding box using Haar Cascade
                roi = frame[y1:y2, x1:x2]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                faces = haar_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5)

                for (fx, fy, fw, fh) in faces:
                    # Adjust face coordinates relative to the original frame
                    face_x1 = x1 + fx
                    face_y1 = y1 + fy
                    face_x2 = face_x1 + fw
                    face_y2 = face_y1 + fh

                    # Estimate distance to the face
                    distance = estimate_distance(FOCAL_LENGTH, KNOWN_WIDTH, fw)
                    print(f"Estimated Distance: {distance:.2f} cm")

                    # Draw bounding box for the face
                    cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Dist: {distance:.2f}cm", (face_x1, face_y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw YOLO bounding box for the human
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

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
