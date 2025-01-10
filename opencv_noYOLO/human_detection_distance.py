import cv2
from ultralytics import YOLO

# Constants for calibration
KNOWN_OBJECT_HEIGHT = 170  # Example: Height of the object in cm (e.g., average human height)
KNOWN_DISTANCE = 200  # Example: Distance from camera to object during calibration in cm

# Focal length will be calculated during calibration
focal_length = None

# YOLO Model
model = YOLO("yolov8n.pt")

# Calibration function to calculate focal length
def calibrate_focal_length(frame, known_distance, known_height):
    results = model(frame)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            confidence = float(box.conf)

            if cls == 0 and confidence > 0.5:  # Class ID 0 = "person"
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                bounding_box_height = y2 - y1
                if bounding_box_height > 0:  # Avoid division by zero
                    return (bounding_box_height * known_distance) / known_height
    return None

# Distance estimation function
def calculate_distance(focal_length, known_height, bounding_box_height):
    if bounding_box_height > 0:  # Avoid division by zero
        return (known_height * focal_length) / bounding_box_height
    return None

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Webcam initialized. Press 'q' to quit.")

# Step 1: Calibrate focal length
print("Calibrating focal length... Place a reference object at a known distance.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Exiting...")
        break

    cv2.imshow("Calibration - Place Reference Object", frame)

    # Press 'c' to capture frame for calibration
    if cv2.waitKey(1) & 0xFF == ord('c'):
        focal_length = calibrate_focal_length(frame, KNOWN_DISTANCE, KNOWN_OBJECT_HEIGHT)
        if focal_length:
            print(f"Focal Length Calculated: {focal_length:.2f}")
            break
        else:
            print("Error: Could not detect reference object. Try again.")

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Step 2: Use YOLO for distance estimation
print("Starting distance estimation using YOLO...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Exiting...")
        break

    # YOLO Detection
    results = model(frame)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            confidence = float(box.conf)

            if cls == 0 and confidence > 0.5:  # Class ID 0 = "person"
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                bounding_box_height = y2 - y1

                # Estimate distance
                distance = calculate_distance(focal_length, KNOWN_OBJECT_HEIGHT, bounding_box_height)
                if distance:
                    print(f"Estimated Distance: {distance:.2f} cm")

                    # Draw bounding box and display distance
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Dist: {distance:.2f}cm", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("YOLO - Distance Estimation", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam and windows successfully released.")
