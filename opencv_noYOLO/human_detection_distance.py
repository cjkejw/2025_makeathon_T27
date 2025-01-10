import cv2
import serial
import time

# Initialize serial communication with ESP32
esp32 = serial.Serial('COM3', 115200)  # Replace 'COM3' with your ESP32 port
time.sleep(2)  # Allow time for the serial connection to initialize

# Initialize OpenCV's HOG descriptor for human detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Start video capture from the laptop camera
cap = cv2.VideoCapture(0)  # 0 = Default laptop camera, change to 1/2 for external cameras

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect humans in the video frame
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in boxes:
        # Draw bounding box around detected humans
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Estimate distance based on bounding box height (simple approximation)
        distance = int(1000 / h)  # Example formula for distance estimation in cm

        # Display the distance on the video feed
        cv2.putText(frame, f"Distance: {distance} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Send the distance to ESP32
        esp32.write(f"{distance}\n".encode())

    # Display the video feed with bounding boxes and distance
    cv2.imshow("Human Detection with Distance Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
