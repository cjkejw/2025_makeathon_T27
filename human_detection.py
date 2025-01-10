import cv2
import os
import serial
import time    # For adding delays

# Set up serial communication with esp32
try:
    ser = serial.Serial('COM7', 9600)  # Replace 'COM7' with the correct port
    print("Serial communication established.")
except Exception as e:
    print(f"Error: Unable to establish serial communication. {e}")
    exit()

fullbody_cascade_path = "haarcascade_fullbody.xml"
upperbody_cascade_path = "haarcascade_upperbody.xml"
frontalface_cascade_path = "haarcascade_frontalface_default.xml"
profileface_cascade_path = "haarcascade_profileface.xml"

if not all(os.path.exists(p) for p in [fullbody_cascade_path, upperbody_cascade_path, frontalface_cascade_path, profileface_cascade_path]):
    print("Error: One or more Haar Cascade files are missing!")
    exit()

# Load Haar Cascade classifiers
fullbody_cascade = cv2.CascadeClassifier(fullbody_cascade_path)
upperbody_cascade = cv2.CascadeClassifier(upperbody_cascade_path)
frontalface_cascade = cv2.CascadeClassifier(frontalface_cascade_path)
profileface_cascade = cv2.CascadeClassifier(profileface_cascade_path)

# Initialize  webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Webcam initialized. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Exiting...")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect full bodies, upper bodies, frontal faces, and profile faces
    fullbodies = fullbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    upperbodies = upperbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    frontalfaces = frontalface_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    profilefaces = profileface_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Flag to indicate if a human is detected
    human_detected = False

    # Draw rectangles for detected humans
    for (x, y, w, h) in fullbodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green for full body
        human_detected = True
    for (x, y, w, h) in upperbodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue for upper body
        human_detected = True
    for (x, y, w, h) in frontalfaces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red for frontal face
        human_detected = True
    for (x, y, w, h) in profilefaces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)  # Yellow for profile face
        human_detected = True

    # Send signal to the microcontroller if a human is detected
    try:
        if human_detected:
            ser.write(b'1')  # Send '1' to trigger the buzzer
            print("Human detected: Signal sent to Arduino ('1')")
        else:
            ser.write(b'0')  # Send '0' to turn off the buzzer
            print("No human detected: Signal sent to Arduino ('0')")
    except Exception as e:
        print(f"Error: Failed to send signal to Arduino. {e}")
        break

    # Display the resulting frame
    cv2.imshow("Webcam Feed - Human Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(100) & 0xFF == ord('q'):
        print("Exiting program...")
        break

# Release the webcam, serial port, and close all OpenCV windows
cap.release()
ser.close()
cv2.destroyAllWindows()
print("Webcam and windows successfully released.")