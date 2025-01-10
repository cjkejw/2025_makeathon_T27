import cv2
import os
import serial
import time    # For adding delays

# Set up serial communication with ESP32
try:
    ser = serial.Serial('COM7', 9600)  #'COM7' - port
    print("Serial communication established.")
except Exception as e:
    print(f"Error: Unable to establish serial communication. {e}")
    exit()

# Paths to Haar Cascade XML files
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

# Initialize variables for motion detection
ret, previous_frame = cap.read()
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
previous_gray = cv2.GaussianBlur(previous_gray, (21, 21), 0)  # Reduce noise

# Track the last signal sent
last_signal = '0'

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Exiting...")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)  # Reduce noise

    # Compute  absolute difference between current frame and prev frame
    frame_delta = cv2.absdiff(previous_gray, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)  # Fill in holes

    # Find contours in thresholded frame
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if motion is detected
    motion_detected = any(cv2.contourArea(contour) > 500 for contour in contours)  # Adjust 500 as needed

    # Update previous frame
    previous_gray = gray

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

    # # Send signal to the microcontroller if a human is detected
    # try:
    #     if human_detected:
    #         ser.write(b'1')  # Send '1' to trigger the buzzer
    #         print("Human detected: Signal sent to Arduino ('1')")
    #     else:
    #         ser.write(b'0')  # Send '0' to turn off the buzzer
    #         print("No human detected: Signal sent to Arduino ('0')")
    # except Exception as e:
    #     print(f"Error: Failed to send signal to Arduino. {e}")
    #     break

    # Combine motion detection and human detection
    try:
        if motion_detected or human_detected:
            if last_signal != '1':
                ser.write(b'1')  # Send '1' to trigger the buzzer
                print("Motion or Human detected: Sent '1'")
                last_signal = '1'
        else:
            if last_signal != '0':
                ser.write(b'0')  # Send '0' to turn off the buzzer
                print("No motion or human detected: Sent '0'")
                last_signal = '0'
    except Exception as e:
        print(f"Error: Failed to send signal to ESP32. {e}")
        break

    # Display the resulting frame
    cv2.imshow("Webcam Feed - Human Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): #waitKey(1) waits for 1ms b4 moving to next frame, lesser delay than 100ms bcs now is life
        print("Exiting program...")
        break

# Release webcam, serial port, and close all OpenCV windows
cap.release()
ser.close()
cv2.destroyAllWindows()
print("Webcam and windows successfully released.")