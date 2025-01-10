import cv2
import os

# Absolute paths to Haar Cascade XML files
fullbody_cascade_path = os.path.join(base_path, "haarcascade_fullbody.xml")
upperbody_cascade_path = os.path.join(base_path, "haarcascade_upperbody.xml")
frontalface_cascade_path = os.path.join(base_path, "haarcascade_frontalface_default.xml")
profileface_cascade_path = os.path.join(base_path, "haarcascade_profileface.xml")

# Ensure all cascade files exist
if not all(os.path.exists(p) for p in [fullbody_cascade_path, upperbody_cascade_path, frontalface_cascade_path, profileface_cascade_path]):
    print("Error: One or more Haar Cascade files are missing!")
    exit()

# Load Haar Cascade classifiers
fullbody_cascade = cv2.CascadeClassifier(fullbody_cascade_path)
upperbody_cascade = cv2.CascadeClassifier(upperbody_cascade_path)
frontalface_cascade = cv2.CascadeClassifier(frontalface_cascade_path)
profileface_cascade = cv2.CascadeClassifier(profileface_cascade_path)

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
    thresh = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]  # Increased threshold to reduce noise
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if motion is detected
    motion_detected = any(cv2.contourArea(contour) > 1000 for contour in contours)  # Adjust sensitivity

    previous_gray = gray_blur  # Update the previous frame

    # Human Detection
    human_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Use unblurred frame
    fullbodies = fullbody_cascade.detectMultiScale(human_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    upperbodies = upperbody_cascade.detectMultiScale(human_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    frontalfaces = frontalface_cascade.detectMultiScale(human_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    profilefaces = profileface_cascade.detectMultiScale(human_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    human_detected = False
    for (x, y, w, h) in fullbodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box for full body
        human_detected = True
    for (x, y, w, h) in upperbodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box for upper body
        human_detected = True
    for (x, y, w, h) in frontalfaces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box for frontal face
        human_detected = True
    for (x, y, w, h) in profilefaces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Yellow box for profile face
        human_detected = True

    # Print detection status
    if motion_detected and human_detected:
        print("Motion and Human detected!")
    elif motion_detected:
        print("Motion detected!")
    elif human_detected:
        print("Human detected!")

    # Display the resulting frame (only with human detection bounding boxes)
    cv2.imshow("Webcam Feed - Human Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(50) & 0xFF == ord('q'):
        print("Exiting program...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Webcam and windows successfully released.")
