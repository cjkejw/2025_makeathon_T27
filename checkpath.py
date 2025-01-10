import cv2

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if cascade.empty():
    print("Failed to load haarcascade_frontalface_default.xml")
else:
    print("Successfully loaded haarcascade_frontalface_default.xml")
