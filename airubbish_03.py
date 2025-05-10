import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import servo_controller  # Make sure this file is in the same folder
import time

# Load your trained model
model = load_model("waste_classifier.h5")

# Categories (make sure they match your training labels)
categories = ["paper", "aluminium can", "plastic bottles", "other waste"]

# Setup camera (USB webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open webcam.")
    exit()

print("[INFO] Starting recognition... Press 'q' to quit.")

def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img = preprocess_frame(frame)
    prediction = model.predict(img)[0]
    class_idx = np.argmax(prediction)
    label = categories[class_idx]

    # Display prediction
    cv2.putText(frame, f"Prediction: {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Waste Classification", frame)

    # Perform servo actions
    print(f"[AI] Detected: {label}")
    
    if label == "paper":
        servo_controller.move_servo(channel=0, angle=30)  # Rotate plate to paper bin
    elif label == "aluminium can":
        servo_controller.move_servo(channel=0, angle=60)
    elif label == "plastic bottles":
        servo_controller.move_servo(channel=0, angle=90)
    else:
        servo_controller.move_servo(channel=0, angle=120)

    time.sleep(1.0)  # Pause briefly before opening trapdoor
    servo_controller.move_servo(channel=1, angle=60)  # Open trapdoor
    time.sleep(1.0)
    servo_controller.move_servo(channel=1, angle=0)   # Close trapdoor

    # Wait for key to exit or continue
    key = cv2.waitKey(1000) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
servo_controller.close()
