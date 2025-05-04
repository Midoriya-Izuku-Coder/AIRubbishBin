import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import pigpio

# Load the trained model
MODEL_PATH = 'final_model_weights.hdf5'  # or waste_classifier.h5
model = load_model(MODEL_PATH)

# Define class labels
class_labels = ['Organic', 'Recyclable']  # Update if your model has more classes

# Servo setup (using pigpio)
pi = pigpio.pi()
SERVO_PIN = 18  # GPIO pin connected to servo signal wire
SERVO_POSITIONS = {
    'Organic': 1000,      # Example: degrees mapped to PWM pulse width
    'Recyclable': 2000,
    'Default': 1500       # Idle position
}

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

def classify_frame(frame):
    image = cv2.resize(frame, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype("float") / 255.0
    preds = model.predict(image)
    label_index = np.argmax(preds[0])
    label = class_labels[label_index]
    confidence = preds[0][label_index]
    return label, confidence

def move_servo_to(label):
    pulse_width = SERVO_POSITIONS.get(label, SERVO_POSITIONS['Default'])
    print(f"Moving servo to: {label} ({pulse_width} µs)")
    pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)
    time.sleep(1.5)  # allow time for servo to rotate
    pi.set_servo_pulsewidth(SERVO_PIN, 0)  # stop sending signal

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        label, confidence = classify_frame(frame)
        print(f"Prediction: {label} ({confidence:.2f})")

        move_servo_to(label)

        # Wait for object to fall and disk to return
        time.sleep(2.5)

        # Return to default position
        move_servo_to('Default')

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
    print("Cleaned up and exited.")
