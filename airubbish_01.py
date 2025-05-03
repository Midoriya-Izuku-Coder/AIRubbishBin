import cv2
import numpy as np
import tensorflow as tf
import time

# 如果是樹莓派要控制GPIO
import pigpio

# ------------------ Configuration ------------------
MODEL_PATH = 'waste_classifier.h5'

# 兩個Servo
DISK_SERVO_PIN = 18  # 控制圓盤旋轉的servo
DOOR_SERVO_PIN = 19  # 控制活門開關的servo

LABELS = ['paper', 'aluminium_can', 'plastic_bottle', 'other']

# 每個分類對應圓盤角度
DISK_ANGLE_MAP = {
    'paper': 30,
    'aluminium_can': 90,
    'plastic_bottle': 150,
    'other': 210
}

# 活門角度
DOOR_CLOSED_ANGLE = 0    # 活門關上
DOOR_OPEN_ANGLE = 90     # 活門打開

# ------------------ Initialization ------------------
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("[INFO] Connecting to pigpio...")
pi = pigpio.pi()
if not pi.connected:
    raise Exception("Could not connect to pigpio daemon!")

print("[INFO] Starting camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Camera not found!")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ------------------ Helper Functions ------------------
def set_servo_angle(pin, angle):
    pulse_width = 500 + (angle / 180.0) * 2000
    pi.set_servo_pulsewidth(pin, pulse_width)
    time.sleep(0.5)  # 等待servo動作完成

def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def rotate_disk_to(label):
    target_angle = DISK_ANGLE_MAP[label]
    set_servo_angle(DISK_SERVO_PIN, target_angle)
    print(f"[ACTION] Rotated disk to {label} position.")

def open_trapdoor():
    set_servo_angle(DOOR_SERVO_PIN, DOOR_OPEN_ANGLE)
    print("[ACTION] Trapdoor opened.")

def close_trapdoor():
    set_servo_angle(DOOR_SERVO_PIN, DOOR_CLOSED_ANGLE)
    print("[ACTION] Trapdoor closed.")

# ------------------ Main Loop ------------------
print("[INFO] System ready. Press 'q' to exit.")
try:
    # 初始把活門關上
    close_trapdoor()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        img = preprocess_frame(frame)
        predictions = model.predict(img)
        label_index = np.argmax(predictions[0])
        confidence = predictions[0][label_index]
        label = LABELS[label_index]

        # Show preview
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("AI Rubbish Bin", frame)

        if confidence > 0.8:
            print(f"[INFO] Detected {label} with {confidence:.2f} confidence.")

            # 旋轉圓盤到對應分類
            rotate_disk_to(label)
            time.sleep(1)

            # 打開活門
            open_trapdoor()
            time.sleep(1.5)  # 等垃圾掉落

            # 關上活門
            close_trapdoor()
            time.sleep(1)

            # 回到初始位置（可選）
            set_servo_angle(DISK_SERVO_PIN, 0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")

# ------------------ Cleanup ------------------
print("[INFO] Cleaning up...")
cap.release()
cv2.destroyAllWindows()
pi.set_servo_pulsewidth(DISK_SERVO_PIN, 0)
pi.set_servo_pulsewidth(DOOR_SERVO_PIN, 0)
pi.stop()
