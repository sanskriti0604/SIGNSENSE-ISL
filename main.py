from ultralytics import YOLO
import cv2
import pyttsx3

# Load model
model = YOLO("best.pt")

# Text to speech
engine = pyttsx3.init()
last_spoken = ""

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, conf=0.25, imgsz=320)

    annotated_frame = results[0].plot()

    # Check if detections exist
    if results[0].boxes is not None and len(results[0].boxes.cls) > 0:

        for box in results[0].boxes.cls:

            class_id = int(box)
            label = model.names[class_id]

            print("Detected:", label)

            if label != last_spoken:
                engine.say(label)
                engine.runAndWait()
                last_spoken = label

    cv2.imshow("SIGN-SENSE ISL Interpreter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()