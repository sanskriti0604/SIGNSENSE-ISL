import threading
import sys
from ultralytics import YOLO
import cv2
import pyttsx3

# --- Configuration ---
MODEL_PATH = "best.pt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.25
INFERENCE_IMGSZ = 320

# --- State ---
last_spoken = ""
is_speaking = False

# --- Initialize Text-to-Speech ---
def init_engine():
    """Initializes and returns a new pyttsx3 engine."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150) # Optional: Adjust speaking rate
    return engine

def speak_worker(text):
    """Background thread function to handle TTS."""
    global is_speaking
    try:
        # We need a new isolated engine instance per thread on some OS/pyttsx versions
        temp_engine = init_engine()
        temp_engine.say(text)
        temp_engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")
    finally:
        is_speaking = False

def speak_async(text):
    """Starts a new background thread to speak text without blocking."""
    global is_speaking
    if not is_speaking:
        is_speaking = True
        threading.Thread(target=speak_worker, args=(text,), daemon=True).start()


def main():
    global last_spoken
    
    print("Loading YOLO model...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model from {MODEL_PATH}: {e}")
        sys.exit(1)

    print(f"Starting webcam (Index: {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {CAMERA_INDEX}.")
        sys.exit(1)

    print("SIGN-SENSE ISL Interpreter started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam. Exiting...")
                break

            # Run YOLO detection
            results = model(frame, conf=CONFIDENCE_THRESHOLD, imgsz=INFERENCE_IMGSZ, verbose=False)

            annotated_frame = results[0].plot()
            current_sign = "None"
            best_conf = 0.0

            # Check if detections exist
            if results[0].boxes is not None and len(results[0].boxes.cls) > 0:
                
                # Find the detection with the highest confidence
                for i in range(len(results[0].boxes.cls)):
                    conf = float(results[0].boxes.conf[i])
                    
                    if conf > best_conf:
                        best_conf = conf
                        class_id = int(results[0].boxes.cls[i])
                        current_sign = model.names[class_id]
                
                # Speak if the highest confidence sign is new and we aren't already speaking
                if current_sign != "None":
                    print(f"Detected: {current_sign} (Conf: {best_conf:.2f})")
                    if current_sign != last_spoken and current_sign != "":
                        speak_async(current_sign)
                        last_spoken = current_sign

            # Draw Custom UI Overlay for Current Sign
            overlay_text = f"Sign: {current_sign}"
            if current_sign != "None":
                overlay_text += f" ({best_conf:.2f})"
                
            cv2.rectangle(annotated_frame, (0, 0), (640, 50), (0, 0, 0), -1)  # Black top bar
            cv2.putText(annotated_frame, overlay_text, (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # FPS could potentially be calculated and added here as well if desired

            cv2.imshow("SIGN-SENSE ISL Interpreter", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()