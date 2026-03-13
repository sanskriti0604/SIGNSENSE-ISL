from ultralytics import YOLO

def export_model():
    print("Loading YOLO model 'best.pt'...")
    try:
        model = YOLO("best.pt")
        
        # ONNX is preferred for CPU inference specifically.
        print("Exporting model to ONNX format. This may take a minute...")
        success = model.export(format="onnx", imgsz=640)
        
        if success:
            print("Model exported successfully!")
            print(f"Exported model path: {success}")
        else:
            print("Export failed.")
            
    except Exception as e:
        print(f"An error occurred during export: {e}")

if __name__ == "__main__":
    export_model()
