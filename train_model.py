from ultralytics import YOLO
import os

def train():
    print("Loading base YOLOv8n model...")
    # Load a pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")
    
    print("Starting training process on ISL_dataset/data.yaml...")
    print("Note: This may take several hours on a CPU.")
    
    # Train the model with recommended augmentations
    results = model.train(
        data="ISL_dataset/data.yaml",
        epochs=25,        # Same as previous training
        imgsz=640,        # Standard resolution
        augment=True,     # Enable data augmentations (flips, rotations, etc.) for better robustness
        workers=0         # Set workers to 0 for Windows CPU stability
    )
    
    print("Training finished!")
    print("The newly trained weights are located in the 'runs/detect/' folder.")
    print("Look for the latest 'train' folder, inside its 'weights' directory (e.g., 'runs/detect/train6/weights/best.pt').")

if __name__ == "__main__":
    train()
