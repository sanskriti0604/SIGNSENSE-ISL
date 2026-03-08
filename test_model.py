from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model("test.jpg", show=True)