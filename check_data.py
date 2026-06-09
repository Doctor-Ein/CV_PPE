from ultralytics import YOLO
import os

# Try to load the dataset which should trigger a download if not found
try:
    model = YOLO('yolov8n.pt')
    # Using a known dataset name that Ultralytics supports or our custom yaml
    # The documentation says we can use 'Construction-PPE.yaml'
    # We created 'ppe_data.yaml' based on it.
    print("Checking dataset...")
    model.train(data='ppe_data.yaml', epochs=1, imgsz=32, batch=1) # Minimal run to trigger download
    print("Dataset check complete.")
except Exception as e:
    print(f"Error during dataset check: {e}")
