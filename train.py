"""Trains YOLOv8s on the speed limit dataset."""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data" / "speed_dataset" / "data.yaml"


def main():
    # starts from the pretrained COCO weights - 'transfer learning'
    model = YOLO("yolov8s.pt") #shifted from yolov8n.pt to yolov8s.pt for better accuracy

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=32,
        workers=8,
        patience=10,  # stop if the validation score hasn't improved for 10 epochs
        name="speed_limit_detector",
    )

    # results go in runs/detect/speed_limit_detector 
    print("Done. The trained model is the weights/best.pt file inside that folder.")


if __name__ == "__main__":
    main()
