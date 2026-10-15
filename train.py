
from pathlib import Path

import torch
from ultralytics import YOLO


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"


def main():
    # Check dataset configuration
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    # Select GPU when available
    if torch.cuda.is_available():
        device = 0
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("CUDA is unavailable. Using CPU.")

    print("PyTorch:", torch.__version__)
    print("Starting Crab Detection training...")

    # Load YOLOv8 Nano pretrained model
    model = YOLO("yolov8n.pt")

    # Train model
    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        workers=4,
        patience=10,
        seed=42,
        project=str(BASE_DIR / "runs" / "crab_detection"),
        name="yolov8n",
        plots=True
    )

    print("Training finished.")
    print(
        "Best model:",
        BASE_DIR / "runs" / "crab_detection"
        / "yolov8n" / "weights" / "best.pt"
    )


if __name__ == "__main__":
    main()
