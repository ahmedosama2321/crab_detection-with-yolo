
from pathlib import Path
from ultralytics import YOLO
import torch

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "runs" / "crab_detection"
    / "yolov8n" / "weights" / "best.pt"
)

DATA_YAML = BASE_DIR / "dataset" / "data.yaml"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=8,
        device=device,
        plots=True,
        project=str(BASE_DIR / "runs" / "crab_detection"),
        name="test_evaluation"
    )

    print("\nTest evaluation completed.")
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
