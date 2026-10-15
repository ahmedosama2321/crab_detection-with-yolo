
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR / "runs" / "crab_detection"
    / "yolov8n" / "weights" / "best.pt"
)


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Train the model first.")
        return

    image_input = input("Enter image path: ").strip()
    image_path = Path(image_input).expanduser()

    if not image_path.is_file():
        print("Image file not found:", image_path)
        return

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        save=True
    )

    print("Prediction completed.")
    print("Check the runs/detect folder for the result.")


if __name__ == "__main__":
    main()
