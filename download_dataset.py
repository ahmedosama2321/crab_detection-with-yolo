import os
from pathlib import Path
from roboflow import Roboflow

API_KEY = os.environ.get("ROBOFLOW_API_KEY")

WORKSPACE = "crab-detection-1wjmy"
PROJECT = "european-crab-wepdb"
VERSION = 5

def main():
    if not API_KEY:
        print("ERROR: Roboflow API key is missing.")
        print("Set ROBOFLOW_API_KEY before running this script.")
        return

    print("Connecting to Roboflow...")

    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)

    output_dir = Path(__file__).resolve().parent / "dataset"

    print("Downloading European Crab Dataset v5...")

    version.download(
        "yolov8",
        location=str(output_dir)
    )

    print(f"Download completed: {output_dir}")

if __name__ == "__main__":
    main()
