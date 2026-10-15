# 🦀 Crab Detection with YOLOv8

A computer vision project for detecting and classifying crabs in images using **YOLOv8** and a custom Roboflow dataset.

## 📌 Project Overview

This project uses a YOLOv8 object detection model to locate crabs in images and identify their classes. The model was trained using a Roboflow dataset and PyTorch with GPU acceleration.

## 🎯 Classes

The model detects three crab classes:

- Jonah Crab
- Rock Crab
- European Crab

## 🛠️ Technologies Used

- Python
- Ultralytics YOLOv8
- PyTorch
- CUDA
- Roboflow
- OpenCV

## 📂 Project Structure

```text
Crab-Detection-YOLO/
├── dataset/           # Dataset (not included in GitHub)
├── runs/              # Training and prediction results
├── train.py           # Model training
├── predict.py         # Object detection on images
├── evaluate.py        # Model evaluation
├── requirements.txt   # Python dependencies
├── .gitignore         # Files excluded from Git
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ahmedosama2321/crab_detection-with-yolo.git
cd crab_detection-with-yolo
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch version compatible with your NVIDIA GPU and CUDA environment.

## 📊 Dataset

The dataset was obtained from Roboflow Universe:

[European Crab Dataset — Version 5](https://universe.roboflow.com/crab-detection-1wjmy/european-crab-wepdb/dataset/5)

The dataset contains training, validation, and test splits. Dataset files are excluded from this repository to keep it lightweight.

After downloading and extracting the dataset, ensure the following structure exists:

```text
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Update the dataset path in `dataset/data.yaml` if necessary.

## 🚀 Training

Train the YOLOv8 model using:

```bash
python train.py
```

The trained model weights and training results are saved under:

```text
runs/crab_detection/yolov8n/weights/
```

The best model is saved as `best.pt`.

## 🔍 Prediction

Run predictions on an image using:

```bash
python predict.py
```

Enter the path to the image when prompted.

Alternatively, run YOLO directly:

```bash
yolo predict model=runs/crab_detection/yolov8n/weights/best.pt source=path/to/image.jpg imgsz=640 conf=0.25
```

Prediction images are saved in the `runs/detect/` directory by default.

## 📈 Evaluation

Evaluate the trained model on the test split:

```bash
python evaluate.py
```

### Test Results

| Metric | Score |
|---|---:|
| Precision | 99.72% |
| Recall | 97.71% |
| mAP@50 | 97.83% |
| mAP@50–95 | 89.15% |

These results were obtained on the project's test split. Performance may vary on unseen images and different underwater environments.

## 💻 Hardware

Training was performed using an NVIDIA GeForce RTX 5060 Laptop GPU with CUDA-enabled PyTorch.

## 🔮 Future Improvements

- Test on more real-world underwater images.
- Improve detection in crowded scenes and challenging lighting conditions.
- Evaluate performance on images from different underwater environments.
- Experiment with larger YOLO models and compare accuracy and inference speed.

## 👨‍💻 Author

**Ahmed Osama**

GitHub: [@ahmedosama2321](https://github.com/ahmedosama2321)

## 📄 License

Choose and add an appropriate license before redistributing this project or its code.
