# 📘 Brain Tumor Detection with YOLOv8  
_Object Detection on MRI Images using Ultralytics YOLOv8_

---

## 🧠 Overview

This project applies **YOLOv8** to detect and localize **brain tumors** in MRI images.  
It uses the Kaggle dataset *"Medical Image Dataset: Brain Tumor Detection"* and demonstrates:

- Training YOLOv8 on MRI scans  
- Evaluating model performance (Precision, Recall, mAP)  
- Running inference on unseen test images  
- Exporting example predictions  
- A full notebook workflow for reproducibility  

The goal is to provide a clean, practical template for medical-image object detection using modern deep learning tools.

---

## 📂 Repository Structure

```
brain-tumor-yolov8/
│
├── notebooks/
│   └── tumor.ipynb            # Main training & inference notebook
│
├── data/
│   └── data.yaml       # YOLO dataset configuration
│
├── results/
│   ├── training_curves.png    # Loss & metric curves
│   └── sample_predictions/    # Few example prediction images
│
├── requirements.txt
└── README.md
```

📌 **Note:**  
The dataset and large YOLO `runs/` directory are excluded from the repo to keep it lightweight.

---


### 🔍 Example Predictions

Detection examples are included under:

```
results/sample_predictions/
```

---

## 📥 Dataset

**Dataset:**  
Medical Image Dataset: Brain Tumor Detection (Kaggle)  
https://www.kaggle.com/code/pkdarabi/brain-tumor-detection-with-yolov8

🔒 *The dataset is not included due to licensing restrictions.*



---

## 🏋️ Training the Model

Training is done inside the notebook:

```
notebooks/tumor.ipynb
```

It includes:

- YOLOv8 setup  
- Dataset loading  
- Training loop  
- Validation  
- Inference  
- Saving predictions  

A typical training command:

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data="data/brain_tumor.yaml",
    epochs=50,
    imgsz=640,
    batch=4
)
```

Training results (best model, loss curves, predictions) will be created locally inside a `runs/` directory.

---

## 🔮 Running Inference

After training:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
model.predict(
    source="data/raw/brain_tumor/images/test",
    save=True
)
```

Predicted images will be saved to:

```
runs/detect/predict/
```

---


## 💡 Notes & Tips

- If your GPU has limited VRAM (6GB or less), use:
  - `yolov8n.pt` or `yolov8s.pt`
  - Lower batch size (`batch=2` or `batch=1`)
  - Smaller image size (`imgsz=512` or `416`)
- Medical images may benefit from preprocessing (CLAHE, normalization)
- Training longer (100–200 epochs) typically increases mAP on medical datasets
- For better localization, segmentation models (e.g., YOLOv8-seg, U-Net) can outperform detection models

---

## 🙏 Acknowledgements

- **Dataset:** Kaggle — Medical Image Dataset: Brain Tumor Detection  
- **Model:** Ultralytics YOLOv8  
- https://github.com/ultralytics/ultralytics

---

## 📜 License

This repository is released for educational and research purposes.
