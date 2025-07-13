# 🤖 Real-Time Sign Language Translator using X3D-XS

This project implements a **real-time sign language translator** using the **AUTSL Dataset** and the **X3D-XS** video classification model. The system processes sign language gestures from a webcam feed and outputs their English meanings. It is built for efficiency and real-time performance using modern deep learning architectures.

---

## 📊 Results

* **Training Accuracy**: **89%**
* **Validation Accuracy**: **81%**
* Model: `X3D-XS` from PyTorchVideo hub
* Optimized for **real-time inference** using `AMP (Mixed Precision)` and `X3D`'s compact architecture

---

## 📁 Project Structure

```
AUTSL-Translator/
├── data/
│   └── raw/
│       └── AUTSL/
│           ├── train.csv
│           ├── val.csv
│           └── test.csv
├── processed_data/
│   ├── train/             # Extracted frames
│   ├── val/
│   └── test/
├── models/
│   └── x3d_model.py       # X3D-XS architecture
├── datasets/
│   └── dataset.py         # Custom PyTorch Dataset
├── checkpoints/
│   └── best_x3d_xs.pth    # Trained model weights
├── train.py               # Training script
├── extract_frames.py      # Frame extractor
├── realtime_inference.py  # Real-time sign recognition
├── sign.csv               # ClassId → EN mappings
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Extract Frames from Videos

To extract frames from raw AUTSL videos (change path inside the code accordingly):

```bash
python extract_frames.py
```

Update paths for train/val/test extraction:

```python
if __name__ == "__main__":
    extract_all_frames(
        video_dir="data/raw/AUTSL/train",
        output_dir="processed_data/train",
        csv_path="data/raw/AUTSL/train.csv"
    )
```

Do this similarly for `val` and `test`.

### 3. Train the Model

```bash
python train.py
```

* Uses `X3D-XS` pretrained model
* Automatically saves best model to `checkpoints/best_x3d_xs.pth`
* Mixed precision and scheduler support included

### 4. Real-Time Inference

Connect a webcam and run:

```bash
python realtime_inference.py
```

Make sure `sign.csv` is in the root and `best_x3d_xs.pth` exists in `checkpoints/`.

---

## 🧠 Technologies Used

* **PyTorch** + **TorchVision** + **TorchAMP** (Mixed Precision)
* **PyTorchVideo** (X3D-XS architecture)
* **OpenCV** for webcam input
* **AUTSL Dataset** with over 226 sign classes
* Real-time frame buffering using **deque**
* Trained on **RTX 4070 8GB** (average 70–90% GPU usage)

---

## 🏁 Final Note

The project is optimized for educational and demonstration purposes. With minimal dependencies and lightweight architecture, it showcases how advanced video models like X3D can be effectively used for real-time translation tasks.

🎉 Validation accuracy above **80%**, trained end-to-end with extracted frames. You can integrate it into AR/VR pipelines, accessibility software, or language learning platforms.

---

Feel free to fork, improve, or extend it for larger vocabularies or faster inference!
