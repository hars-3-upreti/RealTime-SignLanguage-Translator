import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from models.x3d_model import build_x3d_xs
from collections import deque
import csv
import time
from PIL import Image  

def load_label_map(csv_path):
    label_map = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row['ClassId'])
                label = row['EN']
                label_map[idx] = label
            except ValueError:
                continue  # Skip header or invalid rows
    return label_map

# === Load class labels ===
label_map = load_label_map("sign.csv")
known_class_ids = set(label_map.keys())

# === Load Model ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = build_x3d_xs(num_classes=228, pretrained=False)
checkpoint = torch.load("checkpoints/best_x3d_xs.pth", map_location=device)
model.load_state_dict(checkpoint)
model = model.to(device).eval()

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
])

# === Parameters ===
NUM_FRAMES = 16
FRAME_STRIDE = 2

# === Frame buffer ===
frame_buffer = deque(maxlen=NUM_FRAMES * FRAME_STRIDE)

# === Webcam ===
cap = cv2.VideoCapture(0)
print("🚀 Starting Real-Time Sign Recognition... Press 'q' to quit.")

# === Cooldown Timer ===
last_pred_time = 0
PREDICTION_INTERVAL = 3  # seconds
current_prediction = "Waiting..."

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    frame_buffer.append(frame)

    current_time = time.time()

    if len(frame_buffer) >= NUM_FRAMES * FRAME_STRIDE and current_time - last_pred_time > PREDICTION_INTERVAL:
        try:
            sampled_frames = list(frame_buffer)[::FRAME_STRIDE][:NUM_FRAMES]

            clip = [
                transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                for f in sampled_frames
            ]
            clip_tensor = torch.stack(clip)
            clip_tensor = clip_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(clip_tensor)
                probs = F.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()

                if pred_class in known_class_ids:
                    current_prediction = label_map[pred_class]
                else:
                    current_prediction = f"Unknown ({pred_class})"

                print(f"🧠 Prediction: {current_prediction}")
                last_pred_time = current_time  # update cooldown

        except Exception as e:
            print(f"⚠️ Error during inference: {e}")

    cv2.putText(
        frame,
        f"Prediction: {current_prediction}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
