import cv2
import os
from tqdm import tqdm
import pandas as pd

def extract_frames(video_path, output_dir, frame_rate=5):
    """
    Extracts frames from a video at the given frame rate and saves them.
    """
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_folder = os.path.join(output_dir, video_name)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    count = 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_rate == 0:
            frame_path = os.path.join(out_folder, f"{frame_id:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_id += 1
        count += 1

    cap.release()


def extract_all_frames(video_dir, output_dir, csv_path):
    df = pd.read_csv(csv_path, header=None)
    os.makedirs(output_dir, exist_ok=True)

    for idx in tqdm(range(len(df))):
        video_file = df.iloc[idx][0]
        video_path = os.path.join(video_dir, video_file)
        extract_frames(video_path, output_dir)

if __name__ == "__main__":
    extract_all_frames(
        video_dir="data/raw/AUTSL/test",
        output_dir="processed_data/test",
        csv_path="data/raw/AUTSL/test.csv"
    )