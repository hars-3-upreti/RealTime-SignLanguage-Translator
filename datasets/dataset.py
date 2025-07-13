import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

class SignLanguageDataset(Dataset):
    def __init__(self, frames_root, csv_path, num_frames=16, transform=None):
        self.frames_root = frames_root
        self.df = pd.read_csv(csv_path, header=None)
        self.num_frames = num_frames
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])
        self.samples = self._build_sample_list()

    def _build_sample_list(self):
        samples = []
        for idx in range(len(self.df)):
            video_name = os.path.splitext(self.df.iloc[idx][0])[0]
            label = int(self.df.iloc[idx][1])
            frame_dir = os.path.join(self.frames_root, video_name)

            if os.path.isdir(frame_dir):
                frame_files = [f for f in os.listdir(frame_dir) if f.endswith('.jpg')]
                if len(frame_files) >= 1:  
                    samples.append((frame_dir, label))
        return samples


    def _load_frames(self, frame_dir):
        frame_files = sorted([
            os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.jpg')
        ])
        total = len(frame_files)

        if total == 0:
            raise RuntimeError(f"No frames found in directory: {frame_dir}")

        if total >= self.num_frames:
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        else:
            indices = np.concatenate([
                np.arange(total),
                np.full(self.num_frames - total, total - 1)
            ])

        selected_files = [frame_files[i] for i in indices]
        frames = [self.transform(Image.open(f).convert("RGB")) for f in selected_files]
        return torch.stack(frames)


    def __getitem__(self, index):
        frame_dir, label = self.samples[index]
        frames = self._load_frames(frame_dir)  # Shape: (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)
        return frames, label

    def __len__(self):
        return len(self.samples)
