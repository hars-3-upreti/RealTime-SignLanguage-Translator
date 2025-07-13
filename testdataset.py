from dataset import SignLanguageDataset
from torch.utils.data import DataLoader

def main():
    dataset = SignLanguageDataset(
        frames_root="processed_data/frames",
        csv_path="data/raw/AUTSL/train.csv",
        num_frames=16
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)  # Set to 0 for now

    for batch in loader:
        video_tensor, labels = batch  # video_tensor: (B, T, C, H, W)
        print("Batch video shape:", video_tensor.shape)
        print("Labels:", labels)
        break

if __name__ == "__main__":
    main()
