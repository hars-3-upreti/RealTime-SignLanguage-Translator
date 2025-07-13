import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.dataset import SignLanguageDataset
from models.x3d_model import build_x3d_xs
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    # Hyperparameters 
    NUM_CLASSES = 228
    BATCH_SIZE = 4
    EPOCHS = 35
    LR = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 2
    MAX_GRAD_NORM = 1.0

    #  Paths 
    TRAIN_DIR = 'processed_data/train'
    VAL_DIR   = 'processed_data/val'
    TRAIN_CSV = 'data/raw/AUTSL/train.csv'
    VAL_CSV   = 'data/raw/AUTSL/val.csv'
    CHECKPOINT_DIR = 'checkpoints'
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'best_x3d_xs.pth')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Data Transforms
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    ])

    # Datasets & Loaders 
    train_dataset = SignLanguageDataset(TRAIN_DIR, TRAIN_CSV, num_frames=16, transform=train_transforms)
    val_dataset   = SignLanguageDataset(VAL_DIR, VAL_CSV, num_frames=16, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    #  Model, Optimizer, Loss, Scheduler 
    model = build_x3d_xs(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)  # ⬅️ weight decay added
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)

   
    start_epoch = 0
    best_val_acc = 0.0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔁 Loading checkpoint from {CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
        print("✅ Checkpoint loaded. Resuming training...")
        start_epoch = 24  # ⬅️ Update to your latest completed epoch

    #  Training 
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # --- Training ---
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for videos, labels in tqdm(train_loader, desc='Training'):
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                outputs = model(videos)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_loss /= len(train_loader)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for videos, labels in tqdm(val_loader, desc='Validation'):
                videos, labels = videos.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        
        scheduler.step(val_acc)

        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"✅ Saved new best model at epoch {epoch+1} — Val Acc: {val_acc:.2f}%")

if __name__ == '__main__':
    main()
