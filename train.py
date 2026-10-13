# Fine-tune a pretrained ResNet18 to classify chest X-rays (normal vs pneumonia).
#
# Usage:
#   python train.py               (default: 8 epochs)
#   python train.py --epochs 1    (quick test)

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMG_SIZE = 224

# X-rays are grayscale, but ResNet expects 3 channels, so we copy the gray channel 3 times.
# We don't flip left/right because the heart is always on the left side.
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class XrayDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i])
        return self.transform(img), torch.tensor(self.labels[i], dtype=torch.float32)


def make_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 1)  # 1 output: score for pneumonia
    return model


def predict(model, loader, device):
    model.eval()
    labels, probs = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.sigmoid(model(x.to(device)).squeeze(1))
            probs += p.cpu().tolist()
            labels += y.tolist()
    return labels, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = pd.read_csv("splits.csv")
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    train_loader = DataLoader(XrayDataset(train_df, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(XrayDataset(val_df, eval_tf), batch_size=args.batch_size, num_workers=2)

    model = make_model().to(device)

    # There are ~3x more pneumonia images than normal ones.
    # pos_weight makes mistakes on the normal class count more, to balance this.
    n_pos = (train_df["label"] == 1).sum()
    n_neg = (train_df["label"] == 0).sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x).squeeze(1), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        train_loss = total_loss / len(train_df)

        val_labels, val_probs = predict(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})
        print(f"Epoch {epoch}: train loss {train_loss:.4f}, val AUC {val_auc:.4f}")

        # keep the model from the epoch with the best validation AUC
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_model.pt")
            print("  saved best_model.pt")

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv("results/history.csv", index=False)
    print(f"Done. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
