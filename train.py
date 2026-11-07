# Fine-tune a pretrained CNN to classify chest X-rays (normal vs pneumonia).
#
# Usage:
#   python train.py               (baseline: ResNet18, stretch, 8 epochs -> best_model.pt)
#   python train.py --epochs 1    (quick test)
#   python train.py --arch efficientnet_b0 --resize pad --name effnet_pad
#       (saves models/effnet_pad.pt and results/effnet_pad_history.csv)

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

NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def pad_to_square(img):
    # Add black borders so the image becomes square without squashing it.
    # (EDA showed normal and pneumonia images have different shapes, so stretching
    # them to a square distorts the two classes differently.)
    side = max(img.size)
    square = Image.new(img.mode, (side, side))
    square.paste(img, ((side - img.size[0]) // 2, (side - img.size[1]) // 2))
    return square


def get_transforms(resize="stretch"):
    # X-rays are grayscale, but the CNNs expect 3 channels, so we copy the gray channel 3 times.
    # We don't flip left/right because the heart is always on the left side.
    first = [transforms.Grayscale(num_output_channels=3)]
    if resize == "pad":
        first.append(transforms.Lambda(pad_to_square))
    first.append(transforms.Resize((IMG_SIZE, IMG_SIZE)))

    train_tf = transforms.Compose(first + [
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        NORMALIZE,
    ])
    eval_tf = transforms.Compose(first + [transforms.ToTensor(), NORMALIZE])
    return train_tf, eval_tf


# default transforms (used by evaluate.py, gradcam.py, export_onnx.py, benchmark.py)
train_tf, eval_tf = get_transforms("stretch")


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


def make_model(arch="resnet18"):
    # Replace the last layer so the model gives 1 output: a score for pneumonia
    if arch == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    else:
        raise ValueError(f"unknown arch: {arch}")
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
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--resize", default="stretch", choices=["stretch", "pad"])
    parser.add_argument("--name", default=None, help="run name; if not set, saves best_model.pt like before")
    args = parser.parse_args()

    if args.name:
        Path("models").mkdir(exist_ok=True)
        model_path = f"models/{args.name}.pt"
        history_path = f"results/{args.name}_history.csv"
    else:
        model_path = "best_model.pt"
        history_path = "results/history.csv"
    print(f"arch={args.arch}, resize={args.resize}, saving to {model_path}")
    train_tf, eval_tf = get_transforms(args.resize)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = pd.read_csv("splits.csv")
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    train_loader = DataLoader(XrayDataset(train_df, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(XrayDataset(val_df, eval_tf), batch_size=args.batch_size, num_workers=2)

    model = make_model(args.arch).to(device)

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
            torch.save(model.state_dict(), model_path)
            print(f"  saved {model_path}")

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Done. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
