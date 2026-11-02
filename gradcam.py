# Grad-CAM: show which part of the X-ray the model looked at.
#
# Picks a few test images from each group (correct pneumonia, correct normal,
# false positive, false negative) and draws a heatmap on top of each one.
#
# Usage:
#   python gradcam.py
# Needs: best_model.pt, splits.csv, results/test_results.json (for the threshold)

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from train import IMG_SIZE, XrayDataset, eval_tf, make_model, predict

device = "cuda" if torch.cuda.is_available() else "cpu"

model = make_model()
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.to(device)
model.eval()


def gradcam(img_path):
    x = eval_tf(Image.open(img_path)).unsqueeze(0).to(device)

    # Save the output of the last conv block (layer4) when the image goes through the model
    saved = {}
    hook = model.layer4.register_forward_hook(lambda module, inp, out: saved.update(features=out))
    score = model(x)[0, 0]
    hook.remove()

    # How much does each feature map change the pneumonia score?
    grads = torch.autograd.grad(score, saved["features"])[0]   # shape (1, 512, 7, 7)
    weights = grads.mean(dim=(2, 3), keepdim=True)             # one weight per feature map

    # Weighted sum of the feature maps -> 7x7 heatmap, keep only positive evidence
    cam = F.relu((weights * saved["features"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear")[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), torch.sigmoid(score).item()


# 1) Predict the whole test set so we know which images were right or wrong
df = pd.read_csv("splits.csv")
test_df = df[df["split"] == "test"].reset_index(drop=True)
loader = DataLoader(XrayDataset(test_df, eval_tf), batch_size=32)
_, probs = predict(model, loader, device)

threshold = json.load(open("results/test_results.json"))["threshold"]
test_df["pred"] = (np.array(probs) >= threshold).astype(int)

groups = {
    "Pneumonia, correct (TP)": test_df[(test_df["label"] == 1) & (test_df["pred"] == 1)],
    "Normal, correct (TN)": test_df[(test_df["label"] == 0) & (test_df["pred"] == 0)],
    "Normal, called pneumonia (FP)": test_df[(test_df["label"] == 0) & (test_df["pred"] == 1)],
    "Pneumonia, missed (FN)": test_df[(test_df["label"] == 1) & (test_df["pred"] == 0)],
}

# 2) Draw 3 examples per group: X-ray on the left, X-ray + heatmap on the right
n = 3
fig, axes = plt.subplots(len(groups), n * 2, figsize=(n * 5, len(groups) * 2.8))
for row, (name, group) in enumerate(groups.items()):
    examples = group.sample(min(n, len(group)), random_state=0)
    for col in range(n):
        ax_img, ax_cam = axes[row, col * 2], axes[row, col * 2 + 1]
        ax_img.axis("off")
        ax_cam.axis("off")
        if col >= len(examples):
            continue
        path = examples.iloc[col]["path"]
        cam, prob = gradcam(path)
        img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
        ax_img.imshow(img, cmap="gray")
        ax_cam.imshow(img, cmap="gray")
        ax_cam.imshow(cam, cmap="jet", alpha=0.4)
        ax_cam.set_title(f"p = {prob:.2f}", fontsize=9)
    axes[row, 0].set_title(name, fontsize=10, loc="left")

plt.tight_layout()
plt.savefig("results/gradcam_examples.png", dpi=120)
print("Saved results/gradcam_examples.png")
