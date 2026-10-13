# Evaluate best_model.pt on the official test set.
#
# The decision threshold is chosen on the VALIDATION set, then used once on the test set
# (choosing it on the test set would be cheating).
#
# Usage:
#   python evaluate.py

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from train import XrayDataset, eval_tf, make_model, predict

device = "cuda" if torch.cuda.is_available() else "cpu"

model = make_model()
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.to(device)

df = pd.read_csv("splits.csv")
val_loader = DataLoader(XrayDataset(df[df["split"] == "val"], eval_tf), batch_size=32)
test_loader = DataLoader(XrayDataset(df[df["split"] == "test"], eval_tf), batch_size=32)

val_labels, val_probs = predict(model, val_loader, device)
test_labels, test_probs = predict(model, test_loader, device)

# Pick the threshold that gives the best (sensitivity + specificity) on validation
fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
best = np.argmax(tpr - fpr)
threshold = float(thresholds[best])

test_preds = (np.array(test_probs) >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()

results = {
    "threshold": round(threshold, 4),
    "test_auc": round(roc_auc_score(test_labels, test_probs), 4),
    "sensitivity": round(tp / (tp + fn), 4),  # of the sick children, how many we caught
    "specificity": round(tn / (tn + fp), 4),  # of the healthy children, how many we cleared
    "accuracy": round((tp + tn) / len(test_labels), 4),
    "tp": int(tp), "fn": int(fn), "tn": int(tn), "fp": int(fp),
}
print(json.dumps(results, indent=2))

Path("results").mkdir(exist_ok=True)
with open("results/test_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Plots: ROC curve + confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
test_fpr, test_tpr, _ = roc_curve(test_labels, test_probs)
ax[0].plot(test_fpr, test_tpr, label=f"AUC = {results['test_auc']:.3f}")
ax[0].plot([0, 1], [0, 1], "--", color="gray")
ax[0].set_xlabel("False positive rate")
ax[0].set_ylabel("True positive rate")
ax[0].set_title("ROC curve (test set)")
ax[0].legend()

ConfusionMatrixDisplay.from_predictions(
    test_labels, test_preds, display_labels=["Normal", "Pneumonia"], cmap="Blues", ax=ax[1], colorbar=False
)
ax[1].set_title(f"Confusion matrix (threshold {threshold:.2f})")

plt.tight_layout()
plt.savefig("results/test_plots.png", dpi=150)
print("Saved results/test_results.json and results/test_plots.png")
