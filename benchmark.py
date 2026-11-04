# Compare PyTorch vs ONNX FP32 vs ONNX INT8 on CPU:
#   - file size
#   - speed (milliseconds per image, batch size 1)
#   - test AUC (to check that quantization didn't break the model)
#
# Usage:
#   python benchmark.py
# Needs: best_model.pt, model_fp32.onnx, model_int8.onnx, splits.csv

import json
import os
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from train import eval_tf, make_model

# PyTorch model on CPU
torch_model = make_model()
torch_model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
torch_model.eval()


def run_torch(x):
    with torch.no_grad():
        return torch_model(torch.from_numpy(x)).item()


def onnx_runner(path):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return lambda x: session.run(None, {"image": x})[0].item()


models = {
    "PyTorch FP32": ("best_model.pt", run_torch),
    "ONNX FP32": ("model_fp32.onnx", onnx_runner("model_fp32.onnx")),
    "ONNX INT8": ("model_int8.onnx", onnx_runner("model_int8.onnx")),
}

# Load the test images once (as numpy arrays, shape 1x3x224x224)
df = pd.read_csv("splits.csv")
test_df = df[df["split"] == "test"]
images = [eval_tf(Image.open(p)).unsqueeze(0).numpy() for p in test_df["path"]]
labels = test_df["label"].tolist()
print(f"Loaded {len(images)} test images")

results = []
for name, (path, run) in models.items():
    # Speed: warm up first, then time 100 runs on the same image
    x = images[0]
    for _ in range(10):
        run(x)
    start = time.perf_counter()
    for _ in range(100):
        run(x)
    ms = (time.perf_counter() - start) / 100 * 1000

    # Accuracy: score every test image (logit -> AUC; AUC doesn't need a sigmoid)
    scores = [run(img) for img in images]
    auc = roc_auc_score(labels, scores)

    size_mb = os.path.getsize(path) / 1e6
    results.append({"model": name, "size_mb": round(size_mb, 1), "ms_per_image": round(ms, 1), "test_auc": round(auc, 4)})
    print(f"{name:13s}  size {size_mb:6.1f} MB   {ms:7.1f} ms/image   test AUC {auc:.4f}")

os.makedirs("results", exist_ok=True)
with open("results/benchmark.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved results/benchmark.json")
