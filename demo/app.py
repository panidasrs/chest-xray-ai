# Gradio demo: upload a chest X-ray -> pneumonia probability.
# Runs the INT8 ONNX model on CPU (no PyTorch needed).

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

# On Hugging Face the Space runs on ZeroGPU, which needs a function marked with @spaces.GPU.
# Locally / in Docker the "spaces" package isn't installed, so we use a decorator that does nothing.
try:
    import spaces
    gpu = spaces.GPU(duration=10)
except ImportError:
    def gpu(fn):
        return fn

THRESHOLD = 0.59  # chosen on the validation set (see the GitHub repo)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

session = ort.InferenceSession("model_int8.onnx", providers=["CPUExecutionProvider"])


def preprocess(img):
    # Same steps as eval_tf in train.py: grayscale -> 3 channels, resize to 224, normalize
    img = img.convert("L").convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0  # (224, 224, 3), values 0-1
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[np.newaxis]        # (1, 3, 224, 224)


# The model always runs on CPU (ONNX Runtime CPUExecutionProvider)
@gpu
def classify(img):
    if img is None:
        return None, ""
    logit = session.run(None, {"image": preprocess(img)})[0].item()
    prob = 1 / (1 + np.exp(-logit))  # sigmoid
    verdict = "Pneumonia" if prob >= THRESHOLD else "Normal"
    text = f"**Model output: {verdict}** (p = {prob:.2f}, threshold {THRESHOLD})"
    return {"Pneumonia": float(prob), "Normal": float(1 - prob)}, text


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Chest X-ray"),
    outputs=[gr.Label(label="Probability"), gr.Markdown()],
    title="Chest X-ray Pneumonia Classifier",
    description=(
        "ResNet18 fine-tuned on pediatric chest X-rays (Kermany et al., 2018), "
        "quantized to INT8 with ONNX Runtime and running on CPU.\n\n"
        "**Student project for learning only. Not for medical use.** "
        "The model was trained on X-rays of children aged 1–5 from one hospital and may be wrong on other images.\n\n"
        "Code and results: [github.com/panidasrs/chest-xray-ai](https://github.com/panidasrs/chest-xray-ai)"
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
