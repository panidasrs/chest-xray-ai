# Export best_model.pt to ONNX, then make an INT8 version with static quantization.
#
# Output:
#   model_fp32.onnx  - same model, ONNX format (float32)
#   model_int8.onnx  - weights and activations stored as 8-bit integers (~4x smaller)
#
# Usage:
#   python export_onnx.py
# Needs: best_model.pt, splits.csv (uses 100 validation images for calibration)

import pandas as pd
import torch
from onnxruntime.quantization import (
    CalibrationDataReader, QuantFormat, QuantType, quant_pre_process, quantize_static,
)
from PIL import Image

from train import IMG_SIZE, eval_tf, make_model

# 1) PyTorch -> ONNX
model = make_model()
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)  # one example input so ONNX knows the shape
torch.onnx.export(model, dummy, "model_fp32.onnx",
                  input_names=["image"], output_names=["logit"], opset_version=17, dynamo=False)
print("Saved model_fp32.onnx")


# 2) ONNX FP32 -> INT8
# Static quantization needs to see some real images first ("calibration"),
# so it can learn the range of values inside the network and pick good 8-bit scales.
class XrayCalibration(CalibrationDataReader):
    def __init__(self, paths):
        self.paths = iter(paths)

    def get_next(self):
        path = next(self.paths, None)
        if path is None:
            return None  # tells ONNX Runtime we are done
        x = eval_tf(Image.open(path)).unsqueeze(0).numpy()
        return {"image": x}


df = pd.read_csv("splits.csv")
val_df = df[df["split"] == "val"]
calib_paths = val_df.sample(min(100, len(val_df)), random_state=0)["path"].tolist()

quant_pre_process("model_fp32.onnx", "model_prep.onnx")  # recommended clean-up step before quantizing
quantize_static(
    "model_prep.onnx", "model_int8.onnx",
    calibration_data_reader=XrayCalibration(calib_paths),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
)
print("Saved model_int8.onnx")
