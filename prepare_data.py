# Make a list of all images and split them into train / val / test.
#
# Usage:
#   python prepare_data.py                       (downloads the dataset from Kaggle)
#   python prepare_data.py --data-dir chest_xray (if you already have it)

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def get_patient_id(filename):
    # Pneumonia files look like "person1_bacteria_1.jpeg"  -> patient "person1"
    # Normal files look like    "IM-0115-0001.jpeg"         -> patient "IM-0115"
    if filename.startswith("person"):
        return filename.split("_")[0]
    return filename.rsplit("-", 1)[0]


parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=None)
args = parser.parse_args()

if args.data_dir:
    data_dir = Path(args.data_dir)
else:
    import kagglehub
    data_dir = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")) / "chest_xray"

print("Data folder:", data_dir)

rows = []
for folder in ["train", "val", "test"]:
    for label, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
        for img_path in sorted((data_dir / folder / class_name).glob("*.jpeg")):
            rows.append({
                "path": str(img_path),
                "label": label,  # 0 = normal, 1 = pneumonia
                "patient": class_name + "/" + get_patient_id(img_path.name),
                "folder": folder,
            })

df = pd.DataFrame(rows)
if len(df) == 0:
    raise SystemExit("No images found. Check --data-dir (it should contain train/ and test/).")

# The official val folder has only 16 images, so we put it back into train
# and make our own validation set (15% of the patients).
test_df = df[df["folder"] == "test"].copy()
trainval_df = df[df["folder"] != "test"].copy()

# Split by patient so the same child is never in both train and val
splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(splitter.split(trainval_df, groups=trainval_df["patient"]))

trainval_df["split"] = "train"
trainval_df.iloc[val_idx, trainval_df.columns.get_loc("split")] = "val"
test_df["split"] = "test"

out = pd.concat([trainval_df, test_df])
out.to_csv("splits.csv", index=False)

print(out.groupby(["split", "label"]).size().unstack().rename(columns={0: "normal", 1: "pneumonia"}))
print("Saved splits.csv")
