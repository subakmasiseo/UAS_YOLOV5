import pathlib

import torch

# FIX path Linux -> Windows
pathlib.PosixPath = pathlib.WindowsPath

# IZINKAN model YOLOv5
from models.yolo import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])

# load model (trusted source)
model = torch.load("weights/best.pt", map_location="cpu", weights_only=False)

# save ulang versi Windows-safe
torch.save(model, "weights/best_fixed.pt")

print("✅ Model berhasil di-fix → best_fixed.pt")
