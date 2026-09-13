"""Reproducible export of YOLO best.pt -> ONNX for the Rust pipeline.

Same command ultralytics runs; writes best.onnx into rust-moderasi/models.
"""
from ultralytics import YOLO

MODEL_PT = "moderasi-satudata-ml/models/best.pt"
OUT = "rust-moderasi/models/best.onnx"

m = YOLO(MODEL_PT)
out = m.export(format="onnx", imgsz=224, opset=17, simplify=False, dynamic=False)
print("exported:", out, "-> move/copy to", OUT)
print("names:", m.names)