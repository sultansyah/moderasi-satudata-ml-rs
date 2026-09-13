"""Export openai/clip-vit-base-patch32 (HF transformers, same as prototype) to ONNX.

Produces, under rust-moderasi/models/clip/:
  - vision.onnx  : pixel_values (1,3,224,224)            -> image_features (1,512)
  - text.onnx    : input_ids (1,77) + attention_mask     -> text_features (1,512)
  - config.json  : metadata (model id, logit_scale, prompts)
  - tokenizer artifacts (vocab.json, merges.txt, special tokens) under models/clip-tokenizer/
  - reference.json : per-prompt token ids + normalized text features (used to unit-test
                     the Rust tokenizer/encoder against Python/transformers).
Downloaded HF files are cached in rust-moderasi/models/clip-hf/ so offline runs are stable.
"""
import json
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

BLOCK = "openai/clip-vit-base-patch32"
BASE = Path("rust-moderasi/models")
HF_DIR = BASE / "clip-hf"
OUT = BASE / "clip"
TOK = BASE / "clip-tokenizer"

sys.path.insert(0, ".")  # allow importing moderasi logic for the prompt lists

# Import the exact prompt lists from the prototype (do not edit the prototype).
import importlib.util
sys.path.insert(0, "moderasi-satudata-ml")
spec = importlib.util.spec_from_file_location("moderasi_ref", "moderasi-satudata-ml/moderasi.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
VIO = [t for _, t in mod.CLIP_VIOLATIVE_CONCEPTS]
SAFE = [t for _, t in mod.CLIP_SAFE_CONCEPTS]
TEXTS = VIO + SAFE
VIO_LABELS = [l for l, _ in mod.CLIP_VIOLATIVE_CONCEPTS]
SAFE_LABELS = [l for l, _ in mod.CLIP_SAFE_CONCEPTS]

OUT.mkdir(parents=True, exist_ok=True)
TOK.mkdir(parents=True, exist_ok=True)

from transformers import CLIPModel, CLIPProcessor

print("loading CLIP (this downloads ~600MB on first run)...", flush=True)
model = CLIPModel.from_pretrained(BLOCK)
proc = CLIPProcessor.from_pretrained(BLOCK)
print("vision config:", {k: model.config.vision_config.to_dict().get(k) for k in
                          ("image_size", "hidden_size", "num_hidden_layers", "num_attention_heads")})
print("text config  :", {k: model.config.text_config.to_dict().get(k) for k in
                          ("max_position_embeddings", "vocab_size")})
print("projection dims:", model.visual_projection.in_features, "->", model.visual_projection.out_features,
      model.text_projection.in_features, "->", model.text_projection.out_features)
logit_scale = float(model.logit_scale.exp().item())
print("logit_scale (exp):", logit_scale)

model.eval()

# Save HF files to disk for offline Rust-era reproducibility / reference.
model.save_pretrained(HF_DIR)
proc.save_pretrained(HF_DIR)

# --- tokenizer artifacts for Rust ----------
tok = proc.tokenizer
vocab = tok.get_vocab()
TOK.joinpath("vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
merges = tok._merges
TOK.joinpath("merges.txt").write_text("\n".join(" ".join(m) for m in merges) + "\n", encoding="utf-8")
bytes_to_unicode = tok.byte_decoder if hasattr(tok, "byte_decoder") else {}
TOK.joinpath("byte_decoder.json").write_text(json.dumps({str(k): v for k, v in bytes_to_unicode.items()}), encoding="utf-8")
tokenizer_info = {
    "bos_token": tok.bos_token, "eos_token": tok.eos_token, "pad_token": tok.pad_token,
    "bos_id": tok.bos_token_id, "eos_id": tok.eos_token_id, "pad_id": tok.pad_token_id,
    "model_max_length": tok.model_max_length,
}
TOK.joinpath("config.json").write_text(json.dumps(tokenizer_info, indent=2), encoding="utf-8")

# --- ONNX exports ---------------------------
class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_model = model.vision_model
        self.proj = model.visual_projection

    def forward(self, pixel_values):
        x = self.vision_model(pixel_values)[1]
        x = self.proj(x)
        return F.normalize(x, dim=-1)

class TextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_model = model.text_model
        self.proj = model.text_projection

    def forward(self, input_ids, attention_mask):
        x = self.text_model(input_ids=input_ids, attention_mask=attention_mask)[1]
        x = self.proj(x)
        return F.normalize(x, dim=-1)

N = 224
# vision export (fixed batch 1)
image_enc = ImageEncoder(model)
pv = torch.randn(1, 3, N, N)
torch.onnx.export(
    image_enc, (pv,), str(OUT / "vision.onnx"),
    input_names=["pixel_values"], output_names=["image_features"],
    opset_version=17, do_constant_folding=True, dynamo=False,
)
print("vision.onnx written", flush=True)

# text export (fixed 1x77 plus mask)
text_enc = TextEncoder(model)
L = 77
ids = torch.ones(1, L, dtype=torch.long) * tok.pad_token_id
ids[0] = torch.tensor([tok.bos_token_id] + [tok.pad_token_id] * (L - 1))
mask = torch.zeros(1, L, dtype=torch.long)
mask[0, 0] = 1
torch.onnx.export(
    text_enc, (ids, mask), str(OUT / "text.onnx"),
    input_names=["input_ids", "attention_mask"], output_names=["text_features"],
    opset_version=17, do_constant_folding=True, dynamo=False,
    dynamic_axes=None,
)
print("text.onnx written", flush=True)

# --- reference data for parity checks -------
inputs = proc(text=TEXTS, return_tensors="pt", padding=True)
ids_ref = inputs["input_ids"]
mask_ref = inputs["attention_mask"]
print("padded text length used by processor:", ids_ref.shape[1])
with torch.no_grad():
    pooled = model.text_model(input_ids=ids_ref, attention_mask=mask_ref)[1]
    feat_ref = model.text_projection(pooled)
feat_ref = F.normalize(feat_ref, dim=-1)
reference = {
    "texts": TEXTS,
    "padded_length": ids_ref.shape[1],
    "input_ids": ids_ref.tolist(),
    "attention_mask": mask_ref.tolist(),
    "text_features": feat_ref.tolist(),
    "logit_scale": logit_scale,
    "vio_labels": VIO_LABELS,
    "safe_label": "aman",
    "vio_threshold": 0.30,
    "safe_margin": 2.0,
    "clip_mean": [0.48145466, 0.4578275, 0.40821073],
    "clip_std": [0.26862954, 0.26130258, 0.27577711],
    "n_vio": len(VIO),
}
(OUT / "reference.json").write_text(json.dumps(reference, indent=1), encoding="utf-8")
(OUT / "config.json").write_text(json.dumps({
    "model_id": BLOCK, "image_size": N, "text_length": 77, "logit_scale": logit_scale,
    "n_violative_concepts": len(VIO), "n_safe_concepts": len(SAFE),
}, indent=2), encoding="utf-8")
print("config/reference written")
print("DONE. tokenizer special ids:", tokenizer_info)
