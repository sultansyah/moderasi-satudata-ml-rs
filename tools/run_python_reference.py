"""Run the existing Python prototype (moderasi-satudata-ml) over testing-image-from-user/.

This program drives the prototype's own functions (moderasi.moderasi_satu_gambar,
moderasi._clip_classify equivalents, moderasi.ocr_text) WITHOUT modifying the prototype.
It captures, per image, the exact inputs (YOLO + CLIP preprocessed tensors), outputs
(logits/probs), the full moderation decision (YOLO-engine and CLIP-engine), OCR text,
keyword/context hits, and per-stage timings, then stores everything in:
  rust-moderasi/reference/python_reference.json  (+ .bin reference inputs)

Run:
  python rust-moderasi/tools/run_python_reference.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent  # rust-moderasi
PROJ = BASE.parent  # bench-clip
sys.path.insert(0, str(PROJ / "moderasi-satudata-ml"))
sys.path.insert(0, str(PROJ))

import moderasi as MOD  # noqa: E402

IMG_DIR = PROJ / "testing-image-from-user"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
VIO = [t for _, t in MOD.CLIP_VIOLATIVE_CONCEPTS]
VIO_LABELS = [l for l, _ in MOD.CLIP_VIOLATIVE_CONCEPTS]
SAFE = [t for _, t in MOD.CLIP_SAFE_CONCEPTS]
TEXTS = VIO + SAFE
N_VIO = len(VIO)

OUT_DIR = BASE / "reference"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    return torch, model, proc


def clip_call(torch, model, proc, image_path, warm=False):
    """Mirror of moderasi._clip_classify with detailed capture + timings."""
    t_i = time.perf_counter()
    image = MOD.Image.open(image_path).convert("RGB")
    t_pre = time.perf_counter()
    inputs = proc(text=TEXTS, images=image, return_tensors="pt", padding=True)
    t_post = time.perf_counter()
    pixel_values = inputs["pixel_values"].numpy()
    with torch.no_grad():
        logits = model(**inputs).logits_per_image[0].numpy()
    t_fwd = time.perf_counter()
    probs = np.exp(logits - logits.max()) if warm else None
    probs = _softmax(logits)
    vio_scores = probs[:N_VIO]
    safe_scores = probs[N_VIO:]
    vio_sum = float(np.sum(vio_scores))
    safe_sum = float(np.sum(safe_scores))
    floor = 0.30
    margin = 2.0
    best_vio = float(np.max(vio_scores))
    best_safe = float(np.max(safe_scores))
    violative = vio_sum >= floor and vio_sum >= safe_sum * margin
    cls_name = VIO_LABELS[int(np.argmax(vio_scores))] if violative else "aman"
    conf = max(best_vio, best_safe)
    return {
        "logits": logits.tolist(),
        "probs": probs.tolist(),
        "n_vio": N_VIO,
        "vio_sum": vio_sum,
        "safe_sum": safe_sum,
        "best_vio": best_vio,
        "best_safe": best_safe,
        "floor": floor,
        "margin": margin,
        "violative": violative,
        "cls_name": cls_name,
        "conf": round(float(conf), 4),
        "pixel_values": pixel_values.astype("<f4"),
        "t_imgload_ms": (t_pre - t_i) * 1e3,
        "t_preprocess_ms": (t_post - t_pre) * 1e3,
        "t_forward_ms": (t_fwd - t_post) * 1e3,
    }


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def capture_yolo(model, image_path):
    """Replicate ultralytics ClassificationPredictor.preprocess exactly + capture probs."""
    from ultralytics.data.augment import classify_transforms
    import cv2
    from PIL import Image
    im_bgr = cv2.imread(str(image_path))
    tr = classify_transforms(224)
    t0 = time.perf_counter()
    preimg = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    t1 = time.perf_counter()
    tensor = tr(Image.fromarray(preimg)).unsqueeze(0).numpy().astype("<f4")
    t2 = time.perf_counter()
    res = model.predict(image_path, verbose=False)[0]
    t3 = time.perf_counter()
    return {
        "top1": int(res.probs.top1),
        "top1_class": res.names[res.probs.top1],
        "conf": float(res.probs.top1conf),
        "probs": [float(x) for x in res.probs.data.flatten()],
        "names": res.names,
        "input": tensor,
        "t_decode_rgb_ms": (t1 - t0) * 1e3,
        "t_preprocess_ms": (t2 - t1) * 1e3,
        "t_yolo_ms": (t3 - t2) * 1e3,
    }


def main() -> int:
    from ultralytics import YOLO
    torch, clip_model, clip_proc = load_clip()

    model = MOD.load_model()
    MOD.setup_tesseract()

    images = sorted(p for p in IMG_DIR.iterdir() if p.is_file() and p.suffix.lower() in
                    {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"})
    if not images:
        print("no images found in", IMG_DIR)
        return 1
    print(f"running prototype reference on {len(images)} images (this includes OCR+CLIP, may take a while)...")

    entries = []
    for k, path in enumerate(images):
        print(f"[{k + 1}/{len(images)}] {path.name}", flush=True)
        timings = {}

        t0 = time.perf_counter()
        yolo_meta = capture_yolo(model, path)
        timings.update({f"yolo_{kk}": v for kk, v in yolo_meta.items() if kk in ("t_decode_rgb_ms", "t_preprocess_ms", "t_yolo_ms")})

        t1 = time.perf_counter()
        dec_yolo = MOD.moderasi_satu_gambar(model, str(path), lang="ind+eng")
        timings["total_yolo_engine_ms"] = (time.perf_counter() - t1) * 1e3

        clip_meta = clip_call(torch, clip_model, clip_proc, path)
        timings.update({f"clip_{kk}": v for kk, v in clip_meta.items() if kk in ("t_imgload_ms", "t_preprocess_ms", "t_forward_ms")})

        t2 = time.perf_counter()
        dec_clip = MOD.moderasi_satu_gambar(model, str(path), lang="ind+eng", visual="clip")
        timings["total_clip_engine_ms"] = (time.perf_counter() - t2) * 1e3

        (OUT_DIR / f"yolo_input_{k:02d}.bin").write_bytes(yolo_meta["input"].tobytes())
        (OUT_DIR / f"clip_input_{k:02d}.bin").write_bytes(clip_meta["pixel_values"].tobytes())

        entries.append({
            "index": k,
            "filename": path.name,
            "yolo": {kk: v for kk, v in yolo_meta.items() if kk not in ("input",)},
            "clip": {kk: v for kk, v in clip_meta.items() if kk not in ("pixel_values",)},
            "decision_yolo": {
                "keputusan": dec_yolo["keputusan"],
                "visual_engine": dec_yolo["visual_engine"],
                "yolo_class": dec_yolo["yolo_class"],
                "yolo_conf": dec_yolo["yolo_conf"],
                "yolo_violative": dec_yolo["yolo_violative"],
                "ocr_text": dec_yolo["ocr_text"],
                "keyword_hits": dec_yolo["keyword_hits"],
                "context_hits": dec_yolo["context_hits"],
                "transaction_hits": dec_yolo["transaction_hits"],
                "keyword_context_exempt": dec_yolo["keyword_context_exempt"],
                "alasan": dec_yolo["alasan"],
            },
            "decision_clip": {
                "keputusan": dec_clip["keputusan"],
                "visual_engine": dec_clip["visual_engine"],
                "yolo_class": dec_clip["yolo_class"],
                "yolo_conf": dec_clip["yolo_conf"],
                "yolo_violative": dec_clip["yolo_violative"],
                "ocr_text": dec_clip["ocr_text"],
                "keyword_hits": dec_clip["keyword_hits"],
                "context_hits": dec_clip["context_hits"],
                "transaction_hits": dec_clip["transaction_hits"],
                "keyword_context_exempt": dec_clip["keyword_context_exempt"],
                "alasan": dec_clip["alasan"],
            },
            "timings": timings,
        })

    result = {
        "image_dir": str(IMG_DIR),
        "count": len(images),
        "yolo_classes": {0: "dokumen", 1: "normal", 2: "obat_aborsi"},
        "yolo_violative_classes": MOD.KELAS_VIOLATIVE,
        "viol_conf_threshold": MOD.VIOL_CONF_THRESHOLD,
        "clip_violative_threshold": 0.30,
        "clip_safe_margin": 2.0,
        "clip_violative_concepts": MOD.CLIP_VIOLATIVE_CONCEPTS,
        "clip_safe_concepts": MOD.CLIP_SAFE_CONCEPTS,
        "images": entries,
    }
    with open(BASE / "reference" / "python_reference.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("\nwrote", BASE / "reference" / "python_reference.json")
    print("\n=== decisions (python prototype) ===")
    for e in entries:
        d = e["decision_yolo"]
        print(f"YOLO-engine {d['keputusan']:<10} {e['filename']:<45} cls={e['yolo']['top1_class']} conf={e['yolo']['conf']:.3f} ocr_hits={len(d['keyword_hits'])}")
    for e in entries:
        d = e["decision_clip"]
        print(f"CLIP-engine {d['keputusan']:<10} {e['filename']:<45} cls={e['clip']['cls_name']} conf={e['clip']['conf']:.3f} viol={e['clip']['violative']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())