# moderasi-flow (Rust)

Implementasi Rust dari pipeline moderasi gambar `moderasi-satudata-ml` (prototype Python):
**YOLO11-cls/CLIP + Tesseract OCR + keyword blacklist + context gate** → keputusan `LOLOS` / `DIMODERASI`.

Tujuan tahap ini: **paritas perilaku dengan prototype Python** (bukan sekadar "sekali jalan"),
lalu pengukuran performa. Detail kesesuaian dan angka pengukuran ada di
[reports/paritas-dan-performa.md](reports/paritas-dan-performa.md).

## Struktur

| File | Peran |
|---|---|
| `src/main.rs` | CLI, pipeline per gambar, benchmark timings, laporan paritas vs `python_reference.json` |
| `src/yolo.rs` | YOLO11-cls ONNX (output `output0` sudah softmax — jangan di-softmax ulang) |
| `src/clip.rs` | CLIP vision/text ONNX + skor (cosine × 100 → logits → softmax, identik prototype) |
| `src/tokenizer.rs` | Tokenizer GPT2 byte-BPE CLIP (regex kilat + BPE + pad 77), divalidasi 50/50 vs reference |
| `src/keywords.rs` | Daftar blacklist + normalisasi + whole-word match (<3 huruf) — identik `keywords.py` |
| `src/preprocess.rs` | Resize+crop (YOLO: antialiased-bilinear ala torchvision; CLIP: Lanczos3) + normalisasi |
| `src/ocr.rs` | Menjalankan `tesseract` (ind+eng, OEM 1, PSM 3) pada grayscale ≤1600px |
| `src/decision.rs` | Port `moderasi_satu_gambar` (alur yolo & clip + context gate) |
| `src/web.rs` | Server HTTP (axum) — mirror endpoint `server.py` + halaman upload |
| `web/index.html` | UI upload (drag & drop, multi-file, pilih engine) |
| `src/sampler.rs` | Statistik benchmark (avg/stdev/min/max/p50/p95) |

## Model

`models/` berisi artefak runtime (tidak di-commit):

- `best.onnx` — YOLO11-cls (input `images` (1,3,224,224) f32 → `output0` (1,3) **probabilitas**).
- `clip/vision.onnx`, `clip/text.onnx`, `clip/reference.json` (50 prompt → 50 input_ids referensi),
  `clip-tokenizer/*` (kosa kata + merges).

## Build & Jalankan

```sh
cargo build --release
```

```sh
# dari rust-moderasi/
./target/release/moderasi-flow.exe \
  --images ..\testing-image-from-user \
  --models .\models \
  --python-ref ..\rust-moderasi\reference\python_reference.json \
  --warmup 2 --iterations 3
```

Argumen:

| Argumen | Default | Keterangan |
|---|---|---|
| `--images` | (wajib) | Folder/jalur gambar |
| `--models` | (wajib) | Folder model runtime |
| `--python-ref` | otomatis dicari | `reference/python_reference.json` + `yolo_input_XX.bin`/`clip_input_XX.bin` untuk cek paritas |
| `--out` | `results/rust_moderation_results.json` | Output dengan metrik paritas + benchmark |
| `--warmup` | `1` | Iterasi warmup (tidak dihitung) |
| `--iterations` | `1` | Iterasi yang dihitung |
| `--tesseract` | `C:\Program Files\Tesseract-OCR\tesseract.exe` | Binary OCR |
| `--lang` | `ind+eng` | Bahasa OCR |
| `--visual` | `yolo` | Engine default untuk perbandingan |
| `--yolo-filter` / `--clip-filter` | `Triangle` / `Lanczos3` | Hanya untuk eksperimen; YOLO sekarang selalu pakai AA-bilinear |

Output JSON berisi: `meta`, `tokenizer_parity`, `images[]` (keputusan + banding paritas per gambar),
`parity`, `benchmark`, `resources`.

## Web Server (mirror `server.py`)

Mode `server` memulai API HTTP + halaman upload yang sama fungsinya dengan prototype:

```sh
# dari rust-moderasi/ (butuh web/index.html di working directory)
./target/release/moderasi-flow.exe server --models .\models --port 8787
```

Buka `http://127.0.0.1:8787/` (drag & drop satu/banyak gambar, pilih engine, lihat hasil).

Endpoint (skema respons meniru `server.py`; kesalahan = `{"detail": "..."}`):

| Endpoint | Keterangan |
|---|---|
| `GET /` | UI upload |
| `GET /health` | `{status, visual_engine, model, kelas}` |
| `GET /api/engines` | daftar engine + ketersediaan (yolo/clip tersedia; mobilenetv3/smolvlm tidak) |
| `POST /api/engines/default?engine=yolo\|clip` | ganti engine default |
| `GET /api/keywords` | `{total, aborsi, boraks, umum, judi}` (urut) |
| `POST /api/moderasi/satu` | multipart field `file` (+ `?visual=`); balasan `{ringkasan, results:[...]}` |
| `POST /api/moderasi/bulk` | multipart field `files` berulang (max 1000, `?visual=`); diproses paralel |

Flag server: `--host` (127.0.0.1), `--port` (8787), `--models`, `--tesseract`, `--lang` (ind+eng),
`--visual` (yolo), selain mode CLI `--images/--python-ref/--out/--warmup/--iterations` tidak berlaku.

Setiap gambar hasil meniru `moderasi_satu_gambar`: `visual_engine`, `yolo_class`, `yolo_conf`,
`yolo_violative`, `ocr_text`, `keyword_hits`, `context_hits`, `transaction_hits`,
`keyword_context_exempt`, `keputusan`, `alasan`, `elapsed_ms` + `file`/`filename`; gambar tak valid
→ `keputusan: "ERROR"`.

## Status Paritas (hasil warmup 2, iterasi 3)

- Tokenizer: **50/50** prompt → susunan token identik; fitur teks CLIP cos **1.0000000**.
- Keputusan engine YOLO: **36/36** cocok; keputusan engine CLIP: **36/36** cocok.
- Probabilitas YOLO vs reference: cos min **0.9987**; logits CLIP cos min **0.9995**.
- Selisih conf YOLO: rerata **0.0054**, terbesar **0.0299** (1 gambar sensitif dekat ambang 0.70,
  namun keputusan akhir tetap cocok).
- OCR: teks hasil `tesseract` tidak 100% sama untuk 7 dari 12 gambar (beda dekoder JPEG/input OCR
  antar library — di dokumentasikan, tidak mengubah keputusan).

## Catatan Implementasi Penting

1. **`best.onnx` output sudah softmax** — proses awal memakai double-softmax dan menghasilkan
   conf yang terkunci ~0.45–0.58 pada semua gambar; sudah diperbaiki dan diverifikasi
   (nilai sum ≈ 1, selisih vs forward torch ≈ 1e-8).
2. **Resize YOLO** memakai antialiased-bilinear yang diimplementasikan ulang agar cocok dengan
   `torchvision F.interpolate(..., antialias=True)` (cos vs torch = 1.0000002).
3. **Colokan tokenizer**: 50 prompt dipakai buat fitur teks sekali saat startup dan di-cache;
   teks tidak di-encode ulang per gambar.
4. OCR **selalu** dijalankan di Rust agar kedua cabang keputusan (yolo & clip) bisa dibandingkan;
   prototype melewati OCR pada gambar violative-YOLO. Perbandingan `ocr_text` dilakukan
   per-cabang dan hanya gambar yang benar-benar menjalankan OCR dibandingkan teksnya.

### Divergensi yang disengaja (perubahan keputusan)

Fallback **nama file** sekarang hanya menghitung keyword "kuat" (**aborsi / boraks / judi**).
`KEYWORDS_UMUM` (frasa komersial generik: `whatsapp`, `cod`, `wa.me`, dll.) tidak memicu dari
nama file — karena `WhatsApp Image …` adalah penamaan file biasa (screenshot Android), bukan
indikator transaksi. Deteksi melalui **OCR tetap memakai semua daftar** (pola iklan aborsi
terselubung tetap tertangkap). Dampak terhadap parity: 3 gambar `WhatsApp Image …` di dataset
berubah dari `DIMODERASI` menjadi `LOLOS` (prototype mem-flag-nya).

## Keterbatasan / Disparitas yang Diketahui

- Dekoder JPEG: `image` crate (pure-Rust) vs Pillow/OpenCV (libjpeg-turbo) menghasilkan selisih
  piksel kecil → conf di dekat ambang 0.70 bisa beda ~0.03 (1 gambar).
- OCR: input grayscale/resize beda library → beberapa karakter beda; tidak memengaruhi keputusan.
- Drop-in penuh ke protokol server `server.py` sekarang tersedia lewat mode `server` (lihat di atas);
  engine `mobilenetv3`/`smolvlm` di laporkan `available: false` (setara 405 di prototype).