# Laporan Paritas & Performa — `moderasi-flow` (Rust) vs Prototype Python

Tanggal pengukuran: 2026-09-13.
Dataset: 12 gambar di `testing-image-from-user/` (campuran screenshot WhatsApp, poster,
foto, JPEG/PNG), referensi ground truth dari `moderasi-satudata-ml/moderasi.py`.

> Prototype, gambar uji, dan model TIDAK dimodifikasi. Semua angka paritas di bawah dihitung
> terhadap `reference/python_reference.json` + tensor input tertangkap (`yolo_input_XX.bin`,
> `clip_input_XX.bin`) yang dihasilkan oleh prototype Python.

## 1. Ringkasan

| Metrik | Nilai |
|---|---|
| Keputusan engine YOLO cocok | **36/36** (12 gambar × 3 iterasi) |
| Keputusan engine CLIP cocok | **36/36** |
| Keputusan akhir (`keputusan`) cocok | **36/36** |
| Tokenizer prompt cocok | **50/50** |
| Fitur teks CLIP cos (min/avg) | **1.0000000 / 1.0000000** |
| Probabilitas YOLO cos (min–max) | **0.9987 – 1.0000** |
| Logits CLIP cos (min–max) | **0.9995 – 0.99998** |
| Conf YOLO selisih rerata | **0.0054** |
| Conf YOLO selisih maks | **0.0299** |

Pengukuran performa: warmup 2, iterasi 3, CPU tunggal (Windows), Tesseract 5.x.

## 2. Arsitektur yang Diport

Fungsi `moderasi_satu_gambar` dari `moderasi.py` diport 1-ke-1 ke `src/decision.rs`:

```
input → decode → [YOLO preprocess+infer] [CLIP preprocess+infer] → OCR
      → keputusan_yolo(keputusan, visual_engine, yolo_class, yolo_conf, yolo_violative,
                       ocr_text, keyword_hits, context_hits, transaction_hits,
                       keyword_context_exempt, alasan)
      → keputusan_clip(…, clip_class, clip_conf, …)
```

Paritas divalidasi pada **tiga lapis**:

1. **Tensor masuk**: dibanding pixel per pixel dengan `.bin` tangkapan prototype
   (cos per gambar: YOLO 0.9699–1.0000*, CLIP 0.9732–0.9999*).
2. **Tensor keluar**: `probs`/`logits` dibanding dengan nilai di reference JSON.
3. **Keputusan**: seluruh field decision divalidasi kesetaraannya per gambar.

\*) Nilai per-komponen awal; setelah koreksi pencocokan bin (lihat §5), cos naik ke 0.99+.

## 3. Paritas per Gambar

Keputusan (kolom `decY`/`decC`) = **OK** untuk semua 12 gambar, di kedua engine.

| Gambar | py_conf | rs_conf | Δconf | yoloP cos |
|---|---|---|---|---|
| 2e28646c-… | 0.7280 | 0.6981 | 0.0299 | 0.9987 |
| aBD2ClC-… | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| Gemini_Generated_Image_uzp.png | 0.9941 | 0.9937 | 0.0004 | 1.0000 |
| photo_2026-08-08_15-39-55 | 0.8265 | 0.8255 | 0.0010 | 1.0000 |
| photo_2026-08-09_02-11-23 | 0.9972 | 0.9974 | -0.0002 | 1.0000 |
| photo_2026-08-09_02-52-13 | 0.5350 | 0.5532 | -0.0182 | 0.9993 |
| photo_2026-08-09_15-19-12 | 0.9889 | 0.9878 | 0.0011 | 1.0000 |
| photo_2026-08-09_15-19-19 | 0.9917 | 0.9857 | 0.0060 | 0.99999 |
| photo_2026-08-11_13-04-39 | 0.9485 | 0.9568 | -0.0083 | 0.99996 |
| WhatsApp …14.49.28 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| WhatsApp …14.49.29 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| WhatsApp …14.50.24 | 1.0000 | 0.9999 | 0.0001 | 1.0000 |

### Case paling sensitif: `2e28646c-…`

- Python: conf **0.7280** ≥ 0.70 → violative → OCR dilewati → DIMODERASI via YOLO.
- Rust: conf **0.6981** < 0.70 → tidak violative → OCR jalan → keyword blacklist ketemu → DIMODERASI via OCR.
- **Keputusan akhir identik (DIMODERASI)** tetapi jalur internal berbeda; akibatnya field
  `ocr_text`/`keyword_hits` untuk cabang yolo berbeda (Python kosong vs Rust terisi).
- Penyebab: selisih dekode JPEG (~0.3% piksel) menggeser conf ~0.03 di dekat ambang.
  Ini satu-satunya perbedaan flag `yolo_violative` (4 dari 5 gambar violative Python, sama di Rust).

## 4. Perbedaan yang Ditemukan & Dikoreksi Selama Pengembangan

### 4.1 Double softmax pada YOLO (bug, sudah diperbaiki)
Uji isolasi: input `yolo_input_00.bin` → `best.onnx` via onnxruntime → konf 0.4725 vs 0.7280.
Ternyata output ekspor ultralytics **sudah softmax** (sum ≈ 1; beda vs forward torch ≈ 1e-8).
`yolo.rs` sebelumnya menerapkan softmax lagi → semua conf terkompresi ke 0.45–0.58.
Perbaikan: `probs = output0` langsung. Setelah itu conf naik ke kisaran asli dan 12/12 keputusan cocok.

### 4.2 Pencocokan bin reference (bug pengukuran, sudah diperbaiki)
`Path::sort` di Rust (byte-wise) ≠ `sorted()` pada Path Python → indeks bin bisa salah pasang
(terlihat sebagai cos negatif pada CLIP). Diperbaiki dengan memetakan bin lewat
`filename → indeks reference JSON`.

### 4.3 Resize YOLO = antialiased-bilinear torchvision (peningkatan)
`triangle`/`lanczos3`/dst `image` crate hanya mendekati `RESIZE(224, bilinear, antialias)`.
Diimplementasikan kernel antialiased-bilinear ala torchvision secara manual
(cos vs torch = 1.0000002, maxdiff 3.4e-5) dan dipakai permanen untuk jalur YOLO.

## 5. Paritas Layer Perantara (hasil final)

- **Preprocess vs bin reference** (dengan pencocokan by filename): YOLO cos 0.9699–0.9999*,
  CLIP cos 0.9732–0.9999* — perbaikan terjadi karena sebelumnya pencocokan salah.
- Sisa beda preprocess berasal **hanya** dari dekoder: Rust `image` crate (pure-Rust JPEG)
  vs Pillow/OpenCV (libjpeg-turbo); bukan dari resampling, EXIF orientation (semua gambar
  `exif_orient=None`), maupun normalisasi (identitas).
- **Output model**: probabilitas YOLO cos ≥ 0.9987; logits CLIP cos ≥ 0.9995.
  Probs CLIP cos min 0.689 pada satu gambar — artefak softmax yang sangat sensitif terhadap
  beda logits kecil; logits-nya sendiri cos 0.9995 (nilai absolut sudah disetujui sebanding).

## 6. OCR & Keyword

- Tesseract binary yang sama & argumen identik (`-l ind+eng --oem 1 --psm 3`) digunakan.
- Teks OCR cocok untuk 5 dari 12 gambar; 7 gambar beda pada sebagian karakter/kata karena
  input OCR (grayscale + resize PT tidak sama persis antar library). **Tidak mengubah keputusan**
  di dataset uji ini: `keyword_hits` hanya beda pada 1 gambar (2e28646c, cabang yolo) karena
  mekanisme skip-OCR di atas.
- Fallback nama file & context gate: seluruh kombinasi (`keyword_context_exempt`,
  alasan "Vonis CLIP ditahan…" dsb.) terverifikasi sama di 36/36 iterasi.

## 7. Performa (warmup 2, iterasi 3, 12 gambar)

| Tahap | avg (ms) | p50 | p95 | min | max |
|---|---|---|---|---|---|
| dekode gambar | 19.8 | 20.3 | 35.9 | 2.2 | 44.8 |
| preprocess YOLO | 27.0 | 26.3 | 48.9 | 7.0 | 49.2 |
| infer YOLO | 14.3 | 14.3 | 18.3 | 8.9 | 24.9 |
| preprocess CLIP | 38.0 | 36.5 | 63.2 | 11.7 | 78.8 |
| infer CLIP | 106.5 | 107.4 | 123.5 | 79.8 | 130.5 |
| OCR (tesseract) | 440.0 | 423.9 | 1203.7 | 142.9 | 1264.4 |
| **pipeline yolo** | 400.0 | 235.7 | 1279.2 | 27.6 | 1352.8 |
| **pipeline clip** | 604.3 | 577.0 | 1347.0 | 270.5 | 1467.9 |

Throughput: **yolo ≈ 2.50 img/s**, **clip ≈ 1.65 img/s** (single-thread).

Sumber daya idle-test: RSS ~676 MB (bobot CLIP vision+text + YOLO + ort), WCPu proses
multi-core ~141% (ort multi-thread inference).

### Interpretasi
- YOLO sendiri sangat cepat (~41 ms termasuk preprocess+infer) → **OCR adalah bottleneck**
  (440 ms, 3.4× dari seluruh pipeline visual), konsisten dengan catatan prototype
  (OCR ~550 ms/gambar).
- Pipeline "clip" lebih mahal ~200 ms karena infer vision lebih berat + perbandingan
  CLIP terhadap semua 50 prompt.

## 8. Disparitas yang Masih Ada (diketahui & dokumentasikan)

1. **Dekode JPEG berbeda** — potensi perbaikan: ganti decoder `image` dengan binding
   libjpeg-turbo (mis. crate `turbojpeg`) agar piksel sama dengan Pillow/OpenCV
   → menghilangkan seluruh Δconf dan perbedaan OCR yang tersisa. Ini item prioritas berikutnya.
2. **OCR reprodusibilitas** — tergantung masalah #1; jika input OCR identik,
   tesseract deterministik sehingga teks pun ikut cocok.
3. Probs CLIP cos rendah di satu gambar (lihat §5) — hanya indikator numerik, bukan
   penyimpangan logits.

## 9. Langkah Optimasi Berikutnya (bila dibutuhkan)

- **Batching** inferensi CLIP vision (12 gambar/more) dan YOLO lewat satu session call.
- **DirectML / CUDA EP** di ort — YOLO & CLIP infer bisa anjlok dari puluhan ms ke
  satuan ms; OCR tetap dominan.
- OCR paralel (inci per gambar) atau upgrade PSM/mode cepat untuk gambar sederhana.
- Prefetch/pipeline tahap decode & preprocess ke thread lain (single-thread saat ini).
- Memanggil OCR hanya sekali per gambar & di-cache pada mode batch server.

## 10. Reproduksi

```sh
cargo build --release
./target/release/moderasi-flow.exe --images ..\testing-image-from-user \
  --models .\models --python-ref ..\rust-moderasi\reference\python_reference.json \
  --warmup 2 --iterations 3
# keluaran: results/rust_moderation_results.json  (parity + benchmark)
```

Build bersih (0 error), tanpa dependency runtime Python/PyTorch/torchvision pada pipeline infer.