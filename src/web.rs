//! Web server (axum) mirroring `moderasi-satudata-ml/server.py`:
//!   GET  /                        -> halaman upload (web/index.html)
//!   POST /api/moderasi/satu       (multipart field: file,  opsional ?visual=yolo|clip)
//!   POST /api/moderasi/bulk       (multipart field: files, opsional ?visual=..., max 1000)
//!   GET  /api/engines
//!   POST /api/engines/default?engine=yolo|clip
//!   GET  /api/keywords
//!   GET  /health
//!
//! Response schema untuk setiap gambar meniru moderasi.py `moderasi_satu_gambar`
//! (visual_engine, yolo_class/yolo_conf, yolo_violative, ocr_text, keyword_hits,
//! context_hits, transaction_hits, keyword_context_exempt, keputusan, alasan)
//! ditambah file/filename/elapsed_ms. Ringkasan meniru `_ringkasan()`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::extract::{DefaultBodyLimit, Multipart, Query, State};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::clip::ClipClassifier;
use crate::decision::moderasi;
use crate::keywords::{
    KEYWORDS_ABORSI, KEYWORDS_BORAKS, KEYWORDS_JUDI, KEYWORDS_UMUM, keywords_all,
};
use crate::yolo::{CLASSES, YoloClassifier};

const VALID_ENGINES: [&str; 2] = ["yolo", "clip"];
const MAX_BULK: usize = 1000;
const MAX_OCR_TEXT: usize = 500;
const MAX_UPLOAD_BYTES: usize = 500 * 1024 * 1024;
const ALLOWED_EXT: [&str; 6] = ["jpg", "jpeg", "png", "webp", "gif", "bmp"];

pub struct AppState {
    yolo: Mutex<YoloClassifier>,
    clip: Mutex<ClipClassifier>,
    tesseract: String,
    lang: String,
    current_engine: Mutex<String>,
    ui_html: String,
}

fn err(status: axum::http::StatusCode, detail: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "detail": detail.into() }))).into_response()
}

fn internal(detail: impl std::fmt::Display) -> Response {
    err(axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("{detail}"))
}

/// Resolve ?visual= (or default). Prototype returns 400 for invalid/unavailable engines.
fn resolve_engine(q: Option<&str>, state: &AppState) -> Result<String, Response> {
    match q.map(|s| s.trim().to_ascii_lowercase()) {
        Some(v) if VALID_ENGINES.contains(&v.as_str()) => Ok(v),
        Some(v) => Err(err(
            axum::http::StatusCode::BAD_REQUEST,
            format!("Engine visual tidak valid: {v} (pilihan: yolo, clip)"),
        )),
        None => Ok(state
            .current_engine
            .lock()
            .unwrap()
            .clone()),
    }
}

fn temp_image_path(name: &str) -> String {
    let ext = Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .filter(|e| ALLOWED_EXT.contains(&e.as_str()))
        .unwrap_or_else(|| "jpg".to_string());
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir()
        .join(format!("moderasi_{}_{}.{}", std::process::id(), nanos, ext))
        .to_string_lossy()
        .to_string()
}

fn error_result(name: &str, detail: impl std::fmt::Display) -> serde_json::Value {
    serde_json::json!({
        "file": name,
        "filename": name,
        "keputusan": "ERROR",
        "alasan": [format!("Bukan gambar valid: {detail}")],
        "visual_engine": null,
        "yolo_class": null,
        "yolo_conf": null,
        "keyword_hits": [],
        "ocr_text": "",
    })
}

/// Mirrors server.py `_moderate_bytes` + `moderasi_satu_gambar` for yolo/clip.
fn moderate_bytes(
    state: &AppState,
    name: String,
    data: Vec<u8>,
    engine: String,
) -> serde_json::Value {
    let t0 = Instant::now();
    let tmp = temp_image_path(&name);
    if let Err(e) = std::fs::write(&tmp, &data) {
        let mut v = error_result(&name, e);
        v["elapsed_ms"] = serde_json::json!(((elapsed_ms(t0) * 10.0).round() / 10.0));
        return v;
    }

    // decode (PIL open-and-validate pada prototype)
    let img = match image::open(&tmp) {
        Ok(i) => i.into_rgb8(),
        Err(e) => {
            let mut v = error_result(&name, e);
            v["elapsed_ms"] = serde_json::json!(((elapsed_ms(t0) * 10.0).round() / 10.0));
            let _ = std::fs::remove_file(&tmp);
            return v;
        }
    };

    // visual classification
    let (cls, conf, violative) = match engine.as_str() {
        "clip" => match state.clip.lock().unwrap().classify(&img) {
            Ok(r) => (Some(r.cls_name), Some(r.conf), r.violative),
            Err(e) => {
                eprintln!("[WARN] CLIP gagal ({e}); hasil mengikuti OCR fallback.");
                (None, None, false)
            }
        },
        _ => match state.yolo.lock().unwrap().classify(&img) {
            Ok(r) => (
                Some(CLASSES[r.top1].to_string()),
                Some(r.conf),
                r.violative,
            ),
            Err(e) => {
                eprintln!("[WARN] YOLO gagal ({e}); hasil mengikuti OCR fallback.");
                (None, None, false)
            }
        },
    };

    // OCR: dilewati hanya jika yolo_violative (engine yolo/mobilenet). Untuk CLIP
    // selalu dijalankan agar poster berita/edukasi bisa membatalkan vonis.
    let should_run_ocr = !violative || engine == "clip";
    let raw = if should_run_ocr {
        crate::ocr::ocr_text(Path::new(&tmp), &state.tesseract, &state.lang)
    } else {
        String::new()
    };

    let d = moderasi(
        &name,
        &engine,
        cls.as_deref(),
        conf,
        violative,
        &raw,
    );

    let _ = std::fs::remove_file(&tmp);

    let result = serde_json::json!({
        "file": name,
        "filename": name,
        "visual_engine": d.visual_engine,
        "yolo_class": d.yolo_class,
        "yolo_conf": d.yolo_conf,
        "yolo_violative": d.yolo_violative,
        "ocr_text": d.ocr_text.chars().take(MAX_OCR_TEXT).collect::<String>(),
        "keyword_hits": d.keyword_hits,
        "context_hits": d.context_hits,
        "transaction_hits": d.transaction_hits,
        "keyword_context_exempt": d.keyword_context_exempt,
        "keputusan": d.keputusan,
        "alasan": d.alasan,
        "elapsed_ms": (elapsed_ms(t0) * 10.0).round() / 10.0,
    });
    result
}

fn elapsed_ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

fn ringkasan(results: &[serde_json::Value], total_elapsed: f64) -> serde_json::Value {
    let mut dimoderasi = 0;
    let mut lolos = 0;
    let mut error = 0;
    let mut sum = 0.0;
    for r in results {
        match r["keputusan"].as_str().unwrap_or("") {
            "DIMODERASI" => dimoderasi += 1,
            "ERROR" => error += 1,
            _ => lolos += 1,
        }
        sum += r["elapsed_ms"].as_f64().unwrap_or(0.0);
    }
    let avg = if results.is_empty() { 0.0 } else { sum / results.len() as f64 };
    serde_json::json!({
        "total": results.len(),
        "dimoderasi": dimoderasi,
        "lolos": lolos,
        "error": error,
        "avg_elapsed_ms": (avg * 10.0).round() / 10.0,
        "elapsed_ms": (total_elapsed * 10.0).round() / 10.0,
    })
}

// ------------------------------------------------------------------ handlers

async fn index(State(state): State<Arc<AppState>>) -> Html<String> {
    Html(state.ui_html.clone())
}

async fn health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let engine = state.current_engine.lock().unwrap().clone();
    Json(serde_json::json!({
        "status": "ok",
        "visual_engine": engine,
        "model": "best.onnx",
        "kelas": CLASSES,
    }))
}

async fn engines(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(engines_json(&state))
}

fn engines_json(state: &AppState) -> serde_json::Value {
    serde_json::json!({
        "current": *state.current_engine.lock().unwrap(),
        "available": [
            { "name": "yolo", "available": true },
            { "name": "clip", "available": true },
            { "name": "mobilenetv3", "available": false },
            { "name": "smolvlm", "available": false },
        ],
    })
}

async fn set_default(
    State(state): State<Arc<AppState>>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let engine = q.get("engine").cloned().unwrap_or_default();
    let engine = engine.trim().to_ascii_lowercase();
    if !VALID_ENGINES.contains(&engine.as_str()) {
        return err(
            axum::http::StatusCode::BAD_REQUEST,
            format!("Engine visual tidak valid: {engine} (pilihan: yolo, clip)"),
        );
    }
    *state.current_engine.lock().unwrap() = engine;
    Json(engines_json(&state)).into_response()
}

async fn keywords() -> Json<serde_json::Value> {
    let sorted = |v: &[&str]| -> Vec<String> {
        let mut x: Vec<String> = v.iter().map(|s| s.to_string()).collect();
        x.sort();
        x
    };
    Json(serde_json::json!({
        "total": keywords_all().len(),
        "aborsi": sorted(KEYWORDS_ABORSI),
        "boraks": sorted(KEYWORDS_BORAKS),
        "umum": sorted(KEYWORDS_UMUM),
        "judi": sorted(KEYWORDS_JUDI),
    }))
}

async fn moderasi_satu(
    State(state): State<Arc<AppState>>,
    Query(q): Query<HashMap<String, String>>,
    mut mp: Multipart,
) -> Response {
    let engine = match resolve_engine(q.get("visual").map(|x| x.as_str()), &state) {
        Ok(e) => e,
        Err(r) => return r,
    };

    let mut name: Option<String> = None;
    let mut data: Option<Vec<u8>> = None;
    while let Some(field) = mp.next_field().await.map_err(|e| internal(e)).ok().flatten() {
        if field.name() == Some("file") {
            name = field.file_name().map(|s| s.to_string());
            let buf = match field.bytes().await {
                Ok(b) => b.to_vec(),
                Err(e) => return internal(e),
            };
            data = Some(buf);
            break;
        }
    }

    let name = match name {
        Some(n) if !n.is_empty() => n,
        _ => return err(axum::http::StatusCode::BAD_REQUEST, "Nama file kosong"),
    };
    let data = match data {
        Some(d) => d,
        None => return err(axum::http::StatusCode::BAD_REQUEST, "Field 'file' tidak diunggah"),
    };

    let state2 = Arc::clone(&state);
    let name2 = name.clone();
    let t0 = Instant::now();
    let result = tokio::task::spawn_blocking(move || {
        moderate_bytes(&state2, name2, data, engine)
    })
    .await
    .unwrap_or_else(|e| {
        let mut v = error_result(&name, "task panicked");
        v["elapsed_ms"] = serde_json::json!(0.0);
        eprintln!("[WARN] worker error: {e}");
        v
    });

    let ring = ringkasan(&[result.clone()], elapsed_ms(t0));
    Json(serde_json::json!({ "ringkasan": ring, "results": [result] })).into_response()
}

async fn moderasi_bulk(
    State(state): State<Arc<AppState>>,
    Query(q): Query<HashMap<String, String>>,
    mut mp: Multipart,
) -> Response {
    let engine = match resolve_engine(q.get("visual").map(|x| x.as_str()), &state) {
        Ok(e) => e,
        Err(r) => return r,
    };

    let mut items: Vec<(String, Vec<u8>)> = Vec::new();
    while let Some(field) = mp.next_field().await.map_err(|e| internal(e)).ok().flatten() {
        if field.name() == Some("files") {
            let nm = field.file_name().map(|s| s.to_string()).unwrap_or_else(|| {
                format!("unnamed_{}.jpg", items.len())
            });
            let buf = match field.bytes().await {
                Ok(b) => b.to_vec(),
                Err(e) => return internal(e),
            };
            items.push((nm, buf));
        }
    }

    if items.is_empty() {
        return err(axum::http::StatusCode::BAD_REQUEST, "Tidak ada file diunggah");
    }
    if items.len() > MAX_BULK {
        return err(
            axum::http::StatusCode::BAD_REQUEST,
            format!("Maksimal {MAX_BULK} file per request"),
        );
    }

    let t0 = Instant::now();
    let tasks: Vec<_> = items
        .into_iter()
        .map(|(name, data)| {
            let st = Arc::clone(&state);
            let engine = engine.clone();
            tokio::task::spawn_blocking(move || moderate_bytes(&st, name, data, engine))
        })
        .collect();
    let results: Vec<serde_json::Value> = futures_util::future::join_all(tasks)
        .await
        .into_iter()
        .map(|r| r.unwrap_or_else(|e| {
            eprintln!("[WARN] worker error: {e}");
            serde_json::json!({ "keputusan": "ERROR", "alasan": ["worker panicked"] })
        }))
        .collect();

    let ring = ringkasan(&results, elapsed_ms(t0));
    Json(serde_json::json!({ "ringkasan": ring, "results": results })).into_response()
}

// ------------------------------------------------------------------ runner

fn get_arg(args: &[String], key: &str, default: &str) -> String {
    let mut val = default.to_string();
    let mut i = 0;
    while i < args.len() {
        if args[i] == key {
            if i + 1 < args.len() {
                val = args[i + 1].clone();
                i += 2;
                continue;
            }
        }
        i += 1;
    }
    val
}

/// Entry point dijalankan dari CLI mode: `moderasi-flow server [flags]`.
pub fn run_server(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let host = get_arg(args, "--host", "127.0.0.1");
    let port: u16 = get_arg(args, "--port", "8787").parse().unwrap_or(8787);
    let models_dir = get_arg(args, "--models", "models");
    let tesseract = get_arg(args, "--tesseract", "C:\\Program Files\\Tesseract-OCR\\tesseract.exe");
    let lang = get_arg(args, "--lang", "ind+eng");
    let visual = get_arg(args, "--visual", "yolo").to_ascii_lowercase();
    if !VALID_ENGINES.contains(&visual.as_str()) {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Engine visual tidak valid: {visual} (pilihan: yolo, clip)"),
        )));
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let models = std::path::PathBuf::from(&models_dir);
        let yolo = YoloClassifier::new(&models.join("best.onnx").to_string_lossy(), image::imageops::FilterType::Triangle)?;
        let clip = ClipClassifier::new(&models, image::imageops::FilterType::Lanczos3)?;

        let ui_html = load_ui().unwrap_or_else(|| {
            "<h3>web/index.html tidak ditemukan di samping binary</h3>".to_string()
        });

        let state = Arc::new(AppState {
            yolo: Mutex::new(yolo),
            clip: Mutex::new(clip),
            tesseract,
            lang,
            current_engine: Mutex::new(visual.clone()),
            ui_html,
        });

        let app = Router::new()
            .route("/", get(index))
            .route("/health", get(health))
            .route("/api/engines", get(engines))
            .route("/api/engines/default", post(set_default))
            .route("/api/keywords", get(keywords))
            .route("/api/moderasi/satu", post(moderasi_satu))
            .route("/api/moderasi/bulk", post(moderasi_bulk))
            .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES))
            .with_state(state);

        let addr = format!("{host}:{port}");
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        println!("=== Server Moderasi Gambar (Rust) ===");
        println!("  http://{addr}");
        println!("  GET  / -> halaman upload");
        println!("  POST /api/moderasi/satu  (field: file, opsional ?visual=yolo|clip)");
        println!("  POST /api/moderasi/bulk  (field: files, opsional ?visual=...)");
        println!("  GET  /health");
        axum::serve(listener, app).await?;
        Ok(())
    })
}

fn load_ui() -> Option<String> {
    for p in ["web/index.html", "web/ui.html", "../web/index.html"] {
        if let Ok(s) = std::fs::read_to_string(p) {
            return Some(s);
        }
    }
    None
}