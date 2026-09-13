//! Rust moderation-flow replica: faithful decision pipeline (YOLO11-cls + CLIP +
//! Tesseract OCR + keyword/context gates) with per-stage timing, parity checks
//! against the Python prototype reference, and benchmark statistics.

mod clip;
mod decision;
mod keywords;
mod ocr;
mod preprocess;
mod sampler;
mod tokenizer;
mod web;
mod yolo;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use image::imageops::FilterType;
use serde::Serialize;

use crate::clip::ClipClassifier;
use crate::decision::moderasi;
use crate::sampler::ResourceSampler;
use crate::yolo::YoloClassifier;

const DEFAULT_IMAGES: &str = "testing-image-from-user";
const DEFAULT_MODELS: &str = "rust-moderasi/models";
const DEFAULT_TESSERACT: &str = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe";

#[derive(Serialize)]
struct Stats {
    n: usize,
    min: f32,
    max: f32,
    avg: f32,
    p50: f32,
    p95: f32,
}

fn stats(data: &[f32]) -> Stats {
    let mut s = data.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    let avg = if n == 0 {
        0.0
    } else {
        s.iter().sum::<f32>() / n as f32
    };
    let p = |q: f32| -> f32 {
        if n == 0 {
            return 0.0;
        }
        let idx = ((n as f32 - 1.0) * q).round() as usize;
        s[idx.min(n - 1)]
    };
    Stats {
        n,
        min: s.first().copied().unwrap_or(0.0),
        max: s.last().copied().unwrap_or(0.0),
        avg,
        p50: p(0.50),
        p95: p(0.95),
    }
}

#[derive(Serialize)]
struct ImageEntry {
    file: String,
    filename: String,
    width: u32,
    height: u32,
    yolo_class: String,
    yolo_conf: f32,
    yolo_violative: bool,
    clip_class: String,
    clip_conf: f32,
    clip_violative: bool,
    yolo_time_ms: f32,
    clip_time_ms: f32,
    preprocess_time_ms: f32,
    ocr_time_ms: f32,
    yolo_pipeline_ms: f32,
    clip_pipeline_ms: f32,
    total_time_ms: f32,
    keputusan: String,
    keputusan_clip: String,
    ocr_text: String,
    keyword_hits: Vec<String>,
    context_hits: Vec<String>,
    transaction_hits: Vec<String>,
    keyword_context_exempt: bool,
    alasan: Vec<String>,
    // parity vs python reference
    yolo_probs_cos: f32,
    yolo_probs_maxdiff: f32,
    clip_logits_cos: f32,
    clip_probs_cos: f32,
    clip_conf_diff: f32,
    decision_yolo_match: bool,
    decision_clip_match: bool,
    ocr_text_match: bool,
    keyword_hits_match: bool,
}

fn ms(t: Instant) -> f32 {
    t.elapsed().as_secs_f64() as f32 * 1000.0
}

fn parse_filter(s: &str) -> FilterType {
    match s.to_ascii_lowercase().as_str() {
        "lanczos3" | "lanczos" => FilterType::Lanczos3,
        "catmullrom" | "bicubic" => FilterType::CatmullRom,
        "gaussian" => FilterType::Gaussian,
        _ => FilterType::Triangle,
    }
}

fn image_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let ok = |p: &Path| -> bool {
        matches!(
            p.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref(),
            Some("jpg") | Some("jpeg") | Some("png") | Some("webp") | Some("gif") | Some("bmp")
        )
    };
    if dir.is_file() && ok(dir) {
        out.push(dir.to_path_buf());
    } else if let Ok(rd) = fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_file() && ok(&p) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- args -------------------------------------------------------------
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    // Web mode: `moderasi-flow server [--host .. --port .. --models ..]`
    if args.first().map(String::as_str) == Some("server") {
        args.remove(0);
        return web::run_server(&args);
    }

    let mut get = |key: &str, default: &str| -> String {
        let mut val = default.to_string();
        let mut i = 0;
        while i < args.len() {
            if args[i] == key {
                if i + 1 < args.len() {
                    val = args[i + 1].clone();
                    args.remove(i);
                    args.remove(i);
                    continue;
                }
            }
            i += 1;
        }
        val
    };
    let images_dir = PathBuf::from(get("--images", DEFAULT_IMAGES));
    let models_dir = PathBuf::from(get("--models", DEFAULT_MODELS));
    let python_ref = PathBuf::from(get("--python-ref", "reference/python_reference.json"));
    let out_path = PathBuf::from(get("--out", "results/rust_moderation_results.json"));
    let warmup: u32 = get("--warmup", "2").parse().unwrap_or(2);
    let iterations: u32 = get("--iterations", "3").parse().unwrap_or(3);
    let tesseract = get("--tesseract", DEFAULT_TESSERACT);
    let lang = get("--lang", "ind+eng");
    let visual = get("--visual", "yolo");
    let yolo_filter = parse_filter(&get("--yolo-filter", "triangle"));
    let clip_filter = parse_filter(&get("--clip-filter", "lanczos3"));

    let files = image_files(&images_dir);
    if files.is_empty() {
        eprintln!("[ERROR] Tidak ada gambar ditemukan di {}", images_dir.display());
        std::process::exit(1);
    }

    // ---- reference (python prototype) -------------------------------------
    let py_root: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&python_ref)?)?;
    let py_images: &Vec<serde_json::Value> = py_root["images"].as_array().unwrap();
    let py_by_name: BTreeMap<&str, &serde_json::Value> = py_images
        .iter()
        .map(|e| (e["filename"].as_str().unwrap(), e))
        .collect();
    let py_index_by_name: BTreeMap<&str, usize> = py_images
        .iter()
        .enumerate()
        .map(|(i, e)| (e["filename"].as_str().unwrap(), i))
        .collect();

    // ---- models ------------------------------------------------------------
    println!("loading YOLO ONNX ...");
    let mut yolo = YoloClassifier::new(
        &models_dir.join("best.onnx").to_string_lossy(),
        yolo_filter,
    )?;
    println!("loading CLIP ONNX + tokenizer (text feats for {} prompts) ...", 50);
    let mut clip = ClipClassifier::new(&models_dir, clip_filter)?;

    // ---- tokenizer parity ------------------------------------------------
    let clip_ref: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(models_dir.join("clip/reference.json"))?)?;
    let tok = clip.tokenizer_ref();
    let prompts_s: Vec<String> = clip.prompt_refs().iter().map(|s| s.to_string()).collect();
    let mut tok_match = 0usize;
    let mut tok_maxlen_mismatch = 0usize;
    let mut text_feat_cos = Vec::new();
    let ref_ids = clip_ref["input_ids"].as_array().unwrap();
    for (i, p) in prompts_s.iter().enumerate() {
        let (ids, _mask) = tok.encode(p);
        let r: Vec<i64> = ref_ids[i]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        let nref = r.len();
        if ids.len() < nref {
            tok_maxlen_mismatch += 1;
            continue;
        }
        if ids[..nref].iter().map(|&x| x as i64).eq(r.iter().copied()) {
            tok_match += 1;
        }
    }
    // text feature parity (rust reprocesses with 77 pads; reference used length-30 pads)
    let ref_feats: Vec<Vec<f32>> = clip_ref["text_features"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| {
            x.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        })
        .collect();
    for (rf, rf_ref) in clip.text_features_ref().iter().zip(ref_feats.iter()) {
        text_feat_cos.push(crate::yolo::cosine(rf, rf_ref));
    }
    let text_feat_cos = stats(&text_feat_cos);
    println!(
        "tokenizer: {}/{} prompts token-id match (ref padded len ok={})",
        tok_match,
        prompts_s.len(),
        tok_maxlen_mismatch == 0
    );
    println!(
        "text features: cos {:.7} (min) / {:.7} (avg)",
        text_feat_cos.min, text_feat_cos.avg
    );

    // ---- timed pipeline ---------------------------------------------------
    println!("\n=== MODERASI {} GAMBAR (warmup {}, iters {}) ===", files.len(), warmup, iterations);
    let mut sampler = ResourceSampler::start();

    let mut ref_inputs: Option<Vec<(Vec<f32>, Vec<f32>)>> = None; // yolo+clip preprocess (1st run)
    let mut out_images: Vec<ImageEntry> = Vec::new();
    let mut collect_times = false;

    // stage timing buffers (per timed iteration * per image)
    let mut t_load: Vec<f32> = Vec::new();
    let mut t_y_pre: Vec<f32> = Vec::new();
    let mut t_y_inf: Vec<f32> = Vec::new();
    let mut t_c_pre: Vec<f32> = Vec::new();
    let mut t_c_inf: Vec<f32> = Vec::new();
    let mut t_ocr: Vec<f32> = Vec::new();
    let mut t_y_pipe: Vec<f32> = Vec::new();
    let mut t_c_pipe: Vec<f32> = Vec::new();
    let mut all_timings: Vec<Vec<f32>> = Vec::new(); // per-image [8 values] for output

    for run in 0..(warmup + iterations) {
        collect_times = run >= warmup;
        for f in &files {
            let filename = f.file_name().unwrap().to_string_lossy().to_string();

            let t0 = Instant::now();
            let img = preprocess::load_image_rgb(f)?;
            let (w, h) = (img.width(), img.height());
            let dt_load = ms(t0);

            let t1 = Instant::now();
            let (y_input, yrw, yrh) = yolo.preprocess(&img);
            let dt_y_pre = ms(t1);

            let t2 = Instant::now();
            let yr = yolo.infer(&y_input)?;
            let dt_y_inf = ms(t2);

            let t3 = Instant::now();
            let (c_input, crw, crh) = clip.preprocess(&img);
            let dt_c_pre = ms(t3);

            let t4 = Instant::now();
            let cr = clip.infer(&c_input)?;
            let dt_c_inf = ms(t4);

            // OCR: prototype runs when (not yolo_violative) OR engine==clip; we need both
            // engine decisions so run it always, then exclude from the yolo-pipeline
            // accounting for violative images (prototype skips it there).
            let t5 = Instant::now();
            let raw = ocr::ocr_text(f, &tesseract, &lang);
            let dt_ocr = ms(t5);

            let run_ocr_yolo = !yr.violative;

            let d_yolo = moderasi(
                &filename, "yolo",
                Some(yolo::CLASSES[yr.top1]), Some(yr.conf), yr.violative,
                if run_ocr_yolo { &raw } else { "" },
            );
            let d_clip = moderasi(
                &filename, "clip",
                Some(&cr.cls_name), Some(cr.conf), cr.violative,
                &raw,
            );

            if collect_times {
                let yolo_pipe = dt_load + dt_y_pre + dt_y_inf + if run_ocr_yolo { dt_ocr } else { 0.0 };
                let clip_pipe = dt_load + dt_c_pre + dt_c_inf + dt_ocr;
                t_load.push(dt_load);
                t_y_pre.push(dt_y_pre);
                t_y_inf.push(dt_y_inf);
                t_c_pre.push(dt_c_pre);
                t_c_inf.push(dt_c_inf);
                t_ocr.push(dt_ocr);
                t_y_pipe.push(yolo_pipe);
                t_c_pipe.push(clip_pipe);
                all_timings.push(vec![dt_load, dt_y_pre, dt_y_inf, dt_c_pre, dt_c_inf, dt_ocr, yolo_pipe, clip_pipe]);
            }

            let py = py_by_name.get(filename.as_str());

            // parity computation (first timed run)
            let mut p = Parity::default();
            if let Some(entry) = py {
                p = compare_to_python(entry, &yr, &cr, &d_yolo, &d_clip);
            }

            if run == warmup {
                ref_inputs
                    .get_or_insert_with(Vec::new)
                    .push((y_input.clone(), c_input.clone()));
            }

            if collect_times {
                out_images.push(ImageEntry {
                    file: f.display().to_string(),
                    filename: filename.clone(),
                    width: w,
                    height: h,
                    yolo_class: yolo::CLASSES[yr.top1].to_string(),
                    yolo_conf: yr.conf,
                    yolo_violative: yr.violative,
                    clip_class: cr.cls_name.clone(),
                    clip_conf: cr.conf,
                    clip_violative: cr.violative,
                    yolo_time_ms: dt_y_inf,
                    clip_time_ms: dt_c_inf,
                    preprocess_time_ms: dt_y_pre,
                    ocr_time_ms: dt_ocr,
                    yolo_pipeline_ms: dt_load + dt_y_pre + dt_y_inf + if run_ocr_yolo { dt_ocr } else { 0.0 },
                    clip_pipeline_ms: dt_load + dt_c_pre + dt_c_inf + dt_ocr,
                    total_time_ms: dt_load + dt_y_pre + dt_y_inf + dt_c_pre + dt_c_inf + dt_ocr,
                    keputusan: d_yolo.keputusan.clone(),
                    keputusan_clip: d_clip.keputusan.clone(),
                    ocr_text: d_yolo.ocr_text.clone(),
                    keyword_hits: d_yolo.keyword_hits.clone(),
                    context_hits: d_yolo.context_hits.clone(),
                    transaction_hits: d_yolo.transaction_hits.clone(),
                    keyword_context_exempt: d_yolo.keyword_context_exempt,
                    alasan: d_yolo.alasan.clone(),
                    yolo_probs_cos: p.yolo_probs_cos,
                    yolo_probs_maxdiff: p.yolo_probs_maxdiff,
                    clip_logits_cos: p.clip_logits_cos,
                    clip_probs_cos: p.clip_probs_cos,
                    clip_conf_diff: p.clip_conf_diff,
                    decision_yolo_match: p.decision_yolo_match,
                    decision_clip_match: p.decision_clip_match,
                    ocr_text_match: p.ocr_text_match,
                    keyword_hits_match: p.keyword_hits_match,
                });
            }

            let _ = (yrw, yrh, crw, crh);
        }
    }

    // ---- reference preprocess bins -----------------------------------------
    if let Some(inputs) = ref_inputs {
        let ref_dir = python_ref.parent().unwrap_or(Path::new("."));
        let mut cos_y = Vec::new();
        let mut cos_c = Vec::new();
        let mut checked = 0usize;
        for (i, f) in files.iter().enumerate() {
            let (y_in, c_in) = &inputs[i];
            let fname = f.file_name().unwrap().to_string_lossy();
            // bins are indexed by the python reference's own order
            let k = py_index_by_name
                .get(fname.as_ref())
                .copied()
                .unwrap_or(i);
            let yp = ref_dir.join(format!("yolo_input_{k:02}.bin", k = k));
            let cp = ref_dir.join(format!("clip_input_{k:02}.bin", k = k));
            if let (Ok(yb), Ok(cb)) = (fs::read(&yp), fs::read(&cp)) {
                cos_y.push(crate::yolo::cosine(y_in, &to_f32_le(&yb)));
                cos_c.push(crate::yolo::cosine(c_in, &to_f32_le(&cb)));
                checked += 1;
            }
        }
        println!("preprocess cos vs python bins ({checked} images): yolo {:.6}..{:.6}  clip {:.6}..{:.6}",
            cos_y.iter().cloned().fold(f32::MAX, f32::min),
            cos_y.iter().cloned().fold(f32::MIN, f32::max),
            cos_c.iter().cloned().fold(f32::MAX, f32::min),
            cos_c.iter().cloned().fold(f32::MIN, f32::max));
    }

    sampler.stop();
    let res = sampler.snapshot();
    let proc_cpu = stats(&res.process_cpu);
    let rss = stats(&res.process_rss_mb.iter().map(|&v| v as f32).collect::<Vec<_>>());
    let sys_cpu = stats(&res.system_cpu);
    let sys_ram = stats(&res.system_ram_percent.iter().map(|&v| v as f32).collect::<Vec<_>>());

    // ---- benchmark / parity summary ---------------------------------------
    let n_img = files.len() as f32;
    let n_timed = iterations as f32;
    let yolo_ips = if t_y_pipe.is_empty() { 0.0 } else { (n_img * n_timed * 1000.0) / t_y_pipe.iter().sum::<f32>() };
    let clip_ips = if t_c_pipe.is_empty() { 0.0 } else { (n_img * n_timed * 1000.0) / t_c_pipe.iter().sum::<f32>() };

    let dec_y_matches = out_images.iter().filter(|e| e.decision_yolo_match).count();
    let dec_c_matches = out_images.iter().filter(|e| e.decision_clip_match).count();
    let ocr_matches = out_images.iter().filter(|e| e.ocr_text_match).count();

    // console table
    println!("\n--- PARITY (Rust vs Python prototype) ---");
    println!("{:<44} {:>9} {:>9} {:>8} {:>8} {} {}
        ", "", "yoloP", "clipP", "yoloC", "clipC", "decY", "decC");
    for e in &out_images {
        println!(
            "{:<44} {:>9.5} {:>9.5} {:>8} {:>8} {} {}",
            e.filename.chars().take(44).collect::<String>(),
            e.yolo_probs_cos, e.clip_logits_cos,
            if e.decision_yolo_match { "OK" } else { "X" },
            if e.decision_clip_match { "OK" } else { "X" },
            if e.keyword_hits_match { "." } else { "K" },
            if e.ocr_text_match { "." } else { "O" }
        );
    }
    println!("decision match yolo: {}/{}  clip: {}/{}", dec_y_matches, out_images.len(), dec_c_matches, out_images.len());

    let st = |v: &[f32]| -> f32 { stats(v).avg };
    let (avg_load, avg_y_pre, avg_y_inf, avg_c_pre, avg_c_inf, avg_ocr, avg_y_pipe, avg_c_pipe) =
        (st(&t_load), st(&t_y_pre), st(&t_y_inf), st(&t_c_pre), st(&t_c_inf), st(&t_ocr), st(&t_y_pipe), st(&t_c_pipe));
    println!("\n--- TIMING (avg ms) {}-{}-{}-{}-{}-{}-{}-{}", avg_load, avg_y_pre, avg_y_inf, avg_c_pre, avg_c_inf, avg_ocr, avg_y_pipe, avg_c_pipe);
    println!("yolo pipeline avg {:.2} ms -> {:.2} img/s | clip pipeline avg {:.2} ms -> {:.2} img/s",
        avg_y_pipe, yolo_ips, avg_c_pipe, clip_ips);

    // ---- JSON output ------------------------------------------------------
    let out = serde_json::json!({
        "meta": {
            "image_dir": images_dir.display().to_string(),
            "images": out_images.len(),
            "warmup": warmup,
            "iterations": iterations,
            "visual_default": visual,
            "tesseract_cmd": tesseract,
            "lang": lang,
            "yolo_filter": format!("{:?}", yolo_filter),
            "clip_filter": format!("{:?}", clip_filter),
            "viol_conf_threshold": 0.70,
            "clip_viol_threshold": 0.30,
            "clip_viol_margin": 2.0,
        },
        "tokenizer_parity": {
            "prompts_match": tok_match,
            "prompts_total": prompts_s.len(),
            "text_features_cos_avg": text_feat_cos.avg,
            "text_features_cos_min": text_feat_cos.min,
        },
        "images": out_images,
        "parity": {
            "decisions_yolo_match": dec_y_matches,
            "decisions_clip_match": dec_c_matches,
            "ocr_text_match": ocr_matches,
            "ocr_text_samples": out_images.len(),
            "details": out_images.iter().map(|e| {
                serde_json::json!({
                    "filename": e.filename,
                    "decision_yolo_match": e.decision_yolo_match,
                    "decision_clip_match": e.decision_clip_match,
                    "ocr_text_match": e.ocr_text_match,
                    "keyword_hits_match": e.keyword_hits_match,
                    "yolo_probs_cos": e.yolo_probs_cos,
                    "yolo_probs_maxdiff": e.yolo_probs_maxdiff,
                    "clip_logits_cos": e.clip_logits_cos,
                    "clip_probs_cos": e.clip_probs_cos,
                    "clip_conf_diff": e.clip_conf_diff,
                })
            }).collect::<Vec<_>>(),
        },
        "benchmark": {
            "samples": all_timings.len(),
            "per_image_ms": {
                "decode": stats(&t_load),
                "yolo_preprocess": stats(&t_y_pre),
                "yolo_infer": stats(&t_y_inf),
                "clip_preprocess": stats(&t_c_pre),
                "clip_infer": stats(&t_c_inf),
                "ocr": stats(&t_ocr),
                "yolo_pipeline": stats(&t_y_pipe),
                "clip_pipeline": stats(&t_c_pipe),
            },
            "throughput": {
                "yolo_engine_img_s": yolo_ips,
                "clip_engine_img_s": clip_ips,
            },
        },
        "resources": {
            "process_cpu_pct": proc_cpu,
            "process_rss_mb": rss,
            "system_cpu_pct": sys_cpu,
            "system_ram_pct": sys_ram,
        },
    });
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out_path, serde_json::to_string_pretty(&out)?)?;
    println!("\nwrote {}", out_path.display());

    println!("\nDONE.");
    Ok(())
}

// ---- helpers ---------------------------------------------------------------

#[derive(Default)]
struct Parity {
    yolo_probs_cos: f32,
    yolo_probs_maxdiff: f32,
    clip_logits_cos: f32,
    clip_probs_cos: f32,
    clip_conf_diff: f32,
    decision_yolo_match: bool,
    decision_clip_match: bool,
    ocr_text_match: bool,
    keyword_hits_match: bool,
}

fn compare_to_python(
    py: &serde_json::Value,
    yr: &crate::yolo::YoloResult,
    cr: &crate::clip::ClipResult,
    d_yolo: &crate::decision::ModerationDecision,
    d_clip: &crate::decision::ModerationDecision,
) -> Parity {
    let py_yolo = &py["yolo"];
    let py_clip = &py["clip"];
    let py_dy = &py["decision_yolo"];
    let py_dc = &py["decision_clip"];

    let py_probs: Vec<f32> = py_yolo["probs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let py_logits: Vec<f32> = py_clip["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let py_cprobs: Vec<f32> = py_clip["probs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let (yolo_probs_cos, yolo_probs_maxdiff) =
        crate::yolo::prob_stats(&yr.probs, &py_probs);
    let clip_logits_cos = crate::yolo::cosine(&cr.logits, &py_logits);
    let (clip_probs_cos, _) = crate::yolo::prob_stats(&cr.probs, &py_cprobs);
    let clip_conf_diff = (cr.conf - py_clip["conf"].as_f64().unwrap() as f32).abs();

    let mut h_sorted = d_yolo.keyword_hits.clone();
    h_sorted.sort();
    let py_ky: Vec<String> = py_dy["keyword_hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    let mut py_ky_s = py_ky.clone();
    py_ky_s.sort();

    Parity {
        yolo_probs_cos,
        yolo_probs_maxdiff,
        clip_logits_cos,
        clip_probs_cos,
        clip_conf_diff,
        decision_yolo_match: d_yolo.keputusan == py_dy["keputusan"].as_str().unwrap(),
        decision_clip_match: d_clip.keputusan == py_dc["keputusan"].as_str().unwrap(),
        ocr_text_match: d_yolo.ocr_text == py_dy["ocr_text"].as_str().unwrap(),
        keyword_hits_match: py_ky_s == h_sorted,
    }
}

fn to_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}