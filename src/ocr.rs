//! Tesseract OCR replicating moderasi.py: grayscale, LANCZOS resize when the
//! longest edge exceeds MAX_OCR_SIZE (1600), then tesseract -l ind+eng --oem 1 --psm 3.

use image::imageops::FilterType;
use std::path::Path;
use std::process::Command;

pub const MAX_OCR_SIZE: u32 = 1600;

/// Prepare a grayscale PNG for tesseract (mirrors _prep_ocr_image), saving to a
/// temp file. Returns the temp file path, or Err on decode/save failure.
pub fn prep_ocr_png(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let img = image::open(path)?.to_luma8();
    let (w, h) = (img.width(), img.height());
    let longest = w.max(h);
    let out = if longest > MAX_OCR_SIZE {
        let scale = MAX_OCR_SIZE as f32 / longest as f32;
        let nw = ((w as f32 * scale) as u32).max(1);
        let nh = ((h as f32 * scale) as u32).max(1);
        image::imageops::resize(&img, nw, nh, FilterType::Lanczos3)
    } else {
        img
    };
    let tmp = std::env::temp_dir().join(format!(
        "moderasi_ocr_{}_{}.png",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    out.save(&tmp)?;
    Ok(tmp.to_string_lossy().to_string())
}

/// Runs tesseract on `image_path`; returns trimmed OCR text ("" on any failure).
pub fn ocr_text(image_path: &Path, tesseract_cmd: &str, lang: &str) -> String {
    let tmp = match prep_ocr_png(image_path) {
        Ok(p) => p,
        Err(_) => return String::new(),
    };
    let result = Command::new(tesseract_cmd)
        .arg(&tmp)
        .arg("stdout")
        .args(["-l", lang, "--oem", "1", "--psm", "3"])
        .output();
    let _ = std::fs::remove_file(&tmp);
    match result {
        Ok(out) if out.status.success() => {
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        }
        Ok(out) => {
            eprintln!("[WARN] tesseract failed: {}", String::from_utf8_lossy(&out.stderr));
            String::new()
        }
        Err(e) => {
            eprintln!("[WARN] OCR gagal: {e}");
            String::new()
        }
    }
}