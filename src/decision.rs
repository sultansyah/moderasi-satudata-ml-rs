//! Faithful port of moderasi.py `moderasi_satu_gambar` decision flow.

use crate::keywords::{
    covert_abortion_ad_hits, keyword_hits, normalize, public_interest_context,
    KEYWORDS_ABORSI, KEYWORDS_BORAKS, KEYWORDS_JUDI,
};
use serde::Serialize;

const OCR_TEXT_MAX: usize = 500;

#[derive(Serialize, Clone)]
pub struct ModerationDecision {
    pub filename: String,
    pub visual_engine: String,
    pub yolo_class: Option<String>,
    pub yolo_conf: Option<f32>,
    pub yolo_violative: bool,
    pub ocr_text: String,
    pub keyword_hits: Vec<String>,
    pub context_hits: Vec<String>,
    pub transaction_hits: Vec<String>,
    pub keyword_context_exempt: bool,
    pub keputusan: String,
    pub alasan: Vec<String>,
}

/// Runs the decision logic given an engine output + OCR text. Mirrors
/// `moderasi_satu_gambar` for the yolo / clip engines.
pub fn moderasi(
    filename: &str,
    engine: &str,
    cls_name: Option<&str>,
    conf: Option<f32>,
    violative: bool,
    raw_ocr: &str,
) -> ModerationDecision {
    let yolo_class = cls_name.map(|s| s.to_string());
    let yolo_conf = conf;
    let mut yolo_violative = violative;
    let mut result = ModerationDecision {
        filename: filename.to_string(),
        visual_engine: engine.to_string(),
        yolo_class,
        yolo_conf,
        yolo_violative,
        ocr_text: String::new(),
        keyword_hits: Vec::new(),
        context_hits: Vec::new(),
        transaction_hits: Vec::new(),
        keyword_context_exempt: false,
        keputusan: "LOLOS".to_string(),
        alasan: Vec::new(),
    };

    if violative && engine != "smolvlm" {
        let cls = cls_name.unwrap_or("?");
        let c = conf.unwrap_or(0.0);
        result
            .alasan
            .push(format!("{} deteksi kelas violative: {} (conf {:.2})", engine.to_uppercase(), cls, c));
    }

    let should_run_ocr = !violative || engine == "clip";
    if should_run_ocr {
        result.ocr_text = raw_ocr.chars().take(OCR_TEXT_MAX).collect::<String>();
        if !raw_ocr.is_empty() {
            let norm = normalize(raw_ocr);
            let (is_public, context_hits, transaction_hits) = public_interest_context(&norm);
            let public_context = is_public;
            let covert = covert_abortion_ad_hits(&norm, &transaction_hits);

            if !context_hits.is_empty() {
                result.context_hits = context_hits.iter().take(20).cloned().collect();
            }
            if !transaction_hits.is_empty() {
                result.transaction_hits = transaction_hits.iter().take(20).cloned().collect();
            }

            if public_context && covert.is_empty() {
                result.keyword_context_exempt = true;
                result.alasan.push(format!(
                    "Konteks berita/peringatan terdeteksi: {}",
                    context_hits.iter().take(8).cloned().collect::<Vec<_>>().join(", ")
                ));
                if engine == "clip" && yolo_violative {
                    yolo_violative = false;
                    result.yolo_violative = false;
                    result
                        .alasan
                        .push("Vonis CLIP ditahan karena konteks publik/edukatif".to_string());
                }
            }

            let mut hits = keyword_hits(&norm);
            if !covert.is_empty() {
                hits.extend(covert.iter().cloned());
            }
            if !hits.is_empty() {
                result.keyword_hits = hits.iter().take(20).cloned().collect();
                if public_context && covert.is_empty() {
                    result.alasan.push(format!(
                        "OCR match keyword tetapi dikecualikan karena konteks: {}",
                        hits.iter().take(10).cloned().collect::<Vec<_>>().join(", ")
                    ));
                } else if !covert.is_empty() {
                    result.alasan.push(format!(
                        "OCR mendeteksi pola iklan aborsi terselubung: {}",
                        covert.iter().take(8).cloned().collect::<Vec<_>>().join(", ")
                    ));
                } else {
                    result
                        .alasan
                        .push(format!("OCR match keyword: {}", hits.iter().take(10).cloned().collect::<Vec<_>>().join(", ")));
                }
            }
        }

        if result.keyword_hits.is_empty() {
            // Divergensi disengaja dari prototype: fallback nama file hanya memakai
            // keyword "kuat" (aborsi/boraks/judi). Frasa komersial generik
            // (KEYWORDS_UMUM: whatsapp, cod, dll.) diabaikan dari nama file karena
            // "WhatsApp Image ..." adalah penamaan file biasa, bukan indikator
            // transaksi. Deteksi lewat OCR tetap memakai semua daftar.
            let hits: Vec<String> = keyword_hits(&normalize(filename))
                .into_iter()
                .filter(|h| is_strong_filename_keyword(h))
                .collect();
            if !hits.is_empty() {
                result.keyword_hits = hits.iter().take(20).cloned().collect();
                result.alasan.push(format!(
                    "Filename match keyword: {}",
                    hits.iter().take(10).cloned().collect::<Vec<_>>().join(", ")
                ));
            }
        }
    } else {
        result.alasan.push("OCR dilewati (visual sudah violative)".to_string());
    }

    let effective_keyword_hit = !result.keyword_hits.is_empty() && !result.keyword_context_exempt;
    if yolo_violative || effective_keyword_hit {
        result.keputusan = "DIMODERASI".to_string();
    }
    result
}

/// Keyword "kuat" untuk fallback nama file: aborsi, boraks, judi saja.
fn is_strong_filename_keyword(kw: &str) -> bool {
    KEYWORDS_ABORSI.contains(&kw)
        || KEYWORDS_BORAKS.contains(&kw)
        || KEYWORDS_JUDI.contains(&kw)
}