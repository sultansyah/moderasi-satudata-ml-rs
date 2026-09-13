//! YOLO11n-cls ONNX inference.
//! Input "images" (1,3,224,224) f32; output "output0" (1,3) class probabilities
//! (ultralytics classification export bakes the softmax into the graph — verified
//! against torch forward: exact match, values sum to ~1).
//! Decision rule replicates moderasi.py: violative if class in KELAS_VIOLATIVE
//! and confidence >= VIOL_CONF_THRESHOLD.

use image::imageops::FilterType;
use image::RgbImage;
use ort::session::Session;
use std::collections::HashMap;

use crate::preprocess::yolo_preprocess;

pub const CLASSES: [&str; 3] = ["dokumen", "normal", "obat_aborsi"];
pub const KELAS_VIOLATIVE: [&str; 1] = ["obat_aborsi"];
pub const VIOL_CONF_THRESHOLD: f32 = 0.70;

pub struct YoloResult {
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
    pub top1: usize,
    pub conf: f32, // rounded to 4 decimals like prototype
    pub violative: bool,
}

pub struct YoloClassifier {
    session: Session,
    resize_filter: FilterType,
}

impl YoloClassifier {
    pub fn new(onnx_path: &str, resize_filter: FilterType) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?.commit_from_file(onnx_path)?;
        Ok(YoloClassifier { session, resize_filter })
    }

    pub fn preprocess(&self, img: &RgbImage) -> (Vec<f32>, u32, u32) {
        yolo_preprocess(img, self.resize_filter)
    }

    pub fn infer(&mut self, input: &[f32]) -> Result<YoloResult, Box<dyn std::error::Error>> {
        let tensor = ort::value::Tensor::from_array(([1usize, 3, 224, 224], input.to_vec()))?;
        let outputs = self.session.run(ort::inputs!["images" => tensor])?;
        // The exported graph outputs already-softmaxed class probabilities.
        let probs: Vec<f32> = outputs[0].try_extract_tensor::<f32>()?.1.to_vec();
        let logits = probs.clone();

        let mut top1 = 0usize;
        for i in 1..probs.len() {
            if probs[i] > probs[top1] {
                top1 = i;
            }
        }
        let conf = (probs[top1] * 10000.0).round() / 10000.0;
        let cls = CLASSES[top1];
        let violative = KELAS_VIOLATIVE.contains(&cls) && conf >= VIOL_CONF_THRESHOLD;
        Ok(YoloResult {
            logits,
            probs,
            top1,
            conf,
            violative,
        })
    }

    pub fn classify(&mut self, img: &RgbImage) -> Result<YoloResult, Box<dyn std::error::Error>> {
        let (input, _, _) = self.preprocess(img);
        self.infer(&input)
    }
}

pub fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Compare Rust probs vs python reference probs (cosine + max abs diff).
pub fn prob_stats(rust: &[f32], py: &[f32]) -> (f32, f32) {
    let cos = cosine(rust, py);
    let maxdiff = rust
        .iter()
        .zip(py.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    (cos, maxdiff)
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

pub fn class_names_map() -> HashMap<i64, String> {
    CLASSES
        .iter()
        .enumerate()
        .map(|(i, &c)| (i as i64, c.to_string()))
        .collect()
}