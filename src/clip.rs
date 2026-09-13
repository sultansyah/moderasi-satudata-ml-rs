//! CLIP zero-shot engine (openai/clip-vit-base-patch32) via ONNX, faithful to
//! moderasi.py `_clip_classify`: 50 prompts (20 violative + 30 safe), softmax over
//! the 50 logits, violative iff vio_sum >= 0.30 AND vio_sum >= safe_sum*2.0.
//!
//! Text features are computed once at startup (prompts are fixed; the Rust text
//! path reproduces the prototype's pooled features, verified cos ~1.0 vs HF).

use image::imageops::FilterType;
use image::RgbImage;
use ort::session::Session;
use std::path::Path;

use crate::preprocess::clip_preprocess;
use crate::tokenizer::ClipTokenizer;
use crate::yolo::cosine;

pub const CLIP_VIOL_THRESHOLD: f32 = 0.30;
pub const CLIP_VIOL_MARGIN: f32 = 2.0;

pub struct ClipResult {
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
    pub n_vio: usize,
    pub vio_sum: f32,
    pub safe_sum: f32,
    pub best_vio: f32,
    pub best_safe: f32,
    pub cls_name: String,
    pub conf: f32,
    pub violative: bool,
}

pub struct ClipClassifier {
    vision: Session,
    text: Session,
    tokenizer: ClipTokenizer,
    prompts: Vec<String>,
    text_features: Vec<Vec<f32>>,
    logit_scale: f32,
    vio_threshold: f32,
    viol_margin: f32,
    n_vio: usize,
    vio_labels: Vec<String>,
    safe_label: String,
    resize_filter: FilterType,
}

impl ClipClassifier {
    /// `models_dir` should contain clip/ (onnx + reference.json) and clip-tokenizer/.
    pub fn new(
        models_dir: &Path,
        resize_filter: FilterType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let clip_dir = models_dir.join("clip");
        let ref_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(clip_dir.join("reference.json"))?)?;
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(clip_dir.join("config.json"))?)?;

        let mut vision = Session::builder()?.commit_from_file(clip_dir.join("vision.onnx"))?;
        let mut text = Session::builder()?.commit_from_file(clip_dir.join("text.onnx"))?;

        let tokenizer = ClipTokenizer::load(&models_dir.join("clip-tokenizer"))?;

        let prompts: Vec<String> = ref_json["texts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        let vio_labels: Vec<String> = ref_json["vio_labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        let safe_label = ref_json["safe_label"].as_str().unwrap_or("aman");

        // Compute normalized text features for all prompts (once).
        let encoded = tokenizer.encode_batch(
            &prompts.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        );
        let mut text_features = Vec::with_capacity(prompts.len());
        for (ids, mask) in encoded {
            let ids: Vec<i64> = ids.into_iter().map(|x| x as i64).collect();
            let mask: Vec<i64> = mask.into_iter().map(|x| x as i64).collect();
            let ids_t =
                ort::value::Tensor::from_array(([1usize, tokenizer.max_len], ids))?;
            let mask_t =
                ort::value::Tensor::from_array(([1usize, tokenizer.max_len], mask))?;
            let outputs = text.run(ort::inputs!["input_ids" => ids_t, "attention_mask" => mask_t])?;
            let feats = outputs[0].try_extract_tensor::<f32>()?.1.to_vec();
            text_features.push(feats);
        }

        Ok(ClipClassifier {
            vision,
            text,
            tokenizer,
            prompts,
            text_features,
            logit_scale: cfg["logit_scale"].as_f64().unwrap_or(100.0) as f32,
            vio_threshold: ref_json["vio_threshold"].as_f64().unwrap_or(0.30) as f32,
            viol_margin: ref_json["safe_margin"].as_f64().unwrap_or(2.0) as f32,
            n_vio: ref_json["n_vio"].as_u64().unwrap_or(20) as usize,
            vio_labels,
            safe_label: safe_label.to_string(),
            resize_filter,
        })
    }

    pub fn preprocess(&self, img: &RgbImage) -> (Vec<f32>, u32, u32) {
        clip_preprocess(img, self.resize_filter)
    }

    pub fn infer(&mut self, input: &[f32]) -> Result<ClipResult, Box<dyn std::error::Error>> {
        let tensor = ort::value::Tensor::from_array(([1usize, 3, 224, 224], input.to_vec()))?;
        let outputs = self.vision.run(ort::inputs!["pixel_values" => tensor])?;
        let image_features: Vec<f32> = outputs[0].try_extract_tensor::<f32>()?.1.to_vec();

        let n = self.text_features.len();
        let mut logits = Vec::with_capacity(n);
        for f in &self.text_features {
            logits.push(self.logit_scale * cosine(&image_features, f));
        }
        let probs = crate::yolo::softmax(&logits);

        let (vio_scores, safe_scores) = probs.split_at(self.n_vio);
        let vio_sum: f32 = vio_scores.iter().sum();
        let safe_sum: f32 = safe_scores.iter().sum();
        let best_vio = vio_scores.iter().cloned().fold(0.0f32, f32::max);
        let best_safe = safe_scores.iter().cloned().fold(0.0f32, f32::max);
        let violative = vio_sum >= self.vio_threshold && vio_sum >= safe_sum * self.viol_margin;

        let cls_name = if violative {
            let idx = vio_scores
                .iter()
                .position(|x| *x == best_vio)
                .unwrap_or(0);
            self.vio_labels[idx].clone()
        } else {
            self.safe_label.clone()
        };
        let conf = ((best_vio.max(best_safe) * 10000.0).round()) / 10000.0;

        Ok(ClipResult {
            logits,
            probs,
            n_vio: self.n_vio,
            vio_sum,
            safe_sum,
            best_vio,
            best_safe,
            cls_name,
            conf,
            violative,
        })
    }

    pub fn classify(&mut self, img: &RgbImage) -> Result<ClipResult, Box<dyn std::error::Error>> {
        let (input, _, _) = self.preprocess(img);
        self.infer(&input)
    }

    pub fn tokenizer_ref(&self) -> &ClipTokenizer {
        &self.tokenizer
    }

    pub fn prompt_refs(&self) -> Vec<&str> {
        self.prompts.iter().map(|s| s.as_str()).collect()
    }

    pub fn text_features_ref(&self) -> &[Vec<f32>] {
        &self.text_features
    }

    pub fn n_prompts(&self) -> usize {
        self.prompts.len()
    }
}