//! CLIP GPT2 byte-level BPE tokenizer, replicating transformers CLIPTokenizer.
//!
//! Vocabulary / merges are read from the artifacts produced by
//! tools/export_clip.py (vocab.json, merges.txt). Production behaviour matches
//! CLIPProcessor(text=..., padding=True):  bos prompt tokens ... eos, padded with
//! pad_id up to model_max_length (77).

use std::collections::HashMap;
use std::path::Path;

use regex::Regex;

// Same regex as transformers/tokenization_clip.py `self.pat`.
const PATTERN: &str = r#"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#;

pub struct ClipTokenizer {
    vocab: HashMap<String, u64>,
    ranks: HashMap<(String, String), usize>,
    regex: Regex,
    byte_encoder: Vec<char>, // index = byte 0..255
    pub bos_id: u64,
    pub eos_id: u64,
    pub pad_id: u64,
    pub max_len: usize,
}

fn bytes_to_unicode() -> Vec<char> {
    let mut bs: Vec<u32> = Vec::new();
    bs.extend((b'!'..=b'~').map(u32::from));
    bs.extend(0xA1..=0xAC);
    bs.extend(0xAE..=0xFF);
    let mut cs: Vec<char> = bs.iter().map(|&b| char::from_u32(b).unwrap()).collect();
    let mut n: u32 = 0;
    for b in 0_u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(char::from_u32(0x100 + n).unwrap());
            n += 1;
        }
    }
    let mut enc = vec!['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        enc[*b as usize] = *c;
    }
    enc
}

impl ClipTokenizer {
    /// Load from a directory containing vocab.json, merges.txt and config.json.
    pub fn load(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let vocab_raw: HashMap<String, u64> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("vocab.json"))?)?;
        let merges_raw = std::fs::read_to_string(dir.join("merges.txt"))?;
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;

        let mut ranks = HashMap::new();
        for (i, line) in merges_raw.lines().enumerate() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                ranks.insert((parts[0].to_string(), parts[1].to_string()), i);
            }
        }

        Ok(ClipTokenizer {
            vocab: vocab_raw,
            ranks,
            regex: Regex::new(PATTERN)?,
            byte_encoder: bytes_to_unicode(),
            bos_id: cfg["bos_id"].as_u64().unwrap_or(49406),
            eos_id: cfg["eos_id"].as_u64().unwrap_or(49407),
            pad_id: cfg["pad_id"].as_u64().unwrap_or(49407),
            max_len: cfg["model_max_length"].as_u64().unwrap_or(77) as usize,
        })
    }

    fn get_pairs(word: &[String]) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        for w in word.windows(2) {
            pairs.push((w[0].clone(), w[1].clone()));
        }
        pairs
    }

    fn bpe(&self, token: &str) -> String {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        let mut pairs = Self::get_pairs(&word);
        if pairs.is_empty() {
            return word.join(" ");
        }
        loop {
            let mut min_pair: Option<(String, String)> = None;
            let mut min_rank = usize::MAX;
            for p in &pairs {
                if let Some(&r) = self.ranks.get(p) {
                    if r < min_rank {
                        min_rank = r;
                        min_pair = Some(p.clone());
                    }
                }
            }
            let Some((first, second)) = min_pair else { break };
            let mut new_word: Vec<String> = Vec::new();
            let mut i = 0;
            while i < word.len() {
                if word[i] == first
                    && i + 1 < word.len()
                    && word[i + 1] == second
                {
                    new_word.push(format!("{first}{second}"));
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            word = new_word;
            if word.len() == 1 {
                break;
            }
            pairs = Self::get_pairs(&word);
        }
        word.join(" ")
    }

    /// Encode `text` to token ids + attention mask, both padded to max_len.
    pub fn encode(&self, text: &str) -> (Vec<u64>, Vec<u64>) {
        let mut ids: Vec<u64> = vec![self.bos_id];
        let mut mask: Vec<u64> = vec![1];

        for m in self.regex.find_iter(text) {
            let token = &text[m.start()..m.end()];
            if token == "<|startoftext|>" {
                ids.push(self.bos_id);
                mask.push(1);
                continue;
            }
            if token == "<|endoftext|>" {
                ids.push(self.eos_id);
                mask.push(1);
                continue;
            }
            let byte_encoded: String = token
                .as_bytes()
                .iter()
                .map(|&b| self.byte_encoder[b as usize])
                .collect();
            for piece in self.bpe(&byte_encoded).split(' ') {
                if let Some(&id) = self.vocab.get(piece) {
                    ids.push(id);
                    mask.push(1);
                }
            }
        }
        ids.push(self.eos_id);
        mask.push(1);

        if ids.len() < self.max_len {
            let pad = self.max_len - ids.len();
            ids.extend(std::iter::repeat(self.pad_id).take(pad));
            mask.extend(std::iter::repeat(0).take(pad));
        } else {
            ids.truncate(self.max_len);
            mask.truncate(self.max_len);
        }
        (ids, mask)
    }

    pub fn encode_batch(&self, texts: &[&str]) -> Vec<(Vec<u64>, Vec<u64>)> {
        texts.iter().map(|t| self.encode(t)).collect()
    }
}

pub fn special_tokens_of(text: &str) -> bool {
    text == "<|startoftext|>" || text == "<|endoftext|>"
}