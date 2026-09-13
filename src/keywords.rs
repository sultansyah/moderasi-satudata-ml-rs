//! Keyword filtering + context gate, faithful port of moderasi-satudata-ml/moderasi.py
//! and keywords.py (KEYWORDS_ALL, PUBLIC_INTEREST_KEYWORDS, TRANSACTION_KEYWORDS,
//! ABORTION_AD_SIGNAL_GROUPS, CONTACT_RE).

use regex::Regex;
use unicode_normalization::UnicodeNormalization;

pub const KEYWORDS_ABORSI: &[&str] = &[
    "aborsi",
    "penggugur",
    "gugur kandungan",
    "pil aborsi",
    "obat aborsi",
    "obat penggugur",
    "jual obat aborsi",
    "telat datang bulan",
    "terlambat datang bulan",
    "pelancar haid",
    "pelancar mens",
    "obat telat bulan",
    "obat telat datang bulan",
    "penggugur janin",
    "penggugur kandungan",
    "aborsi aman",
    "aborsi tuntas",
    "aborsi secara alami",
    "cytotec",
    "misoprostol",
    "mifepristone",
    "gastrul",
    "prostadel",
    "aborsiva",
    "kuretase",
    "jual cytotec",
    "obat kuret",
    "penggugur kehamilan",
];

pub const KEYWORDS_BORAKS: &[&str] = &[
    "boraks", "borax", "bleng", "bahan pengenyal", "pengenyal bakso", "boraks makanan",
    "jual boraks", "bubuk boraks", "kristal boraks", "formalin", "pengawet mayat",
];

pub const KEYWORDS_UMUM: &[&str] = &[
    "100% original",
    "100% ampuh",
    "order langsung",
    "wa.me",
    "whatsapp",
    "cod",
    "dijamin tuntas",
    "ampuh tuntas",
    "rasa aman",
];

pub const KEYWORDS_JUDI: &[&str] = &[
    // Indonesia
    "judi online", "situs judi", "bandar judi", "agen judi", "judi bola", "taruhan online",
    "judol", "judi slot", "slot gacor", "gacor", "maxwin", "link slot", "situs slot",
    "daftar slot", "slot online", "slot demo", "idn poker", "poker online", "domino qq",
    "baccarat", "roulette", "sicbo",
    // Vietnam (ASCII)
    "danh bac", "danh bai", "ca cuoc", "ca do", "nha cai", "lo de", "xo so", "no hu",
    "quay hu", "ban ca", "doi thuong", "tai xiu", "xoc dia", "co bac", "bau cua", "da ga",
    "song bai", "game bai",
    // brand
    "w88", "m88", "fun88", "f88", "188bet", "1xbet", "bet365", "sbobet", "sbotop", "kubet",
    "jun88", "vn88", "shbet", "hi88", "new88", "ok9", "78win", "bk8", "debet", "fcb8",
    "8xbet", "bwing", "tf88", "v9bet", "dabet", "may88", "rikvip", "go88", "kingfun",
    "red88", "ee88", "zbet", "12bet", "letou", "win55", "fb88", "cmd368", "oppabet",
    "789bet",
    // games
    "bet", "zeus", "olympus", "sweet bonanza", "bonanza", "starlight princess",
    "mahjong ways", "sugar rush", "big bass", "wolf gold", "aztec gems", "book of dead",
    "wisdom of athena", "gates of hades", "fruit party", "pragmatic", "jili", "slot88",
    "cq9", "habanero", "pussy888", "mega888", "pg soft", "ante bet",
];

pub fn keywords_all() -> Vec<&'static str> {
    let mut all: Vec<&'static str> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for kw in [
        KEYWORDS_ABORSI.iter(),
        KEYWORDS_BORAKS.iter(),
        KEYWORDS_UMUM.iter(),
        KEYWORDS_JUDI.iter(),
    ]
    .into_iter()
    .flatten()
    {
        if seen.insert(*kw) {
            all.push(*kw);
        }
    }
    all
}

pub const PUBLIC_INTEREST_KEYWORDS: &[&str] = &[
    "berita", "berita nasional", "skandal", "siber", "cyber", "spam", "pemerintah", "kominfo",
    "investigasi", "keamanan data", "dipertanyakan", "publik resah", "desakan", "peringatan",
    "waspada", "hati hati", "bahaya", "dampak", "edukasi", "sosialisasi", "kampanye",
    "pencegahan", "cegah", "larangan", "anti judi", "jangan judi", "jangan biarkan",
    "berhenti sekarang", "konseling", "adiksi", "lapor", "imbauan", "edaran", "informasi",
];

pub const TRANSACTION_KEYWORDS: &[&str] = &[
    "jual", "beli", "order", "pesan", "pemesanan", "stok", "ready", "harga", "promo",
    "diskon", "wa", "whatsapp", "cod", "kontak", "hubungi", "nomor", "daftar", "deposit",
    "bonus", "link", "slot gacor", "gacor", "maxwin", "terpercaya", "tuntas", "ampuh",
    "terbatas",
];

pub const STRONG_PUBLIC_MARKERS: &[&str] = &[
    "berita",
    "berita nasional",
    "peringatan",
    "edukasi",
    "sosialisasi",
    "kampanye",
    "pencegahan",
    "anti judi",
    "jangan judi",
    "berhenti sekarang",
    "konseling",
    "investigasi",
    "kominfo",
    "pemerintah",
];

pub const ABORTION_AD_SIGNAL_GROUPS: &[&str] = &[
    "masalah", "hamil", "risiko", "klaim",
];

pub const ABORTION_AD_SIGNAL_PHRASES: &[(&str, &[&str])] = &[
    ("masalah", &["mengatasi masalah", "mengatasi masalah anda"]),
    ("hamil", &["bisa hamil kembali", "hamil kembali"]),
    ("risiko", &["aman tanpa resiko", "aman tanpa risiko", "tanpa resiko", "tanpa risiko"]),
    ("klaim", &["100 aman", "100 ampuh", "dijamin aman", "dijamin tuntas"]),
];

fn regex_diacritics() -> Regex {
    // compiled lazily via OnceCell-like static
    static NON_ALNUM: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    NON_ALNUM
        .get_or_init(|| Regex::new(r"[^a-z0-9\s]").unwrap())
        .clone()
}

fn regex_marks() -> Regex {
    static MARKS: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    MARKS
        .get_or_init(|| Regex::new(r"\p{M}").unwrap())
        .clone()
}

/// Mirrors moderasi.normalize(): lowercase, strip Vietnamese diacritics (NFKD),
/// drop non [a-z0-9\s], collapse whitespace.
pub fn normalize(s: &str) -> String {
    let lower = s.to_lowercase();
    let nfkd: String = lower.nfkd().collect();
    let no_marks = regex_marks().replace_all(&nfkd, "");
    let mut cleaned = String::with_capacity(no_marks.len());
    for c in no_marks.chars() {
        match c {
            'đ' => cleaned.push('d'),
            'ư' => cleaned.push('u'),
            'ơ' => cleaned.push('o'),
            _ => cleaned.push(c),
        }
    }
    let replaced = regex_diacritics().replace_all(&cleaned, " ");
    replaced.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Mirrors moderasi._phrase_hits().
pub fn phrase_hits(text_normalized: &str, phrases: &[&str]) -> Vec<String> {
    let words: std::collections::HashSet<&str> =
        text_normalized.split_whitespace().collect();
    let mut hits = Vec::new();
    for phrase in phrases {
        let n = normalize(phrase);
        if n.is_empty() {
            continue;
        }
        if n.chars().count() <= 3 {
            if words.contains(n.as_str()) {
                hits.push(phrase.to_string());
            }
        } else if text_normalized.contains(&n) {
            hits.push(phrase.to_string());
        }
    }
    hits
}

/// Mirrors moderasi.keyword_hits() (uses KEYWORDS_ALL).
pub fn keyword_hits(text_normalized: &str) -> Vec<String> {
    // NOTE: len <= 3 must be matched as a whole token (word boundary), longer as substring.
    phrase_hits(text_normalized, &keywords_all())
}

/// Mirrors moderasi._public_interest_context().
pub fn public_interest_context(
    text_normalized: &str,
) -> (bool, Vec<String>, Vec<String>) {
    let public_hits = phrase_hits(text_normalized, PUBLIC_INTEREST_KEYWORDS);
    let transaction_hits = phrase_hits(text_normalized, TRANSACTION_KEYWORDS);
    let has_strong_public = public_hits
        .iter()
        .any(|h| STRONG_PUBLIC_MARKERS.contains(&h.as_str()));
    let is_public_interest = public_hits.len() >= 2 && has_strong_public;
    (is_public_interest, public_hits, transaction_hits)
}

fn contact_re() -> Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\b(?:wa|whatsapp|hubungi|kontak|telp|telepon|sms)\b|(?:\+?62|0)8\d{7,12}\b")
            .unwrap()
    })
    .clone()
}

/// Mirrors moderasi._covert_abortion_ad_hits().
pub fn covert_abortion_ad_hits(
    text_normalized: &str,
    transaction_hits: &[String],
) -> Vec<String> {
    let contact_hit = contact_re().is_match(text_normalized);
    if !(contact_hit || !transaction_hits.is_empty()) {
        return Vec::new();
    }
    let mut grouped: std::collections::HashMap<&str, Vec<String>> = Default::default();
    for (group, phrases) in ABORTION_AD_SIGNAL_PHRASES {
        let hits = phrase_hits(text_normalized, phrases);
        if !hits.is_empty() {
            grouped.insert(group, hits);
        }
    }
    if !grouped.contains_key("hamil") || grouped.len() < 2 {
        return Vec::new();
    }
    let mut hits = vec!["indikasi iklan aborsi terselubung".to_string()];
    for group in ABORTION_AD_SIGNAL_GROUPS {
        if let Some(g) = grouped.get(group) {
            hits.extend(g.iter().take(2).cloned());
        }
    }
    if contact_hit {
        hits.push("kontak/nomor/wa".to_string());
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_and_hits() {
        assert_eq!(normalize("JUAL OBAT Aborsi"), "jual obat aborsi");
        assert_eq!(normalize("Đánh bạc!"), "danh bac");
        assert_eq!(normalize("  ROKOK& alkohol,."), "rokok alkohol");
        let hits = keyword_hits(&normalize("jual obat aborsi"));
        assert!(hits.contains(&"jual obat aborsi".to_string()));
    }

    #[test]
    fn short_keywords_whole_word_only() {
        // "bet" must not match inside "alphabet"
        let hits = keyword_hits("alphabet zeus");
        assert!(!hits.iter().any(|h| h == "bet"));
        assert!(hits.contains(&"zeus".to_string()));
    }
}