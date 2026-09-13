//! Image loading + preprocessing faithful to the Python prototype.
//!
//! - YOLO engine  : ultralytics classify_transforms(224) = shortest-edge Resize(224)
//!                  (bilinear, antialias) -> CenterCrop(224) -> /255 (mean/std are
//!                  identity for ultralytics classification). The resize replicates
//!                  torchvision's antialiased bilinear (validated: cos 1.0000002 vs
//!                  torch on real images).
//! - CLIP engine  : CLIPProcessor = shortest-edge Resize(224, bicubic) -> CenterCrop(224)
//!                  -> /255 -> Normalize(CLIP mean/std) in RGB order.
//!
//! Residual divergence between Rust and the reference comes only from the image
//! decoders (Rust `image` crate vs Pillow's libjpeg-turbo), not from resampling.

use image::imageops::FilterType;
use image::{GenericImageView, ImageBuffer, Rgb, RgbImage};
use std::path::Path;

pub const IMAGE_SIZE: u32 = 224;

const CLIP_MEAN_RGB: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD_RGB: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

pub fn load_image_rgb(path: &Path) -> Result<RgbImage, Box<dyn std::error::Error>> {
    Ok(image::open(path)?.into_rgb8())
}

pub fn load_image_gray(path: &Path) -> Result<image::GrayImage, Box<dyn std::error::Error>> {
    Ok(image::open(path)?.to_luma8())
}

/// Shortest-edge resize preserving aspect ratio (rounds like torchvision).
pub fn resize_shortest_edge(
    img: &RgbImage,
    new_short: u32,
    filter: FilterType,
) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let shortest = w.min(h).max(1);
    let scale = new_short as f32 / shortest as f32;
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    image::imageops::resize(img, nw, nh, filter)
}

/// Center crop of `size` x `size`.
pub fn center_crop(img: &RgbImage, size: u32) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    if w == size && h == size {
        return img.clone();
    }
    let x0 = w.saturating_sub(size) / 2;
    let y0 = h.saturating_sub(size) / 2;
    let (cw, ch) = (size.min(w - x0), size.min(h - y0));
    let mut out = ImageBuffer::new(size, size);
    for (dx, dy, p) in img.view(x0, y0, cw, ch).pixels() {
        out.put_pixel(dx, dy, p);
    }
    for (x, y, p) in out.enumerate_pixels_mut() {
        if x >= cw || y >= ch {
            *p = Rgb([0, 0, 0]);
        }
    }
    out
}

/// Torchvision-style antialiased bilinear resize (F.interpolate, mode="bilinear",
/// align_corners=False, antialias=True), implemented separably. Validated against torch
/// on real images: cosine 1.0000002 (max abs diff 3.4e-5). Returns f32 RGB triplets in
/// row-major order at out_w x out_h (values in [0,255], unnormalized).
fn aa_bilinear_resize(img: &RgbImage, out_w: u32, out_h: u32) -> Vec<[f32; 3]> {
    fn weights(in_size: usize, out_size: usize) -> Vec<(Vec<usize>, Vec<f32>)> {
        let s = in_size as f32 / out_size as f32;
        let mut taps = Vec::with_capacity(out_size);
        for o in 0..out_size {
            let c = (o as f32 + 0.5) * s - 0.5;
            let mut lo = ((c - s).ceil() as isize).max(0) as usize;
            let mut hi = ((c + s).floor() as isize).min(in_size as isize - 1) as usize;
            if hi < lo {
                hi = lo;
            }
            let mut k: Vec<f32> = (lo..=hi)
                .map(|i| (s - (i as f32 - c).abs()).max(0.0))
                .collect();
            let sum: f32 = k.iter().sum();
            if sum > 0.0 {
                for v in k.iter_mut() {
                    *v /= sum;
                }
            }
            taps.push(((lo..=hi).collect(), k));
        }
        taps
    }

    let (iw, ih) = (img.width() as usize, img.height() as usize);
    let (ow, oh) = (out_w as usize, out_h as usize);
    let wx = weights(iw, ow);
    let wy = weights(ih, oh);
    let mut out = vec![[0f32; 3]; ow * oh];
    for oy in 0..oh {
        for ox in 0..ow {
            let acc = &mut out[oy * ow + ox];
            for (iy, ky) in wy[oy].0.iter().zip(wy[oy].1.iter()) {
                for (ix, kx) in wx[ox].0.iter().zip(wx[ox].1.iter()) {
                    let p = img.get_pixel(*ix as u32, *iy as u32);
                    let w = ky * kx;
                    acc[0] += p[0] as f32 * w;
                    acc[1] += p[1] as f32 * w;
                    acc[2] += p[2] as f32 * w;
                }
            }
        }
    }
    out
}

/// YOLO preprocess: shortest-edge 224 -> CenterCrop 224 -> /255.
/// Resize uses torchvision antialiased bilinear (the `filter` arg is ignored — kept for
/// CLI parity with the experimental filter variants).
/// Returns CHW NCHW flat buffer (1,3,224,224). mean/std are identity.
pub fn yolo_preprocess(
    img: &RgbImage,
    _filter: FilterType,
) -> (Vec<f32>, u32, u32) {
    let (w, h) = (img.width(), img.height());
    let shortest = w.min(h).max(1);
    let scale = IMAGE_SIZE as f32 / shortest as f32;
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    let resized = aa_bilinear_resize(img, nw, nh);
    let x0 = (nw - IMAGE_SIZE) / 2;
    let y0 = (nh - IMAGE_SIZE) / 2;
    let mut data = Vec::with_capacity(3 * IMAGE_SIZE as usize * IMAGE_SIZE as usize);
    for c in 0..3 {
        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                data.push(resized[(y0 + y) as usize * nw as usize + (x0 + x) as usize][c as usize] / 255.0);
            }
        }
    }
    (data, nw, nh)
}

/// CLIP preprocess: shortest-edge 224 (bicubic) -> CenterCrop 224 -> /255 -> CLIP norm.
pub fn clip_preprocess(
    img: &RgbImage,
    filter: FilterType,
) -> (Vec<f32>, u32, u32) {
    let resized = resize_shortest_edge(img, IMAGE_SIZE, filter);
    let (rw, rh) = (resized.width(), resized.height());
    let cropped = center_crop(&resized, IMAGE_SIZE);
    let mut data = Vec::with_capacity(3 * IMAGE_SIZE as usize * IMAGE_SIZE as usize);
    for c in 0..3 {
        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                let p = cropped.get_pixel(x, y);
                let v = p[c] as f32 / 255.0;
                data.push((v - CLIP_MEAN_RGB[c]) / CLIP_STD_RGB[c]);
            }
        }
    }
    (data, rw, rh)
}