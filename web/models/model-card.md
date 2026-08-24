# Baloot Card Detector Model Card (v1.1.0 - 704px High Resolution)

## Model Summary

- **Model Architecture**: YOLO11n (Ultralytics)
- **Task**: Object detection for 32 Saudi Baloot playing cards + 1 `other` class (ranks 2–6)
- **Input Resolution**: 704 × 704 px (Stretch resized, 2.86× higher pixel density than 416px)
- **Export Format**: ONNX (opset 17, simplified, no embedded NMS)
- **Runtime Targets**: WebGPU (FP16) on iOS Safari & modern browsers; WASM (INT8/FP16) fallback

---

## Model Variants & Performance

| Format | File | Size | Primary Target | Expected Latency |
| :--- | :--- | :--- | :--- | :--- |
| **FP16 ONNX** | `web/models/baloot-v1.fp16.onnx` | 5.1 MB | WebGPU (iOS / Desktop) | ~20–40 ms |
| **INT8 ONNX** | `web/models/baloot-v1.int8.onnx` | 3.16 MB | WASM (CPU fallback) | ~110–220 ms |
| **PyTorch** | `runs/detect/runs/hakim/baloot/weights/best.pt` | 5.2 MB | Native Training / Evaluation | — |

### Evaluation Metrics (Holdout Validation Set at 704×704)

- **Precision (B)**: 99.97%
- **Recall (B)**: 100.0%
- **mAP@50 (B)**: 99.50%
- **mAP@50-95 (B)**: 83.25%
- **Per-Class Average**: >99.5% AP across all 32 Baloot cards and `other`

### First-Load Payload (gzip, as served by GitHub Pages)

Both variants ship. INT8 cannot run on WebGPU — the JSEP backend rejects `DequantizeLinear`
on the int32 conv bias ("no zero point") — so dropping FP16 would mean dropping GPU support
entirely and paying ~290 ms per detection instead of ~100 ms.

| Backend | Assets fetched | Total |
| :--- | :--- | :--- |
| WebGPU | `ort.webgpu.min.js` 0.08 + `*.jsep.mjs` 0.02 + `*.jsep.wasm` 4.93 + FP16 4.61 | **9.64 MB** |
| WASM | `ort.webgpu.min.js` 0.08 + `*.jsep.mjs` 0.02 + `*.jsep.wasm` 4.93 + INT8 1.99 | **7.02 MB** |

The WebGPU path is 1.6 MB over the ≤8 MB first-load budget. Accepted deliberately: the
service worker caches both `runtime/` and `models/` cache-first (they are content-versioned
by filename), so this is a one-time cost per client. The non-jsep ORT builds
(`ort-wasm-simd-threaded.wasm`/`.mjs`, 10.75 MB) were removed — `ort.webgpu.min.js` only ever
loads the jsep artifacts, on both backends.

---

## Dataset & Training Pipeline

1. **Source Dataset**:
   - 20,000 images from Roboflow cards dataset (14,000 train / 4,000 valid / 2,000 test).
   - 52 standard card classes remapped to 33 classes (32 Baloot cards + `other` class).
2. **Hygiene & Deduplication**:
   - Filtered boxes smaller than 6 px at base resolution.
   - Coordinates clipped to `[0.0, 1.0]`.
   - dHash perceptual deduplication pruned 35 near-duplicate leaked images from evaluation splits.
3. **Training Hyperparameters**:
   - **Hardware**: NVIDIA GeForce RTX 2080 Ti (11 GB VRAM)
   - **Resolution**: 704 × 704 px
   - **Batch size**: 32 with AMP (`amp=True`, `half=True`)
   - **Workers**: 8 (multiprocessing data loaders)
   - **Augmentations**:
     - `fliplr=0.0` & `flipud=0.0` (Cards are never mirrored — preserves rank glyph integrity)
     - `degrees=180.0` (Full rotational invariance)
     - `scale=0.5` & `hsv_v=0.4`
     - `close_mosaic=10` (Mosaic disabled in last 10 epochs for fine localization)

---

## INT8 Quantization Constraints

Static PTQ (QDQ, per-channel INT8 weights, UINT8 activations, MinMax calibration over 200
train images). Two settings are load-bearing — without either, the model still loads and
still emits plausible boxes, but class scores collapse and nothing clears the 0.35 threshold:

- **`reduce_range=True`.** The u8s8 convolution kernels accumulate into int16, which
  saturates on targets without VNNI — including the WASM SIMD build this variant exists for.
  Without it, peak confidence drops from ~0.90 to ~0.05.
- **Detection-head decode tail excluded from quantization.** The graph output is a `Concat`
  of box coordinates (0–704) and class probabilities (0–1). A single activation scale cannot
  span both: it is set by the coordinate range, so every probability quantizes to zero. The
  excluded nodes are derived from the graph at quantize time (`_decode_tail_nodes`), walking
  back from the output up to and including the first `Conv` on each path.

Calibration method is *not* a factor here — MinMax, Entropy, and Percentile all produce
identical results. Regenerate with `scripts/requantize_704_int8.py`.

### Verified parity against FP16 (real photos, conf ≥ 0.30)

| Photo | FP16 | INT8 |
| :--- | :--- | :--- |
| `hand-aces-kings.jpg` | 5 cards, 69.5–89.2% | same 5 cards, 57.6–87.0% |
| `hand-royal-hearts.jpg` | 7 boxes / 5 cards, 80.3–90.2% | same 7 boxes, 80.7–91.5% |

Boxes agree within 2 px. Both photos yield duplicate labels for some cards because labels
mark corner indices; `dedupeByCard` in `web/model-runner.js` collapses them.

---

## 33 Detection Classes

```
0: Ah, 1: Kh, 2: Qh, 3: Jh, 4: 10h, 5: 9h, 6: 8h, 7: 7h (Hearts)
8: Ad, 9: Kd, 10: Qd, 11: Jd, 12: 10d, 13: 9d, 14: 8d, 15: 7d (Diamonds)
16: Ac, 17: Kc, 18: Qc, 19: Jc, 20: 10c, 21: 9c, 22: 8c, 23: 7c (Clubs)
24: As, 25: Ks, 26: Qs, 27: Js, 28: 10s, 29: 9s, 30: 8s, 31: 7s (Spades)
32: other (Ranks 2–6) -> maps to `card: null` (rendered in UI as '؟')
```

---

## Known Limitations & Caveats

> [!WARNING]
> **Synthetic Dataset Domain Gap**: The dataset consists of synthetic composites (cards pasted onto backgrounds without human hands). In real-world photos where human fingers occlude corners or where lighting differs significantly, recall may drop. The UI maintains manual tap-to-correct for this reason.
