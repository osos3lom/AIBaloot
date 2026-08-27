# Baloot Card Detector: Architecture & Pipeline

## Overview

The `hakim-vision` card detector is a high-performance in-browser object detector built with YOLO11 and exported to ONNX for client-side WebGPU (FP16) and WebAssembly (INT8/FP16) inference.

It detects and classifies all 32 Saudi Baloot playing cards (ranks A, K, Q, J, 10, 9, 8, 7 in hearts `h`, diamonds `d`, clubs `c`, spades `s`) plus an `other` class representing ranks 2–6.

---

## 1. Model & Dataset Specifications

- **Classes (33)**: 32 canonical Baloot cards + 1 `other` class (ranks 2–6).
- **Resolution**: 416 × 416 px with stretch resize (no letterbox, preserving training domain alignment).
- **Labels**: Corner indices (median box ~26 × 26 px), matching human fanned card hands.
- **Targets**:
  - **FP16 ONNX**: Primary target on WebGPU (iOS Safari 26+, Chrome 113+, Edge).
  - **INT8 ONNX**: Static quantized fallback for WASM / CPU Execution Provider.

---

## 2. Pipeline Stages

### Phase 1 — Dataset Preparation & Hygiene
1. **Inspection**: `hakim-vision dataset inspect --source <dir>` flags orphan, empty, or out-of-range labels.
2. **Remapping**: `hakim-vision dataset remap --source <dir> --output data/baloot-dataset --unmapped other` collapses ranks 2–6 into `other` (class 32) and discards boxes < 6 px.
3. **Deduplication**: `hakim-vision dataset dedupe --source <dir> --threshold 4 --drop-leaks` detects perceptual leaks across splits using dHash (difference hash).
4. **Visual Preview**: `hakim-vision dataset preview --source <dir> --output data/preview` validates bounding boxes and class names before training.

### Phase 2 — Training
- `hakim-vision train --data data/baloot-dataset/data.yaml --model yolo11n.pt --imgsz 416`
- **Key Hyperparameters**:
  - `fliplr=0.0` & `flipud=0.0`: Prevents mirrored rank glyphs.
  - `degrees=180.0`: Full rotational invariance.
  - `close_mosaic=10`: Disables mosaic augmentation in the final 10 epochs for sharp localization.

### Phase 3 — Export & Static Quantization
1. **FP16 Export**: `export_yolo_to_onnx(model, half=True, imgsz=416, opset=17, simplify=True)`. NMS is omitted from the graph to maximize WebGPU shader compatibility.
2. **INT8 Static Quantization**: `quantize_onnx_static(fp32_model, output, calib_images)` using ONNX Runtime MinMax calibration with representative training images.
3. **Parity Check**: Evaluate both models against the holdout test set to ensure mAP@50 delta stays within target threshold.

### Phase 4 — Client Runtime & Browser Execution
- **`web/model-runner.js`**: Lazy-loads `ort.webgpu.min.js` on first user interaction, initializes session, warms shaders with a dummy tensor, runs stretch preprocessing, and decodes the `1×37×3549` tensor with IoU-based NMS.
- **`web/detect.js`**: Connects via `HakimDetector.registerDetector()`, falling back to heuristic `findCardRegions()` if WebGPU/WASM is unsupported.
- **`web/sw.js`**: Service Worker caches model weights and WASM runtimes for instant offline loading.

---

## 5. v2 Retraining: Fanned Hands

The v1 dataset is 2–3 card composites with at most 10% corner occlusion. A held
Baloot hand is 8–13 cards sharing one pivot, where each card covers all but a
narrow strip of the one behind it and fingers cover the bottom — so most cards
are represented by a rank/suit index alone, sometimes a partial one. That gap is
what the "Synthetic Dataset Domain Gap" warning in the model card describes.

`scripts/train_fanned_detector.py` retrains against that geometry:

```bash
python scripts/train_fanned_detector.py all --device 0
```

| Stage | What it does |
| :--- | :--- |
| `atlas` | Downloads a slice of HuggingFace `JackFurby/playing-cards` and warps each annotated card quad back onto the canonical 228×348 card, producing 52 BGRA templates. Only the images used are fetched — a few hundred, not the dataset's 23 GB. |
| `preview` | Renders annotated scenes to `preview/` so labels can be checked by eye. A geometry bug here is invisible in the loss curve and obvious in one picture. |
| `dataset` | Renders fanned hands in parallel (`--jobs`), emits YOLO labels in the same 33-class scheme, and optionally folds in existing datasets with `--merge`. |
| `train` | YOLO11 at 704 px with `fliplr=0`, `flipud=0`, `degrees=180`, `scale=0.6`, and mosaic closed for the final epochs. |
| `export` | FP16 + INT8 ONNX plus a `model.json` matching `web/models/`. |

### Label semantics

Corner-index quads come from `REF_CORNER_HL` / `REF_CORNER_LR` projected through
each card's placement homography. Both corners of every card are candidates; each
is tested against the union of the cards drawn on top of it and the hand, and

* survives only if `--min-visible` (default 0.45) of its area is still visible, and
* is boxed around the **visible remainder**, not the whole index.

That is looser than v1's 10% occlusion cut, deliberately: a fan whose labels only
covered near-unoccluded indices would teach the model nothing about the case that
currently fails.
