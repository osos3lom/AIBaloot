# hakim-vision | استوديو حكيم للرؤية الحاسوبية

[![CI](https://github.com/osos3lom/AIBaloot/actions/workflows/ci.yml/badge.svg)](https://github.com/osos3lom/AIBaloot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![UI: Arabic First](https://img.shields.io/badge/UI-Arabic%20First%20%D8%A7%D9%84%D8%B9%D8%B1%D8%A8%D9%8A%D8%A9-10b981.svg)](web/index.html)

> 🇸🇦 **منظومة الرؤية الحاسوبية والذكاء الاصطناعي للعبة البلوت السعودي**
> توليد بيانات تركيبية، استكشاف وتحديد أوراق اللعب، وتدريب نماذج التعلم العميق في المتصفح عبر WebGPU.
>
> Synthetic playing-card dataset generation and detector training pipeline.
> The computer-vision pillar of the **Hakim** open-source Baloot AI platform.

This repository — originally named `AIBaloot` and forked from [`geaxgx/playing-card-detection`](https://github.com/geaxgx/playing-card-detection) — is being modernized as `hakim-vision`: one of four pillars of [Hakim](#about-hakim), a research-credible, OSS-first Baloot AI platform.

The job of `hakim-vision` is to:

1. **Generate** labeled synthetic scenes of playing cards on textured backgrounds, suitable for training modern object detectors.
2. **Train** a card detector (YOLO11 / RT-DETRv2) on those scenes.
3. **Export** quantized models (ONNX, CoreML, TFLite) for the Hakim mobile AR companion that overlays Baloot strategy hints on a physical card table.

## Status

The original 2018 notebook has been replaced by a typed, tested, containerized Python package built around the 32-card Baloot deck. See [the roadmap](#roadmap) for what's next.

| Surface | State |
|---|---|
| Modern Python toolchain (`uv`, `ruff`, `mypy --strict`, `pytest`) | ✅ |
| Pinned, modern deps (OpenCV 4.10, NumPy 2; dropped `imgaug`) | ✅ |
| CI: lint / type-check / test (Linux + macOS + Windows) / Docker | ✅ |
| Multi-stage Dockerfile, non-root runtime | ✅ |
| Notebook → `src/hakim_vision/` modules | ✅ |
| Pickle → tar shards | ✅ |
| 32-card Baloot deck across package & studio | ✅ |
| Player-facing hand-value app (`web/index.html`) | ✅ |
| Static browser dataset studio (`web/studio.html`) | ✅ |
| Baloot scoring engine + tests (`web/scoring.js`) | ✅ |
| Card **region** proposals in-browser (`web/detect.js`) | ✅ |
| Bring-your-own-dataset: inspect, remap to 32 Baloot classes, train, export | ✅ |
| Local job API so the studio runs real training (`hakim-vision studio`) | ✅ |
| Rank/suit recognition model (regions are named manually until one is trained) | ⏳ train it with the pipeline below |
| RT-DETRv2 + CoreML/TFLite export | ⏳ planned |
| Gradio demo on Hugging Face Spaces | ⏳ planned |

## Quick start

### 1. Zero-Install Web Surfaces (Interactive In-Browser)

| Page | For | What it does |
|---|---|---|
| `web/index.html` | players | Photograph the cards in your hand, confirm what Hakim found, and read the hand's value in Sun or Hokum, projects included. |
| `web/studio.html` | dataset work | Synthetic scenes and box annotation, plus the dataset → training → export lab described below. |

Open either file directly, or serve both with the CLI:
```bash
uv run hakim-vision studio
```

**How much of the "AI" is real today:** `detect.js` finds *where* card-shaped
regions are, entirely in-browser and with no model download. It does not yet
name the rank and suit — that needs the trained detector on the roadmap — so
the app asks you to confirm each card instead of guessing. Register a model
with `HakimDetector.registerClassifier()` and the same UI fills the cards in
automatically. Scoring is exact either way, and covered by tests:

```bash
node --test web/scoring.test.js
```

### 2. Train a detector on your own dataset

The studio's lab takes a third-party card dataset and walks it to a trained,
browser-ready model. Every step is also a CLI command, so nothing is trapped in
the UI.

```bash
# 0. Get a dataset (this one is 52 poker classes; Baloot needs 32).
scripts/fetch_kaggle_dataset.sh
# or: curl -L -o playing-cards.zip #       https://www.kaggle.com/api/v1/datasets/download/andy8744/playing-cards-object-detection-dataset

# 1. What is actually in it: images, boxes, classes, and label problems.
uv run hakim-vision dataset inspect --source data/downloads/playing-cards-object-detection-dataset

# 2. Map it onto the 32-card Baloot deck (drops 2-6, re-indexes the rest).
uv run hakim-vision dataset remap   --source data/downloads/playing-cards-object-detection-dataset   --output data/baloot-dataset

# 3. Train (needs the training extra: uv sync --extra train).
uv run hakim-vision train --data data/baloot-dataset/data.yaml --model yolo11n.pt --epochs 100

# 4. Export for the browser.
uv run hakim-vision export --weights runs/hakim/baloot/weights/best.pt
```

Or do all four in the studio:

```bash
uv run hakim-vision studio
```

That serves both pages **and** a loopback job API, so the studio can inspect
folders, build the Baloot dataset, start and stop training, and stream the real
`results.csv` metrics into its chart. The URL it prints carries a per-run token;
the API is bound to `127.0.0.1`, requires that token, and refuses cross-origin
requests — it runs training commands, so do not expose it. Opened as a plain
file instead, the studio still inspects a folder you pick in the browser and
hands you the commands to run yourself.

Once exported, wire the model into the player app so cards are named
automatically instead of by hand:

```js
HakimDetector.registerClassifier(async (imageData, region) => {
  // run your ONNX session, then:
  return { card: "Ah", confidence: 0.93 };
});
```

### 3. Python Package & CLI
```bash
# Install uv and sync dependencies
uv sync --all-extras

# Run tests and check version
uv run pytest -q
uv run hakim-vision version
uv run hakim-vision config-show
```

Run with Docker:

```bash
docker build -t hakim-vision .
docker run --rm hakim-vision hakim-vision version
```

## What's inside

```
.
├── src/hakim_vision/         # The package (typed, tested)
│   ├── cli.py                # `hakim-vision` CLI (typer: generate, studio, pack)
│   ├── config.py             # Pydantic settings (HAKIM_VISION_* env vars)
│   ├── geometry.py           # YOLO/VOC box conversion
│   ├── datasets/             # Discover, inspect, and remap third-party datasets
│   ├── models/               # Training (Ultralytics) and ONNX export
│   ├── jobs.py               # Subprocess job runner for the studio
│   ├── server.py             # Loopback JSON API behind the studio
│   └── synthetic/            # Scene generation, asset loaders, augmentation
├── web/                      # Zero-install browser surfaces
│   ├── index.html            # Player app: photo -> cards -> hand value
│   ├── app.js                # App controller (capture, review, score)
│   ├── scoring.js            # Baloot scoring engine (pure, tested)
│   ├── scoring.test.js       # node --test web/scoring.test.js
│   ├── detect.js             # Card-region proposals + classifier seam
│   ├── i18n.js               # Arabic-first bilingual strings
│   └── studio.html/.js       # Dataset studio: synth scenes, annotation, YOLO export
├── tests/                    # pytest suite
├── scripts/                  # Dataset download helpers
├── data/cards.names          # 32 Baloot card-class names (Ah, 7s, …)
├── docs/roadmap.md           # 12-month technical roadmap
├── Dockerfile                # multi-stage, non-root
└── .github/workflows/ci.yml  # CI
```

## About Hakim

`hakim-vision` is one of four planned pillars of **Hakim** (Arabic: حكيم, "the sage") — an open, free, Arabic-first Baloot AI platform:

| Pillar | Role |
|---|---|
| `hakim-engine` | Authoritative Baloot rules + scoring (Python + Rust/WASM) |
| `hakim-agent` | Self-play RL + search agents (ISMCTS → Deep CFR → ReBeL-style) |
| `hakim-coach` | Arabic LLM commentary, replay analysis, voice play-by-play |
| **`hakim-vision`** | **This repo.** Card detection + AR mobile companion for physical Baloot tables |

The strategic bet: Baloot is the dominant card game in the Gulf (60M+ population, tens of millions of players), and no public research-grade engine exists. Hakim aims to be Lichess + Stockfish + an Arabic AI coach, for Baloot.

## Roadmap

See [`docs/roadmap.md`](docs/roadmap.md) for the detailed 12-month plan. Headline phases:

- **Weeks 1–2 — Quick wins:** modernize toolchain, kill pickle, ship a Gradio demo. _(in progress)_
- **Months 1–3 — Mid-term:** YOLO11 / RT-DETRv2, auto-labeling with Grounding DINO + SAM 2, Kubric-based photorealistic synthesis.
- **Months 3–12 — Long-term:** on-device CoreML / TFLite export, AR companion in Hakim mobile, real-time inference < 30 ms on iPhone.

## Contributing

We want help. Start with [`CONTRIBUTING.md`](CONTRIBUTING.md) and look for issues tagged `good first issue`. Security reports go to [`SECURITY.md`](SECURITY.md).

## Credits

This project began as a fork of the excellent [`geaxgx/playing-card-detection`](https://github.com/geaxgx/playing-card-detection) by Géraud Cardona Gimenez. The original synthetic-data trick (cards composited onto VGG's [Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/)) is preserved while the surrounding engineering is modernized.

## License

MIT — see [LICENSE](LICENSE).
