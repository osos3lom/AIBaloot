# hakim-vision

[![CI](https://github.com/osos3lom/AIBaloot/actions/workflows/ci.yml/badge.svg)](https://github.com/osos3lom/AIBaloot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

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
| 32-card Baloot deck across package + tester notebook | ✅ |
| YOLO11 / RT-DETRv2 + ONNX/CoreML/TFLite export | ⏳ planned |
| Gradio demo on Hugging Face Spaces | ⏳ planned |

## Quick start

```bash
# 1. Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh                 # macOS / Linux
# powershell: irm https://astral.sh/uv/install.ps1 | iex

# 2. Clone + sync
git clone https://github.com/osos3lom/AIBaloot.git
cd AIBaloot
uv sync --all-extras

# 3. Sanity check
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
│   ├── cli.py                # `hakim-vision` CLI (typer)
│   ├── config.py             # Pydantic settings (HAKIM_VISION_* env vars)
│   ├── geometry.py           # YOLO/VOC box conversion
│   └── synthetic/            # Scene generation, asset loaders, augmentation
├── tests/                    # pytest suite
├── baloot_dataset_tester_workflow.ipynb   # tester smoke-test for the Baloot pipeline
├── data/cards.names          # 32 Baloot card-class names (Ah, 7s, …)
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
