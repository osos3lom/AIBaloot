# Contributing to Hakim Vision

Thanks for thinking about contributing. This repository is `hakim-vision` — the synthetic-data and computer-vision pillar of the broader [Hakim Baloot AI platform](https://github.com/osos3lom/AIBaloot). It generates labeled playing-card scenes for training detectors and (eventually) powers the AR companion in the Hakim mobile app.

## Quick start

```bash
# 1. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh        # macOS / Linux
# powershell: irm https://astral.sh/uv/install.ps1 | iex

# 2. Clone and sync the environment
git clone https://github.com/osos3lom/AIBaloot.git
cd AIBaloot
uv sync --all-extras

# 3. Install pre-commit hooks
uv run pre-commit install

# 4. Sanity check
uv run pytest -q
uv run ruff check .
uv run mypy
```

## Ground rules

1. **No new code in notebooks.** Notebooks are for exploration only and live in `notebooks/`. Anything that needs to be tested, reviewed, or shipped goes in `src/hakim_vision/`.
2. **Type everything.** `mypy --strict` is the floor; PRs that loosen it need justification in the description.
3. **Test what you ship.** A PR that adds behavior without a test will not be merged unless the behavior is genuinely untestable (e.g., a one-off migration script).
4. **No `pickle` on disk.** Use `webdataset`, `parquet`, `npz`, or `safetensors`. Pickle is an RCE risk on any shared artifact.
5. **No secrets in commits.** Gitleaks runs in CI; the pre-commit hook will catch most cases.

## Branching & commits

- Branch from `master` as `feat/<short-name>`, `fix/<short-name>`, `chore/<short-name>`, or `docs/<short-name>`.
- Conventional Commits encouraged: `feat:`, `fix:`, `refactor:`, `perf:`, `test:`, `docs:`, `chore:`.
- One logical change per PR. Easier to review, easier to revert.

## What we welcome

- Replacing the legacy notebook flow with modular functions in `src/hakim_vision/`.
- Migrating augmentation from `imgaug` (unmaintained) to `albumentations`.
- Replacing pickle artifacts with `webdataset` shards.
- Modernizing the detector (YOLOv3 → YOLO11 / RT-DETRv2) and adding ONNX/CoreML/TFLite export.
- Auto-labeling pipelines using Grounding DINO + SAM 2 to bootstrap real-world data.
- Synthetic scene generation via Kubric (Blender) for photorealistic training data.
- Bug reports with reproducible failing tests.

## What we do not want

- New notebooks added to `src/`.
- Loosening typing or test coverage without discussion.
- Dependencies on closed-source SaaS where an OSS equivalent exists.
- Code copied from any closed-source Baloot app — clean-room only.

## Communication

- Open an issue before large PRs so we can align on direction.
- Bigger discussions happen on Discord (link in README) and GitHub Discussions.

## Licensing

By contributing, you agree your contribution is licensed under the MIT License of this repository.
