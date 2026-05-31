# Migrating from the legacy notebook to `hakim-vision`

The original repository was a single Jupyter notebook
(`creating_playing_cards_dataset.ipynb`) that loaded pickled asset bundles and
serialized generated scenes to a working directory. That notebook has been
removed; the entire pipeline now lives in `src/hakim_vision/`. This page is
the conversion guide for anyone with legacy artifacts on disk.

## What changed

| Concern | Legacy (2018 notebook) | Modern (`hakim-vision`) |
|---|---|---|
| Asset format | `data/backgrounds.pck`, `data/cards.pck` (pickle) | Plain tar shards (`backgrounds-NNNN.tar`, `cards-NNNN.tar`) loaded via stdlib `tarfile` |
| Background source | Runtime `!wget` of DTD | Pre-packed shards (offline or HF Hub cached) |
| API surface | Globals + cells | `hakim_vision.synthetic.{Backgrounds, Cards, extract_card, render_random_scene, …}` |
| Randomness | `random.random`, `random.randint` (global state) | `numpy.random.Generator` injected per loader |
| Augmentation | `imgaug` (unmaintained since 2020) | OpenCV affine via `random_affine_card`; photometric/elastic layers can be added with `albumentations` later |
| OpenCV API | 3.x `findContours` signature (`(_, cnts, _)`) | 4.x signature (`(cnts, hierarchy)`) |
| Image typing | `np.int0`, `np.int` (removed in NumPy 2) | `np.intp`, builtin `int` |
| Display side effects | `cv2.imshow` baked into `extract_card` | Pure functions; callers display themselves |
| Deck | 52-card standard | 36-card Baloot (A, K, Q, J, 10, 9, 8, 7, 6 × 4 suits) |
| Tests | None | `pytest` suite under `tests/` |

## Migrating an existing dataset

If you already produced assets with the legacy notebook, you can convert them
in two steps. The migration is one-way; nothing in `hakim-vision` reads pickle.

### 1. Backgrounds

The legacy `data/backgrounds.pck` is a pickled `list[np.ndarray]` of BGR
images. Unpickle it once in a trusted shell (pickle is RCE — do this on a
machine you own, from a `.pck` you produced), dump each image as a PNG, then
pack:

```bash
# Untrusted .pck files: DO NOT do this. Re-generate from the textures source.
python -c "
import pickle, cv2, os
images = pickle.load(open('data/backgrounds.pck','rb'))
os.makedirs('data/backgrounds_png', exist_ok=True)
for i, img in enumerate(images):
    cv2.imwrite(f'data/backgrounds_png/{i:06d}.png', img)
"

uv run hakim-vision pack-backgrounds data/backgrounds_png data/shards \
    --shard-size 1024 --prefix backgrounds
```

For a clean start, point `pack-backgrounds` at an unpacked DTD instead:

```bash
# Pre-cached DTD (replace with your local copy of the Describable Textures Dataset).
uv run hakim-vision pack-backgrounds path/to/dtd/images data/shards
```

### 2. Cards

The legacy `data/cards.pck` is a `dict[card_name -> list[(image, hull_hl, hull_lr)]]`.
The recommended migration path is to re-extract from the original card videos
(if you still have them):

```bash
# (Coming in a follow-up PR — replaces extract_cards_from_video.)
# uv run hakim-vision extract-cards data/video data/cards_root

uv run hakim-vision pack-cards data/cards_root data/shards \
    --shard-size 512 --prefix cards
```

If you only have the `.pck`, unpickle it on a trusted machine and write each
`(image, hull_hl, hull_lr)` triple to disk in the layout
`{root}/{card_name}/{NN}.png`, then run `pack-cards` with `--no-hulls` (hulls
will be re-extracted on the fly during packing).

## Loading the new shards

```python
import numpy as np
from pathlib import Path

from hakim_vision.synthetic import Backgrounds, Cards

shards_dir = Path("data/shards")
bg_shards = sorted(shards_dir.glob("backgrounds-*.tar"))
card_shards = sorted(shards_dir.glob("cards-*.tar"))

rng = np.random.default_rng(42)
backgrounds = Backgrounds(bg_shards, rng=rng)
cards = Cards(card_shards, rng=rng)

print(f"{len(backgrounds)} backgrounds, {len(cards)} cards across "
      f"{len(cards.class_names)} classes")  # 36 for a packed Baloot deck

card = cards.sample("Ah")
bg = backgrounds.sample(size=720)
```

## What's still missing (roadmap)

- `extract-cards` CLI to replace the notebook's video-frame capture loop.
- HF Datasets-cached DTD release so users don't need to source backgrounds
  themselves.
- Photometric/elastic augmentation pass (`albumentations`) on top of the
  current affine pipeline.
- Modern detector training (YOLO11 / RT-DETRv2) on the new shards.

See [the project plan](https://github.com/osos3lom/AIBaloot) for the full
12-month roadmap.
