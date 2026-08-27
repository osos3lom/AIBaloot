"""Turn the HuggingFace ``JackFurby/playing-cards`` dataset into Baloot assets.

That dataset labels every card as a **quadrilateral over the whole card** plus a
52-deck class id. The Baloot detector labels something different: the small
rank/suit index in a card corner. Two conversions bridge the gap, and both are
pure geometry because the quad already tells us the card pose:

* :func:`extract_card_template` warps a quad back onto the canonical
  ``CARD_WIDTH x CARD_HEIGHT`` card, producing a clean BGRA template. Feeding
  those to :mod:`hakim_vision.synthetic.fan` is what this module mainly exists
  for - a 52-card atlas costs a few hundred image downloads instead of the
  dataset's 23 GB.
* :func:`index_quads_for_card` maps the canonical corner rectangles
  (``REF_CORNER_HL`` / ``REF_CORNER_LR``) forward through the same homography,
  which turns any annotated image into corner-index labels.

Downloads go through plain HTTPS against the Hub resolve endpoint, so nothing
here needs ``huggingface_hub`` or ``datasets`` installed on the training box.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from hakim_vision.datasets.remap import normalise_card_name
from hakim_vision.synthetic.constants import (
    CARD_HEIGHT,
    CARD_WIDTH,
    REF_CARD,
    REF_CORNER_HL,
    REF_CORNER_LR,
)
from hakim_vision.synthetic.focus import variance_of_laplacian

logger = logging.getLogger(__name__)

#: The dataset this module understands.
HF_REPO: str = "JackFurby/playing-cards"

#: Subsets, by cards per image. ``single`` is what the atlas is built from.
HF_SUBSETS: tuple[str, ...] = ("single", "three", "three_card_poker")

#: Corner radius of a real card, in canonical-template pixels. Rounds off the
#: alpha channel so a pasted card carries no square background corners.
CARD_CORNER_RADIUS: int = 14

_USER_AGENT = "hakim-vision/0.1 (+https://github.com/osos3lom/AIBaloot)"


def _resolve_url(path: str, repo: str = HF_REPO) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path.lstrip('/')}"


def download_file(path: str, dest: Path, *, repo: str = HF_REPO, retries: int = 3) -> Path:
    """Download one file from the dataset repo, skipping work if it already exists.

    Raises:
        urllib.error.URLError: if every attempt fails.
    """
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    # `repo` and `path` reach here from the CLI, so check the composed URL rather
    # than trusting the https prefix in `_resolve_url` - a crafted repo id could
    # otherwise steer the opener at a `file:` or other local scheme.
    url = _resolve_url(path, repo)
    if not url.startswith("https://huggingface.co/"):
        raise ValueError(f"refusing to fetch off-Hub URL: {url}")

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                payload = response.read()
            temporary = dest.with_suffix(dest.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            logger.warning("download failed (%d/%d) for %s: %s", attempt, retries, path, error)
    raise urllib.error.URLError(f"could not download {path}: {last_error}")


@dataclass(frozen=True)
class CardAnnotation:
    """One annotated card: its quad on the image, and its deck name."""

    quad: NDArray[np.float32]
    """``(4, 2)`` corner points as stored by the dataset."""

    source_name: str
    """Deck name from ``concepts.txt``, e.g. ``"10H"``."""

    @property
    def baloot_name(self) -> str:
        """The Baloot class this card belongs to - ``other`` for ranks 2-6."""
        return normalise_card_name(self.source_name) or "other"


@dataclass(frozen=True)
class ImageAnnotation:
    """Every card annotated on one dataset image, front-most first."""

    image_path: str
    cards: tuple[CardAnnotation, ...]


def read_label_names(path: Path) -> list[str]:
    """Parse a ``"<name> <index>"``-per-line label file into an index-ordered list."""
    pairs: dict[int, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pairs[int(parts[1])] = parts[0]
        except ValueError:
            continue
    return [pairs[index] for index in sorted(pairs)]


def read_deck_names(cache_dir: Path, subset: str, *, repo: str = HF_REPO) -> list[str]:
    """The 52 deck names indexing a subset's ``card_points``.

    Which file holds them depends on the subset: ``single`` classifies the card
    itself, so ``classes.txt`` is the deck and ``concepts.txt`` is 17 rank/suit
    attributes; the multi-card subsets classify a poker hand, which pushes the
    deck into ``concepts.txt``. Rather than hard-code that per subset, take
    whichever file parses to a full 52-card deck.

    Raises:
        ValueError: if neither file is a deck, which means the subset is not
            laid out the way this module understands.
    """
    for filename in ("classes.txt", "concepts.txt"):
        local = download_file(
            f"splits/{subset}/{filename}", cache_dir / subset / filename, repo=repo
        )
        names = read_label_names(local)
        if len(names) == 52 and all(
            normalise_card_name(name) or _is_low_rank(name) for name in names
        ):
            return names
    raise ValueError(f"no 52-card deck listing found for subset {subset!r}")


def _is_low_rank(name: str) -> bool:
    """True for a poker card with no Baloot equivalent (ranks 2-6)."""
    text = name.strip().lower()
    return len(text) >= 2 and text[0] in "23456" and text[1] in "hdcs"


def parse_split(path: Path, deck_names: list[str]) -> list[ImageAnnotation]:
    """Parse a split JSON into per-image annotations.

    The dataset lists each image's cards front-most first, which is the draw
    order the occlusion logic downstream depends on. Cards whose class id falls
    outside ``deck_names`` are dropped rather than guessed at.
    """
    raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    annotations: list[ImageAnnotation] = []
    for entry in raw.values():
        cards: list[CardAnnotation] = []
        for points, class_id in entry.get("card_points", []):
            index = int(class_id)
            if not 0 <= index < len(deck_names):
                continue
            quad = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            if quad.shape != (4, 2):
                continue
            cards.append(CardAnnotation(quad=quad, source_name=deck_names[index]))
        if cards:
            annotations.append(
                ImageAnnotation(image_path=str(entry["img_path"]), cards=tuple(cards))
            )
    return annotations


def order_quad(quad: NDArray[np.float32]) -> NDArray[np.float32]:
    """Reorder a dataset quad into ``[corner, +width, +width+height, +height]``.

    The dataset stores a parallelogram as ``[P, P+u, P+v, P+u+v]`` without saying
    which of ``u`` / ``v`` is the card's short side. The short one is the width,
    so measuring the two edges recovers the winding that matches
    :data:`REF_CARD`.

    Which physical corner ``P`` is stays ambiguous - a card rotated 180 degrees
    is indistinguishable - but that does not matter: a playing card is symmetric
    under that rotation, and both index corners are emitted anyway.
    """
    points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    origin = points[0]
    edge_a = float(np.linalg.norm(points[1] - origin))
    edge_b = float(np.linalg.norm(points[2] - origin))
    width_corner, height_corner = (
        (points[1], points[2]) if edge_a <= edge_b else (points[2], points[1])
    )
    far_corner = width_corner + height_corner - origin
    ordered: NDArray[np.float32] = np.stack(
        [origin, width_corner, far_corner, height_corner]
    ).astype(np.float32)
    return ordered


def card_homography(quad: NDArray[np.float32]) -> NDArray[np.float32]:
    """Homography mapping canonical card space onto a quad's image position."""
    matrix = cv2.getPerspectiveTransform(REF_CARD, order_quad(quad))
    result: NDArray[np.float32] = matrix.astype(np.float32)
    return result


def project(matrix: NDArray[np.float32], points: NDArray[np.float32]) -> NDArray[np.float32]:
    """Apply a ``3x3`` homography to ``(N, 2)`` points."""
    source = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(source, matrix).reshape(-1, 2)
    result: NDArray[np.float32] = projected.astype(np.float32)
    return result


def index_quads_for_card(quad: NDArray[np.float32]) -> tuple[NDArray[np.float32], ...]:
    """Both corner-index quads of a card, in image coordinates.

    Returns the top-left and bottom-right index regions. A card shows the same
    rank/suit in both, so both are valid labels; whichever one a neighbouring
    card covers is filtered by the caller's occlusion test.
    """
    matrix = card_homography(quad)
    return tuple(project(matrix, corner) for corner in (REF_CORNER_HL, REF_CORNER_LR))


def rounded_alpha(width: int, height: int, radius: int) -> NDArray[np.uint8]:
    """A full-card alpha mask with rounded corners."""
    alpha = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(alpha, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(alpha, (0, radius), (width, height - radius), 255, -1)
    for x, y in (
        (radius, radius),
        (width - radius, radius),
        (radius, height - radius),
        (width - radius, height - radius),
    ):
        cv2.circle(alpha, (x, y), radius, 255, -1)
    return alpha


def extract_card_template(
    image: NDArray[np.uint8],
    quad: NDArray[np.float32],
) -> NDArray[np.uint8]:
    """Warp an annotated card out of a scene into a canonical BGRA template."""
    matrix = cv2.getPerspectiveTransform(order_quad(quad), REF_CARD)
    warped = cv2.warpPerspective(
        image[:, :, :3],
        matrix,
        (CARD_WIDTH, CARD_HEIGHT),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    alpha = rounded_alpha(CARD_WIDTH, CARD_HEIGHT, CARD_CORNER_RADIUS)
    template: NDArray[np.uint8] = np.dstack([warped, alpha]).astype(np.uint8)
    return template


def iter_class_candidates(
    annotations: list[ImageAnnotation],
    *,
    per_class: int,
) -> Iterator[tuple[str, ImageAnnotation, CardAnnotation]]:
    """Yield up to ``per_class`` annotated cards for each deck class."""
    seen: dict[str, int] = {}
    for annotation in annotations:
        for card in annotation.cards:
            count = seen.get(card.source_name, 0)
            if count >= per_class:
                continue
            seen[card.source_name] = count + 1
            yield card.source_name, annotation, card


def build_card_atlas(
    output_dir: Path,
    *,
    cache_dir: Path,
    per_class: int = 4,
    subset: str = "single",
    repo: str = HF_REPO,
    min_focus: float = 0.0,
) -> dict[str, list[Path]]:
    """Download a slice of the dataset and write a canonical card-template atlas.

    Only the images actually used are fetched - ``per_class=4`` over a 52-card
    deck is ~208 downloads, not the full repository.

    Args:
        output_dir: Where ``<DECK NAME>_<n>.png`` templates are written.
        cache_dir: Download cache; re-runs reuse whatever is already there.
        per_class: Templates to keep per deck class.
        subset: Dataset subset to source cards from.
        repo: Dataset repo id on the Hub.
        min_focus: Reject templates blurrier than this Laplacian variance.

    Returns:
        ``{deck name: [template paths]}``, e.g. ``{"10H": [.../10H_0.png, ...]}``.
    """
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    deck_names = read_deck_names(cache_dir, subset, repo=repo)
    split_file = download_file(
        f"splits/{subset}/train.json", cache_dir / subset / "train.json", repo=repo
    )
    annotations = parse_split(split_file, deck_names)
    logger.info("parsed %d annotated %s images", len(annotations), subset)

    atlas: dict[str, list[Path]] = {}
    for deck_name, annotation, card in iter_class_candidates(annotations, per_class=per_class):
        local = download_file(annotation.image_path, cache_dir / annotation.image_path, repo=repo)
        image = cv2.imread(str(local), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("unreadable image: %s", local)
            continue
        template = extract_card_template(image.astype(np.uint8), card.quad)
        if min_focus > 0.0 and variance_of_laplacian(template[:, :, :3]) < min_focus:
            logger.debug("dropped blurry template for %s", deck_name)
            continue
        destination = output_dir / f"{deck_name}_{len(atlas.get(deck_name, []))}.png"
        cv2.imwrite(str(destination), template)
        atlas.setdefault(deck_name, []).append(destination)

    logger.info("atlas: %d classes, %d templates", len(atlas), sum(len(v) for v in atlas.values()))
    return atlas


__all__ = [
    "CARD_CORNER_RADIUS",
    "HF_REPO",
    "HF_SUBSETS",
    "CardAnnotation",
    "ImageAnnotation",
    "build_card_atlas",
    "card_homography",
    "download_file",
    "extract_card_template",
    "index_quads_for_card",
    "iter_class_candidates",
    "order_quad",
    "parse_split",
    "project",
    "read_deck_names",
    "read_label_names",
    "rounded_alpha",
]
