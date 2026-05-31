"""Synthetic playing-card scene generation pipeline.

Typed, testable modules covering card extraction, corner-hull detection,
random-affine augmentation, and 2-/3-card scene composition with YOLO labels.
"""

from hakim_vision.synthetic.assets import Backgrounds, CardSample, Cards
from hakim_vision.synthetic.card_extraction import ExtractedCard, extract_card
from hakim_vision.synthetic.pack import pack_backgrounds, pack_cards
from hakim_vision.synthetic.scene import (
    Scene,
    SceneLabel,
    compose_scene,
    render_random_scene,
    write_yolo_label,
)
from hakim_vision.synthetic.transforms import (
    AugmentRange,
    PlacedCard,
    random_affine_card,
)
from hakim_vision.synthetic.constants import (
    CARD_HEIGHT,
    CARD_WIDTH,
    REF_CARD,
    REF_CARD_ROT,
    REF_CORNER_HL,
    REF_CORNER_LR,
    SCENE_SIZE,
)
from hakim_vision.synthetic.focus import variance_of_laplacian
from hakim_vision.synthetic.hull import (
    HullDetection,
    find_corner_hull,
    hull_to_points,
    points_to_bbox,
    points_to_polygon,
)

__all__ = [
    "CARD_HEIGHT",
    "CARD_WIDTH",
    "REF_CARD",
    "REF_CARD_ROT",
    "REF_CORNER_HL",
    "REF_CORNER_LR",
    "SCENE_SIZE",
    "AugmentRange",
    "Backgrounds",
    "CardSample",
    "Cards",
    "ExtractedCard",
    "HullDetection",
    "PlacedCard",
    "Scene",
    "SceneLabel",
    "compose_scene",
    "extract_card",
    "find_corner_hull",
    "hull_to_points",
    "pack_backgrounds",
    "pack_cards",
    "points_to_bbox",
    "points_to_polygon",
    "random_affine_card",
    "render_random_scene",
    "variance_of_laplacian",
    "write_yolo_label",
]
