"""Tests for fanned-hand rendering and the HuggingFace card conversion.

Everything here is offline: the dataset's geometry conventions are reproduced
from its documented shape rather than downloaded, so the suite stays fast and
does not depend on the Hub being reachable.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from hakim_vision.datasets.hf_cards import (
    card_homography,
    extract_card_template,
    index_quads_for_card,
    order_quad,
    parse_split,
    project,
    read_label_names,
    rounded_alpha,
)
from hakim_vision.synthetic.constants import (
    CARD_HEIGHT,
    CARD_WIDTH,
    CORNER_X_MAX,
    CORNER_X_MIN,
    CORNER_Y_MAX,
    CORNER_Y_MIN,
    REF_CARD,
)
from hakim_vision.synthetic.fan import (
    CardAtlas,
    FanConfig,
    load_backgrounds,
    render_fan_scene,
)

DECK = ["2C", "2D", "2H", "2S", "AH", "KS", "10D", "7C", "QH", "JS"]


def make_quad(origin: tuple[float, float], width: float, height: float) -> np.ndarray:
    """A dataset-style quad: ``[P, P+u, P+v, P+u+v]`` with unspecified edge order."""
    x, y = origin
    return np.array(
        [[x, y], [x + width, y], [x, y + height], [x + width, y + height]],
        dtype=np.float32,
    )


def make_atlas(names: list[str] = DECK) -> CardAtlas:
    """An atlas of flat colour cards - enough for geometry, no artwork needed."""
    templates: dict[str, list[np.ndarray]] = {}
    for index, name in enumerate(names):
        card = np.zeros((CARD_HEIGHT, CARD_WIDTH, 4), dtype=np.uint8)
        card[:, :, :3] = 20 * (index % 12) + 30
        card[:, :, 3] = 255
        templates[name] = [card]
    return CardAtlas(templates)


class TestQuadGeometry:
    def test_order_quad_puts_the_short_edge_first(self) -> None:
        ordered = order_quad(make_quad((10, 20), width=100, height=200))

        assert np.linalg.norm(ordered[1] - ordered[0]) == pytest.approx(100.0)
        assert np.linalg.norm(ordered[3] - ordered[0]) == pytest.approx(200.0)

    def test_order_quad_is_stable_when_edges_arrive_swapped(self) -> None:
        upright = make_quad((0, 0), width=100, height=200)
        swapped = upright[[0, 2, 1, 3]]

        assert np.allclose(order_quad(upright), order_quad(swapped))

    def test_homography_maps_the_reference_card_onto_the_quad(self) -> None:
        quad = make_quad((40, 60), width=120, height=240)

        mapped = project(card_homography(quad), REF_CARD)

        assert np.allclose(mapped, order_quad(quad), atol=1e-3)

    def test_index_quads_land_inside_the_card(self) -> None:
        quad = make_quad((0, 0), width=CARD_WIDTH, height=CARD_HEIGHT)

        top_left, bottom_right = index_quads_for_card(quad)

        assert top_left[:, 0].min() == pytest.approx(CORNER_X_MIN, abs=1.0)
        assert top_left[:, 1].max() == pytest.approx(CORNER_Y_MAX, abs=1.0)
        assert bottom_right[:, 0].max() == pytest.approx(CARD_WIDTH - CORNER_X_MIN, abs=1.0)
        assert bottom_right[:, 1].min() == pytest.approx(CARD_HEIGHT - CORNER_Y_MAX, abs=1.0)

    def test_index_quads_follow_a_rotated_card(self) -> None:
        rotated = np.array(
            [[300, 100], [300, 200], [100, 100], [100, 200]],
            dtype=np.float32,
        )

        for corner in index_quads_for_card(rotated):
            assert corner[:, 0].min() >= 99.0
            assert corner[:, 0].max() <= 301.0


class TestTemplateExtraction:
    def test_extract_card_template_recovers_the_canonical_size(self) -> None:
        scene = np.full((400, 400, 3), 255, dtype=np.uint8)
        cv2.rectangle(scene, (50, 40), (150, 240), (0, 0, 200), -1)

        template = extract_card_template(scene, make_quad((50, 40), width=100, height=200))

        assert template.shape == (CARD_HEIGHT, CARD_WIDTH, 4)
        assert template[CARD_HEIGHT // 2, CARD_WIDTH // 2, 2] > 150

    def test_rounded_alpha_clears_the_corners_and_fills_the_middle(self) -> None:
        alpha = rounded_alpha(100, 200, radius=20)

        assert alpha[0, 0] == 0
        assert alpha[100, 50] == 255


class TestSplitParsing:
    def test_read_label_names_orders_by_index(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "classes.txt"
        path.write_text("QH 1\nAS 0\n", encoding="utf-8")

        assert read_label_names(path) == ["AS", "QH"]

    def test_parse_split_keeps_draw_order_and_maps_names(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "train.json"
        path.write_text(
            json.dumps(
                {
                    "0": {
                        "img_path": "imgs/three/0.png",
                        "card_points": [
                            [[[0, 0], [10, 0], [0, 20], [10, 20]], 4],
                            [[[5, 5], [15, 5], [5, 25], [15, 25]], 0],
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )

        annotations = parse_split(path, DECK)

        assert len(annotations) == 1
        assert [card.source_name for card in annotations[0].cards] == ["AH", "2C"]
        assert [card.baloot_name for card in annotations[0].cards] == ["Ah", "other"]

    def test_parse_split_drops_out_of_range_classes(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "train.json"
        path.write_text(
            json.dumps(
                {
                    "0": {
                        "img_path": "a.png",
                        "card_points": [[[[0, 0], [1, 0], [0, 2], [1, 2]], 99]],
                    }
                }
            ),
            encoding="utf-8",
        )

        assert parse_split(path, DECK) == []


class TestFanRendering:
    def test_scene_matches_the_configured_canvas(self) -> None:
        config = FanConfig(canvas=320, cards_min=6, cards_max=8)

        scene = render_fan_scene(make_atlas(), rng=np.random.default_rng(1), config=config)

        assert scene.image.shape == (320, 320, 3)
        assert scene.canvas_size == 320

    def test_labels_are_normalised_and_inside_the_frame(self) -> None:
        config = FanConfig(canvas=384, cards_min=6, cards_max=9, hand_prob=0.0)

        scene = render_fan_scene(make_atlas(), rng=np.random.default_rng(3), config=config)

        assert scene.labels
        for label in scene.labels:
            assert 0.0 <= label.yolo.cx <= 1.0
            assert 0.0 <= label.yolo.cy <= 1.0
            assert 0.0 < label.yolo.w <= 1.0
            assert 0.0 < label.yolo.h <= 1.0

    def test_low_ranks_are_labelled_other(self) -> None:
        atlas = make_atlas(["2C", "2D", "2H", "2S", "3C", "3D"])
        config = FanConfig(canvas=384, cards_min=6, cards_max=6, hand_prob=0.0)

        scene = render_fan_scene(atlas, rng=np.random.default_rng(5), config=config)

        assert {label.class_name for label in scene.labels} == {"other"}

    def test_rendering_is_reproducible_from_a_seed(self) -> None:
        config = FanConfig(canvas=256, cards_min=5, cards_max=7)

        first = render_fan_scene(make_atlas(), rng=np.random.default_rng(11), config=config)
        second = render_fan_scene(make_atlas(), rng=np.random.default_rng(11), config=config)

        assert np.array_equal(first.image, second.image)
        assert [label.voc for label in first.labels] == [label.voc for label in second.labels]

    def test_a_fan_hides_more_indices_than_a_spread_hand(self) -> None:
        """The point of the module: crowding cards costs labels to occlusion."""
        atlas = make_atlas()
        crowded = FanConfig(
            canvas=512, cards_min=10, cards_max=10, spread_deg=(8.0, 8.0), hand_prob=0.0
        )
        spread = FanConfig(
            canvas=512, cards_min=10, cards_max=10, spread_deg=(110.0, 110.0), hand_prob=0.0
        )

        crowded_labels = sum(
            len(render_fan_scene(atlas, rng=np.random.default_rng(s), config=crowded).labels)
            for s in range(6)
        )
        spread_labels = sum(
            len(render_fan_scene(atlas, rng=np.random.default_rng(s), config=spread).labels)
            for s in range(6)
        )

        assert crowded_labels < spread_labels

    def test_visible_only_boxes_never_exceed_the_full_index(self) -> None:
        config = FanConfig(canvas=448, cards_min=8, cards_max=10, hand_prob=1.0)

        scene = render_fan_scene(make_atlas(), rng=np.random.default_rng(17), config=config)

        limit = max(CORNER_X_MAX - CORNER_X_MIN, CORNER_Y_MAX - CORNER_Y_MIN)
        for label in scene.labels:
            xmin, ymin, xmax, ymax = label.voc
            assert 0 <= xmin < xmax <= config.canvas
            assert 0 <= ymin < ymax <= config.canvas
            assert max(xmax - xmin, ymax - ymin) <= limit * 2

    def test_backgrounds_are_used_when_supplied(self) -> None:
        config = FanConfig(
            canvas=256, cards_min=5, cards_max=5, hand_prob=0.0, perspective=0.0, noise_sigma=(0, 0)
        )
        magenta = np.zeros((64, 64, 3), dtype=np.uint8)
        magenta[:, :, 0] = 255
        magenta[:, :, 2] = 255

        scene = render_fan_scene(
            make_atlas(), rng=np.random.default_rng(2), config=config, backgrounds=[magenta]
        )

        corner = scene.image[:16, :16]
        assert corner[:, :, 0].mean() > 180
        assert corner[:, :, 1].mean() < 80

    def test_empty_atlas_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            CardAtlas({})


class TestAtlasLoading:
    def test_from_directory_groups_variants_by_deck_name(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        for name in ("AH_0.png", "AH_1.png", "KS_0.png"):
            card = np.zeros((CARD_HEIGHT, CARD_WIDTH, 4), dtype=np.uint8)
            card[:, :, 3] = 255
            cv2.imwrite(str(tmp_path / name), card)

        atlas = CardAtlas.from_directory(tmp_path)

        assert atlas.names == ["AH", "KS"]

    def test_load_backgrounds_without_a_directory_is_empty(self) -> None:
        assert load_backgrounds(None) == []
