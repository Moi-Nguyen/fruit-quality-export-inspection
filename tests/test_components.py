import numpy as np

from src.components import (
    compute_component_stats,
    connected_component_labeling,
    get_neighbor_offsets,
    keep_largest_component,
    keep_largest_non_border_component,
    keep_largest_reasonable_component,
)


def test_get_neighbor_offsets_for_4_connectivity():
    offsets = get_neighbor_offsets(4)

    assert len(offsets) == 4
    assert (-1, 0) in offsets
    assert (1, 1) not in offsets


def test_get_neighbor_offsets_for_8_connectivity():
    offsets = get_neighbor_offsets(8)

    assert len(offsets) == 8
    assert (1, 1) in offsets


def test_connected_component_labeling_detects_one_component():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1

    labels, num_components = connected_component_labeling(mask)

    assert num_components == 1
    assert set(np.unique(labels).tolist()) == {0, 1}


def test_connected_component_labeling_detects_two_components():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[0, 0] = 1
    mask[4, 4] = 1

    labels, num_components = connected_component_labeling(mask)

    assert num_components == 2
    assert labels[0, 0] != labels[4, 4]


def test_compute_component_stats_area_and_bounding_box():
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[1:3, 2:5] = 1

    stats = compute_component_stats(labels, 1)

    assert stats[0]["area"] == 6
    assert stats[0]["min_row"] == 1
    assert stats[0]["min_col"] == 2
    assert stats[0]["max_row"] == 2
    assert stats[0]["max_col"] == 4
    assert stats[0]["height"] == 2
    assert stats[0]["width"] == 3


def test_keep_largest_component_keeps_largest_object():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[0, 0] = 1
    mask[2:5, 2:5] = 1

    largest = keep_largest_component(mask)

    assert int(np.sum(largest)) == 9
    assert largest[0, 0] == 0


def test_empty_mask_returns_zero_components_and_zero_largest_mask():
    mask = np.zeros((4, 4), dtype=np.uint8)

    labels, num_components = connected_component_labeling(mask)
    largest = keep_largest_component(mask)

    assert num_components == 0
    assert int(np.sum(labels)) == 0
    assert int(np.sum(largest)) == 0

def test_keep_largest_non_border_component_ignores_border_component():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0:3, 0:3] = 1
    mask[4:6, 4:6] = 1

    selected = keep_largest_non_border_component(mask, border_margin=1)

    assert int(np.sum(selected)) == 4
    assert selected[4, 4] == 1
    assert selected[0, 0] == 0

def test_keep_largest_non_border_component_falls_back_when_all_touch_border():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    mask[4:6, 4:6] = 1

    selected = keep_largest_non_border_component(mask, border_margin=1)

    assert int(np.sum(selected)) == 4


def test_keep_largest_reasonable_component_keeps_large_object_touching_border():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:8, 4:16] = 1
    mask[15, 15] = 1

    selected = keep_largest_reasonable_component(mask, min_area_ratio=0.005, max_area_ratio=0.80)

    assert int(np.sum(selected)) == 96
    assert selected[0, 4] == 1
    assert selected[15, 15] == 0

def test_keep_largest_reasonable_component_ignores_tiny_noise():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    mask[1, 1] = 1
    mask[28, 28] = 1

    selected = keep_largest_reasonable_component(mask, min_area_ratio=0.005, max_area_ratio=0.80)

    assert int(np.sum(selected)) == 100
    assert selected[1, 1] == 0

def test_keep_largest_reasonable_component_falls_back_safely_if_no_valid_component_exists():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1, 1] = 1
    mask[5:7, 5:7] = 1

    selected = keep_largest_reasonable_component(mask, min_area_ratio=0.20, max_area_ratio=0.30)

    assert int(np.sum(selected)) == 4
