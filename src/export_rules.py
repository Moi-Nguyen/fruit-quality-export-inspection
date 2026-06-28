"""Rule-based export suitability assessment."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import (
    DARK_BRIGHTNESS_THRESHOLD,
    DOMESTIC_MAX_DEFECT_RATIO,
    EXPORT_MAX_DEFECT_RATIO,
    HIGH_DEFECT_RATIO_THRESHOLD,
    HIGH_NOISE_THRESHOLD,
    LOW_CIRCULARITY_THRESHOLD,
    MAX_REASONABLE_MASK_RATIO,
    MEDIUM_DEFECT_RATIO_THRESHOLD,
    MIN_MASK_AREA_RATIO_THRESHOLD,
    MIN_REASONABLE_MASK_RATIO,
)


def _safe_float(features: dict[str, Any], feature_name: str) -> float | None:
    """Convert a feature value to float, or return None."""
    try:
        value = float(features.get(feature_name))
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None
    return value


def assess_export_suitability(
    fruit_type: str,
    quality: str,
    features: dict[str, object],
) -> dict[str, object]:
    """Return rule-based export suitability result."""
    normalized_fruit_type = fruit_type.strip().lower()
    normalized_quality = quality.strip().lower()

    brightness = _safe_float(features, "brightness")
    noise_level = _safe_float(features, "noise_level")
    defect_ratio = _safe_float(features, "defect_ratio")
    circularity = _safe_float(features, "circularity")
    mask_area_ratio = _safe_float(features, "mask_area_ratio")

    rule_flags = {
        "is_rotten": normalized_quality == "rotten",
        "is_too_dark": brightness is not None and brightness < DARK_BRIGHTNESS_THRESHOLD,
        "is_too_noisy": noise_level is not None and noise_level > HIGH_NOISE_THRESHOLD,
        "has_high_defect": defect_ratio is not None
        and defect_ratio >= HIGH_DEFECT_RATIO_THRESHOLD,
        "has_medium_defect": defect_ratio is not None
        and MEDIUM_DEFECT_RATIO_THRESHOLD <= defect_ratio < HIGH_DEFECT_RATIO_THRESHOLD,
        "has_abnormal_shape": normalized_fruit_type in {"apple", "orange"}
        and circularity is not None
        and circularity < LOW_CIRCULARITY_THRESHOLD,
        "has_small_mask": mask_area_ratio is not None
        and mask_area_ratio < MIN_MASK_AREA_RATIO_THRESHOLD,
    }

    if rule_flags["is_rotten"]:
        return {
            "suitability": "Not Suitable",
            "reasons": ["The fruit is predicted as rotten."],
            "rule_flags": rule_flags,
        }

    if rule_flags["is_too_dark"]:
        return {
            "suitability": "Need Recheck",
            "reasons": ["Image is too dark for reliable inspection."],
            "rule_flags": rule_flags,
        }

    if rule_flags["is_too_noisy"]:
        return {
            "suitability": "Need Recheck",
            "reasons": ["Image is too noisy for reliable inspection."],
            "rule_flags": rule_flags,
        }

    if rule_flags["has_high_defect"]:
        return {
            "suitability": "Not Suitable",
            "reasons": [f"Defect ratio is high: {defect_ratio:.2f}."],
            "rule_flags": rule_flags,
        }

    if rule_flags["has_medium_defect"]:
        return {
            "suitability": "Need Recheck",
            "reasons": [f"Defect ratio is medium: {defect_ratio:.2f}."],
            "rule_flags": rule_flags,
        }

    if rule_flags["has_abnormal_shape"]:
        return {
            "suitability": "Need Recheck",
            "reasons": ["Fruit shape looks abnormal for apple or orange."],
            "rule_flags": rule_flags,
        }

    if rule_flags["has_small_mask"]:
        return {
            "suitability": "Need Recheck",
            "reasons": ["Mask area ratio is too small; segmentation may be unreliable."],
            "rule_flags": rule_flags,
        }

    return {
        "suitability": "Suitable",
        "reasons": ["The fruit is predicted as fresh and defect ratio is acceptable."],
        "rule_flags": rule_flags,
    }

def decide_market_grade(
    quality: str,
    defect_ratio: float,
    mask_area_ratio: float,
    circularity: float | None = None,
) -> tuple[str, list[str]]:
    """
    Decide final market grade for fruit sorting.

    Returns:
        A tuple containing:
        - market grade: "Export Grade", "Domestic Grade", or "Reject"
        - list of explanation reasons
    """
    normalized_quality = quality.strip().lower()
    reasons: list[str] = []

    if normalized_quality == "rotten":
        return "Reject", ["Rotten fruit should not be used for export or domestic sale."]

    if mask_area_ratio < MIN_REASONABLE_MASK_RATIO:
        if mask_area_ratio < MIN_REASONABLE_MASK_RATIO / 2.0:
            return "Reject", [
                "Fruit mask area is much too small, so the segmentation is not reliable enough for sorting.",
            ]
        reasons.append(
            "Fruit mask area is small, so segmentation reliability is questionable."
        )
        return "Domestic Grade", reasons

    if mask_area_ratio > MAX_REASONABLE_MASK_RATIO:
        if mask_area_ratio > min(1.0, MAX_REASONABLE_MASK_RATIO + 0.10):
            return "Reject", [
                "Fruit mask area is much too large, so the image may include too much background or a failed mask.",
            ]
        reasons.append(
            "Fruit mask area is large, so segmentation reliability is questionable."
        )
        return "Domestic Grade", reasons

    if circularity is not None and circularity < LOW_CIRCULARITY_THRESHOLD:
        reasons.append("Fruit shape looks abnormal, so it is safer for domestic review.")
        return "Domestic Grade", reasons

    if defect_ratio <= EXPORT_MAX_DEFECT_RATIO:
        return "Export Grade", [
            "Fresh fruit with low defect ratio is suitable for export.",
        ]

    if defect_ratio <= DOMESTIC_MAX_DEFECT_RATIO:
        return "Domestic Grade", [
            "Fruit is fresh but has visible defects, so it is better for the domestic market.",
        ]

    return "Reject", [
        "Defect ratio is high, so the fruit should be rejected for final sorting.",
    ]
