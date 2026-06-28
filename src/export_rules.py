"""Rule-based export suitability assessment."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import (
    DARK_BRIGHTNESS_THRESHOLD,
    HIGH_DEFECT_RATIO_THRESHOLD,
    HIGH_NOISE_THRESHOLD,
    LOW_CIRCULARITY_THRESHOLD,
    MEDIUM_DEFECT_RATIO_THRESHOLD,
    MIN_MASK_AREA_RATIO_THRESHOLD,
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
