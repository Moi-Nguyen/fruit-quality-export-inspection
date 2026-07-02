"""Tests for rule-based export suitability assessment."""

from src.export_rules import (
    LOW_DEFECT_ROTTEN_RECHECK_REASON,
    assess_export_suitability,
    decide_market_grade,
)


def good_features() -> dict[str, object]:
    """Return feature values that should pass export rules."""
    return {
        "brightness": 120.0,
        "noise_level": 5.0,
        "defect_ratio": 0.02,
        "circularity": 0.80,
        "mask_area_ratio": 0.30,
    }


def test_rotten_quality_returns_not_suitable() -> None:
    result = assess_export_suitability("orange", "rotten", good_features())

    assert result["suitability"] == "Not Suitable"
    assert result["rule_flags"]["is_rotten"] is True


def test_high_defect_ratio_returns_not_suitable() -> None:
    features = good_features()
    features["defect_ratio"] = 0.24

    result = assess_export_suitability("apple", "fresh", features)

    assert result["suitability"] == "Not Suitable"
    assert result["rule_flags"]["has_high_defect"] is True


def test_medium_defect_ratio_returns_need_recheck() -> None:
    features = good_features()
    features["defect_ratio"] = 0.12

    result = assess_export_suitability("apple", "fresh", features)

    assert result["suitability"] == "Need Recheck"
    assert result["rule_flags"]["has_medium_defect"] is True


def test_too_low_brightness_returns_need_recheck() -> None:
    features = good_features()
    features["brightness"] = 50.0

    result = assess_export_suitability("banana", "fresh", features)

    assert result["suitability"] == "Need Recheck"
    assert result["rule_flags"]["is_too_dark"] is True


def test_high_noise_level_returns_need_recheck() -> None:
    features = good_features()
    features["noise_level"] = 20.0

    result = assess_export_suitability("banana", "fresh", features)

    assert result["suitability"] == "Need Recheck"
    assert result["rule_flags"]["is_too_noisy"] is True


def test_low_circularity_apple_returns_need_recheck() -> None:
    features = good_features()
    features["circularity"] = 0.20

    result = assess_export_suitability("apple", "fresh", features)

    assert result["suitability"] == "Need Recheck"
    assert result["rule_flags"]["has_abnormal_shape"] is True


def test_low_circularity_banana_does_not_trigger_abnormal_shape_rule() -> None:
    features = good_features()
    features["circularity"] = 0.20

    result = assess_export_suitability("banana", "fresh", features)

    assert result["suitability"] == "Suitable"
    assert result["rule_flags"]["has_abnormal_shape"] is False


def test_fresh_good_features_returns_suitable() -> None:
    result = assess_export_suitability("orange", "fresh", good_features())

    assert result["suitability"] == "Suitable"
    assert len(result["reasons"]) >= 1


def test_missing_feature_values_do_not_crash() -> None:
    result = assess_export_suitability("orange", "fresh", {})

    assert result["suitability"] == "Suitable"
    assert len(result["reasons"]) >= 1


def test_fresh_low_defect_returns_export_grade() -> None:
    market_grade, reasons = decide_market_grade(
        quality="fresh",
        defect_ratio=0.02,
        mask_area_ratio=0.40,
        circularity=0.80,
    )

    assert market_grade == "Export Grade"
    assert reasons

def test_fresh_medium_defect_returns_domestic_grade() -> None:
    market_grade, reasons = decide_market_grade(
        quality="fresh",
        defect_ratio=0.10,
        mask_area_ratio=0.40,
        circularity=0.80,
    )

    assert market_grade == "Domestic Grade"
    assert reasons

def test_rotten_quality_returns_reject_market_grade() -> None:
    market_grade, reasons = decide_market_grade(
        quality="rotten",
        defect_ratio=0.20,
        mask_area_ratio=0.40,
        quality_confidence=0.90,
    )

    assert market_grade == "Reject"
    assert reasons

def test_rotten_low_defect_high_confidence_returns_need_recheck() -> None:
    market_grade, reasons = decide_market_grade(
        quality="rotten",
        defect_ratio=0.0,
        mask_area_ratio=0.40,
        brown_dark_ratio=0.01,
        quality_confidence=0.97,
    )

    assert market_grade == "Need Recheck"
    assert reasons == [LOW_DEFECT_ROTTEN_RECHECK_REASON]


def test_rotten_high_defect_still_returns_reject() -> None:
    market_grade, reasons = decide_market_grade(
        quality="rotten",
        defect_ratio=0.24,
        mask_area_ratio=0.40,
        brown_dark_ratio=0.01,
        quality_confidence=0.97,
    )

    assert market_grade == "Reject"
    assert reasons

def test_rotten_high_brown_dark_still_returns_reject() -> None:
    market_grade, reasons = decide_market_grade(
        quality="rotten",
        defect_ratio=0.0,
        mask_area_ratio=0.40,
        brown_dark_ratio=0.20,
        quality_confidence=0.97,
    )

    assert market_grade == "Reject"
    assert reasons

def test_fresh_low_defect_consistency_rule_does_not_apply() -> None:
    market_grade, reasons = decide_market_grade(
        quality="fresh",
        defect_ratio=0.0,
        mask_area_ratio=0.40,
        brown_dark_ratio=0.01,
        quality_confidence=0.97,
    )

    assert market_grade == "Export Grade"
    assert LOW_DEFECT_ROTTEN_RECHECK_REASON not in reasons

def test_rotten_low_visible_evidence_export_returns_need_recheck() -> None:
    features = good_features()
    features["defect_ratio"] = 0.0
    features["brown_dark_ratio"] = 0.01
    features["quality_confidence"] = 0.97

    result = assess_export_suitability("apple", "rotten", features)

    assert result["suitability"] == "Need Recheck"
    assert result["reasons"] == [LOW_DEFECT_ROTTEN_RECHECK_REASON]
    assert result["rule_flags"]["rotten_prediction_has_low_visible_evidence"] is True
def test_unreasonable_mask_ratio_returns_domestic_or_reject() -> None:
    market_grade, reasons = decide_market_grade(
        quality="fresh",
        defect_ratio=0.02,
        mask_area_ratio=0.03,
    )

    assert market_grade in {"Domestic Grade", "Reject"}
    assert reasons


def test_fresh_moderately_high_defect_returns_need_recheck_not_reject() -> None:
    market_grade, reasons = decide_market_grade(
        quality="fresh",
        defect_ratio=0.18,
        mask_area_ratio=0.40,
        circularity=0.80,
    )

    assert market_grade == "Need Recheck"
    assert reasons



