"""Tests for rule-based export suitability assessment."""

from src.export_rules import assess_export_suitability


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
