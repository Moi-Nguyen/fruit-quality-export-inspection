"""Smoke tests for the Tkinter GUI module."""

from __future__ import annotations


class FakeStringVar:
    def __init__(self) -> None:
        self.value = "-"

    def set(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


class FakeText:
    def __init__(self) -> None:
        self.content = ""
        self.states = []

    def configure(self, **kwargs: object) -> None:
        self.states.append(kwargs)

    def delete(self, start: str, end: str) -> None:
        self.content = ""

    def insert(self, index: str, text: str) -> None:
        self.content += text

class FakeFeatureTable:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str]] = []

    def get_children(self) -> list[int]:
        return list(range(len(self.rows)))

    def delete(self, item_id: int) -> None:
        if self.rows:
            self.rows.pop(0)

    def insert(self, parent: str, index: str, values: tuple[str, str]) -> None:
        self.rows.append(values)

class FakeAxis:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def clear(self) -> None:
        self.calls.append(("clear", None))

    def set_title(self, title: str) -> None:
        self.calls.append(("set_title", title))

    def axis(self, value: str) -> None:
        self.calls.append(("axis", value))

    def text(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("text", args))

    def set_facecolor(self, color: str) -> None:
        self.calls.append(("set_facecolor", color))

class FakeFigure:
    def __init__(self) -> None:
        self.tight_layout_called = False

    def tight_layout(self) -> None:
        self.tight_layout_called = True

class FakeCanvas:
    def __init__(self) -> None:
        self.draw_called = False

    def draw(self) -> None:
        self.draw_called = True

class FakeLabel:
    def __init__(self) -> None:
        self.options: dict[str, object] = {}

    def configure(self, **kwargs: object) -> None:
        self.options.update(kwargs)


def test_gui_app_can_be_imported() -> None:
    import src.gui_app as gui_app

    assert gui_app is not None


def test_run_gui_exists() -> None:
    from src.gui_app import run_gui

    assert callable(run_gui)


def test_format_confidence_handles_missing_invalid_and_numeric_values() -> None:
    from src.gui_app import FruitQualityGUI

    gui = FruitQualityGUI.__new__(FruitQualityGUI)

    assert gui._format_confidence(None) == "-"
    assert gui._format_confidence("not-a-number") == "-"
    assert gui._format_confidence(float("nan")) == "-"
    assert gui._format_confidence(0.923) == "92.3%"
    assert gui._format_confidence("0.5") == "50.0% (low)"
    assert gui._format_confidence(92.345) == "92.3%"
    assert gui._format_confidence(44) == "44.0% (low)"


def test_update_result_text_displays_market_grade_and_all_reasons() -> None:
    from src.gui_app import FruitQualityGUI

    gui = FruitQualityGUI.__new__(FruitQualityGUI)
    gui.fruit_type_var = FakeStringVar()
    gui.quality_var = FakeStringVar()
    gui.fruit_type_confidence_var = FakeStringVar()
    gui.quality_confidence_var = FakeStringVar()
    gui.export_var = FakeStringVar()
    gui.market_grade_var = FakeStringVar()
    gui.processing_time_var = FakeStringVar()
    gui.explanation_text = FakeText()
    gui.colored_market_grade = None
    gui.colored_export_suitability = None

    def fake_update_market_grade_color(market_grade: str) -> None:
        gui.colored_market_grade = market_grade

    def fake_update_export_suitability_color(export_suitability: str) -> None:
        gui.colored_export_suitability = export_suitability

    gui.update_market_grade_color = fake_update_market_grade_color
    gui.update_export_suitability_color = fake_update_export_suitability_color

    gui.update_result_text(
        {
            "fruit_type": "orange",
            "quality": "rotten",
            "fruit_type_confidence": 0.923,
            "quality_confidence": 87.65,
            "export_suitability": "Not Suitable",
            "market_grade": "Reject",
            "processing_time_seconds": 1.234,
            "export_reasons": ["Fruit is rotten."],
            "consistency_warnings": ["Model evidence is inconsistent; manual review is recommended."],
            "market_grade_reasons": ["Rotten fruit must be rejected."],
        }
    )

    assert gui.fruit_type_var.get() == "orange"
    assert gui.quality_var.get() == "rotten"
    assert gui.fruit_type_confidence_var.get() == "92.3%"
    assert gui.quality_confidence_var.get() == "87.7%"
    assert gui.export_var.get() == "Not Suitable"
    assert gui.market_grade_var.get() == "Reject"
    assert gui.processing_time_var.get() == "1.23 s"
    assert gui.colored_market_grade == "Reject"
    assert gui.colored_export_suitability == "Not Suitable"
    assert "Fruit is rotten." in gui.explanation_text.content
    assert "Consistency warning:" in gui.explanation_text.content
    assert "Model evidence is inconsistent" in gui.explanation_text.content
    assert "Rotten fruit must be rejected." in gui.explanation_text.content

def test_update_feature_table_skips_missing_optional_features_safely() -> None:
    from src.gui_app import FruitQualityGUI

    gui = FruitQualityGUI.__new__(FruitQualityGUI)
    gui.feature_table = FakeFeatureTable()

    gui.update_feature_table(
        {
            "area": 1234,
            "circularity": 0.81234,
            "brightness": 141.23456,
            "defect_ratio": 0.021,
            "fruit_type_confidence": 0.923,
        }
    )

    assert gui.feature_table.rows == [
        ("area", "1234.0000"),
        ("circularity", "0.8123"),
        ("brightness", "141.2346"),
        ("defect_ratio", "0.0210"),
        ("fruit_type_confidence", "0.9230"),
    ]


def test_clear_results_resets_labels_tables_explanation_and_images() -> None:
    from src.gui_app import FruitQualityGUI

    gui = FruitQualityGUI.__new__(FruitQualityGUI)
    gui.selected_image_path = "demo.png"
    gui.path_var = FakeStringVar()
    gui.fruit_type_var = FakeStringVar()
    gui.quality_var = FakeStringVar()
    gui.fruit_type_confidence_var = FakeStringVar()
    gui.quality_confidence_var = FakeStringVar()
    gui.export_var = FakeStringVar()
    gui.market_grade_var = FakeStringVar()
    gui.processing_time_var = FakeStringVar()
    gui.status_var = FakeStringVar()
    gui.status_label = None
    gui.export_value_label = None
    gui.feature_table = FakeFeatureTable()
    gui.feature_table.rows = [("area", "123.0000")]
    gui.explanation_text = FakeText()
    gui.axes = [FakeAxis(), FakeAxis(), FakeAxis()]
    gui.figure = FakeFigure()
    gui.canvas = FakeCanvas()
    gui.market_grade_label = None

    gui.clear_results()

    assert gui.selected_image_path is None
    assert gui.path_var.get() == "No image selected"
    assert gui.fruit_type_var.get() == "-"
    assert gui.quality_var.get() == "-"
    assert gui.fruit_type_confidence_var.get() == "-"
    assert gui.quality_confidence_var.get() == "-"
    assert gui.export_var.get() == "-"
    assert gui.market_grade_var.get() == "-"
    assert gui.processing_time_var.get() == "-"
    assert gui.feature_table.rows == []
    assert "Decision details will appear here" in gui.explanation_text.content
    assert gui.status_var.get() == "Ready"
    assert gui.figure.tight_layout_called is True
    assert gui.canvas.draw_called is True
    assert all(("text", (0.5, 0.5, "No image")) in axis.calls for axis in gui.axes)

def test_set_status_updates_badge_style() -> None:
    from src.gui_app import FruitQualityGUI

    gui = FruitQualityGUI.__new__(FruitQualityGUI)
    gui.status_var = FakeStringVar()
    gui.status_label = FakeLabel()

    gui._set_status("Processing...")

    assert gui.status_var.get() == "Processing..."
    assert gui.status_label.options["style"] == "StatusProcessing.TLabel"

def test_grade_and_export_color_helpers_apply_expected_cues() -> None:
    from src.gui_app import FruitQualityGUI

    gui = FruitQualityGUI.__new__(FruitQualityGUI)
    gui.market_grade_label = FakeLabel()
    gui.export_value_label = FakeLabel()

    gui.update_market_grade_color("Domestic Grade")
    gui.update_export_suitability_color("Not Suitable")

    assert gui.market_grade_label.options["foreground"] == "#c46a00"
    assert gui.export_value_label.options["foreground"] == "#b42318"
