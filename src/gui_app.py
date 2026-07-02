"""Tkinter GUI for fruit quality inspection and export assessment."""

from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from src.config import MODELS_DIR
from src.export_rules import LOW_CONFIDENCE_THRESHOLD
from src.predict import predict_image

IMAGE_FILE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp"),
    ("PNG files", "*.png"),
    ("JPEG files", "*.jpg *.jpeg"),
    ("Bitmap files", "*.bmp"),
    ("All files", "*.*"),
]

DEMO_FEATURES = [
    "area",
    "perimeter",
    "circularity",
    "aspect_ratio",
    "mask_area_ratio",
    "mean_r",
    "mean_g",
    "mean_b",
    "brightness",
    "contrast",
    "noise_level",
    "defect_ratio",
    "brown_dark_ratio",
    "red_ratio",
    "yellow_ratio",
    "orange_ratio",
    "green_ratio",
    "fruit_type_confidence",
    "quality_confidence",
]


class FruitQualityGUI:
    """Simple Tkinter desktop app for the fruit inspection pipeline."""

    PANEL_TITLES = ("Original", "Fruit Mask", "Defect Map")

    def __init__(self, root: tk.Tk) -> None:
        """Create the GUI layout and initial empty state."""
        self.root = root
        self.root.title("Fruit Quality Inspection")
        self.root.geometry("1280x860")
        self.root.minsize(1080, 720)
        self.selected_image_path: Path | None = None

        self.path_var = tk.StringVar(value="No image selected")
        self.fruit_type_var = tk.StringVar(value="-")
        self.quality_var = tk.StringVar(value="-")
        self.fruit_type_confidence_var = tk.StringVar(value="-")
        self.quality_confidence_var = tk.StringVar(value="-")
        self.export_var = tk.StringVar(value="-")
        self.market_grade_var = tk.StringVar(value="-")
        self.status_var = tk.StringVar(value="Ready")
        self.processing_time_var = tk.StringVar(value="-")
        self.market_grade_label: ttk.Label | None = None
        self.status_label: ttk.Label | None = None
        self.export_value_label: ttk.Label | None = None
        self.clear_button: ttk.Button | None = None
        self.load_button: ttk.Button | None = None
        self.run_button: ttk.Button | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        self._configure_styles()
        self.root.configure(bg="#F4F5EE")

        page_frame = ttk.Frame(self.root, style="App.TFrame", padding=(20, 18, 20, 18))
        page_frame.pack(fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(page_frame, style="HeaderBar.TFrame", padding=(18, 14, 18, 14))
        header_frame.pack(fill=tk.X)
        ttk.Label(
            header_frame,
            text="Fruit Quality Inspection and Export Suitability Assessment",
            style="Header.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            header_frame,
            text="Traditional Image Processing + Machine Learning",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        top_frame = ttk.Frame(page_frame, style="Toolbar.TFrame", padding=(14, 12, 14, 12))
        top_frame.pack(fill=tk.X, pady=(12, 14))
        top_frame.columnconfigure(1, weight=1)

        ttk.Label(top_frame, text="Selected image", style="Muted.TLabel").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(top_frame, textvariable=self.path_var, style="Path.TLabel", anchor=tk.W).grid(
            row=0, column=1, sticky="ew", padx=(10, 16)
        )
        self.load_button = ttk.Button(top_frame, text="Load Image", command=self.select_image)
        self.load_button.grid(row=0, column=2, padx=4)
        self.run_button = ttk.Button(top_frame, text="Run Analysis", command=self.run_analysis, style="Accent.TButton")
        self.run_button.grid(row=0, column=3, padx=4)
        self.clear_button = ttk.Button(top_frame, text="Clear", command=self.clear_results)
        self.clear_button.grid(row=0, column=4, padx=4)
        self.status_label = ttk.Label(top_frame, textvariable=self.status_var, style="StatusReady.TLabel")
        self.status_label.grid(row=0, column=5, padx=(12, 0))

        main_frame = ttk.Frame(page_frame, style="App.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=5)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)

        image_frame = ttk.LabelFrame(main_frame, text="Image Analysis", padding=14, style="Card.TLabelframe")
        image_frame.grid(row=0, column=0, sticky="nsew")

        self.figure = Figure(figsize=(8.8, 4.8), dpi=100, facecolor="#FEFEFB")
        self.axes = self.figure.subplots(1, 3)
        for axis, title in zip(self.axes, self.PANEL_TITLES):
            self._style_image_axis(axis, title)
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        result_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding=16, style="Card.TLabelframe")
        result_frame.grid(row=0, column=1, sticky="nsew", padx=(14, 0))
        result_frame.columnconfigure(1, weight=1)
        self._add_result_row(result_frame, "Fruit type:", self.fruit_type_var, 0)
        self._add_result_row(result_frame, "Quality:", self.quality_var, 1)
        self._add_result_row(result_frame, "Fruit confidence:", self.fruit_type_confidence_var, 2)
        self._add_result_row(result_frame, "Quality confidence:", self.quality_confidence_var, 3)
        self.export_value_label = self._add_result_row(result_frame, "Export suitability:", self.export_var, 4)
        self.market_grade_label = self._add_result_row(
            result_frame,
            "Final market grade:",
            self.market_grade_var,
            5,
            value_style="MarketGradeBadge.TLabel",
        )
        self._add_result_row(result_frame, "Processing time:", self.processing_time_var, 6)

        bottom_frame = ttk.Frame(page_frame, style="App.TFrame")
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=(14, 0))
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)
        bottom_frame.rowconfigure(0, weight=1)

        feature_frame = ttk.LabelFrame(bottom_frame, text="Important Features", padding=12, style="Card.TLabelframe")
        feature_frame.grid(row=0, column=0, sticky="nsew")

        self.feature_table = ttk.Treeview(
            feature_frame,
            columns=("feature", "value"),
            show="headings",
            height=14,
        )
        self.feature_table.heading("feature", text="Feature")
        self.feature_table.heading("value", text="Value")
        self.feature_table.column("feature", width=160, anchor=tk.W)
        self.feature_table.column("value", width=120, anchor=tk.E)
        feature_scrollbar = ttk.Scrollbar(feature_frame, orient=tk.VERTICAL, command=self.feature_table.yview)
        self.feature_table.configure(yscrollcommand=feature_scrollbar.set)
        self.feature_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        feature_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        explanation_frame = ttk.LabelFrame(bottom_frame, text="Decision Explanation", padding=12, style="Card.TLabelframe")
        explanation_frame.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        explanation_text_frame = ttk.Frame(explanation_frame, style="Card.TFrame")
        explanation_text_frame.pack(fill=tk.BOTH, expand=True)
        self.explanation_text = tk.Text(
            explanation_text_frame,
            height=12,
            wrap=tk.WORD,
            bg="#FEFEFB",
            fg="#1F2A1A",
            relief=tk.FLAT,
            padx=14,
            pady=12,
            font=("Segoe UI", 10),
            spacing1=2,
            spacing3=8,
        )
        explanation_scrollbar = ttk.Scrollbar(
            explanation_text_frame,
            orient=tk.VERTICAL,
            command=self.explanation_text.yview,
        )
        self.explanation_text.configure(yscrollcommand=explanation_scrollbar.set)
        self.explanation_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        explanation_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.explanation_text.configure(state=tk.DISABLED)

    def _configure_styles(self) -> None:
        """Configure ttk styles for a cleaner dashboard presentation."""
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background="#F4F5EE")
        style.configure("HeaderBar.TFrame", background="#F4F5EE")
        style.configure("Toolbar.TFrame", background="#EAF1E3", relief="solid", borderwidth=1)
        style.configure("Card.TFrame", background="#FEFEFB")
        style.configure("Header.TLabel", background="#F4F5EE", foreground="#0B3D16", font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background="#F4F5EE", foreground="#5E6F5B", font=("Segoe UI", 11))
        style.configure("Muted.TLabel", background="#EAF1E3", foreground="#5E6F5B", font=("Segoe UI", 9, "bold"))
        style.configure("Path.TLabel", background="#F7F8F2", foreground="#2E3F30", padding=(10, 6), font=("Consolas", 9))
        style.configure("ResultKey.TLabel", background="#FEFEFB", foreground="#5E6F5B", font=("Segoe UI", 10))
        style.configure("ResultValue.TLabel", background="#FEFEFB", foreground="#0B3D16", font=("Segoe UI", 11, "bold"))
        style.configure("MarketGradeBadge.TLabel", background="#E3E8DD", foreground="#0B3D16", padding=(14, 8), font=("Segoe UI", 13, "bold"))
        style.configure("StatusReady.TLabel", background="#E3E8DD", foreground="#315A3A", padding=(14, 6), font=("Segoe UI", 9, "bold"), relief="solid", borderwidth=1)
        style.configure("StatusProcessing.TLabel", background="#FFE8B8", foreground="#875400", padding=(14, 6), font=("Segoe UI", 9, "bold"), relief="solid", borderwidth=1)
        style.configure("StatusComplete.TLabel", background="#DCF2D1", foreground="#1F7A3F", padding=(14, 6), font=("Segoe UI", 9, "bold"), relief="solid", borderwidth=1)
        style.configure("StatusError.TLabel", background="#F8D7DA", foreground="#9B1C1C", padding=(14, 6), font=("Segoe UI", 9, "bold"), relief="solid", borderwidth=1)
        style.configure("Card.TLabelframe", background="#FEFEFB", bordercolor="#CCD8C4", relief="solid", borderwidth=1)
        style.configure("Card.TLabelframe.Label", background="#F4F5EE", foreground="#0B3D16", font=("Segoe UI", 13, "bold"))
        style.configure("Treeview", background="#FEFEFB", fieldbackground="#FEFEFB", foreground="#1F2A1A", rowheight=26, font=("Consolas", 10))
        style.configure("Treeview.Heading", background="#EAF1E3", foreground="#0B3D16", font=("Segoe UI", 9, "bold"))
        style.configure("TButton", padding=(10, 5), font=("Segoe UI", 9))
        style.configure("Accent.TButton", background="#1F7A3F", foreground="#FFFFFF", padding=(14, 6), font=("Segoe UI", 9, "bold"))
        style.map("TButton", background=[("active", "#E1EBD8")])
        style.map("Accent.TButton", background=[("active", "#17652F"), ("disabled", "#9FB7A5")])

    def _add_result_row(
        self,
        parent: ttk.Frame,
        label_text: str,
        value_var: tk.StringVar,
        row: int,
        value_style: str = "ResultValue.TLabel",
    ) -> ttk.Label:
        ttk.Label(parent, text=label_text, style="ResultKey.TLabel").grid(row=row, column=0, sticky=tk.W, pady=8)
        value_label = ttk.Label(parent, textvariable=value_var, style=value_style)
        value_label.grid(
            row=row,
            column=1,
            sticky=tk.E,
            pady=8,
            padx=(8, 0),
        )
        return value_label

    def select_image(self) -> None:
        """Open a file picker and store the selected image path."""
        file_name = filedialog.askopenfilename(
            title="Select fruit image",
            filetypes=IMAGE_FILE_TYPES,
        )
        if not file_name:
            return

        self.selected_image_path = Path(file_name)
        self.path_var.set(str(self.selected_image_path))
        self._set_status("Ready")

    def run_analysis(self) -> None:
        """Run the existing prediction pipeline for the selected image."""
        if self.selected_image_path is None:
            messagebox.showwarning("No image selected", "Please load an image before running analysis.")
            return

        fruit_model_path = MODELS_DIR / "fruit_type_model.pkl"
        quality_model_path = MODELS_DIR / "quality_model.pkl"
        if not fruit_model_path.exists() or not quality_model_path.exists():
            messagebox.showerror(
                "Missing model files",
                "Model files are missing. Please train the models first:\n\npython main.py --train-models",
            )
            return

        self.set_processing_state(True)
        image_path = self.selected_image_path

        worker = threading.Thread(
            target=self._run_analysis_worker,
            args=(image_path, fruit_model_path, quality_model_path),
            daemon=True,
        )
        worker.start()

    def _run_analysis_worker(
        self,
        image_path: Path,
        fruit_model_path: Path,
        quality_model_path: Path,
    ) -> None:
        """Run prediction away from Tkinter and schedule UI updates safely."""
        try:
            result = predict_image(
                image_path=image_path,
                fruit_model_path=fruit_model_path,
                quality_model_path=quality_model_path,
                save_figure=False,
            )
        except Exception as error:
            self.root.after(0, self._handle_analysis_error, error)
            return

        self.root.after(0, self._handle_analysis_success, result)

    def _handle_analysis_success(self, result: dict[str, object]) -> None:
        """Update Tkinter widgets after successful background analysis."""
        self.update_result_text(result)
        self.update_feature_table(result)
        self.update_image_display(result)
        self.set_processing_state(False)
        self._set_status("Complete")

    def _handle_analysis_error(self, error: Exception) -> None:
        """Restore controls and report a background prediction error."""
        self.set_processing_state(False)
        self._set_status("Error")
        messagebox.showerror("Prediction failed", f"Could not analyze the image:\n\n{error}")

    def set_processing_state(self, is_processing: bool) -> None:
        """Enable or disable controls during background processing."""
        state = tk.DISABLED if is_processing else tk.NORMAL
        if self.run_button is not None:
            self.run_button.configure(state=state)
        if self.load_button is not None:
            self.load_button.configure(state=state)
        if self.clear_button is not None:
            self.clear_button.configure(state=state)
        self._set_status("Processing..." if is_processing else "Ready")

    def _set_status(self, status: str) -> None:
        """Update status text and badge style."""
        self.status_var.set(status)
        if self.status_label is None:
            return

        style_by_status = {
            "Ready": "StatusReady.TLabel",
            "Processing...": "StatusProcessing.TLabel",
            "Complete": "StatusComplete.TLabel",
            "Error": "StatusError.TLabel",
        }
        self.status_label.configure(style=style_by_status.get(status, "StatusReady.TLabel"))

    def clear_results(self) -> None:
        """Reset GUI state without deleting any files."""
        self.selected_image_path = None
        self.path_var.set("No image selected")
        self.fruit_type_var.set("-")
        self.quality_var.set("-")
        self.fruit_type_confidence_var.set("-")
        self.quality_confidence_var.set("-")
        self.export_var.set("-")
        self.market_grade_var.set("-")
        self.processing_time_var.set("-")
        self.update_export_suitability_color("-")
        self.update_market_grade_color("-")
        self.update_feature_table({})
        self._set_explanation("Decision details will appear here after analysis.")
        self._clear_image_display()
        self._set_status("Ready")

    def update_result_text(self, result: dict[str, object]) -> None:
        """Update prediction labels and explanation text."""
        self.fruit_type_var.set(str(result.get("fruit_type", "-")))
        self.quality_var.set(str(result.get("quality", "-")))
        if hasattr(self, "fruit_type_confidence_var"):
            self.fruit_type_confidence_var.set(self._format_confidence(result.get("fruit_type_confidence")))
        if hasattr(self, "quality_confidence_var"):
            self.quality_confidence_var.set(self._format_confidence(result.get("quality_confidence")))
        self.export_var.set(str(result.get("export_suitability", "-")))
        self.update_export_suitability_color(str(result.get("export_suitability", "-")))
        market_grade = str(result.get("market_grade", "-"))
        self.market_grade_var.set(market_grade)
        self.update_market_grade_color(market_grade)
        processing_time = result.get("processing_time_seconds", None)
        if isinstance(processing_time, (int, float, np.floating)):
            self.processing_time_var.set(f"{float(processing_time):.2f} s")
        else:
            self.processing_time_var.set("-")

        explanation_sections = []
        export_reasons = result.get("export_reasons", [])
        if isinstance(export_reasons, list) and export_reasons:
            explanation_sections.append(
                "Export suitability:\n" + "\n".join(f"- {reason}" for reason in export_reasons)
            )

        consistency_warnings = result.get("consistency_warnings", [])
        if isinstance(consistency_warnings, list) and consistency_warnings:
            explanation_sections.append(
                "Consistency warning:\n" + "\n".join(f"- {warning}" for warning in consistency_warnings)
            )

        market_grade_reasons = result.get("market_grade_reasons", [])
        if isinstance(market_grade_reasons, list) and market_grade_reasons:
            explanation_sections.append(
                "Market grade:\n" + "\n".join(f"- {reason}" for reason in market_grade_reasons)
            )

        if explanation_sections:
            explanation = "\n\n".join(explanation_sections)
        else:
            explanation = "No decision reasons were returned."

        self._set_explanation(explanation)

    def _set_explanation(self, explanation: str) -> None:
        """Replace explanation text while preserving read-only behavior."""
        self.explanation_text.configure(state=tk.NORMAL)
        self.explanation_text.delete("1.0", tk.END)
        self.explanation_text.insert(tk.END, explanation)
        self.explanation_text.configure(state=tk.DISABLED)

    def _format_confidence(self, value: object) -> str:
        """Format classifier confidence values for display."""
        if value is None:
            return "-"

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return "-"

        if not np.isfinite(confidence):
            return "-"

        if 0 <= confidence <= 1:
            confidence *= 100

        confidence_ratio = confidence / 100.0 if confidence > 1 else confidence
        low_marker = " (low)" if confidence_ratio < LOW_CONFIDENCE_THRESHOLD else ""
        return f"{confidence:.1f}%{low_marker}"


    def update_market_grade_color(self, market_grade: str) -> None:
        """Apply a simple color cue for the final market grade."""
        if self.market_grade_label is None:
            return

        grade_colors = {
            "Export Grade": ("#1f7a3f", "#E4F4E8"),
            "Domestic Grade": ("#c46a00", "#FFF0D6"),
            "Reject": ("#b42318", "#FDE1E4"),
            "Need Recheck": ("#b77900", "#FFF3C4"),
        }
        foreground, background = grade_colors.get(market_grade, ("#0B3D16", "#E3E8DD"))
        self.market_grade_label.configure(foreground=foreground, background=background)

    def update_export_suitability_color(self, export_suitability: str) -> None:
        """Apply a subtle color cue for export suitability."""
        if getattr(self, "export_value_label", None) is None:
            return

        text = export_suitability.lower()
        if "not" in text or "reject" in text:
            color = "#b42318"
        elif "recheck" in text or "review" in text:
            color = "#b77900"
        elif "suitable" in text or "export" in text:
            color = "#1f7a3f"
        else:
            color = "#0B3D16"
        self.export_value_label.configure(foreground=color)

    def update_feature_table(self, features: dict[str, object]) -> None:
        """Show the important handcrafted feature values."""
        for item_id in self.feature_table.get_children():
            self.feature_table.delete(item_id)

        for feature_name in DEMO_FEATURES:
            if feature_name not in features:
                continue

            value = features[feature_name]
            if isinstance(value, (int, float, np.floating)):
                display_value = f"{float(value):.4f}"
            else:
                display_value = str(value)
            self.feature_table.insert("", tk.END, values=(feature_name, display_value))

    def update_image_display(self, result: dict[str, object]) -> None:
        """Draw the original image, fruit mask, and defect map."""
        display_items = [
            ("Original", result.get("original_image"), None),
            ("Fruit Mask", result.get("fruit_mask"), "gray"),
            ("Defect Map", result.get("defect_map"), "gray"),
        ]

        for axis, (title, image_data, color_map) in zip(self.axes, display_items):
            axis.clear()
            self._style_image_axis(axis, title)
            if isinstance(image_data, np.ndarray):
                axis.imshow(image_data, cmap=color_map)
            else:
                axis.text(0.5, 0.5, "Not available", ha="center", va="center")

        self.figure.tight_layout()
        self.canvas.draw()

    def _clear_image_display(self) -> None:
        """Reset matplotlib panels to their empty card state."""
        for axis, title in zip(self.axes, self.PANEL_TITLES):
            axis.clear()
            self._style_image_axis(axis, title)
            axis.text(0.5, 0.5, "No image", ha="center", va="center", color="#6b7d70")
        self.figure.tight_layout()
        self.canvas.draw()

    @staticmethod
    def _style_image_axis(axis: object, title: str) -> None:
        """Apply consistent lightweight card styling to a matplotlib image axis."""
        axis.set_title(title)
        if hasattr(axis, "title"):
            axis.title.set_color("#0B3D16")
            axis.title.set_fontsize(11)
            axis.title.set_fontweight("bold")
            axis.title.set_position((0.5, 1.03))
        axis.set_facecolor("#F7F8F2")
        if hasattr(axis, "set_xticks"):
            axis.set_xticks([])
        if hasattr(axis, "set_yticks"):
            axis.set_yticks([])
        if hasattr(axis, "set_frame_on"):
            axis.set_frame_on(True)
        for spine in getattr(axis, "spines", {}).values():
            spine.set_visible(True)
            spine.set_edgecolor("#CCD8C4")
            spine.set_linewidth(1.2)


def run_gui() -> None:
    """Start the Tkinter fruit inspection GUI."""
    root = tk.Tk()
    FruitQualityGUI(root)
    root.mainloop()

