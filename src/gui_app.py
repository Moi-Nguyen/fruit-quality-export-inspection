"""Tkinter GUI for fruit quality inspection and export assessment."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from src.config import MODELS_DIR
from src.predict import IMPORTANT_FEATURES, predict_image

IMAGE_FILE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp"),
    ("PNG files", "*.png"),
    ("JPEG files", "*.jpg *.jpeg"),
    ("Bitmap files", "*.bmp"),
    ("All files", "*.*"),
]


class FruitQualityGUI:
    """Simple Tkinter desktop app for the fruit inspection pipeline."""

    def __init__(self, root: tk.Tk) -> None:
        """Create the GUI layout and initial empty state."""
        self.root = root
        self.root.title("Fruit Quality Inspection")
        self.root.geometry("1100x750")
        self.selected_image_path: Path | None = None

        self.path_var = tk.StringVar(value="No image selected")
        self.fruit_type_var = tk.StringVar(value="-")
        self.quality_var = tk.StringVar(value="-")
        self.export_var = tk.StringVar(value="-")
        self.market_grade_var = tk.StringVar(value="-")
        self.market_grade_label: ttk.Label | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Selected image:").pack(side=tk.LEFT)
        ttk.Label(top_frame, textvariable=self.path_var).pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)
        ttk.Button(top_frame, text="Load Image", command=self.select_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_frame, text="Run Analysis", command=self.run_analysis).pack(side=tk.LEFT, padx=4)

        main_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        main_frame.pack(fill=tk.BOTH, expand=True)

        image_frame = ttk.LabelFrame(main_frame, text="Image Analysis", padding=8)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(7.5, 4.2), dpi=100)
        self.axes = self.figure.subplots(1, 3)
        for axis, title in zip(self.axes, ["Original", "Fruit Mask", "Defect Map"]):
            axis.set_title(title)
            axis.axis("off")
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        result_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding=10)
        result_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self._add_result_row(result_frame, "Fruit type:", self.fruit_type_var, 0)
        self._add_result_row(result_frame, "Quality:", self.quality_var, 1)
        self._add_result_row(result_frame, "Export suitability:", self.export_var, 2)
        self.market_grade_label = self._add_result_row(
            result_frame,
            "Final market grade:",
            self.market_grade_var,
            3,
        )

        bottom_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        feature_frame = ttk.LabelFrame(bottom_frame, text="Important Features", padding=8)
        feature_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.feature_table = ttk.Treeview(
            feature_frame,
            columns=("feature", "value"),
            show="headings",
            height=12,
        )
        self.feature_table.heading("feature", text="Feature")
        self.feature_table.heading("value", text="Value")
        self.feature_table.column("feature", width=160, anchor=tk.W)
        self.feature_table.column("value", width=120, anchor=tk.E)
        self.feature_table.pack(fill=tk.BOTH, expand=True)

        explanation_frame = ttk.LabelFrame(bottom_frame, text="Decision Explanation", padding=8)
        explanation_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.explanation_text = tk.Text(explanation_frame, height=12, wrap=tk.WORD)
        self.explanation_text.pack(fill=tk.BOTH, expand=True)
        self.explanation_text.configure(state=tk.DISABLED)

    def _add_result_row(
        self,
        parent: ttk.Frame,
        label_text: str,
        value_var: tk.StringVar,
        row: int,
    ) -> ttk.Label:
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=4)
        value_label = ttk.Label(parent, textvariable=value_var, font=("TkDefaultFont", 10, "bold"))
        value_label.grid(
            row=row,
            column=1,
            sticky=tk.W,
            pady=4,
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

        try:
            result = predict_image(
                image_path=self.selected_image_path,
                fruit_model_path=fruit_model_path,
                quality_model_path=quality_model_path,
            )
        except Exception as error:
            messagebox.showerror("Prediction failed", f"Could not analyze the image:\n\n{error}")
            return

        self.update_result_text(result)
        self.update_feature_table(result)
        self.update_image_display(result)

    def update_result_text(self, result: dict[str, object]) -> None:
        """Update prediction labels and explanation text."""
        self.fruit_type_var.set(str(result.get("fruit_type", "-")))
        self.quality_var.set(str(result.get("quality", "-")))
        self.export_var.set(str(result.get("export_suitability", "-")))

        reasons = result.get("export_reasons", [])
        if isinstance(reasons, list) and reasons:
            explanation = "\n".join(f"- {reason}" for reason in reasons)
        else:
            explanation = "No export suitability reasons were returned."

        self.explanation_text.configure(state=tk.NORMAL)
        self.explanation_text.delete("1.0", tk.END)
        self.explanation_text.insert(tk.END, explanation)
        self.explanation_text.configure(state=tk.DISABLED)


    def update_market_grade_color(self, market_grade: str) -> None:
        """Apply a simple color cue for the final market grade."""
        if self.market_grade_label is None:
            return

        grade_colors = {
            "Export Grade": "green",
            "Domestic Grade": "orange",
            "Reject": "red",
        }
        self.market_grade_label.configure(foreground=grade_colors.get(market_grade, "black"))

    def update_feature_table(self, features: dict[str, object]) -> None:
        """Show the important handcrafted feature values."""
        for item_id in self.feature_table.get_children():
            self.feature_table.delete(item_id)

        for feature_name in IMPORTANT_FEATURES:
            value = features.get(feature_name, "-")
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
            axis.set_title(title)
            axis.axis("off")
            if isinstance(image_data, np.ndarray):
                axis.imshow(image_data, cmap=color_map)
            else:
                axis.text(0.5, 0.5, "Not available", ha="center", va="center")

        self.figure.tight_layout()
        self.canvas.draw()


def run_gui() -> None:
    """Start the Tkinter fruit inspection GUI."""
    root = tk.Tk()
    FruitQualityGUI(root)
    root.mainloop()
