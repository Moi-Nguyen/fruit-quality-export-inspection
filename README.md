# Fruit Quality Inspection and Export Suitability Assessment

Traditional image processing and machine learning project for classifying fruit type, estimating fruit quality, and deciding export suitability from images.

## Project Overview

This project analyzes fruit images using explainable computer vision methods. The system loads an image, evaluates image quality, preprocesses it adaptively, segments the fruit region, extracts handcrafted features, predicts fruit type and quality with machine learning, and produces an export suitability decision.

The project is designed for a university image processing and computer vision course. Core image-processing algorithms are implemented with NumPy instead of OpenCV so that each stage can be explained during review or oral defense.

## Main Objectives

- Build a complete image-processing pipeline for apple, banana, and orange images.
- Classify fruit quality as `fresh` or `rotten` using handcrafted features.
- Assess export suitability with transparent rule-based decisions.
- Provide confidence handling and `Need Recheck` decisions for uncertain cases.
- Provide a polished Tkinter GUI for demonstration.
- Keep the implementation explainable, testable, and suitable for academic defense.

## Technology Constraints

- Use NumPy for core image processing algorithms.
- Use PIL only for image loading, saving, and resizing.
- Use matplotlib for visualization outputs.
- Use scikit-learn only for machine learning models and evaluation.
- Do not use OpenCV image-processing functions.
- Do not use scipy, scikit-image, deep learning, or copied external algorithms.
- Keep algorithms readable and explainable for a traditional computer vision project.

## Complete Pipeline

1. **Image loading**: PIL loads the image and converts it to NumPy arrays for manual processing.
2. **Image quality analysis**: The system measures brightness, contrast, and estimated noise level.
3. **Adaptive preprocessing**: The system selects enhancement or smoothing steps based on image quality.
4. **Segmentation**: The fruit region is separated using thresholding, morphology, and connected components.
5. **Feature extraction**: Shape, color, texture, quality, and defect-related features are extracted from the fruit mask.
6. **Machine learning training**: KNN, SVM, and Random Forest models are trained and compared.
7. **Single image prediction**: The best saved models predict fruit type and quality for one image.
8. **Export suitability assessment**: Rule-based logic combines prediction, evidence, confidence, and defect indicators.
9. **GUI application**: A Tkinter interface allows image selection, prediction, and visual explanation.
10. **End-to-end evaluation**: The full prediction pipeline is tested on sampled test images.
11. **External image robustness**: The pipeline includes safeguards for difficult lighting, rotation, and background cases.
12. **Model-evidence consistency check**: The final decision checks whether model output agrees with image evidence.

## Project Folder Structure

```text
fruit-quality-export-inspection/
├── data/
│   ├── raw/                 # Original dataset folders
│   └── sample/              # Balanced train/test sample dataset
├── models/                  # Saved trained models and metadata
├── outputs/
│   ├── features/            # Exported CSV feature files
│   ├── figures/             # Prediction and processing visualizations
│   ├── masks/               # Segmentation masks
│   └── reports/             # Evaluation reports and prediction CSV files
├── reports/                 # Project notes and defense preparation reports
├── scripts/                 # Helper scripts
├── src/                     # Main source code modules
├── tests/                   # Automated tests
├── main.py                  # Command-line entry point and GUI launcher
├── requirements.txt         # Python dependencies
└── README.md                # Final project guide
```

## Installation

Run these commands in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, run PowerShell as a user with script execution permission or activate the environment from another terminal.

## Dataset Layout

The project expects a fruit dataset arranged by class folders. The sampled project dataset uses train/test folders and fruit-quality categories.

Example layout:

```text
data/sample/
├── train/
│   ├── freshapples/
│   ├── freshbanana/
│   ├── freshoranges/
│   ├── rottenapples/
│   ├── rottenbanana/
│   └── rottenoranges/
└── test/
    ├── freshapples/
    ├── freshbanana/
    ├── freshoranges/
    ├── rottenapples/
    ├── rottenbanana/
    └── rottenoranges/
```

The labels are inferred from folder names. For example, `freshapples` maps to fruit type `apple` and quality `fresh`.

## How to Run

Run commands from the project root. Image paths containing spaces must be wrapped in quotes.

### Run Tests

```powershell
python -m pytest -q
```

### Export Features

```powershell
python main.py --export-features
```

This command extracts features from the sample dataset and writes CSV files under `outputs/features/`.

### Train Models

```powershell
python main.py --train-models
```

This command trains and compares KNN, SVM, and Random Forest models for fruit type and quality classification.

### Predict One Image

```powershell
python main.py --predict-image "path/to/image.jpg"
```

Use quotes when the image path contains spaces:

```powershell
python main.py --predict-image "C:/Users/Student/Desktop/test fruit.jpg"
```

### Evaluate System

```powershell
python main.py --evaluate-system
```

For a faster demo evaluation, limit the number of images per class:

```powershell
python main.py --evaluate-system --eval-max-per-class 5
```

### Launch GUI

```powershell
python main.py --gui
```

The GUI lets the user select an image, run prediction, view the processed result, and explain the final grade.

## Final Evaluation Results

The latest saved reports in `outputs/reports/` contain these documented results.

| Evaluation | Fruit Type Accuracy | Quality Accuracy | Notes |
| --- | ---: | ---: | --- |
| Model evaluation report | `0.9467` best Random Forest | `0.9600` best SVM / Random Forest | Report file: `outputs/reports/ml_evaluation_report.txt` |
| End-to-end evaluation report | `0.97` | `0.93` | Report file: `outputs/reports/end_to_end_evaluation_report.txt`, evaluated on 30 sampled test images |

The known full-system performance is therefore approximately `0.95` fruit type accuracy and approximately `0.94` quality accuracy, depending on whether the metric comes from model evaluation or full end-to-end evaluation.

## GUI Demo Notes

- Start with `python main.py --gui`.
- Select a clear fruit image first to show the normal pipeline.
- Explain the displayed prediction as a combination of model output, confidence, visual evidence, and rule-based grading.
- Use `Need Recheck` as a safety mechanism, not as a failure state.
- For external images, mention that unusual lighting, cluttered backgrounds, occlusion, or multiple fruits may reduce reliability.

## Current Limitations

- The system is trained on a limited fruit set: apple, banana, and orange.
- The segmentation pipeline works best when one fruit is visible and the background is not too complex.
- External internet images may contain lighting, compression, scale, shadows, or backgrounds unlike the training data.
- The quality decision depends on handcrafted features, so subtle defects may be missed.
- The GUI is intended for demonstration and not for industrial real-time deployment.

## Future Improvements

- Add more fruits, more defect types, and more varied capture conditions.
- Capture a conveyor-belt dataset with consistent camera distance and lighting.
- Add calibration panels or controlled illumination to reduce color variation.
- Improve segmentation for cluttered backgrounds and multiple fruit instances.
- Add real-time camera input and batch inspection logging.
- Compare the traditional pipeline with a deep learning baseline in future research, while preserving explainability.

## Defense Notes / Lecturer Questions

### Why use traditional image processing instead of deep learning?

The project goal is to demonstrate image processing concepts directly. Traditional methods make every step explainable: preprocessing, segmentation, feature extraction, and classification. Deep learning may improve accuracy with enough data, but it is harder to explain and requires more training resources.

### Why avoid OpenCV?

Avoiding OpenCV forces the team to implement the core algorithms manually with NumPy. This helps prove understanding of thresholding, morphology, connected components, and feature extraction instead of calling ready-made library functions.

### Why use adaptive preprocessing?

Fruit images have different brightness, contrast, and noise. Adaptive preprocessing lets the system choose a suitable correction for the current image instead of applying one fixed operation to every case.

### How does Otsu thresholding work?

Otsu thresholding searches for the intensity threshold that best separates pixels into two groups. It chooses the threshold that maximizes separation between foreground and background intensity distributions.

### Why use morphology and connected components?

Morphology removes small holes and noise in the binary mask. Connected components then identify separate regions so the system can keep the most likely fruit region and ignore small background artifacts.

### What features are extracted?

The system extracts shape features, color statistics, normalized color histograms, grayscale and gradient statistics, brightness and contrast values, estimated noise, defect area, and defect ratio.

### Why use KNN, SVM, and Random Forest?

These models are suitable for handcrafted tabular features. KNN provides a simple distance-based baseline, SVM handles decision boundaries well, and Random Forest handles nonlinear feature interactions while remaining easier to inspect than deep learning.

### Why use confidence scores?

Confidence scores help the system avoid overconfident decisions. If a prediction is uncertain, the final rule can choose `Need Recheck` instead of forcing an export or reject decision.

### Why does the system use Need Recheck?

`Need Recheck` protects the final decision when model confidence is low or image evidence conflicts with the predicted label. This is important because export inspection should prefer safety over risky automatic approval.

### Why can external images still fail?

External images may have different backgrounds, lighting, camera angles, image compression, multiple objects, or fruit varieties not represented in the training set. These changes can affect segmentation and extracted features.

### What are current limitations?

The system supports only a small set of fruits, relies on handcrafted features, and assumes a mostly visible single fruit. It is not yet a production system for uncontrolled factory environments.

### How can the system be improved for conveyor-belt deployment?

Use fixed lighting, a fixed camera, controlled fruit placement, real-time image capture, batch logging, rejection hardware integration, and a larger dataset collected from the actual conveyor-belt environment.
