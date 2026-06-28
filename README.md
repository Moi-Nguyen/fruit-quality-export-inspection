# Fruit Quality Inspection and Export Suitability Assessment

A university Image Processing and Computer Vision project for analyzing fruit images using traditional image processing and machine learning.

## Project Overview

This project will build a Python application with a graphical user interface that loads fruit images, extracts manually computed image-processing features, predicts fruit type, classifies fruit quality as Fresh or Rotten, and decides whether the fruit is suitable for export.

The project is intentionally designed for learning traditional image processing. OpenCV is not used. Core image processing algorithms will be implemented manually with NumPy, while scikit-learn will be used only for the final machine learning stage.

## Goals

- Load and inspect fruit images from a balanced sample dataset.
- Estimate image quality using brightness, contrast, and noise metrics.
- Apply adaptive preprocessing based on image quality.
- Segment fruit regions using manually implemented thresholding, morphology, and connected components.
- Extract shape, color, quality, and defect features.
- Train machine learning models for fruit type and Fresh/Rotten prediction.
- Assess export suitability using transparent rule-based logic.
- Provide a beginner-friendly Tkinter GUI for demonstrations.

## Dataset Structure

Original Kaggle dataset: `fruits-fresh-and-rotten-for-classification`

The original dataset should remain unchanged under `data/raw/`:

```text
data/raw/
+-- train/
�   +-- freshapples/
�   +-- freshbanana/
�   +-- freshoranges/
�   +-- rottenapples/
�   +-- rottenbanana/
�   +-- rottenoranges/
+-- test/
    +-- freshapples/
    +-- freshbanana/
    +-- freshoranges/
    +-- rottenapples/
    +-- rottenbanana/
    +-- rottenoranges/
```

A later sampling script will create a smaller balanced dataset under `data/sample/` using random seed `42`:

- Train: 150 images per class
- Test: 50 images per class

```text
data/sample/
+-- train/
�   +-- freshapples/
�   +-- freshbanana/
�   +-- freshoranges/
�   +-- rottenapples/
�   +-- rottenbanana/
�   +-- rottenoranges/
+-- test/
    +-- freshapples/
    +-- freshbanana/
    +-- freshoranges/
    +-- rottenapples/
    +-- rottenbanana/
    +-- rottenoranges/
```

## Technical Constraints

- Do not use OpenCV.
- Do not use `cv2.GaussianBlur`, `cv2.threshold`, `cv2.equalizeHist`, `cv2.erode`, `cv2.dilate`, `cv2.connectedComponents`, `cv2.findContours`, or similar built-in image processing functions.
- Use Pillow only for reading and writing image files.
- Use NumPy to manually implement image processing algorithms.
- Use matplotlib for visualization only.
- Use scikit-learn only for final machine learning models and evaluation.
- Use Tkinter for the GUI.
- Keep the code modular, readable, and suitable for a student project.

## Full Pipeline

1. Image loading
2. Image quality analysis
   - Brightness estimation
   - Contrast estimation
   - Noise estimation
3. Adaptive preprocessing
   - Dark image ? histogram equalization
   - Noisy image ? median filter
   - Normal image ? gaussian filter
4. Segmentation
   - RGB to grayscale
   - Otsu threshold
   - Morphology: erosion, dilation, opening, closing
   - Connected component labeling
5. Feature extraction
   - Area
   - Perimeter
   - Circularity
   - Bounding box
   - Aspect ratio
   - Color histogram
   - Brightness
   - Contrast
   - Noise level
   - Defect ratio
6. Machine Learning
   - Random Forest
   - KNN
   - SVM
7. Evaluation
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix
8. Export suitability assessment
9. GUI display and explanation

## Planned Algorithms

- Manual RGB to grayscale conversion with NumPy.
- Manual histogram computation and histogram equalization.
- Manual gaussian and median filtering.
- Manual Otsu thresholding.
- Manual binary morphology operations.
- Manual connected component labeling.
- Manual feature extraction from masks and image arrays.
- scikit-learn classifiers for fruit type and Fresh/Rotten prediction.

## Export Suitability Logic

Initial planned rules:

- Rotten ? Not Suitable
- Fresh + low defect ratio ? Suitable
- Too dark or high noise ? Need Recheck
- High defect ratio ? Not Suitable
- Abnormal circularity or shape ? Not Suitable or Need Recheck

The final decision should include a short human-readable explanation for the GUI.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate with:

```bash
source .venv/bin/activate
```

## Running the Project

Run the placeholder entry point:

```bash
python main.py
```


## Step 1: Image Loading and Quality Analysis

Purpose: this first algorithmic step loads one fruit image, converts it from RGB to grayscale using the manual weighted formula, and measures basic image quality using brightness, contrast, and an estimated noise level.

Run the analysis on one sampled image:

```bash
python main.py --analyze-image data/sample/test/freshapples/<image_name>
```

The command prints the image quality metrics and saves a side-by-side original/grayscale figure under `outputs/figures/`.

This step supports later adaptive preprocessing decisions:

- Dark image -> histogram equalization
- Noisy image -> median filter
- Normal image -> gaussian filter

The thresholds are initial empirical values and will be adjusted later using parameter sweep experiments.

## Step 4: Feature Extraction

Purpose: feature extraction converts each segmented fruit image into a clear numeric feature vector that can be explained in the project defense and later used for machine learning.

The current handcrafted feature groups are:

- Shape features: fruit area, perimeter, circularity, bounding box, aspect ratio, centroid, and mask area ratio.
- Color features: RGB mean, RGB standard deviation, and normalized RGB histograms inside the fruit mask.
- Texture/quality features: grayscale statistics, gradient statistics, brightness, contrast, and noise level.
- Defect-related features: a simple heuristic defect map, defect area, and defect ratio inside the fruit region.

Note: the defect map is a heuristic feature based on brightness/color rules inside the fruit mask; it supports machine learning, but is not perfect defect segmentation.

Run feature extraction on one sampled image:

```bash
python main.py --extract-features data/sample/test/freshapples/<image_name>
```

The command prints important features and saves a visualization under `outputs/figures/` showing the original image, fruit mask, and defect map.

These handcrafted features will later be exported to CSV and used to train Random Forest, KNN, and SVM models.
## Dataset Preparation

Place the original Kaggle dataset under `data/raw/` with `train/` and `test/` splits. Keep the class folder names unchanged:

- `freshapples`
- `freshbanana`
- `freshoranges`
- `rottenapples`
- `rottenbanana`
- `rottenoranges`

The raw dataset is ignored by Git and should not be committed. Create a smaller balanced dataset under `data/sample/` with:

- Train: 150 images per class
- Test: 50 images per class
- Random seed: 42

Check the current raw and sampled dataset counts:

```bash
python main.py --check-data
```

Create the sampled dataset:

```bash
python main.py --sample-data
```

Check counts again after sampling:

```bash
python main.py --check-data
```

Future commands will include model training, prediction, and GUI launch.

## Current Project Status

This repository currently contains the initial project skeleton only. Modules include docstrings, type hints, and TODO placeholders. Full image processing algorithms, model training, prediction, and GUI behavior are not implemented yet.

## Next Development Steps

1. Implement dataset sampling from `data/raw/` to `data/sample/`.
2. Implement image loading and NumPy conversion utilities.
3. Add quality metrics for brightness, contrast, and noise.
4. Implement manual filters, histogram equalization, thresholding, morphology, and connected components.
5. Add feature extraction and export suitability rules.
6. Train and evaluate baseline ML models.
7. Build the Tkinter GUI and visualization outputs.


## Step 2: Adaptive Preprocessing

The purpose of this step is to prepare fruit images before future segmentation and feature extraction. It uses the Step 1 quality analysis result to choose a preprocessing method that has a clear purpose in the computer vision pipeline.

Implemented algorithms:

- Gaussian filter: smooths normal images and reduces small intensity variations.
- Median filter: reduces impulse-style noise while preserving edges better than averaging.
- Histogram equalization: improves dark or low-contrast grayscale images by spreading intensity values.

Adaptive decision logic:

- Dark image -> histogram equalization
- Low contrast image -> histogram equalization
- Noisy image -> median filter
- Normal image -> gaussian filter

Run adaptive preprocessing on one image:

```bash
python main.py --preprocess-image data/sample/test/freshapples/<image_name>
```

The command prints brightness, contrast, noise level, quality label, and the selected preprocessing method. It also saves a 1x3 visual output under `outputs/figures/` showing the original image, grayscale image, and preprocessed result.

Later parameter sweep experiments will compare:

- Gaussian sigma values
- Median kernel sizes
- Histogram equalization effects

## Step 3: Fruit Segmentation

Purpose: this step separates the main fruit region from the background so later stages can measure fruit shape, color, and defects only inside the fruit area. It makes segmentation a clear major technique in the computer vision pipeline and saves intermediate visual outputs for explanation during a defense.

Implemented algorithms:

- Manual Otsu thresholding with NumPy.
- Binary mask creation with optional inversion for bright backgrounds.
- Manual morphology: erosion, dilation, opening, and closing.
- Manual connected component labeling using BFS.
- Simple non-white background mask for fruit pixels near white backgrounds.
- Simple color difference mask for yellow and orange fruit pixels.
- Black border removal for rotated dataset images.
- Largest non-border component selection to keep the main fruit object.

Segmentation improvement for rotated dataset images:

- Black triangular border artifacts are removed from the candidate mask.
- Non-white background masking helps separate fruit from white backgrounds.
- Color difference masking helps banana and orange images where grayscale Otsu alone is weak.
- Largest non-border component selection avoids border artifacts and background regions touching the image edge.

Segmentation pipeline:

```text
original image
-> grayscale
-> adaptive preprocessing
-> Otsu threshold
-> combine Otsu, non-white, and color-difference masks
-> morphology cleanup
-> largest non-border fruit component
```

Run segmentation on one sampled image:

```bash
python main.py --segment-image data/sample/test/freshapples/<image_name>
```

The command prints brightness, contrast, noise level, quality label, selected preprocessing method, Otsu threshold, combined mask area, and final fruit mask area. It also saves a visualization under `outputs/figures/` showing the original image, grayscale image, preprocessed image, initial Otsu mask, combined candidate mask, cleaned morphology mask, and final fruit mask.

Later parameter sweep experiments will compare:

- Morphology kernel size: 3, 5, 7
- Connected component connectivity: 4 vs 8
- Otsu segmentation before and after preprocessing

## Step 5: Export Feature CSV

Purpose: convert the sampled image dataset into feature tables that can be used for machine learning.

Output files:

- `outputs/features/train_features.csv`
- `outputs/features/test_features.csv`

Run batch feature extraction:

```bash
python main.py --export-features
```

Each CSV row contains metadata, labels, and handcrafted image features. The `class_name` value comes from the dataset folder name. The `fruit_type` label is one of `apple`, `banana`, or `orange`. The `quality` label is either `fresh` or `rotten`.

These CSV files will be used in the next step to train Random Forest, KNN, and SVM models.
