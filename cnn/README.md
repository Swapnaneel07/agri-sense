# Module 3: Leaf Disease Detection and Edge Deployment

Part of the **AgriSense – AI-Powered Smart Agriculture System**

## 1. Overview

Module 3 focuses on developing a deep learning-based plant leaf disease detection system using Convolutional Neural Networks (CNNs).

The module uses a fine-tuned **MobileNetV2** architecture to classify plant leaf images into their corresponding disease categories. The trained model is also converted into the ExecuTorch `.pte` format to support lightweight edge inference.

The objective is to provide an efficient and deployable AI-based solution for identifying plant diseases from leaf images.

## 2. Key Features

* CNN-based plant disease classification.
* Transfer learning using MobileNetV2.
* Image preprocessing and normalization.
* Model training and validation.
* Performance evaluation using classification metrics.
* Confusion matrix visualization.
* Model checkpoint saving.
* PyTorch model export to ExecuTorch.
* Edge inference using the `.pte` model.
* Top-3 prediction results with confidence scores.

## 3. Technology Stack

| Component               | Technology   |
| ----------------------- | ------------ |
| Programming Language    | Python       |
| Deep Learning Framework | PyTorch      |
| CNN Architecture        | MobileNetV2  |
| Image Processing        | Pillow       |
| Computer Vision         | Torchvision  |
| Edge Deployment         | ExecuTorch   |
| Data Visualization      | Matplotlib   |
| Model Evaluation        | Scikit-learn |

## 4. Model Architecture

The module uses MobileNetV2, a lightweight CNN architecture designed for computational efficiency.

The model employs transfer learning with pretrained ImageNet weights and is fine-tuned for plant disease classification.

### Model Workflow

1. Input leaf image.
2. Resize image to 224 × 224 pixels.
3. Convert image into a tensor.
4. Apply ImageNet normalization.
5. Pass the image through the MobileNetV2 network.
6. Generate classification logits.
7. Apply Softmax to obtain class probability estimates.
8. Identify the predicted disease category.

## 5. Project Structure

```text
Module-3/
│
├── checkpoints/
│   ├── best_model.pt
│   └── class_names.json
│
├── train_cnn.py
├── evaluate_model.py
├── export_to_executorch.py
├── test_pte.py
│
├── disease_cnn.pte
│
├── pytorch_confusion_matrix.png
├── pte_confusion_matrix.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 6. Dataset

The model is developed using the PlantVillage leaf image dataset.

The dataset contains images of healthy and diseased plant leaves across multiple plant categories.

Dataset organization:

```text
plantvillage dataset/
└── color/
    ├── Apple___Cedar_apple_rust/
    ├── Grape___healthy/
    ├── Tomato___Bacterial_spot/
    └── ...
```

The dataset is maintained separately and is not included in this GitHub module because of its size.

## 7. Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/Swapnaneel07/agri-sense.git
```

### Step 2: Navigate to the project directory

```bash
cd agri-sense
```

Switch to the Module 3 branch:

```bash
git checkout module-3
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

## 8. Model Training

To train the CNN model:

```bash
python train_cnn.py
```

The training pipeline uses MobileNetV2 and saves the best-performing checkpoint.

Generated model files:

* `checkpoints/best_model.pt`
* `checkpoints/class_names.json`

## 9. Model Evaluation

Run the evaluation script:

```bash
python evaluate_model.py
```

The evaluation module supports model performance analysis and confusion matrix generation.

Evaluation visualizations:

* `pytorch_confusion_matrix.png`
* `pte_confusion_matrix.png`

## 10. Edge Deployment Using ExecuTorch

The trained PyTorch model can be exported into the ExecuTorch `.pte` format.

Run:

```bash
python export_to_executorch.py
```

The exported model is saved as:

```text
disease_cnn.pte
```

ExecuTorch enables the model to execute inference using its lightweight runtime.

## 11. Running Edge Inference

The `test_pte.py` script performs plant disease classification using the exported ExecuTorch model.

Before execution, configure the image path inside the script:

```python
IMAGE_PATH = Path(
    r"path/to/your/leaf_image.JPG"
)
```

Run:

```bash
python test_pte.py
```

The inference pipeline produces:

* Image information.
* Predicted disease category.
* Prediction confidence score.
* Top-3 predicted classes.
* Inference execution time.
* Detection status (healthy or diseased).

## 12. Expected Output

The inference script generates a formatted detection report.

Example:

```text
============================================================
           AGRISENSE - AI PLANT DISEASE DETECTION
============================================================

[OK] Model files verified.
[OK] Image successfully processed.
[OK] ExecuTorch model loaded.

                  DETECTION REPORT

Image Name       : leaf_image.JPG
Detection Status : HEALTHY LEAF
Predicted Class  : Grape - healthy
Confidence Score : Model-generated value
Inference Time   : Measured at runtime

                 TOP 3 PREDICTIONS

1. Grape - healthy
2. Grape - Black rot
3. Grape - Esca (Black Measles)

============================================================
       AI INFERENCE COMPLETED SUCCESSFULLY
============================================================
```

*The output above is illustrative. Actual predictions and confidence scores depend on the input image and trained model.*

## 13. Important Notes

* The PlantVillage dataset must be downloaded separately.
* Update dataset and image paths according to your local environment.
* The `.pt` checkpoint is used for PyTorch model operations.
* The `.pte` file is used for ExecuTorch inference.
* `class_names.json` maintains the mapping between model output indices and disease labels.
* Softmax confidence values are model-generated probability estimates and should not be treated as guaranteed prediction accuracy.

## 14. Future Enhancements

* Real-time plant disease detection through a camera.
* Mobile application integration.
* Further optimization for resource-constrained edge devices.
* Disease severity estimation.
* Treatment and prevention recommendations.
* Integration with the complete AgriSense agricultural monitoring platform.

## 15. Contribution

This module is part of the AgriSense project.

**Module:** Module 3 – Leaf Disease Detection and Edge Deployment

**Repository:** AgriSense

**Branch:** module-3

---

*AgriSense – AI-Powered Smart Agriculture System*
