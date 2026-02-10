# Marine Species Identification Using EfficientNetB0 and Test-Time Augmentation

## Project Overview
This project implements a deep learning pipeline for **visual classification of marine species** across **16 taxonomic categories**. The objective is to support marine biological surveys by enabling accurate species identification in challenging underwater environments.

The model classifies a wide range of marine life, from invertebrates such as **Corals** and **Starfish** to large marine mammals including **Dolphins** and **Whales**, using transfer learning and inference-time optimization techniques.

---

## Technical Methodology

### Model Architecture
- **Base Model:** EfficientNetB0 pre-trained on ImageNet  
- Selected for its strong trade-off between **computational efficiency** and **classification accuracy**

### Custom Classification Head
- Global Average Pooling layer  
- Dense layer with **512 units** and ReLU activation  
- **L2 regularization** for weight penalization  
- **Dropout (0.4)** to reduce overfitting  
- Final **Softmax layer** for 16-class prediction  

### Optimization and Training
- **Optimizer:** Adam  
- **Callbacks Used:**
  - EarlyStopping (monitored on validation loss)
  - ReduceLROnPlateau for dynamic learning rate adjustment

### Inference Strategy: Test-Time Augmentation (TTA)
During evaluation, predictions are generated from multiple augmented versions of each test image. The final prediction is obtained by aggregating these outputs, improving robustness and prediction stability.

---

## Dataset and Preprocessing

### Dataset
- RGB images spanning **16 marine species categories**

### Data Augmentation
Applied during training to improve generalization:
- Random rotations
- Horizontal flipping
- Zoom transformations

### Input Standardization
- Images resized to a fixed resolution  
- Pixel values rescaled for compatibility with EfficientNetB0  

---

## Experimental Results

### Classification Performance
- The fine-tuned model achieved strong accuracy across diverse marine species
- Demonstrated robustness to orientation and lighting variations

### Evaluation Metrics
- Confusion matrices
- Precision, recall, and F1-score via classification reports
- Per-class performance analysis

---

## Repository Structure

```text
.
├── notebooks/
│   ├── training_pipeline.ipynb
│   ├── tta_evaluation.ipynb
│   └── results_analysis.ipynb
│
├── models/
│   └── marine_species_classifier.keras
│
├── docs/
│   └── technical_report.pdf
│
└── README.md
