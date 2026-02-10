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
```
---

## Installation and Requirements

### Python Version
- Python 3.11

### Required Libraries
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python
```
---

## Real-Time Demo

This project includes a fully functional web interface built with **Gradio**. You can upload any image of a sea animal, and the EfficientNet model will classify it in real-time.

### Option 1: Run in the Cloud (No Installation Required)
The easiest way to try the app is using GitHub Codespaces.

1. Click the **Code** button at the top of this repository.
2. Select the **Codespaces** tab and click **"Create codespace on main"**.
3. Wait for the terminal to load, then run these two commands:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
Click "Open in Browser": When the app starts, a **popup** will appear in the bottom-right corner saying "Your application running on port 7860 is available." Click the Open in Browser button to use the app. If you don't see any popup, you can just copy the url saying: **"Running on local URL"**

### Option 2: Run Locally
If you prefer to run it on your own machine:

1. Clone the repository:

```bash
git clone [https://github.com/burhanuddink/Sea-Animal-Classification-Deep-Learning.git](https://github.com/burhanuddink/Sea-Animal-Classification-Deep-Learning.git)
cd Sea-Animal-Classification-Deep-Learning
```
2. Install dependencies: Make sure you have Python 3.11 installed.

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

4. Open the Interface: The terminal will provide a URL (usually http://127.0.0.1:7860). Open this link in your browser.

---
## Usage

1. Clone the repository:
```bash
git clone https://github.com/burhanuddink/Sea-Animal-Classification-Deep-Learning.git
```
2. Navigate to the project directory:
```bash
cd Sea-Animal-Classification-Deep-Learning
```
3. Execute the Jupyter notebooks in the notebooks/ directory sequentially to train the model, perform test-time augmentation evaluation, and analyze results.

---

## Future Improvements

- Extend the model to finer-grained marine species classification
- Experiment with newer architectures such as EfficientNetV2 or Vision Transformers
- Integrate attention mechanisms to improve feature localization

---

## Acknowledgements

- EfficientNet architecture by Google Research
- ImageNet pre-trained weights
- TensorFlow and the open-source deep learning ecosystem

---
