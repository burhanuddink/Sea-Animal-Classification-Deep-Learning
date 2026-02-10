# Sea Animal Classification using Deep Learning (EfficientNetB0)

### 🚀 Project Overview
This project focuses on the development of a high-precision visual classification system designed to identify marine species across **16 different categories**. The dataset presents a wide taxonomic range, encompassing everything from simple invertebrates like **Corals and Starfish** to complex mammals like **Dolphins and Whales**.

The primary objective was to leverage **Transfer Learning** architectures to solve intricate computer vision challenges, ensuring the model generalizes effectively across varied and complex marine environments.

### 🛠️ Technical Methodology
Instead of training a model from scratch, this implementation utilizes the **EfficientNetB0** architecture, pre-trained on ImageNet. This approach was chosen for its optimal balance between parameter efficiency and classification accuracy.

**Key Implementation Details:**
* **Architectural Optimization:** Integrated a custom head consisting of **Global Average Pooling**, followed by Dense layers with **ReLU activation** and a final **Softmax** output for the 16 marine classes.
* **Regularization Strategy:** To mitigate overfitting and improve robust performance, **L2 Regularization** and **Dropout layers** were strategically implemented within the training pipeline.
* **Inference Enhancement:** Employed **Test-Time Augmentation (TTA)** during the evaluation phase, allowing the model to aggregate predictions from multiple augmented versions of test images to ensure higher reliability.
* **Dynamic Learning:** Utilized `ReduceLROnPlateau` and `EarlyStopping` callbacks to monitor validation loss and prevent model divergence during the 20-epoch training cycle.

### 📊 Evaluation & Results
* **High-Precision Classification:** Achieved superior accuracy by fine-tuning the model's top layers and optimizing hyperparameters.
* **Data Robustness:** Engineered a preprocessing pipeline to standardize RGB image inputs, ensuring the model remains resilient to variations in lighting and orientation found in real-world visual scenarios.

### 📂 Repository Structure
* **`notebooks/`**: Contains the full pipeline—from data augmentation and training to TTA evaluation and confusion matrix analysis.
* **`models/`**: Includes the final saved `.keras` model file.
* **`docs/`**: Features the comprehensive technical report detailing the research and experimental findings.

### ⚙️ Environment Setup
To replicate the results, install the necessary dependencies:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python
