
**OPTIMISING RETAIL INVENTORY MANAGEMENT USING MACHINE LEARNING AND SALES DATA** 

---

# Overview
This notebook investigates the effectiveness of deep learning models in classifying pediatric chest X-ray images into different types of pneumonia (bacterial, viral) and normal cases. It compares a custom CNN and three pre-trained transfer learning models (DenseNet121, MobileNetV2, VGG16), performs data augmentation, and builds an ensemble classifier to explore improvements in sensitivity and generalization.

---

# Dataset
- **Path/URL:** `/content/drive/MyDrive/ChestXRay2017/chest_xray`
- **Target column:** `label`
- **Feature column(s):** `filepath` (input image), `label` (normal, bacterial, viral), `split` (train/val/test)
- **Feature count/types:** Not explicitly printed; includes image paths and categorical labels

---

# Features & Preprocessing
- Image resizing to 224x224 pixels (`IMG_SIZE = (224, 224)`)
- Dataset split into train/validation/test using `train_test_split` with stratification (15% validation)
- Normalization using `Rescaling(1./255)`
- Augmentation with `RandomFlip`, `RandomRotation`, and `RandomZoom`
- Batch loading using `image_dataset_from_directory`
- Model-specific preprocessing (e.g., `preprocess_input` from `keras.applications`)

---

# Models
- `DenseNet121` (Transfer Learning, `include_top=False`, `weights='imagenet'`, frozen base)
- `MobileNetV2` (Transfer Learning, `include_top=False`, `weights='imagenet'`, frozen base)
- `VGG16` (Transfer Learning, `include_top=False`, `weights='imagenet'`, frozen base, later fine-tuned)
- `Custom CNN` (3 Conv2D layers with ReLU + MaxPool2D + Dropout, followed by Dense layers)
- `Ensemble Model` (Majority vote ensemble from predictions of the 4 models)

---

# Evaluation
- **Metrics:** `accuracy_score`, `f1_score`, `precision_score`, `recall_score`, `classification_report`
- **Visualizations:** Confusion matrix heatmaps (Seaborn), class distribution plots, ROC curves
- **Tuning:** Fine-tuning of VGG16 (unfreezing layers, adjusting learning rate)
- Performance comparisons on validation and test sets

---

# Environment & Requirements
- **Libraries:**
  - `tensorflow`, `keras`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `glob`, `tqdm`
- **Install example:**
  ```bash
  pip install tensorflow keras pandas matplotlib seaborn scikit-learn tqdm
  ```