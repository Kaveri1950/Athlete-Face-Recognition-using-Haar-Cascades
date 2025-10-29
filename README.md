# Athlete Face Recognition using Haar Cascades, Wavelet Transforms & Stacked Machine Learning Models  

This project implements an **Athlete Face Recognition System** leveraging **OpenCV Haar Cascade Classifiers**, **Wavelet Transforms**, and a range of **Machine Learning models**, including **SVM**, **KNN**, **Random Forest**, **Logistic Regression**, **XGBoost**, and **Stacking Classifiers**.  

The system automatically detects, crops, and processes athlete faces, extracts powerful features combining spatial and frequency domains, and classifies them into athlete categories using supervised learning techniques.

***

## Dataset  

- Dataset used: [Athletes Face Dataset (Google Drive)](https://drive.google.com/drive/folders/1Gduv8Qd97gfbwuIu-42uU66VUSgX4NcC?usp=sharing)
  
- Structure:
```
celebrity_dataset_final/
├── Athlete1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Athlete2/
├── Athlete3/
└── ...
```

- Each subfolder contains images of a single athlete.  
- Cropped faces are automatically generated and stored in:
```
celebrity_dataset_final/cropped/
```

After filtering out classes with fewer than **40 valid cropped images**, **3 athlete classes** were retained for model training.

***

## Methodology  

### 1. Face & Eye Detection  
- Haar Cascades (`haarcascade_frontalface_default.xml`, `haarcascade_eye.xml`) are used for face detection.  
- Only faces containing **at least two eyes** are considered valid.  
- Cropped regions are saved automatically for each identified athlete.  

### 2. Preprocessing & Feature Extraction  
- Images are resized to 32×32 and converted to grayscale.  
- **Discrete Wavelet Transform (DWT)** is applied using the `db1` wavelet at level 5 to extract frequency and edge features.  
- Raw image (32×32×3) and wavelet-transformed image (32×32) are vertically combined, resulting in a **4096-dimensional feature vector**.  

### 3. Dataset Cleaning  
- Classes with fewer than 40 images are removed to ensure reliable training data.  
- Final dataset consists of 3 valid athlete categories.  

### 4. Model Training & Selection  
Multiple models were trained and compared using **GridSearchCV** and cross-validation:

- **Support Vector Machine (SVM)** with linear and RBF kernels  
- **Random Forest**  
- **Logistic Regression**  
- **K-Nearest Neighbors (KNN)** with optimized parameters  
- **XGBoost Classifier**  
- **Stacking Classifier** combining SVM, Random Forest, KNN, and XGBoost with Logistic Regression as meta-model  

Each model’s performance was optimized via grid search for best hyperparameters.

***

## Model Performance  

### Model Comparison (GridSearchCV)

| Model | Cross-Validation Accuracy | Best Parameters |
|--------|----------------------------|-----------------|
| SVM | 84.3% | C=1, kernel=linear |
| Random Forest | 69.5% | n_estimators=10 |
| Logistic Regression | 86.0% | C=1 |
| KNN | 73.6% | n_neighbors=11, p=1, weights=distance |
| XGBoost | 78.1% | n_estimators=200, max_depth=4, lr=0.01 |
| Stacking (SVM + RF + KNN + XGB) | 82.9% | Default combination |
| Final Tuned Stacking Model | **87.8%** | Tuned base learners |

***

## Evaluation  

**Final Stacking Model Results (3 Classes)**

| Metric | Class 0 | Class 1 | Class 2 |
|---------|----------|----------|----------|
| Precision | 0.88 | 0.85 | 0.80 |
| Recall | 0.64 | 0.85 | 0.94 |
| F1-Score | 0.74 | 0.85 | 0.86 |

**Overall Accuracy:** 87.8%  
**Macro Avg F1-Score:** 0.82  

The stacked model significantly outperformed individual classifiers, achieving highly reliable classification on the athlete dataset.

***

## Model Export  

The best-performing tuned stacked model and class mapping were saved as:

```
models/stacking_tuned_model.pkl
models/class_dict.pkl
```

***

## Installation & Usage  

### Requirements  
Install dependencies using:
```bash
pip install opencv-python numpy matplotlib pywavelets scikit-learn seaborn joblib xgboost pandas
```

### Run the Project  

1. Clone the repository and download the dataset.  
2. Place Haar Cascade XML files in the directory:  
   ```
   ./opencv/haarcascades/
   ```
3. Run the main script:  
   ```bash
   python athlete_face_recognition.py
   ```
4. Cropped face images will be saved in:  
   ```
   celebrity_dataset_final/cropped/
   ```
5. Training, evaluation, and model saving will be executed automatically.  

***

## Key Highlights  

- Automated face and eye detection pipeline using OpenCV.  
- DWT-based feature extraction enhancing frequency domain recognition.  
- Multi-model pipeline with hyperparameter tuning and cross-validation.  
- Final stacked ensemble model achieving 87.8% accuracy.  
- Saved model ready for real-time prediction and deployment.  
```
```

This will render your entire new README as a code block when copied into your documentation, following GitHub markdown code-block conventions.[1][3]
