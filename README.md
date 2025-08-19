# Athlete Face Recognition using Haar Cascades & Machine Learning  

This project implements an **Athlete Face Recognition system** using **OpenCV Haar Cascade Classifiers**, **Wavelet Transforms**, and **Machine Learning models**.  
The pipeline automatically detects, crops, and processes athlete faces from a dataset, then classifies them into different athlete categories using supervised learning.  

## Dataset  

- Dataset used: [Athletes Face Dataset (Google Drive)](https://drive.google.com/drive/folders/1Gduv8Qd97gfbwuIu-42uU66VUSgX4NcC?usp=sharing)  
- Structure:  
```
athletes\_dataset\_4/
├── Athlete1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Athlete2/
├── Athlete3/
└── ...
```
- Each folder contains images of a single athlete.  
- Cropped and preprocessed faces are saved in:  


athletes\_dataset\_4/cropped/

## Methodology  

### 1. Face & Eye Detection  
- Haar Cascades (`haarcascade_frontalface_default.xml`, `haarcascade_eye.xml`) are used to detect faces and ensure at least **two eyes** are present for a valid crop.  
- Only valid cropped face regions are kept.  

### 2. Preprocessing  
- Convert images to grayscale.  
- Apply **Discrete Wavelet Transform (DWT)** to extract edge & frequency features.  
- Combine raw image features (32×32×3) with wavelet-transformed features (32×32).  
- Flatten into a **feature vector of size 4096**.  

### 3. Dataset Cleaning  
- Classes (athletes) with fewer than **50 valid cropped images** are removed.  
- Final dataset: **12 athlete classes** kept out of 29.  

### 4. Model Training  
- Train/Test split using `train_test_split()`.  
- Models tried:  
- **Support Vector Machine (SVM)** with RBF/Linear kernels  
- **Random Forest**  
- **Logistic Regression**  

- Hyperparameter tuning with **GridSearchCV**.  
- Best performing model: **SVM (Linear Kernel, C=1)**.  

### 5. Evaluation  
- Accuracy: **~66%** (12 athlete classes).  
- Metrics used: **Precision, Recall, F1-score**.  
- Confusion Matrix plotted for misclassification analysis.  


## Results  

- **Best Model:** SVM (Linear Kernel, C=1)  
- **Accuracy:** ~66% on test set  
- **Classification Report (excerpt):**


          precision    recall  f1-score   support


Athlete1       0.62      0.44      0.52
Athlete2       0.63      0.86      0.73
Athlete3       0.69      0.85      0.76
...
Overall        0.66      0.66      0.66



- **Confusion Matrix:**  
Visualized with Seaborn heatmap to inspect per-class performance.  


## Model Export  

- Best classifier is saved as:  

saved\_model.pkl

- Athlete → Class index mapping stored in:  

class\_dictionary.json

## Installation & Usage  

### Requirements  
Install dependencies:  
```bash
pip install opencv-python numpy matplotlib pywavelets scikit-learn seaborn joblib
````

### Run the Project

1. Clone the repo & download dataset.
2. Place Haar Cascade XML files in `./opencv/haarcascades/`.
3. Run the script:

   ```bash
   python athlete_face_recognition.py
   ```
4. Preprocessed cropped images will be generated in `athletes_dataset_4/cropped/`.
5. Training, evaluation, and model saving will run automatically.


## Future Improvements

* Improve accuracy using **Deep Learning (CNNs / Transfer Learning)**.
* Balance dataset with **data augmentation**.
* Add a **real-time recognition pipeline** using webcam feed.


## Acknowledgements

* OpenCV Haar Cascades for face & eye detection.
* PyWavelets for DWT-based feature extraction.
* Scikit-learn for model training and evaluation.


