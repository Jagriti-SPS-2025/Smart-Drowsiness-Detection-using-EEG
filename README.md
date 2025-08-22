# IEEE SPS Bangalore Chapter - MathWorks Challenge Report  
## Smart Drowsiness Detection using EEG  

**Challenge:** [IEEE SPS Bangalore Chapter - MathWorks Challenge on Smart Drowsiness Detection using EEG](https://sps.ieeebangalore.org/sps-forum-2025_2/)  

**Project Duration:** 7th–23rd Aug 2025  

**Authors:**  
- Prof. Ankitha A¹ – Dept. of E&C, Vemana IT, Bengaluru – Vemana IT, India  
- Dr. Vijayalakshmi² – Dept. of E&C, BMSCE, Bengaluru, India  
- Kavyashree K³ – UG Student, Dept. of E&C, Vemana IT, Bengaluru – Vemana IT, India  
- Margaret Sheela C³ – UG Student, Dept. of E&C, Vemana IT, Bengaluru – Vemana IT, India  

¹²³ Emails: ankitha@vemanait.edu.in, kv.ml@bmsce.ac.in, {kavyashreek, margaretsheelac}@vemanait.edu.in}   

---

## Objective
Design a lightweight machine-learning pipeline that analyzes single-channel frontal EEG data (Fp1/Fp2) and accurately detects driver drowsiness in real-time.  

---

## Dataset
- EEG signals sampled at 512 Hz.  
- Two classes: **Active** and **Drowsy**.  
- Raw datasets downloaded from IEEE DataPort ([EEG dataset for drowsiness detection](https://ieee-dataport.org/documents/eeg-dataset-drowsiness-detection))  

---

## Methodology & Feature Engineering
1. **Preprocessing:**  
   - 50 Hz IIR notch filter + 0.5–30 Hz Butterworth band-pass filter  
   - Artifact/spike removal (±100 µV threshold, replaced by segment average)  
   - Segmentation into 1-second windows (512 samples) with 50% overlap  

2. **Feature Extraction:**  
   - Band powers: Delta, Theta, Alpha, Beta  
   - Ratios: α/β, β/θ, θ/α, Slow/Fast  
   - Spectral features: Peak Frequency, Spectral Centroid, Spectral Slope, SEF-95%, Entropy  
   - Hjorth descriptors: Mobility, Complexity  
   - Time-domain statistics: Mean, Variance, IQR, Skewness, Kurtosis  

3. **Feature Labeling:**  
   - Active = 1, Drowsy = 0  
   - Extracted 27 features per segment  
   - Merged into `features.csv` for modeling  

---

## Model Architecture
- Classifier: **Radial Basis Function SVM (RBF-SVM)**  
- Hyperparameters auto-tuned via **Bayesian optimization**  
- Evaluated with **stratified 5-fold cross-validation**  
- Compared with kNN, Decision Tree, Naive Bayes, Ensemble, MLP – **SVM achieved highest performance**  
- Input features: EEG band powers, ratios, statistical measures  

---

## Model Evaluation
- **Performance Metrics:** Accuracy and F1-score  
- **Outcome:** 94% accuracy and F1 = 0.93  
- Confusion matrix and ROC curve used for evaluation  

---

## Challenges Faced
- Heavy noise and artifacts in EEG signals requiring extensive preprocessing  
- Class imbalance between Active and Drowsy samples  
- Selecting meaningful features from a large pool  
- Hyperparameter tuning to avoid overfitting  

---

## Advantages
- High accuracy (94%) and F1 score (0.93)  
- Single-channel EEG reduces hardware cost  
- Lightweight model suitable for real-time deployment  

## Drawbacks
- Sensitive to noise/artifacts  
- Manual feature engineering required  
- Dataset contains ~1.2 million EEG samples, which is substantial, but still relatively small compared to datasets typically used for deep learning. 

---

## Model Usage (Detection Process)
1. Input raw EEG  
2. Preprocessing (filtering + artifact removal)  
3. Feature extraction (band powers, ratios, spectral, Hjorth, statistical)  
4. Prediction using trained RBF-SVM → outputs “Active” or “Drowsy” in real-time  

---

## Insights Gained
- Learned MATLAB functionalities and efficient workflows  
- Understood EEG preprocessing and feature engineering importance  
-Compared GUI and code-based model training; both achieved similar accuracy, but code-based training was preferred for full control over preprocessing and feature engineering.  

---

## Future Scope
- Extend to multi-channel EEG  
- Explore deep learning (CNN/LSTM) for automated feature extraction  
- Real-time driver drowsiness detection with IoT hardware  
- Integration into wearable headbands or vehicle dashboards  

---

## Project Flowchart

EEG Drowsiness Detection Pipeline:

Preprocessing
  │
  ├─ Load Datasets
  ├─ Filtering (0.5–30 Hz Butterworth + 50 Hz notch)
  ├─ Artifact/Spike Removal (±100 µV threshold)
  └─ Segmentation (1-sec windows, 50% overlap)
       │
Feature Extraction
  │
  ├─ Band Powers (Delta, Theta, Alpha, Beta)
  ├─ Ratios (α/β, β/θ, θ/α, Slow/Fast)
  ├─ Spectral Features (Peak Freq, Centroid, Slope, SEF-95%, Entropy)
  ├─ Hjorth Parameters (Mobility, Complexity)
  └─ Time-Domain Statistics (Mean, Variance, IQR, Skewness, Kurtosis)
       │
Model Training
  │
  ├─ RBF-SVM Classifier
  ├─ Hyperparameter Tuning (Bayesian Optimization)
  └─ Stratified 5-Fold Cross-Validation
       │
Model Testing
  │
  ├─ Predict Unseen EEG Data
  ├─ Confusion Matrix
  └─ ROC Curve
       │
Model Evaluation
  │
  ├─ Accuracy
  ├─ F1-Score
  └─ Final Report

---

## Repository Structure
root/
├── datasets/ ← EEG CSV datasets
├── model/ ← trained SVM models (.mat, .pkl)
├── scripts/ ← all code (.py, .m)
├── submissions/ ← predictions.csv & technical report
├── images/ ← flowcharts, diagrams
└── README.md

---
## How to Run

### Prerequisites

#### Python:
- Python 3.8 or higher  
- Required libraries:  
```bash
pip install numpy pandas scipy scikit-learn
```

#### MATLAB:
- MATLAB R2023a or later (or compatible version)  
- Signal Processing Toolbox  
- Statistics and Machine Learning Toolbox  

---

### Repository Setup
Ensure the repository structure is as follows:
```
root/
├── datasets/       ← EEG CSV datasets (active.csv, drowsy.csv)
├── scripts/        ← all code (.py, .m)
├── model/          ← trained SVM models (.mat, .pkl)
├── submissions/    ← predictions.csv & technical report
├── images/         ← flowcharts, diagrams
└── README.md
```

---

### Running the Python Pipeline

1. **Preprocessing EEG Data**  
```bash
python scripts/eeg_preprocessing.py
```
- Output: `cleaned_segments.csv` (preprocessed EEG segments)

2. **Feature Extraction**  
```bash
python scripts/feature_extraction.py
```
- Output: `features.csv`  

3. **Train the SVM Model**  
```bash
python scripts/train_svm.py
```
- Output: `model/SVM_trained_model.pkl`  

4. **Test the Model / Generate Predictions**  
```bash
python scripts/test_model.py
```
- Output: `submissions/predictions.csv`  

---

### Running the MATLAB Pipeline

1. Open `scripts/eeg_preprocessing.m` → Run → Output preprocessed EEG segments  
2. Open `scripts/feature_extraction.m` → Run → Output `features.csv`  
3. Open `scripts/train_svm.m` → Run → Output `SVM_trained_model.mat` in `model/`  
4. Open `scripts/test_model.m` → Run → Output `submissions/predictions.csv`  

---

### Notes
- Ensure all datasets are in `datasets/` before running scripts.  
- Scripts are reproducible; preprocessing parameters can be changed in the respective `.py` or `.m` files.  
- Python and MATLAB pipelines are interchangeable but require the correct model file format.  
- `predictions.csv` must be placed in the `submissions/` folder for submission.  
- Verify folder paths and Python/MATLAB environments are correctly set.  

--

## References
- IEEE SPS Forum 2025: [Challenge Details](https://sps.ieeebangalore.org/sps-forum-2025_2/)  
- EEG Dataset: [IEEE DataPort](https://ieee-dataport.org/documents/eeg-dataset-drowsiness-detection)  
- MATLAB Documentation: [MathWorks MATLAB Documentation](https://www.mathworks.com/help/matlab/)  
- MATLAB Classification Learner App: [MathWorks Video Tutorials](https://www.mathworks.com/videos/)  
- EEG Signal Processing Tutorials: [EEG in MATLAB](https://www.mathworks.com/help/signal/eeg.html)  
- Feature Extraction Techniques in EEG: [Hjorth Parameters and Spectral Features](https://www.sciencedirect.com/topics/neuroscience/hjorth-parameters)  
- Machine Learning with EEG Signals: [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)  

 ---

 ## Contact

For any questions or clarifications regarding this project, please contact:  

- Kavyashree K – Email: kavyashreek.ec2022@vemanait.edu.in  
- Margaret Sheela C – Email: margaretsheelac.ec2022@vemanait.edu.in
- Dr. Vijayalakshmi - Email: kv.ml@ bmsce.ac.in
- Prof. Ankitha A - Email: ankitha@vemanait.edu.in 


