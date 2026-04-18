# Motor Fault Identification System

This repository contains a complete Machine Learning pipeline for diagnosing electric motor faults based on raw vibration signal data. The system extracts 16 distinct time-domain and frequency-domain features and utilizes a Random Forest Classifier to identify whether a motor falls into the following categories:

* `healthy`
* `bearing`
* `misalignment`
* `others`

## 📂 Project Structure

* `extract_features_vibration.py`: The core feature extraction module. Computes features like RMS, Peak, Skewness, Kurtosis, Crest Factor, Spectral Centroid, and Bearing fault ratio metrics (BPFO, BPFI, BSF, FTF).
* `train_rf.py`: A high-performance training script that scans the `dataset/` directory, extracts features dynamically across all cores using Multiprocessing, applies `SMOTE` to handle imbalanced datasets, and saves the final `RandomForestClassifier`.
* `inference.py`: The single-data inference pipeline. This dynamically loads a chosen raw `.csv` vibration signal, runs the exact feature extraction logic, and predicts its condition.
* `inference_v2.py`: A complete automated evaluation suite. It automatically searches for specific test paths (e.g., *Healthy*, *Bearing*, *Misalignment*) and tests the reliability of the `rf_model.pkl` against these expected realities in a beautifully logged terminal output.
* `models/`: Contains the generated serialized `rf_model.pkl` and `label_encoder.pkl` state.
* `notebooks/`: Contains the original Jupyter Notebooks used for R&D.

## 🚀 Quick Start

### 1. Requirements

Ensure you are running on Python 3.9+ and have a virtual environment enabled. Install all the necessary packages via the generated `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Retraining the Model

In order to establish a strong alignment between your dataset features and the Random Forest logic, use the `train_rf.py` script. (Ensure your raw `.csv` folders are inside the `dataset/` directory):

```bash
python train_rf.py
```
*Depending on the depth of your dataset, this process might take up to a few minutes. It is utilizing parallel processing to speed up the FFT transformations.*

### 3. Evaluating Scenarios

To verify how well the newly trained model handles real-life scenarios, execute:

```bash
python inference_v2.py
```
This script will print out class probabilities (e.g., 99.5% Healthy, 0% Misalignment) and compare results against the expected directory label.

### 4. Direct Inference Use

To use the model in production (single signal pipeline):

```bash
python inference.py
```
*Tip: Change the target CSV file inside `inference.py` when linking it up to a live sensor database.*
