import pandas as pd
import numpy as np
import joblib
import importlib.util
import os

import extract_features_vibration as extract_module

def load_feature_extractor():
    return extract_module.extract_features

def main():
    print("Initializing Fault Identification Inference pipeline...\n")
    
    # 1. Load the trained RF model and Label Encoder
    model_path = "models/rf_model.pkl"
    encoder_path = "models/label_encoder.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print("Error: Could not find model files.")
        print(f"Make sure '{model_path}' and '{encoder_path}' are in the current directory.")
        return
        
    try:
        model = joblib.load(model_path)
        le = joblib.load(encoder_path)
        print("✅ Model and Label Encoder loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return

    # 2. Dynamic Import of feature extractor
    try:
        extract_features = load_feature_extractor()
        print("✅ Feature extraction module loaded.")
    except Exception as e:
        print(f"❌ Failed to load feature extraction module: {e}")
        return

    # 3. Set Parameters (Should match training configuration)
    rpm = 1500           # Change this based on your inference settings
    n_ball = 8
    Bd = 15.5            # Bearing ball diameter
    Pd = 72.5            # Bearing pitch diameter
    sampling_rate = 12000 # Sample rate in Hz

    import glob
    
    # 4. Load a real vibration signal from the dataset folder for testing
    print("\nFetching data...")
    csv_files = glob.glob("dataset/**/*.csv", recursive=True)
    if not csv_files:
        print("❌ Error: No CSV files found in the 'dataset' directory.")
        return
        
    # Use the first dataset file found
    test_file = csv_files[0]
    print(f"ℹ️ Loading real test data from: {test_file}")
    
    df = pd.read_csv(test_file)
    
    # Check if df has at least 2 columns (assuming col 0 is time, col 1 is signal)
    if df.shape[1] >= 2:
        # Preprocess exactly how training did: fill NaNs
        signal = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        signal = signal.interpolate().ffill().bfill().values
    else:
        signal = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        signal = signal.interpolate().ffill().bfill().values
        
    print(f"ℹ️ Loaded real data: {len(signal)} samples.")


    # 5. Feature Extraction
    print("\nExtracting features...")
    try:
        features = extract_features(signal, rpm, n_ball, Bd, Pd, sampling_rate)
        # Reshape to (1, n_features) since sklearn expects a 2D array
        features_2d = np.array(features).reshape(1, -1)
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return

    # Enforce correct feature count
    expected_feature_count = model.n_features_in_
    if features_2d.shape[1] != expected_feature_count:
        print(f"❌ Error: Model expects {expected_feature_count} features, but exactor output {features_2d.shape[1]}.")
        print(f"Please ensure `extract_features_vibration.py` returns all {expected_feature_count} features.")
        return
        
    print(f"✅ Successfully extracted {features_2d.shape[1]} features.")

    print("\nRunning Model Prediction...")
    predicted_encoded = model.predict(features_2d)[0]
    probabilities = model.predict_proba(features_2d)[0]
    
    if np.issubdtype(type(predicted_encoded), np.integer) or str(predicted_encoded).isdigit():
        predicted_label = le.inverse_transform([int(predicted_encoded)])[0]
    else:
        predicted_label = predicted_encoded

    # 7. Print the results
    print("\n" + "="*40)
    print(f"🛠️  DIAGNOSIS RESULT: {predicted_label.upper()}")
    print("="*40)
    print("\nClass Probabilities:")
    for cls, prob in zip(le.classes_, probabilities):
        print(f"  [{cls.ljust(15)}]: {prob*100:.2f}%")

if __name__ == "__main__":
    main()
