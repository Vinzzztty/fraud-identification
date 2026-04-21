import pandas as pd
import numpy as np
import os
import glob
import joblib
from colorama import init, Fore, Style

init(autoreset=True)
import extract_features_vibration as extract_module

def main():
    print(Style.BRIGHT + "Initializing Run-To-Failure Evaluation Pipeline (AI Model Mode)...\n")
    
    # 1. Load trained model and encoder
    model_path = "models/rf_model.pkl"
    encoder_path = "models/label_encoder.pkl"
    
    try:
        model = joblib.load(model_path)
        le = joblib.load(encoder_path)
        print(Fore.GREEN + "✅ AI Model & Label Encoder loaded successfully.")
    except Exception as e:
        print(Fore.RED + f"❌ Failed to load models: {e}. Please run train_rf.py first.")
        return

    scaler_path = "models/rtf_scaler.pkl"
    try:
        rtf_scaler = joblib.load(scaler_path)
        print(Fore.GREEN + "✅ RTF Scaler loaded successfully.")
    except Exception as e:
        print(Fore.YELLOW + f"⚠️  RTF Scaler not found: {e}. Run train_rf.py to generate scaler. Falling back to baseline normalization.")
        rtf_scaler = None

    extract_features = extract_module.extract_features
    # RTF-specific bearing parameters
    n_ball = 9
    Bd = 7.938
    Pd = 38.5
    sampling_rate = 25600
    rpm = 1775
    
    # 2. Get Run-to-Failure dataset
    target_dir = "dataset/Vibration_Bearing_RuntoFailure"
    if not os.path.exists(target_dir):
        print(Fore.RED + f"❌ Error: Directory not found - {target_dir}")
        return
        
    vibrasi_files = glob.glob(os.path.join(target_dir, "[Vv]ibrasi*.csv"))
    if vibrasi_files:
        csv_files = vibrasi_files
    else:
        print(Fore.YELLOW + "⚠️  No 'vibrasi*' prefix files found. Processing all CSV files.")
        csv_files = glob.glob(os.path.join(target_dir, "*.csv"))

    if not csv_files:
        print(Fore.RED + "❌ Error: No CSV files found in the dataset.")
        return
        
    # Sort files chronologically (String sorting works perfectly for YYYY-MM-DD timestamps)
    csv_files.sort()
    
    print(Fore.CYAN + f"Found {len(csv_files)} files. Chronological order established.")
    print("Evaluating timeline using Random Forest Model & Baseline Normalization...\n")
    
    # Formatting table
    print(f"{'FILE (Waktu)':<15} | {'ML DIAGNOSIS':<20} | {'CONFIDENCE'} | {'DEVIATION'}")
    print("-" * 75)
    
    baseline_features = None

    for idx, f in enumerate(csv_files):
        filename = os.path.basename(f)
        try:
            # RTF files have no header; col 0 & 1 = vibration
            df = pd.read_csv(f, header=None)
            signal = (pd.to_numeric(df.iloc[:, 0], errors='coerce') + pd.to_numeric(df.iloc[:, 1], errors='coerce')) / 2
            signal = signal.interpolate().ffill().bfill().values

            if len(signal) == 0: continue

            # Extract raw features
            features = extract_features(signal, rpm, n_ball, Bd, Pd, sampling_rate)
            features_np = np.array(features)

            # --- Normalization using training scaler ---
            if rtf_scaler is not None:
                normalized_features = rtf_scaler.transform(features_np.reshape(1, -1))[0]
                deviation_score = np.mean(np.abs(normalized_features))
            else:
                # Fallback: baseline subtraction from first file
                if baseline_features is None:
                    baseline_features = features_np
                    normalized_features = np.zeros_like(features_np)
                    deviation_score = 0.0
                else:
                    normalized_features = features_np - baseline_features
                    relative_diff = np.abs(features_np - baseline_features) / (np.abs(baseline_features) + 1e-6)
                    deviation_score = np.mean(relative_diff)

            # Reshape for prediction
            X_input = normalized_features.reshape(1, -1)
            
            # Predict using RF Model
            pred_encoded = model.predict(X_input)[0]
            probs = model.predict_proba(X_input)[0]
            confidence = np.max(probs)
            
            if np.issubdtype(type(pred_encoded), np.integer) or str(pred_encoded).isdigit():
                pred_label = le.inverse_transform([int(pred_encoded)])[0]
            else:
                pred_label = pred_encoded

            # UI Color logic
            color = Fore.GREEN if pred_label.upper() == 'HEALTHY' else Fore.RED
            
            print(f"{filename:<15} | {color}{pred_label.upper():<20}{Style.RESET_ALL} | {confidence*100:>8.1f}% | {deviation_score*100:>6.2f} %")
            
        except Exception as e:
            print(Fore.RED + f"{filename:<15} | FAILED TO PROCESS    | ERROR: {e}")
            
    print("-" * 75)
    print(Style.BRIGHT + "\n🏁 Timeline Evaluation Complete using AI Model.")

if __name__ == "__main__":
    main()
