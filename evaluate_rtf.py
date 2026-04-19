import pandas as pd
import numpy as np
import joblib
import os
import glob
from colorama import init, Fore, Style

init(autoreset=True)
import extract_features_vibration as extract_module

def main():
    print(Style.BRIGHT + "Initializing Run-To-Failure Evaluation Pipeline...\n")
    
    # 1. Load Models
    model_path = "models/rf_model.pkl"
    encoder_path = "models/label_encoder.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print(Fore.RED + "❌ Error: Could not find model files.")
        return
        
    try:
        model = joblib.load(model_path)
        extract_features = extract_module.extract_features
    except Exception as e:
        print(Fore.RED + f"❌ Failed to load models or module: {e}")
        return
    
    # Constants for feature extraction (fallback defaults)
    n_ball = 8
    Bd = 15.5
    Pd = 72.5
    sampling_rate = 12000
    rpm = 1500 
    
    # 2. Get Run-to-Failure dataset
    target_dir = "dataset/DATASET TA/Vibration_Bearing_RuntoFailure (Pengujian dengan Dataset Berbeda)"
    if not os.path.exists(target_dir):
        print(Fore.RED + f"❌ Error: Directory not found - {target_dir}")
        return
        
    csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
    if not csv_files:
        print(Fore.RED + "❌ Error: No CSV files found in the dataset.")
        return
        
    # Sort files chronologically based on numeric filename
    def extract_file_number(filepath):
        basename = os.path.basename(filepath)
        name_without_ext = os.path.splitext(basename)[0]
        try:
            return int(name_without_ext)
        except ValueError:
            return 999999 # fallback end
            
    csv_files.sort(key=extract_file_number)
    print(Fore.CYAN + f"Found {len(csv_files)} files. Chronological order established.")
    print("Evaluating timeline...\n")
    
    # Formatting table
    print(f"{'FILE (Waktu)':<15} | {'PREDIKSI MODEL':<20} | {'CONFIDENCE (%)'}")
    print("-" * 55)
    
    for idx, f in enumerate(csv_files):
        filename = os.path.basename(f)
        try:
            df = pd.read_csv(f)
            if df.shape[1] >= 2:
                signal = pd.to_numeric(df.iloc[:, 1], errors='coerce')
            else:
                signal = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                
            signal = signal.interpolate().ffill().bfill().values
            if len(signal) == 0:
                continue
                
            # Extract features
            features = extract_features(signal, rpm, n_ball, Bd, Pd, sampling_rate)
            features_2d = np.array(features).reshape(1, -1)
            
            # Predict
            pred_label = model.predict(features_2d)[0] # direct output since rf trained on encoded/string ? Wait...
            # Actually train_rf.py converted y to numeric before training! Let me verify.
            # y_pred = rf_model.predict(X_val)
            # If so, model outputs numeric, we must decode it.
            # Let's handle both string and numeric safely:
            le = joblib.load(encoder_path)
            if np.issubdtype(type(pred_label), np.integer) or str(pred_label).isdigit():
                pred_label_str = le.inverse_transform([int(pred_label)])[0]
            else:
                # If model somehow outputs string directly
                pred_label_str = pred_label
                
            probabilities = model.predict_proba(features_2d)[0]
            confidence = probabilities.max() * 100
            
            # Coloring output
            if pred_label_str.lower() == 'healthy':
                color = Fore.GREEN
            elif pred_label_str.lower() == 'bearing':
                color = Fore.RED
            else:
                color = Fore.YELLOW
                
            print(f"{filename:<15} | {color}{pred_label_str.upper():<20}{Style.RESET_ALL} | {confidence:.2f}%")
            
        except Exception as e:
            print(Fore.RED + f"{filename:<15} | FAILED TO PROCESS    | {e}")
            
    print("-" * 55)
    print(Style.BRIGHT + "\n🏁 Timeline Evaluation Complete.")

if __name__ == "__main__":
    main()
