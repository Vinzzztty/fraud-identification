import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from colorama import init, Fore, Style
from concurrent.futures import ProcessPoolExecutor, as_completed

init(autoreset=True)

import extract_features_vibration as extract_module
extract_features = extract_module.extract_features

def get_expected_label(path):
    path_lower = path.lower()
    if "bearing" in path_lower:
        return "bearing"
    elif "healthy" in path_lower or "new motor" in path_lower or "noise" in path_lower:
        return "healthy"
    elif "loose" in path_lower or "soft" in path_lower:
        return "misalignment"
    else:
        return "others"

def extract_rpm_from_path(path):
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part.endswith("rpm") and part[:-3].isdigit():
            return int(part[:-3])
    return 1500

def process_file(f):
    label = get_expected_label(f)
    rpm = extract_rpm_from_path(f)
    
    n_ball = 8
    Pd = 72.5
    Bd = 15.5
    sampling_rate = 12000
    
    X_local = []
    y_local = []
    
    try:
        df = pd.read_csv(f)
        if df.shape[1] >= 2:
            signal_cols = df.columns[1:]
        else:
            signal_cols = df.columns
            
        for col in signal_cols:
            signal = pd.to_numeric(df[col], errors='coerce')
            signal = signal.interpolate().ffill().bfill().values
            
            if len(signal) < 1000 or np.isnan(signal).all():
                continue
                
            try:
                features = extract_features(signal, rpm, n_ball, Bd, Pd, sampling_rate)
                X_local.append(features)
                y_local.append(label)
            except Exception as e:
                pass
    except Exception as e:
        pass
        
    return X_local, y_local

def train():
    print(Style.BRIGHT + "Initiating FAST Training Pipeline...\n")
    
    csv_files = glob.glob("dataset/**/*.csv", recursive=True)
    if not csv_files:
        print(Fore.RED + "No datasets found in 'dataset/' directory.")
        return
        
    print(f"Found {len(csv_files)} CSV files. Extracting features using all cores...")
    
    X = []
    y = []
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, f): f for f in csv_files}
        for i, future in enumerate(as_completed(futures)):
            X_loc, y_loc = future.result()
            X.extend(X_loc)
            y.extend(y_loc)
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(csv_files)} files...")
                
    print(Fore.GREEN + f"\n✅ Successfully extracted {len(X)} samples.\n")
    
    X = np.array(X)
    y = np.array(y)
    
    # Train validation split isn't strictly necessary just for saving but SMOTE is requested
    print("Encoding labels and applying SMOTE...")
    le = LabelEncoder()
    le.fit(y)
    
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    
    print("Training RandomForestClassifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_res, y_res)
    print(Fore.GREEN + "✅ Training complete.")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/rf_model.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    
    print(Style.BRIGHT + f"\n🎉 Model saved successfully! Expected features: {rf_model.n_features_in_}")

if __name__ == "__main__":
    train()
