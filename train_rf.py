import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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

def get_motor_group(path):
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[1] # e.g., 'DATASET TA 18'
    return 'UNKNOWN'

def process_file(f):
    label = get_expected_label(f)
    rpm = extract_rpm_from_path(f)
    group_id = get_motor_group(f)
    
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
        
    return X_local, y_local, group_id

def process_file_with_label(file_info):
    f, label = file_info
    rpm = extract_rpm_from_path(f)
    group_id = get_motor_group(f)
    is_rtf = "Vibration_Bearing_RuntoFailure" in f.replace("\\", "/")
    
    # --- Parameter sets ---
    if is_rtf:
        n_ball = 9
        Bd = 7.938
        Pd = 38.5
        sampling_rate = 25600
        rpm = 1775
    else:
        n_ball = 8
        Pd = 72.5
        Bd = 15.5
        sampling_rate = 12000
    
    X_loc, y_loc = [], []
    try:
        if is_rtf:
            # RTF files have no header; col 0 & 1 = vibration, col 2 & 3 = temperature
            df = pd.read_csv(f, header=None)
            signal_cols = [0, 1]
        else:
            df = pd.read_csv(f)
            signal_cols = list(df.columns[1:]) if df.shape[1] >= 2 else list(df.columns)
            
        for col in signal_cols:
            signal = pd.to_numeric(df[col], errors='coerce').interpolate().ffill().bfill().values
            if len(signal) < 1000 or np.isnan(signal).all(): continue
            features = extract_features(signal, rpm, n_ball, Bd, Pd, sampling_rate)
            X_loc.append(features)
            y_loc.append(label)
    except: pass
    return X_loc, y_loc, group_id

def train():
    print(Style.BRIGHT + "Initiating FAST Training Pipeline...\n")
    
    # 1. Glob all CSV files
    all_csv_files = glob.glob("dataset/**/*.csv", recursive=True)
    if not all_csv_files:
        print(Fore.RED + "No datasets found in 'dataset/' directory.")
        return

    # 2. Intellectual Labeling (Handling RTF vs Standard Folders)
    labeled_files = []
    
    # Check for RTF folder
    rtf_dir = "dataset/Vibration_Bearing_RuntoFailure"
    rtf_files = [f for f in all_csv_files if rtf_dir in f.replace("\\", "/")]
    
    if rtf_files:
        print(f"Detected RTF sequence in {rtf_dir}. Applying timeline labeling...")
        rtf_files.sort() # Sort by timestamp in filename
        # Based on brief: Last 70 are Fault, Total 129. => First 59 are Healthy.
        for idx, f in enumerate(rtf_files):
            label = "healthy" if idx < 59 else "bearing"
            labeled_files.append((f, label))
            
    # Add other folders if they exist
    other_files = [f for f in all_csv_files if rtf_dir not in f.replace("\\", "/")]
    for f in other_files:
        label = get_expected_label(f)
        labeled_files.append((f, label))

    if not labeled_files:
        print(Fore.RED + "No valid labeled files found.")
        return

    print(f"Total files to process: {len(labeled_files)}")
    
    X = []
    y = []
    groups = []

    print(f"Extracting features using all cores...")
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file_with_label, info): info for info in labeled_files}
        for i, future in enumerate(as_completed(futures)):
            X_loc, y_loc, grp_id = future.result()
            for x_val, y_val in zip(X_loc, y_loc):
                X.append(x_val)
                y.append(y_val)
                groups.append(grp_id)
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(labeled_files)}...")
                
    print(Fore.GREEN + f"\n✅ Successfully extracted {len(X)} samples.\n")
    
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    
    print("Normalizing features per-motor (Step 3: Domain Adaptation)...")
    X_normalized = np.zeros_like(X)
    for grp in np.unique(groups):
        idx = (groups == grp)
        scaler = StandardScaler()
        if np.sum(idx) > 1:
            X_normalized[idx] = scaler.fit_transform(X[idx])
        else:
            X_normalized[idx] = X[idx] - np.mean(X[idx], axis=0)
            
    print("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Classes found: {le.classes_}")
    
    print("Splitting dataset (80% Train, 20% Validation)...")
    X_train, X_val, y_train, y_val = train_test_split(X_normalized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    print("Applying SMOTE on Training Data...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    print("\nTraining RandomForestClassifier...")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_res, y_train_res)
    
    print(Fore.GREEN + "✅ Training complete.")
    print(Style.BRIGHT + "\nEvaluating on Validation Data...")
    y_pred = rf_model.predict(X_val)
    print(classification_report(y_val, y_pred, target_names=le.classes_))
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/rf_model.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    print(Style.BRIGHT + f"\n🎉 Model saved successfully!")

if __name__ == "__main__":
    train()
