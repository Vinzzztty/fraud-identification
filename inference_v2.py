import pandas as pd
import numpy as np
import joblib
import os
import glob
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

import extract_features_vibration as extract_module

def load_feature_extractor():
    return extract_module.extract_features

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
    # e.g., looks for /1470rpm/ or \\1470rpm\\
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part.endswith("rpm") and part[:-3].isdigit():
            return int(part[:-3])
    # Fallback default if parsing fails
    return 1500

def run_scenarios():
    print(Style.BRIGHT + "Initializing Inference Scenarios...\n")
    
    # 1. Load the trained RF model
    model_path = "models/rf_model.pkl"
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(Fore.RED + f"❌ Failed to load models: {e}")
        return

    extract_features = load_feature_extractor()

    # 2. Pick explicit scenarios to test
    # Find one file for each category if possible
    base_dir = "dataset"
    all_files = glob.glob(f"{base_dir}/**/*.csv", recursive=True)
    
    if not all_files:
        print(Fore.RED + "❌ No datasets found for scenarios.")
        return

    # Categorize available files to pick one of each
    scenarios = {
        "healthy": None,
        "bearing": None,
        "misalignment": None,
        "others": None
    }
    
    for f in all_files:
        cat = get_expected_label(f)
        if scenarios[cat] is None:
            scenarios[cat] = f
        # Break early if we found one for each
        if all(v is not None for v in scenarios.values()):
            break

    # 3. Test each scenario
    for scenario_name, file_path in scenarios.items():
        if file_path is None:
            continue
            
        print(Style.BRIGHT + "-" * 60)
        print(Fore.CYAN + f"🧪 Scenario: Expected '{scenario_name.upper()}'")
        print(f"📁 From file: {file_path}")
        
        rpm = extract_rpm_from_path(file_path)
        print(f"🔄 Detected RPM: {rpm}")

        # Set Constants (Matching training setup)
        n_ball = 8
        Bd = 15.5
        Pd = 72.5
        sampling_rate = 12000 
        
        try:
            df = pd.read_csv(file_path)
            # Use the second column if available, else first
            if df.shape[1] >= 2:
                signal = pd.to_numeric(df.iloc[:, 1], errors='coerce')
            else:
                signal = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                
            signal = signal.interpolate().ffill().bfill().values
            
            features = extract_features(signal, rpm, n_ball, Bd, Pd, sampling_rate)
            features_2d = np.array(features).reshape(1, -1)
            
            expected_feature_count = model.n_features_in_
            if features_2d.shape[1] != expected_feature_count:
                print(Fore.RED + f"❌ Error: Model expects {expected_feature_count} features, extracting outputted: {features_2d.shape[1]}.")
                continue

            # Predict
            predicted_label = model.predict(features_2d)[0]
            probabilities = model.predict_proba(features_2d)[0]
            
            if predicted_label.lower() == scenario_name.lower():
                print(Fore.GREEN + f"✅ SUCCESS: Model correctly predicted {predicted_label.upper()}")
            else:
                print(Fore.RED + f"⚠️ MISMATCH: Model predicted {predicted_label.upper()} (Expected {scenario_name.upper()})")
                
            print("\nProbabilities:")
            for cls, prob in zip(model.classes_, probabilities):
                print(f"  [{cls.ljust(15)}]: {prob*100:.2f}%")
                
        except Exception as e:
            print(Fore.RED + f"❌ Failed to process scenario: {e}")

    print(Style.BRIGHT + "-" * 60)
    print("🏁 All scenarios completed.")

if __name__ == "__main__":
    run_scenarios()
