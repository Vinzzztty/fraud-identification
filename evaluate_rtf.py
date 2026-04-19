import pandas as pd
import numpy as np
import os
import glob
from colorama import init, Fore, Style

init(autoreset=True)
import extract_features_vibration as extract_module

def main():
    print(Style.BRIGHT + "Initializing Run-To-Failure Evaluation Pipeline (Anomaly Detection Mode)...\n")
    
    extract_features = extract_module.extract_features
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
    print("Evaluating timeline using Baseline Deviation Scoring...\n")
    
    # Formatting table
    print(f"{'FILE (Waktu)':<15} | {'PREDIKSI MODEL':<20} | {'DEVIASI DARI AWAL'}")
    print("-" * 57)
    
    baseline_features = None
    DEVIATION_THRESHOLD = 0.25  # Threshold deviasi rata-rata 25% dari baseline awal (file 44)
    
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
            features_np = np.array(features)
            
            # Anomaly Detection Logic: Bandingkan profil data ini dengan TITIK NOL (File pertama yang diasumsikan sehat)
            if baseline_features is None:
                baseline_features = features_np
                deviation_score = 0.0
            else:
                # Mengukur persentase perubahan rata-rata secara absolut pada keseluruhan 16 fitur
                relative_diff = np.abs(features_np - baseline_features) / (np.abs(baseline_features) + 1e-6)
                deviation_score = np.mean(relative_diff)
            
            # Classify based on deviation score
            if deviation_score < DEVIATION_THRESHOLD:
                pred_label_str = 'HEALTHY'
                color = Fore.GREEN
            else:
                pred_label_str = 'BEARING FAULT'
                color = Fore.RED
                
            print(f"{filename:<15} | {color}{pred_label_str:<20}{Style.RESET_ALL} | {deviation_score*100:>6.2f} %")
            
        except Exception as e:
            print(Fore.RED + f"{filename:<15} | FAILED TO PROCESS    | {e}")
            
    print("-" * 57)
    print(Style.BRIGHT + "\n🏁 Timeline Evaluation Complete.")

if __name__ == "__main__":
    main()
