# eeg_preprocessing.py
# Script to preprocess EEG signals:
#  - Apply 50 Hz notch filter
#  - Apply 0.5–30 Hz Butterworth bandpass filter
#  - Save filtered signals
#  - Remove spikes/artifacts
#  - Save cleaned signals

import pandas as pd
import numpy as np
from scipy.signal import iirnotch, butter, filtfilt

# --- Sampling frequency (from dataset description) ---
fs = 512  

def apply_filters(signal, fs):
    """Apply 50 Hz notch + 0.5–30 Hz bandpass filter to EEG signal."""
    # --- 50 Hz notch filter ---
    f0 = 50  
    Q = 30   
    b_notch, a_notch = iirnotch(f0, Q, fs)

    # --- 0.5–30 Hz bandpass filter ---
    lowcut = 0.5
    highcut = 30
    b_band, a_band = butter(4, [lowcut / (fs/2), highcut / (fs/2)], btype='band')

    # --- Apply filters sequentially ---
    filtered = filtfilt(b_notch, a_notch, signal)
    filtered = filtfilt(b_band, a_band, filtered)
    return filtered

def remove_spikes(signal, threshold=100):
    """Remove large amplitude spikes/artifacts using interpolation."""
    cleaned = signal.copy()
    spike_indices = np.where(np.abs(cleaned) > threshold)[0]
    for idx in spike_indices:
        if 1 <= idx < len(cleaned) - 1:
            cleaned[idx] = (cleaned[idx-1] + cleaned[idx+1]) / 2
    return cleaned

def preprocess_eeg():
    try:
        # --- Load datasets ---
        active = pd.read_csv("active.csv")
        drowsy = pd.read_csv("drowsy.csv")
        print("✅ Datasets loaded successfully!")

        # --- Select EEG channel (2nd column assumed) ---
        if active.shape[1] < 2 or drowsy.shape[1] < 2:
            raise ValueError("CSV files must have at least 2 columns (ID/time + EEG signal).")

        active_signal = active.iloc[:, 1].values
        drowsy_signal = drowsy.iloc[:, 1].values

        # --- Apply filters ---
        active_filtered = apply_filters(active_signal, fs)
        drowsy_filtered = apply_filters(drowsy_signal, fs)

        # --- Save filtered signals ---
        pd.DataFrame({"Filtered_Signal": active_filtered}).to_csv("active_filtered.csv", index=False)
        pd.DataFrame({"Filtered_Signal": drowsy_filtered}).to_csv("drowsy_filtered.csv", index=False)
        print("✅ Filtered signals saved: 'active_filtered.csv', 'drowsy_filtered.csv'")

        # --- Remove spikes/artifacts ---
        active_cleaned = remove_spikes(active_filtered, threshold=100)
        drowsy_cleaned = remove_spikes(drowsy_filtered, threshold=100)

        # --- Save cleaned signals ---
        pd.DataFrame({"Cleaned_Signal": active_cleaned}).to_csv("active_cleaned.csv", index=False)
        pd.DataFrame({"Cleaned_Signal": drowsy_cleaned}).to_csv("drowsy_cleaned.csv", index=False)
        print("✅ Cleaned signals saved: 'active_cleaned.csv', 'drowsy_cleaned.csv'")

        print("🎯 Preprocessing complete!")
        print(f"Active samples: {len(active_cleaned)} | Drowsy samples: {len(drowsy_cleaned)}")

        return active_cleaned, drowsy_cleaned

    except Exception as e:
        print("❌ Error during preprocessing:", e)

if __name__ == "__main__":
    preprocess_eeg()
