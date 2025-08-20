# eeg_preprocessing.py
# Script to apply notch (50 Hz) and bandpass (0.5–30 Hz) filters to EEG data
# and save filtered signals to CSV

import pandas as pd
import numpy as np
from scipy.signal import iirnotch, butter, filtfilt

# --- Sampling frequency (from dataset description) ---
fs = 512  

def apply_filters(signal, fs):
    """Apply 50 Hz notch + 0.5–30 Hz bandpass filter to EEG signal."""

    # --- 50 Hz notch filter ---
    f0 = 50  # Frequency to remove (Hz)
    Q = 30   # Quality factor (sharpness)
    b_notch, a_notch = iirnotch(f0, Q, fs)

    # --- 0.5–30 Hz bandpass filter ---
    lowcut = 0.5
    highcut = 30
    b_band, a_band = butter(4, [lowcut / (fs/2), highcut / (fs/2)], btype='band')

    # --- Apply filters sequentially ---
    filtered = filtfilt(b_notch, a_notch, signal)
    filtered = filtfilt(b_band, a_band, filtered)

    return filtered

def preprocess_eeg():
    try:
        # --- Load datasets ---
        active = pd.read_csv("active.csv")
        drowsy = pd.read_csv("drowsy.csv")

        print("✅ Datasets loaded successfully!")

        # --- Select EEG channel (2nd column assumed) ---
        active_signal = active.iloc[:, 1].values
        drowsy_signal = drowsy.iloc[:, 1].values

        # --- Apply filters ---
        active_filtered = apply_filters(active_signal, fs)
        drowsy_filtered = apply_filters(drowsy_signal, fs)

        print("✅ Filtering complete.")
        print(f"Active signal: {len(active_filtered)} samples processed")
        print(f"Drowsy signal: {len(drowsy_filtered)} samples processed")

        # --- Save filtered signals into CSV ---
        pd.DataFrame({"Filtered_Signal": active_filtered}).to_csv("active_filtered.csv", index=False)
        pd.DataFrame({"Filtered_Signal": drowsy_filtered}).to_csv("drowsy_filtered.csv", index=False)

        print("✅ Filtered signals saved as 'active_filtered.csv' and 'drowsy_filtered.csv'.")

        return active_filtered, drowsy_filtered

    except Exception as e:
        print("❌ Error during preprocessing:", e)

if __name__ == "__main__":
    preprocess_eeg()
