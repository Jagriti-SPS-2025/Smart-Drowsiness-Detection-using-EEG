# segmentation.py
import numpy as np
import pandas as pd

# --- Step 1: Load cleaned signals ---
active_cleaned = pd.read_csv("active_cleaned.csv")["EEG"].values
drowsy_cleaned = pd.read_csv("drowsy_cleaned.csv")["EEG"].values

# --- Step 2: Segmentation function ---
def segment_signal(signal, window_size=512, step_size=256):
    segments = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        segment = signal[start:start + window_size]
        segments.append(segment)
    return np.array(segments)

# --- Step 3: Apply segmentation ---
active_segments = segment_signal(active_cleaned, 512, 256)
drowsy_segments = segment_signal(drowsy_cleaned, 512, 256)

print(f"Active segments shape: {active_segments.shape}")
print(f"Drowsy segments shape: {drowsy_segments.shape}")

# --- Step 4: Save segmented data ---
# Save as .npy
np.save("active_segments.npy", active_segments)
np.save("drowsy_segments.npy", drowsy_segments)

# Save as .csv (flattened, each row = one segment)
pd.DataFrame(active_segments).to_csv("active_segments.csv", index=False)
pd.DataFrame(drowsy_segments).to_csv("drowsy_segments.csv", index=False)

print("Segmentation complete. Segments saved as .npy and .csv files.")
