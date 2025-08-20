# feature_extraction.py
# total 27 features
import numpy as np
import pandas as pd
from scipy.signal import welch

fs = 512  # Sampling frequency (Hz)

# --- Feature extraction function ---
def extract_features(segment):
    freqs, psd = welch(segment, fs=fs, nperseg=fs)

    # Frequency bands
    delta_band, theta_band, alpha_band, beta_band = (0.5, 4), (4, 8), (8, 13), (13, 30)

    # Band masks
    delta_mask = (freqs >= delta_band[0]) & (freqs < delta_band[1])
    theta_mask = (freqs >= theta_band[0]) & (freqs < theta_band[1])
    alpha_mask = (freqs >= alpha_band[0]) & (freqs < alpha_band[1])
    beta_mask  = (freqs >= beta_band[0]) & (freqs < beta_band[1])

    # Band powers
    delta_power = np.trapezoid(psd[delta_mask], freqs[delta_mask])
    theta_power = np.trapezoid(psd[theta_mask], freqs[theta_mask])
    alpha_power = np.trapezoid(psd[alpha_mask], freqs[alpha_mask])
    beta_power  = np.trapezoid(psd[beta_mask],  freqs[beta_mask])

    # Relative powers
    total_power = delta_power + theta_power + alpha_power + beta_power
    delta_relative = delta_power / total_power if total_power else 0
    theta_relative = theta_power / total_power if total_power else 0
    alpha_relative = alpha_power / total_power if total_power else 0
    beta_relative  = beta_power  / total_power if total_power else 0

    # Ratios
    beta_theta_ratio = beta_power / theta_power if theta_power else 0
    slow_fast_ratio = delta_power / beta_power if beta_power else 0
    beta_delta_theta_ratio = beta_power / (delta_power + theta_power) if (delta_power + theta_power) else 0
    alpha_beta_ratio = alpha_power / beta_power if beta_power else 0
    delta_theta_ratio = delta_power / theta_power if theta_power else 0
    delta_alpha_ratio = delta_power / alpha_power if alpha_power else 0
    delta_beta_ratio  = delta_power / beta_power if beta_power else 0
    theta_alpha_ratio = theta_power / alpha_power if alpha_power else 0
    theta_beta_ratio  = theta_power / beta_power if beta_power else 0

    # Additional spectral features
    peak_freq = freqs[np.argmax(psd)]
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) else 0

    log_freqs, log_psd = np.log(freqs[1:]), np.log(psd[1:] + 1e-10)
    spectral_slope = np.polyfit(log_freqs, log_psd, 1)[0] if len(log_freqs) > 1 else 0

    entropy = -(psd / np.sum(psd) * np.log(psd / np.sum(psd) + 1e-10)).sum() if np.sum(psd) else 0
    cumulative_power = np.cumsum(psd)
    sef95 = freqs[np.where(cumulative_power >= 0.95 * cumulative_power[-1])[0][0]] if cumulative_power[-1] else 0

    # Hjorth parameters
    diff_segment = np.diff(segment)
    mobility = np.sqrt(np.var(diff_segment) / np.var(segment)) if np.var(segment) else 0
    complexity = (np.sqrt(np.var(np.diff(diff_segment)) / np.var(diff_segment)) / mobility) if mobility else 0

    return [
        delta_power, theta_power, alpha_power, beta_power,
        delta_relative, theta_relative, alpha_relative, beta_relative,
        beta_theta_ratio, slow_fast_ratio, beta_delta_theta_ratio,
        spectral_slope, entropy, sef95, mobility, complexity,
        np.mean(segment), np.std(segment), np.var(segment),
        alpha_beta_ratio, delta_theta_ratio, delta_alpha_ratio,
        delta_beta_ratio, theta_alpha_ratio, theta_beta_ratio,
        peak_freq, spectral_centroid
    ]

# --- Load segmented data ---
active_segments = pd.read_csv("active_segments.csv", header=None).values
drowsy_segments = pd.read_csv("drowsy_segments.csv", header=None).values

# --- Extract features ---
active_features = np.array([extract_features(seg) for seg in active_segments])
drowsy_features = np.array([extract_features(seg) for seg in drowsy_segments])

# --- Column names ---
columns = [
    "Delta Power", "Theta Power", "Alpha Power", "Beta Power",
    "Delta Rel", "Theta Rel", "Alpha Rel", "Beta Rel",
    "Beta/Theta", "Slow/Fast", "Beta/(Delta+Theta)",
    "Spectral Slope", "Entropy", "SEF95",
    "Hjorth Mobility", "Hjorth Complexity",
    "Mean", "Std", "Variance",
    "Alpha/Beta", "Delta/Theta", "Delta/Alpha",
    "Delta/Beta", "Theta/Alpha", "Theta/Beta",
    "Peak Freq", "Spectral Centroid"
]

# --- Save to CSV ---
active_df = pd.DataFrame(active_features, columns=columns)
drowsy_df = pd.DataFrame(drowsy_features, columns=columns)

active_df.to_csv("activefeatures.csv", index=False)
drowsy_df.to_csv("drowsyfeatures.csv", index=False)

# Combined with labels
active_df["label"], drowsy_df["label"] = "active", "drowsy"
combined_df = pd.concat([active_df, drowsy_df], ignore_index=True)
combined_df.to_csv("features.csv", index=False)

print("✅ Feature extraction complete. Saved activefeatures.csv, drowsyfeatures.csv, and features.csv")
