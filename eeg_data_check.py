# eeg_data_check.py
# Script to load and inspect EEG datasets (Active vs Drowsy)

import pandas as pd

def load_and_preview():
    try:
        # Step 1: Load the datasets
        active = pd.read_csv("active.csv")
        drowsy = pd.read_csv("drowsy.csv")

        # Step 2: Print dataset shapes
        print("✅ Datasets loaded successfully!\n")
        print(f"Active dataset shape : {active.shape}")
        print(f"Drowsy dataset shape : {drowsy.shape}\n")

        # Step 3: Print column names
        print("Active dataset columns:", list(active.columns))
        print("Drowsy dataset columns:", list(drowsy.columns), "\n")

        # Step 4: Show first 5 rows for quick check
        print("🔹 Active data sample:")
        print(active.head(), "\n")

        print("🔹 Drowsy data sample:")
        print(drowsy.head(), "\n")

        # Step 5: Check column consistency
        if list(active.columns) == list(drowsy.columns):
            print("✅ Column names match between Active and Drowsy datasets.")
        else:
            print("⚠️ Column names do NOT match!")
            print("Active columns:", list(active.columns))
            print("Drowsy columns:", list(drowsy.columns))

        return active, drowsy

    except FileNotFoundError as e:
        print("❌ Error: File not found. Make sure 'active.csv' and 'drowsy.csv' are in the same folder.")
        print(e)
    except Exception as e:
        print("❌ Unexpected error occurred:")
        print(e)

if __name__ == "__main__":
    load_and_preview()
