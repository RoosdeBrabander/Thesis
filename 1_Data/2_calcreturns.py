import pandas as pd
import numpy as np

# Load your file
file_path = r"C:\Users\roosd\Downloads\econometrie jaar 3\Thesis\sp500_weekly.csv"
df = pd.read_csv(file_path)

# --- FIX: ensure Close is numeric ---
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Compute weekly log returns
df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

# Drop rows with missing or invalid values
df = df.dropna(subset=["Close", "LogReturn"])

# Save back to CSV
df.to_csv(file_path, index=False)

print("Added numeric weekly log returns successfully!")
print(df.head())
