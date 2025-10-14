# === Basic data exploration for your weekly S&P 500 data ===
# pip install pandas matplotlib numpy

import pandas as pd
import matplotlib.pyplot as plt

# --- 1) Load your file ---
file_path = r"C:\Users\roosd\Downloads\econometrie jaar 3\Thesis\sp500_weekly_closed.csv"
df = pd.read_csv(file_path)

# --- 2) Ensure numeric types ---
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df["LogReturn"] = pd.to_numeric(df["LogReturn"], errors="coerce")

# --- 3) Basic info ---
print("=== Basic Info ===")
print(df.info())
print("\n=== Descriptive Statistics ===")
print(df[["Close", "LogReturn"]].describe().T)

# --- 4) Check if any missing values ---
print("\nMissing values:\n", df.isna().sum())

# --- 5) Plot price and log returns ---
plt.figure(figsize=(10,5))
plt.plot(df["Date"], df["Close"], label="S&P 500 Weekly Close")
plt.title("S&P 500 Weekly Closing Prices (10 Years)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df["Date"], df["LogReturn"], label="Weekly Log Return")
plt.axhline(0, color="gray", lw=1)
plt.title("Weekly Log Returns of S&P 500")
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.legend()
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # show only ~10 date ticks
plt.tight_layout()
plt.show()


# --- 6) Histogram of weekly returns ---
plt.figure(figsize=(8,5))
plt.hist(df["LogReturn"], bins=50, edgecolor="black")
plt.title("Distribution of Weekly Log Returns")
plt.xlabel("Log Return")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --- 7) Print some extra summary metrics ---
mean_ret = df["LogReturn"].mean()
std_ret = df["LogReturn"].std()
annual_mean = mean_ret * 52
annual_vol = std_ret * (52 ** 0.5)
print("\n=== Key Summary ===")
print(f"Mean weekly return: {mean_ret:.4f}")
print(f"Std weekly return: {std_ret:.4f}")
print(f"Annualized mean: {annual_mean:.4f}")
print(f"Annualized volatility: {annual_vol:.4f}")
