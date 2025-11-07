import yfinance as yf
import pandas as pd

# Download 20 years of weekly data for S&P 500 (^GSPC)
df = yf.download("^GSPC", period="20y", interval="1wk", auto_adjust=True)

# Keep only Date + Close
df = df.reset_index()[["Date", "Close"]]

# Save locally
output_path = r"C:\Users\roosd\Downloads\econometrie jaar 3\Thesis\sp500_weekly.csv"
df.to_csv(output_path, index=False)

print(f"Saved 20y weekly S&P500 data to:\n{output_path}")
print(df.head())
