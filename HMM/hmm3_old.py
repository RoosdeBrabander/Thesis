# === 3-state HMM on your weekly S&P 500 CSV + regime-specific GBM sims ===
# pip install pandas numpy matplotlib hmmlearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

# ---------------------------
# 0) Load & prepare the data
# ---------------------------
file_path = r"C:\Users\roosd\Downloads\econometrie jaar 3\Thesis\sp500_weekly_closed.csv"

df = pd.read_csv(file_path)

# Ensure types
if "Date" not in df.columns or ("Close" not in df.columns and "Adj Close" not in df.columns):
    raise ValueError("CSV must contain at least 'Date' and 'Close' (or 'Adj Close').")

price_col = "Close" if "Close" in df.columns else "Adj Close"
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

# If LogReturn not present, compute it
if "LogReturn" not in df.columns:
    df["LogReturn"] = np.log(df[price_col] / df[price_col].shift(1))

# Clean
df = df.dropna(subset=["Date", price_col, "LogReturn"]).sort_values("Date")
prices = df.set_index("Date")[[price_col]].rename(columns={price_col: "price"})
rets = df.set_index("Date")["LogReturn"]

# ---------------------------
# 1) Fit a 3-state Gaussian HMM
# ---------------------------
X = rets.values.reshape(-1, 1)      # shape (T, 1)
hmm = GaussianHMM(
    n_components=3,
    covariance_type="diag",
    n_iter=1000,
    random_state=42,                # seed for reproducibility
    verbose=False
)
hmm.fit(X)

# ---------------------------
# 2) Summaries and ordering
# ---------------------------
means_w = hmm.means_.ravel()  # weekly means
if hmm.covariance_type == "diag":
    vars_w = np.array([np.diag(c) for c in hmm.covars_]).ravel()
else:
    vars_w = hmm.covars_.ravel()
stds_w = np.sqrt(vars_w)

# Order by mean: bear < neutral < bull
order = np.argsort(means_w)
means_w = means_w[order]
stds_w  = stds_w[order]
A = hmm.transmat_[order][:, order]
pi = hmm.startprob_[order]

# Annualize (52 weeks)
W = 52.0
means_a = means_w * W
stds_a  = stds_w * np.sqrt(W)

summary = pd.DataFrame({
    "State": [f"S{i+1}" for i in range(3)],
    "Mean (weekly)": means_w,
    "Vol (weekly)": stds_w,
    "Mean (annual)": means_a,
    "Vol (annual)": stds_a
})

print("\n=== State parameter summary ===")
print(summary.round(6))

print("\n=== Transition matrix (rows->next state) ===")
print(pd.DataFrame(A, index=summary["State"], columns=summary["State"]).round(4))

print("\n=== Start probabilities ===")
print(pd.Series(pi, index=summary["State"]).round(4))

# ---------------------------
# 3) Decoding: posteriors & Viterbi
# ---------------------------
post = hmm.predict_proba(X)[:, order]           # smoothed probabilities per ordered state
viterbi_raw = hmm.predict(X)                    # original labels
label_map = {orig: new for new, orig in enumerate(order)}
viterbi = np.vectorize(label_map.get)(viterbi_raw)  # remapped to our order

# ---------------------------
# 4) Quick EDA: stats & plots
# ---------------------------
mean_w = rets.mean()
std_w  = rets.std()
print("\n=== Basic returns stats ===")
print(f"Mean weekly: {mean_w:.6f} | Std weekly: {std_w:.6f} | "
      f"Annualized mean: {mean_w*W:.4f} | Annualized vol: {std_w*np.sqrt(W):.4f}")

# Price with regimes (points colored by inferred regime)
plt.figure(figsize=(10,5))
plt.plot(prices.index, prices["price"], label="S&P 500")
plt.scatter(rets.index, prices.loc[rets.index, "price"], c=viterbi, s=10, label="Regime (Viterbi)")
plt.title("S&P 500 Price with Inferred Regimes (3-state HMM)")
plt.xlabel("Date"); plt.ylabel("Price")
plt.legend(); plt.tight_layout(); plt.show()

# Weekly log returns over time (with fewer ticks)
plt.figure(figsize=(10,5))
plt.plot(rets.index, rets.values, label="Weekly log return")
plt.axhline(0, lw=1, color="gray")
plt.title("Weekly Log Returns")
plt.xlabel("Date"); plt.ylabel("Log return")
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
plt.xticks(rotation=45)
plt.legend(); plt.tight_layout(); plt.show()

# Histogram of returns
plt.figure(figsize=(8,5))
plt.hist(rets.values, bins=50, edgecolor="black")
plt.title("Distribution of Weekly Log Returns")
plt.xlabel("Log return"); plt.ylabel("Frequency")
plt.tight_layout(); plt.show()

# Posterior probability per state (3 separate figures)
for i in range(3):
    plt.figure(figsize=(10,3.5))
    plt.plot(rets.index, post[:, i])
    plt.title(f"Smoothed Posterior Probability: {summary['State'].iloc[i]}")
    plt.xlabel("Date"); plt.ylabel("Probability")
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.xticks(rotation=45)
    plt.tight_layout(); plt.show()

# ---------------------------
# 5) Regime-specific GBM simulations (no switching)
# ---------------------------
def simulate_gbm(S0, mu_annual, sigma_annual, steps=104, dt=1/52, seed=123):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(steps)
    S = np.empty(steps+1); S[0] = S0
    drift = (mu_annual - 0.5*sigma_annual**2) * dt
    for t in range(1, steps+1):
        S[t] = S[t-1] * np.exp(drift + sigma_annual*np.sqrt(dt)*Z[t-1])
    return S

S0 = float(prices.iloc[-1, 0])
horizon = 104  # ~2 years
gbm_paths = []
for i in range(3):
    gbm_paths.append(
        simulate_gbm(S0, mu_annual=float(means_a[i]), sigma_annual=float(stds_a[i]),
                     steps=horizon, dt=1/W, seed=100+i)
    )

plt.figure(figsize=(10,5))
for i, path in enumerate(gbm_paths):
    plt.plot(range(horizon+1), path, label=f"{summary['State'].iloc[i]} GBM")
plt.title("Regime-specific GBM Simulations (no switching)")
plt.xlabel("Weeks ahead"); plt.ylabel("Simulated price")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,5))
plt.scatter(df["Date"], rets, c=viterbi, s=10)
plt.axhline(0, color="gray", lw=1)
plt.title("Weekly Log Returns with Regime Coloring")
plt.xlabel("Date"); plt.ylabel("Log Return")
plt.tight_layout(); plt.show()

print(summary[["Mean (weekly)", "Vol (weekly)"]])
