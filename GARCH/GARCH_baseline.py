import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

# === Load and preprocess data ===
file_path = r"C:\Users\roosd\Downloads\econometrie jaar 3\Thesis\sp500_weekly_closed.csv"

df = pd.read_csv(file_path)
df.columns = [col.lower() for col in df.columns]

# Try to detect price column
price_col = [c for c in df.columns if 'close' in c][0]
df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
df.dropna(inplace=True)

returns = df['log_return'] * 100  # Convert to percentages for better scaling

print(f"Sample size: {len(returns)} weekly returns")

# === Fit GARCH(1,1) ===
model = arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
res = model.fit(update_freq=5, disp='off')
print(res.summary())

# === Extract conditional volatility ===
df['cond_vol'] = res.conditional_volatility

# === Simulate returns from fitted GARCH model ===
sim_data = model.simulate(res.params, nobs=len(returns))
sim_returns = sim_data['data']

# === Stylized facts evaluation ===
print("\n=== Stylized Facts Evaluation ===")

# 1. Volatility clustering → autocorrelation of |r_t| and r_t^2
abs_acf = [returns.abs().autocorr(lag=i) for i in range(1, 11)]
sq_acf = [(returns**2).autocorr(lag=i) for i in range(1, 11)]

print("Volatility Clustering (ACF of |r_t| and r_t^2):")
for lag, (a, b) in enumerate(zip(abs_acf, sq_acf), 1):
    print(f"Lag {lag}: |r_t|={a:.3f}, r_t^2={b:.3f}")

# 2. Heavy tails (Kurtosis)
kurt_real = stats.kurtosis(returns, fisher=False)
kurt_sim = stats.kurtosis(sim_returns, fisher=False)
print(f"\nKurtosis: Empirical={kurt_real:.2f}, Simulated={kurt_sim:.2f}")

# 3. Skewness
skew_real = stats.skew(returns)
skew_sim = stats.skew(sim_returns)
print(f"Skewness: Empirical={skew_real:.2f}, Simulated={skew_sim:.2f}")

# 4. Return independence (Ljung–Box test on raw returns)
lb_test = acorr_ljungbox(returns, lags=[10], return_df=True)
print(f"\nLjung–Box test (returns): p-value={lb_test['lb_pvalue'].iloc[0]:.3f}")
if lb_test['lb_pvalue'].iloc[0] > 0.05:
    print("→ No significant autocorrelation (stylized fact confirmed)")
else:
    print("→ Some autocorrelation detected")

# === 5. Visualizations ===
sns.set_style("whitegrid")

# Plot 1: Returns and conditional volatility
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(df['log_return'], color='black', lw=0.7)
ax[0].set_title('Weekly Log Returns (S&P 500)')
ax[1].plot(df['cond_vol'], color='darkred', lw=0.8)
ax[1].set_title('Conditional Volatility (GARCH(1,1))')
plt.tight_layout()
plt.show()

# Plot 2: Distribution comparison
plt.figure(figsize=(8, 4))
sns.kdeplot(returns, label='Empirical', fill=True, alpha=0.5)
sns.kdeplot(sim_returns, label='Simulated (GARCH)', fill=True, alpha=0.5)
plt.title('Distribution of Returns: Empirical vs. GARCH Simulation')
plt.legend()
plt.show()

# Plot 3: ACF of absolute returns
from statsmodels.graphics.tsaplots import plot_acf

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_acf(returns.abs(), ax=axes[0], lags=20, title='ACF of |r_t| (Empirical)')
plot_acf(sim_returns.abs(), ax=axes[1], lags=20, title='ACF of |r_t| (Simulated)')
plt.tight_layout()
plt.show()
