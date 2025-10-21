import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from tabulate import tabulate

# === Load Data ===
file_path = r"C:\Users\roosd\Downloads\econometrie jaar 3\Thesis\sp500_weekly_closed.csv"
df = pd.read_csv(file_path)
df.columns = [c.lower() for c in df.columns]
price_col = [c for c in df.columns if 'close' in c][0]

df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
df.dropna(inplace=True)
returns = df['log_return'] * 100  # scale to percentages for stability
returns = returns.reset_index(drop=True)

# === Split Train/Test ===
split_idx = int(len(returns) * 0.8)
train = returns.iloc[:split_idx]
test = returns.iloc[split_idx:]

print(f"Train size: {len(train)}, Test size: {len(test)}")

# === Fit GARCH(1,1) ===
model = arch_model(train, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
res = model.fit(update_freq=5, disp='off')
print(res.summary())

# === Forecast 1-step ahead variance ===
# Refit GARCH on the full series but only use training data for estimation
model = arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
res = model.fit(first_obs=0, last_obs=split_idx - 1, update_freq=5, disp='off')

# Forecast volatility for the test period
fcst = res.forecast(horizon=1, start=split_idx)

# Extract 1-step-ahead conditional variances
test_vol_forecast = fcst.variance.dropna().iloc[:, 0].to_numpy()

# Match realized variance (squared returns) for the same window
realized_vol = (test ** 2).to_numpy()

print("Shapes (realized, forecast):", realized_vol.shape, test_vol_forecast.shape)

# === Simulate returns from fitted model ===
sim_data = model.simulate(res.params, nobs=len(returns))['data']

# === Stylized Facts Evaluation ===
def stylized_facts(data, sim_data=None):
    acf_abs = [data.abs().autocorr(lag=i) for i in range(1, 11)]
    acf_sq = [(data**2).autocorr(lag=i) for i in range(1, 11)]
    kurt = stats.kurtosis(data, fisher=False)
    skew = stats.skew(data)
    lb_p = acorr_ljungbox(data, lags=[10], return_df=True)['lb_pvalue'].iloc[0]
    res_dict = {
        "Mean ACF(|r|, lags 1–10)": np.mean(acf_abs),
        "Mean ACF(r², lags 1–10)": np.mean(acf_sq),
        "Kurtosis": kurt,
        "Skewness": skew,
        "Ljung–Box p(returns)": lb_p
    }
    if sim_data is not None:
        ks_stat, ks_p = stats.ks_2samp(data, sim_data)
        res_dict["KS test p-value"] = ks_p
    else:
        res_dict["KS test p-value"] = np.nan
    return res_dict

sf_emp = stylized_facts(returns)
sf_sim = stylized_facts(sim_data, sim_data=returns)

# === Quantitative Evaluation ===
def qlike(y_true, y_pred):
    return np.mean(np.log(y_pred) + y_true / y_pred)

rmse = np.sqrt(np.mean((realized_vol - test_vol_forecast) ** 2))
qlike_val = qlike(realized_vol, test_vol_forecast)

aic = res.aic
bic = res.bic
llf = res.loglikelihood

# === Display Results ===
print("\n=== Stylized Facts Summary ===")
table_sf = [[k, sf_emp[k], sf_sim[k]] for k in sf_emp.keys()]
print(tabulate(table_sf, headers=["Statistic", "Empirical", "GARCH(1,1)"], floatfmt=".4f"))

print("\n=== Quantitative Evaluation ===")
table_q = [
    ["Log-Likelihood", llf],
    ["AIC", aic],
    ["BIC", bic],
    ["RMSE (vol forecast)", rmse],
    ["QLIKE", qlike_val]
]
print(tabulate(table_q, headers=["Metric", "Value"], floatfmt=".4f"))

# === Visualization Section ===
sns.set_style("whitegrid")

# 1. Returns and conditional volatility
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(df['log_return'], color='black', lw=0.8)
plt.title("S&P 500 Weekly Log Returns")
plt.subplot(2, 1, 2)
plt.plot(res.conditional_volatility, color='darkred', lw=0.8)
plt.title("Conditional Volatility (GARCH(1,1))")
plt.tight_layout()
plt.show()

# 2. Distribution comparison
plt.figure(figsize=(8, 4))
sns.kdeplot(returns, label='Empirical', fill=True, alpha=0.5)
sns.kdeplot(sim_data, label='Simulated (GARCH)', fill=True, alpha=0.5)
plt.title("Distribution: Empirical vs Simulated GARCH Returns")
plt.legend()
plt.show()

# 3. ACF of absolute returns
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_acf(returns.abs(), ax=axes[0], lags=20, title='Empirical |r_t|')
plot_acf(sim_data.abs(), ax=axes[1], lags=20, title='Simulated |r_t| (GARCH)')
plt.tight_layout()
plt.show()

