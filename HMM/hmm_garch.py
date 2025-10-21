import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
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
returns = df['log_return'] * 100
returns = returns.reset_index(drop=True)

split_idx = int(len(returns) * 0.8)
train = returns.iloc[:split_idx]
test = returns.iloc[split_idx:]

print(f"Train size: {len(train)}, Test size: {len(test)}")

# === STEP 1: Fit Gaussian HMM (2 states) on returns ===
hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=500, random_state=42)
hmm.fit(train.values.reshape(-1, 1))

# Predict state probabilities
posteriors = hmm.predict_proba(returns.values.reshape(-1, 1))
high_vol_state = np.argmax(hmm.means_)  # the state with higher mean volatility
prob_high_vol = posteriors[:, high_vol_state]
df['prob_high_vol'] = prob_high_vol

# === STEP 2: Define a hybrid volatility driver ===
# Combine squared returns and regime probability
df['hmm_weighted_ret_sq'] = (returns ** 2) * (1 + df['prob_high_vol'])

# Fit standard GARCH but using weighted squared returns indirectly
train_idx = np.arange(len(train))
test_idx = np.arange(split_idx, len(returns))

model_hmm_garch = arch_model(train, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
res_hmm_garch = model_hmm_garch.fit(update_freq=5, disp='off')
print(res_hmm_garch.summary())

# === Forecast as in baseline ===
# === Forecast as in baseline (FIXED) ===
# Refit GARCH on full data, but train only up to the split index
model_hmm_garch = arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
res_hmm_garch = model_hmm_garch.fit(first_obs=0, last_obs=split_idx - 1, update_freq=5, disp='off')

# Forecast volatility for the test period (one-step ahead rolling)
fcst_hmm = res_hmm_garch.forecast(horizon=1, start=split_idx)
test_vol_forecast = fcst_hmm.variance.dropna().iloc[:len(test), 0].to_numpy()

# Match realized variance (squared returns) for test period
realized_vol = (test ** 2).to_numpy()

print("Shapes (realized, forecast):", realized_vol.shape, test_vol_forecast.shape)

# === STEP 3: Stylized Facts & Evaluation ===
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

# Simulate HMM–GARCH returns (weighted volatility simulation)
sim_data = model_hmm_garch.simulate(res_hmm_garch.params, nobs=len(returns))["data"]

sf_emp = stylized_facts(returns)
sf_sim = stylized_facts(sim_data, sim_data=returns)

# === Quantitative Evaluation ===
def qlike(y_true, y_pred):
    return np.mean(np.log(y_pred) + y_true / y_pred)

rmse = np.sqrt(np.mean((realized_vol - test_vol_forecast) ** 2))
qlike_val = qlike(realized_vol, test_vol_forecast)

aic = res_hmm_garch.aic
bic = res_hmm_garch.bic
llf = res_hmm_garch.loglikelihood

# === Display Results ===
print("\n=== Stylized Facts Summary (HMM–GARCH) ===")
table_sf = [[k, sf_emp[k], sf_sim[k]] for k in sf_emp.keys()]
print(tabulate(table_sf, headers=["Statistic", "Empirical", "HMM–GARCH(1,1)"], floatfmt=".4f"))

print("\n=== Quantitative Evaluation (HMM–GARCH) ===")
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

# 1. Returns + HMM probability overlay
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(returns, color='black', lw=0.8, label='Returns')
ax2 = ax1.twinx()
ax2.plot(df['prob_high_vol'], color='red', alpha=0.5, label='Prob(High Vol Regime)')
ax1.set_title("Returns and HMM-Inferred Regime Probability")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.show()

# 2. Conditional volatility
plt.figure(figsize=(8, 4))
plt.plot(res_hmm_garch.conditional_volatility, color='darkblue', lw=0.8)
plt.title("Conditional Volatility (HMM–GARCH(1,1))")
plt.show()

# 3. Distribution comparison
plt.figure(figsize=(8, 4))
sns.kdeplot(returns, label='Empirical', fill=True, alpha=0.5)
sns.kdeplot(sim_data, label='Simulated (HMM–GARCH)', fill=True, alpha=0.5)
plt.title("Distribution: Empirical vs Simulated HMM–GARCH Returns")
plt.legend()
plt.show()

# 4. ACF of absolute returns
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_acf(returns.abs(), ax=axes[0], lags=20, title='Empirical |r_t|')
plot_acf(sim_data.abs(), ax=axes[1], lags=20, title='Simulated |r_t| (HMM–GARCH)')
plt.tight_layout()
plt.show()
