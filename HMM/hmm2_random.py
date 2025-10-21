# === Robust 2-state HMM (weekly S&P500 returns): loop over 20 seeds, simulate 1000 paths/seed ===
# pip install numpy pandas matplotlib hmmlearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    # scikit-learn’s convergence warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass


# ---------------------------
# I/O
# ---------------------------
def load_returns_from_csv(path, return_col="LogReturn", date_col="Date"):
    df = pd.read_csv(path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    if return_col not in df.columns:
        raise ValueError(f"Column '{return_col}' not found in CSV.")
    rets = pd.to_numeric(df[return_col], errors="coerce").dropna()
    return rets  # pd.Series (Date index if present)

# ---------------------------
# HMM fit + utilities
# ---------------------------
def fit_hmm_gaussian_nstate(returns, n_states=2, random_state=42, n_iter=1000):
    X = returns.values.reshape(-1, 1)
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False
    )
    hmm.fit(X)
    return hmm

def order_states_by_mean(hmm):
    means = hmm.means_.ravel()
    if hmm.covariance_type == "diag":
        vars_ = np.array([np.diag(c) for c in hmm.covars_]).ravel()
    else:
        vars_ = hmm.covars_.ravel()
    order = np.argsort(means)  # ascending mean
    means_ord = means[order]
    vars_ord  = vars_[order]
    trans_ord = hmm.transmat_[order][:, order]
    start_ord = hmm.startprob_[order]
    return means_ord, vars_ord, trans_ord, start_ord, order

def expected_durations(trans):
    """Expected length in each state: 1/(1 - A_ii)"""
    diag = np.diag(trans)
    return 1.0 / (1.0 - diag)

# ---------------------------
# Simulation from fitted HMM
# ---------------------------
def simulate_hmm_returns(means, vars_, transmat, startprob, T, n_paths=1000, seed=2025):
    """Simulate n_paths sequences of length T from a Gaussian HMM."""
    rng = np.random.default_rng(seed)
    n_states = len(means)
    stds = np.sqrt(vars_)

    sim = np.empty((T, n_paths), dtype=float)

    trans_cum = np.cumsum(transmat, axis=1)
    start_cum = np.cumsum(startprob)

    # initial state per path
    states = np.searchsorted(start_cum, rng.random(n_paths))
    sim[0, :] = means[states] + stds[states] * rng.standard_normal(n_paths)

    for t in range(1, T):
        u = rng.random(n_paths)
        next_states = np.empty_like(states)
        for s in range(n_states):
            idx = (states == s)
            if idx.any():
                next_states[idx] = np.searchsorted(trans_cum[s], u[idx])
        states = next_states
        sim[t, :] = means[states] + stds[states] * rng.standard_normal(n_paths)

    return pd.DataFrame(sim, columns=[f"path_{i}" for i in range(n_paths)])

# ---------------------------
# Stylized facts: stats & plots
# ---------------------------
def series_stats(x: pd.Series):
    mean = x.mean()
    std  = x.std(ddof=1)
    skew = x.skew()
    ex_kurt = x.kurt()  # excess
    kurt = ex_kurt + 3.0
    return pd.Series({"mean": mean, "std": std, "skew": skew,
                      "excess_kurtosis": ex_kurt, "kurtosis": kurt})

def compare_real_vs_sim_across_seeds(real_rets: pd.Series, sim_dfs: list):
    # real stats
    real_stats = series_stats(real_rets)

    # For each seed's sim_df (T x 1000), compute per-path stats then average
    seed_rows = []
    for i, sdf in enumerate(sim_dfs):
        per_path_stats = pd.DataFrame(series_stats(sdf[c]) for c in sdf.columns)
        row = per_path_stats.mean(axis=0)
        row.name = f"seed_{i}"
        seed_rows.append(row)

    per_seed = pd.DataFrame(seed_rows)  # rows: seeds
    comp = pd.DataFrame({
        "real": real_stats,
        "sim_mean_across_seeds": per_seed.mean(axis=0),
        "sim_p05_across_seeds":  per_seed.quantile(0.05),
        "sim_p95_across_seeds":  per_seed.quantile(0.95),
    })

    print("\n=== Stylized facts: real vs simulated (mean per-seed, 1000 paths/seed) ===")
    print(comp.round(6))

    return comp, per_seed

def plot_histograms(real_rets: pd.Series, sim_all: pd.DataFrame, bins=50, title_suffix=""):
    pooled = sim_all.values.ravel()
    plt.figure(figsize=(10,5))
    plt.hist(real_rets.values, bins=bins, alpha=0.7, density=True, label="Real")
    plt.hist(pooled,          bins=bins, alpha=0.5, density=True, label="Simulated (pooled)")
    plt.title(f"Histogram (density) of Weekly Log Returns: Real vs Simulated {title_suffix}")
    plt.xlabel("Weekly log return"); plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

def acf(x: np.ndarray, max_lag=20):
    # Ensure 1-D float array
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x - x.mean()
    var = np.dot(x, x) / x.size
    ac = np.empty(max_lag + 1, dtype=float)
    ac[0] = 1.0
    for k in range(1, max_lag + 1):
        ac[k] = np.dot(x[:-k], x[k:]) / ((x.size - k) * var)
    return ac

def plot_acf_squared(real_rets: pd.Series, sim_all: pd.DataFrame, max_lag=20, sample_paths=1500):
    # Real series (1-D)
    ac_real = acf(np.square(np.asarray(real_rets.values, dtype=float)), max_lag=max_lag)

    # Choose a manageable subset of columns for the band
    rng = np.random.default_rng(123)
    all_cols = list(sim_all.columns)
    if sample_paths < len(all_cols):
        cols = list(rng.choice(all_cols, size=sample_paths, replace=False))
    else:
        cols = all_cols

    # Compute ACF per selected path (force 1-D each time)
    ac_sims = []
    for c in cols:
        x = np.asarray(sim_all[c].values, dtype=float).reshape(-1)
        ac_sims.append(acf(np.square(x), max_lag=max_lag))
    ac_sims = np.vstack(ac_sims)

    ac_sim_mean = ac_sims.mean(axis=0)
    ac_sim_p05  = np.quantile(ac_sims, 0.05, axis=0)
    ac_sim_p95  = np.quantile(ac_sims, 0.95, axis=0)

    lags = np.arange(max_lag + 1)
    plt.figure(figsize=(10, 5))
    markerline, stemlines, baseline = plt.stem(lags, ac_real, label="Real (squared returns)")
    plt.setp(markerline, markersize=4)
    plt.setp(stemlines, linewidth=1)
    plt.plot(lags, ac_sim_mean, label="Simulated mean (squared returns)")
    plt.fill_between(lags, ac_sim_p05, ac_sim_p95, alpha=0.2, label="Simulated 5–95% band")
    plt.title("ACF of Squared Weekly Returns (Volatility Clustering)")
    plt.xlabel("Lag (weeks)"); plt.ylabel("Autocorrelation")
    plt.legend(); plt.tight_layout(); plt.show()


# ---------------------------
# Main (20 seeds x 1000 paths)
# ---------------------------
def main():
    # 1) Load weekly returns
    file_path = r"C:\Users\roosd\Downloads\econometrie jaar 3\Thesis\sp500_weekly_closed.csv"
    rets = load_returns_from_csv(file_path, return_col="LogReturn", date_col="Date")
    T = len(rets)

    # 2) Loop over seeds: fit HMM & simulate 1000 paths per seed
    n_states = 2
    seeds = list(range(20))  # 0..19
    sim_dfs = []             # list of DataFrames (T x 1000) per seed
    param_rows = []          # store parameters per seed (means/vols/A_ii/durations)

    for s in seeds:
        hmm = fit_hmm_gaussian_nstate(rets, n_states=n_states, random_state=s, n_iter=1000)
        means, vars_, trans, start, order = order_states_by_mean(hmm)
        durations = expected_durations(trans)

        # record parameters (ordered S1<S2 by mean)
        param_rows.append({
            "seed": s,
            "mean_S1": means[0], "vol_S1": np.sqrt(vars_[0]),
            "mean_S2": means[1], "vol_S2": np.sqrt(vars_[1]),
            "A11": trans[0,0], "A22": trans[1,1],
            "dur_S1": durations[0], "dur_S2": durations[1]
        })

        # simulate 1000 paths with a deterministic simulation seed per (seed)
        sim_df = simulate_hmm_returns(
            means=means, vars_=vars_, transmat=trans, startprob=start,
            T=T, n_paths=1000, seed=10_000 + s
        )
        sim_dfs.append(sim_df)

        print(f"Seed {s:2d} → means {means.round(5)}, vols {np.sqrt(vars_).round(5)}, "
              f"Aii [{trans[0,0]:.4f}, {trans[1,1]:.4f}], "
              f"dur [{durations[0]:.2f}, {durations[1]:.2f}]")

    # 3) Robustness: parameters across seeds
    params = pd.DataFrame(param_rows)
    print("\n=== Parameter robustness across 20 seeds (ordered by mean) ===")
    print(params.describe().T.round(6))

    # 4) Stylized facts: real vs simulated, aggregated across seeds
    comp, per_seed_stats = compare_real_vs_sim_across_seeds(rets, sim_dfs)

    # 5) Combine all simulations (20,000 paths total) for pooled plots
    sim_all = pd.concat(sim_dfs, axis=1)  # shape: (T, 20000)

    # 6) Histograms (real vs pooled simulated)
    plot_histograms(rets, sim_all, bins=50, title_suffix="(2-state HMM, 20 seeds, 20k paths)")

    # 7) (Optional) ACF of squared returns (subset for speed)
    plot_acf_squared(rets, sim_all, max_lag=20, sample_paths=1500)

if __name__ == "__main__":
    main()
