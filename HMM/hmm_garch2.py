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

def simulate_hmm_garch(hmm_model, garch_params, n_obs, random_state=42):
    """
    Simulate returns from HMM-GARCH hybrid model
    """
    np.random.seed(random_state)
    
    # Step 1: Simulate HMM state sequence
    states = []
    current_state = np.random.choice(hmm_model.n_components, 
                                      p=hmm_model.startprob_)
    states.append(current_state)
    
    for t in range(1, n_obs):
        current_state = np.random.choice(
            hmm_model.n_components,
            p=hmm_model.transmat_[current_state]
        )
        states.append(current_state)
    
    states = np.array(states)
    
    # Step 2: Simulate GARCH with state-dependent scaling
    # Get regime probabilities for volatility weighting
    regime_probs = np.zeros(n_obs)
    high_vol_state = np.argmax(hmm_model.means_)
    for t in range(n_obs):
        regime_probs[t] = 1.0 if states[t] == high_vol_state else 0.5
    
    # Simulate base GARCH process
    omega = garch_params['omega']
    alpha = garch_params['alpha[1]']
    beta = garch_params['beta[1]']
    mu = garch_params['mu']
    
    returns = np.zeros(n_obs)
    sigma2 = np.zeros(n_obs)
    sigma2[0] = omega / (1 - alpha - beta)  # Unconditional variance
    
    for t in range(n_obs):
        # Apply regime-dependent volatility scaling
        regime_scale = 1 + regime_probs[t]
        
        # GARCH variance equation with regime scaling
        if t > 0:
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        
        # Scale by regime
        sigma2[t] *= regime_scale
        
        # Generate return
        returns[t] = mu + np.sqrt(sigma2[t]) * np.random.randn()
    
    return returns, states, sigma2

# Use the function
sim_returns, sim_states, sim_vol = simulate_hmm_garch(
    hmm, 
    res_hmm_garch.params,
    len(returns),
    random_state=42
)

sim_data = pd.Series(sim_returns)