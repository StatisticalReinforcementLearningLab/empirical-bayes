import numpy as np
from pathlib import Path
import pandas as pd
from scipy.optimize import minimize_scalar
from openpyxl import Workbook

def empirical_bayes_sequential(csv_file, sigma2=5000):

    # Quick fix for random csv file path issue 
    csv_path = Path(csv_file)
    if not csv_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        csv_path = (repo_root / csv_file).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Import csv as df
    df = pd.read_csv(csv_path)

    # Group once to keep everything aligned by individual
    g = (df.sort_values('individual')
           .groupby('individual', sort=True)
           .agg(x_i=('steps','mean'),
                true_mu_i=('true_mu_i','mean'),
                true_mu_0=('true_mu_0','first'))   # constant per dataset
           .reset_index())

    ids         = g['individual'].values
    x_i         = g['x_i'].astype(float).values
    true_mu_i   = g['true_mu_i'].astype(float).values
    true_mu_0   = float(g['true_mu_0'].iloc[0])      # same for all rows
    n = len(g)

    # Known within-person variance for a single observation
    sigma_i_sq = np.full(n, float(sigma2))

    # Estimate tau^2 (profile ll) using single-day x_i and sigma2
    def neg_loglik(tau2):
        if tau2 < 0:
            return np.inf
        w_i = 1.0 / (sigma_i_sq + tau2)
        mu_0 = np.sum(w_i * x_i) / np.sum(w_i)
        Q = np.sum(w_i * (x_i - mu_0)**2)
        return 0.5 * (np.sum(np.log(sigma_i_sq + tau2)) + Q)

    tau2_hat = float(minimize_scalar(neg_loglik, bounds=(0, 1e8), method='bounded').x)
    tau2_hat = max(tau2_hat, 1e-12)  # small floor for stability

    # Estimate mu_0 using those weights
    w_i = 1.0 / (sigma_i_sq + tau2_hat)
    mu_0_hat = float(np.sum(w_i * x_i) / np.sum(w_i))

    # Posterior for one observation per person
    posterior_var = 1.0 / (1.0/sigma_i_sq + 1.0/tau2_hat)
    mu_i_posterior = posterior_var * (x_i / sigma_i_sq + mu_0_hat / tau2_hat)

    print(f"mu_0={mu_0_hat:.2f}, tau2={tau2_hat:.2f}")

    # Final table (now includes true values)
    results_df = pd.DataFrame({
        'individual': ids,
        'x_i': x_i,
        'mu_i_posterior': mu_i_posterior,
        'posterior_var': posterior_var,
        'mu_0_hat': mu_0_hat,
        'tau2_hat': tau2_hat,
        'true_mu_0': true_mu_0,     # ← included
        'true_mu_i': true_mu_i      # ← included
    }).sort_values('individual')
    
    # Save to "single-empirical-bayes" folder inside the current directory
    out_dir = Path.cwd() / "single-empirical-bayes"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{csv_path.stem}_sequential_posterior.xlsx"
    results_df.to_excel(out_path, index=False)
    print(f"\nWrote results to {out_path}")
    
    return results_df

# Run
results = empirical_bayes_sequential('single-empirical-bayes/generated-data.csv')
print(results.head(10))