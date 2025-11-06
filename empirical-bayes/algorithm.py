import numpy as np
from pathlib import Path
import pandas as pd
from scipy.optimize import minimize_scalar

def empirical_bayes_update(csv_file, sigma2=5000):
    # resolve csv path relative to repo root
    csv_path = Path(csv_file)
    if not csv_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        csv_path = (repo_root / csv_file).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # per-individual summary (x_i = mean steps per individual)
    x_i = df.groupby('individual')['steps'].mean().values
    n = len(x_i)
    sigma_i_sq = np.full(n, sigma2)

    # negative log-likelihood for tau^2
    def neg_loglik(tau2):
        w_i = 1 / (sigma_i_sq + tau2)
        mu_0 = np.sum(w_i * x_i) / np.sum(w_i)
        Q = np.sum(w_i * (x_i - mu_0)**2)
        return 0.5 * (np.sum(np.log(sigma_i_sq + tau2)) + Q)

    tau2_hat = minimize_scalar(neg_loglik, bounds=(0, 1e8), method='bounded').x

    # estimate mu_0
    w_i = 1 / (sigma_i_sq + tau2_hat)
    mu_0_hat = np.sum(w_i * x_i) / np.sum(w_i)

    # posterior per individual
    lambda_i = tau2_hat / (sigma_i_sq + tau2_hat)
    mu_i_posterior = lambda_i * x_i + (1 - lambda_i) * mu_0_hat
    posterior_var = 1 / (1/sigma_i_sq + 1/tau2_hat)

    print(f"mu_0 = {mu_0_hat:.2f}, tau2 = {tau2_hat:.2f}")

    # results DataFrame (one row per individual)
    results = pd.DataFrame({
        'individual': range(1, n+1),
        'x_i': x_i,
        'mu_i_posterior': mu_i_posterior,
        'posterior_var': posterior_var
    })

    # Merge posterior info back into the full (original) dataframe
    merged = df.merge(results, on='individual', how='left')

    # write merged dataframe next to input CSV
    out_path = csv_path.parent / f"{csv_path.stem}_with_posterior.csv"
    merged.to_csv(out_path, index=False)
    print(f"Wrote merged results to {out_path}")

    return merged

# Run
results = empirical_bayes_update('data-generation/dp_steps1.csv')
print(results.head(10))