import numpy as np
from pathlib import Path
import pandas as pd
from scipy.optimize import minimize_scalar

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
    max_day = df['day'].max()
    
    results_list = []
    
    # For each day, calculate posterior using data up to that day
    for current_day in range(1, max_day + 1):
        # Filter data up to current day
        df_subset = df[df['day'] <= current_day].copy()
        
        # Calculate mean for each individual using data up to current_day
        x_i = df_subset.groupby('individual')['steps'].mean().values
        n = len(x_i)
        sigma_i_sq = np.full(n, sigma2)
        
        # Estimate tau^2
        def neg_loglik(tau2):
            w_i = 1 / (sigma_i_sq + tau2)
            mu_0 = np.sum(w_i * x_i) / np.sum(w_i)
            Q = np.sum(w_i * (x_i - mu_0)**2)
            return 0.5 * (np.sum(np.log(sigma_i_sq + tau2)) + Q)
        
        tau2_hat = minimize_scalar(neg_loglik, bounds=(0, 1e8), method='bounded').x
        
        # Estimate mu_0
        w_i = 1 / (sigma_i_sq + tau2_hat)
        mu_0_hat = np.sum(w_i * x_i) / np.sum(w_i)
        
        # Posterior
        lambda_i = tau2_hat / (sigma_i_sq + tau2_hat)
        mu_i_posterior = lambda_i * x_i + (1 - lambda_i) * mu_0_hat
        posterior_var = 1 / (1/sigma_i_sq + 1/tau2_hat)
        
        print(f"Day {current_day}: mu_0={mu_0_hat:.2f}, tau2={tau2_hat:.2f}")
        
        # Store results for this day
        for i in range(n):
            results_list.append({
                'day': current_day,
                'individual': i + 1,
                'x_i': x_i[i],
                'mu_i_posterior': mu_i_posterior[i],
                'posterior_var': posterior_var[i],
                'mu_0_hat': mu_0_hat,
                'tau2_hat': tau2_hat
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results_list)
    
    # Merge with original data
    merged = df.merge(results_df, on=['individual', 'day'], how='left')
    
    # Save to correct folder in the project 
    
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / f"{csv_path.stem}_sequential_posterior.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nWrote sequential results to {out_path}")
    
    return merged

# Run
results = empirical_bayes_sequential('data-generation/dp_steps1.csv')
print(results.head(10))