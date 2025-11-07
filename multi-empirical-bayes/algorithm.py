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
    max_day = df['day'].max()
    
    results_list = []
    
    # For each day, calculate posterior using data up to that day
    for current_day in range(1, max_day + 1):
        # Filter data up to current day
        df_subset = df[df['day'] <= current_day].copy()
        
        # CHANGE: compute per-person running mean and count
        grp = df_subset.groupby('individual')['steps']              
        xbar = grp.mean()                                               
        n_i  = grp.count().astype(float)                           
        ids  = xbar.index                                                   
        n    = len(xbar)                                                    

        # CHANGE: heteroskedastic variance for means: Var(\bar{x}) = sigma2 / n_i
        sigma_i_sq = (sigma2 / n_i.values)                          
        
        # Estimate tau^2 (profile ll) using xbar and sigma2/n_i
        def neg_loglik(tau2):
            w_i = 1 / (sigma_i_sq + tau2)                              
            mu_0 = np.sum(w_i * xbar.values) / np.sum(w_i)               
            Q = np.sum(w_i * (xbar.values - mu_0)**2)               
            return 0.5 * (np.sum(np.log(sigma_i_sq + tau2)) + 
        
        tau2_hat = minimize_scalar(neg_loglik, bounds=(0, 1e8), method='bounded').x
        
        # Estimate mu_0 using those weights
        w_i = 1 / (sigma_i_sq + tau2_hat)                                   # CHANGE
        mu_0_hat = np.sum(w_i * xbar.values) / np.sum(w_i)                  # CHANGE
        
        # CHANGE: posterior given \bar{x} and n_i
        # posterior_var = (n_i/sigma2 + 1/tau2_hat)^(-1)
        posterior_var = 1.0 / (n_i.values / sigma2 + 1.0 / tau2_hat)        # CHANGE
        # mu_i_posterior = v * ((n_i/sigma2)*\bar{x} + (1/tau2_hat)*mu_0_hat)
        mu_i_posterior = posterior_var * ( (n_i.values / sigma2) * xbar.values
                                           + (mu_0_hat / tau2_hat) )        # CHANGE
        
        print(f"Day {current_day}: mu_0={mu_0_hat:.2f}, tau2={tau2_hat:.2f}")
        
        # Store results for this day
        for i in range(n):
            results_list.append({
                'day': current_day,
                'individual': ids[i],                                       # CHANGE: keep true ID
                'x_i': xbar.values[i],                                      # CHANGE: store running mean (or rename to xbar_i)
                'n_i': int(n_i.values[i]),                                  # CHANGE: store count (helpful)
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
    
    # Save to "multi-empirical-bayes-updated" folder inside the current directory
    out_dir = Path.cwd() / "multi-empirical-bayes-updated"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{csv_path.stem}_sequential_posterior.xlsx"
    merged.to_excel(out_path, index=False)
    print(f"\nWrote sequential results to {out_path}")
    
    return merged

# Run
results = empirical_bayes_sequential('multi-empirical-bayes-all/generated-data.csv')
print(results.head(10))