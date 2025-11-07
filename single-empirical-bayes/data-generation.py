# single-empirical-bayes/generate_one_day.py
import numpy as np
import pandas as pd
from pathlib import Path

# --- True hyperparameters for simulation ---
mu0   = 10000.0     # population mean
tau2  = 15000.0     # population variance
sigma2 = 5000.0     # within-person/device variance (constant here)
n     = 30          # number of individuals

np.random.seed(888)

rows = []
for i in range(1, n + 1):
    # Draw true person mean
    mu_i = np.random.normal(mu0, np.sqrt(tau2))
    # One observation for that person
    x_i  = np.random.normal(mu_i, np.sqrt(sigma2))
    rows.append({
        "individual": i,
        "steps": int(max(0, x_i)),
        "true_mu_i": mu_i,
        "true_mu_0": mu0,       # ← ADD THIS
        "sigma2_i": sigma2
    })

df = pd.DataFrame(rows)

# Save next to the algorithm, as a CSV (the algorithm will read this)
out_dir = Path.cwd() / "single-empirical-bayes"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "generated-data.csv"
df.to_csv(out_path, index=False)