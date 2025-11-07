import numpy as np
import pandas as pd

# TEST SET 1

mu0=10000
tau2=15000
sigma2=5000
n=30
days=10

np.random.seed(888)

data=[]

for i in range(1, n + 1):
    # Draw person's true mean from population distribution
    mu_i = np.random.normal(mu0, np.sqrt(tau2))
    
    # Draw step counts for each day from person's distribution
    for day in range(1, days + 1):
        x_i = np.random.normal(mu_i, np.sqrt(sigma2))
        data.append({
            'individual': i,
            'day': day,
            'steps': int(max(0, x_i)),
            'true_mu_i': mu_i,
            'true_tau2_+sigma2_i': tau2+sigma2
        })

df = pd.DataFrame(data)
df.to_csv('multi-empirical-bayes/generated-data.csv', index=False)