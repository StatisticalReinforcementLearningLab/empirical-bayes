import pandas as pd
import matplotlib.pyplot as plt

csv_path = "mean_regret_over_time_500_runs.csv"
out_path = "cumulative_regret_over_time_500_runs.png"

df = pd.read_csv(csv_path)

# Keep these in sync with 030926.cpp constants.
params = {
	"n": 50,
	"T": 40,
	"n_runs": 500,
	"lambda_prior": 1.0,
	"mu_theta": [0.0, 1.0, 0.5, -0.25],
	"Sigma_theta_diag": [0.02, 0.02, 0.02, 0.02],
	"alpha": 0.0,
	"beta": 0.8,
	"gamma": 0.4,
	"sigma_s": 0.5,
	"sigma_r": 1.5,
	"mu_s0": 0.0,
	"sigma_s0": 1.0,
}

cum_unpooled = df["unpooled_ts_mean_regret"].cumsum()
cum_pooled = df["pooled_ts_mean_regret"].cumsum()
cum_eb = df["eb_ts_mean_regret"].cumsum()

plt.figure(figsize=(8, 5))
plt.plot(df["time"], cum_unpooled, label="Unpooled TS")
plt.plot(df["time"], cum_pooled, label="Pooled TS")
plt.plot(df["time"], cum_eb, label="EB TS")
plt.xlabel("Time")
plt.ylabel("Cumulative Regret")
plt.title("Cumulative Regret Over Time (500 Runs)")
plt.legend()

param_text = (
	f"n={params['n']}, T={params['T']}, runs={params['n_runs']}\n"
	f"lambda_prior={params['lambda_prior']}\n"
	f"mu_theta={params['mu_theta']}\n"
	f"Sigma_diag={params['Sigma_theta_diag']}\n"
	f"alpha={params['alpha']}, beta={params['beta']}, gamma={params['gamma']}\n"
	f"sigma_s={params['sigma_s']}, sigma_r={params['sigma_r']}\n"
	f"mu_s0={params['mu_s0']}, sigma_s0={params['sigma_s0']}"
)

plt.gca().text(
	1.02,
	0.5,
	param_text,
	transform=plt.gca().transAxes,
	va="center",
	fontsize=8,
	bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
)

plt.subplots_adjust(right=0.72)
plt.tight_layout()
plt.savefig(out_path, dpi=200)
print(f"Saved plot to {out_path}")
