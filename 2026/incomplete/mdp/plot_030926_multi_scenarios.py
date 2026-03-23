import matplotlib.pyplot as plt
import pandas as pd

csv_path = "multi_scenario_mean_regret_500_runs.csv"
out_path = "multi_scenario_cumulative_regret_500_runs.png"

df = pd.read_csv(csv_path)
scenarios = list(df["scenario"].unique())

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False, sharey=False)
axes = axes.flatten()

for i, scenario in enumerate(scenarios):
    ax = axes[i]
    d = df[df["scenario"] == scenario].sort_values("time")

    ax.plot(d["time"], d["unpooled_ts_mean_regret"].cumsum(), label="Unpooled TS")
    ax.plot(d["time"], d["pooled_ts_mean_regret"].cumsum(), label="Pooled TS")
    ax.plot(d["time"], d["eb_ts_mean_regret"].cumsum(), label="EB TS")
    ax.set_title(scenario)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Regret")
    ax.grid(alpha=0.2)

for j in range(len(scenarios), 4):
    axes[j].axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955))
fig.suptitle("Cumulative Regret Across Four Scenarios (500 Runs Each)")
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(out_path, dpi=220)
print(f"Saved plot to {out_path}")
