import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}'

def normal_pdf(x, mu, var):
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mu)**2 / (2 * var))

def reward_mean(mu_vec, c):
    return mu_vec[0] + mu_vec[1] * c

def reward_var(Sigma, sigma_r, c):
    x = np.array([1.0, float(c)])
    S = np.array([[Sigma[0], Sigma[1]],
                  [Sigma[2], Sigma[3]]])
    return float(x @ S @ x + sigma_r**2)

def main():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11

    colors = {
        'unpooled': '#1f77b4',
        'pooled':   '#d62728',
        'eb':       '#ff7f0e',
    }

    dist_colors = {
        'a0c0': '#1f77b4',
        'a0c1': '#2ca02c',
        'a1c0': '#d62728',
        'a1c1': '#9467bd',
    }

    df = pd.read_csv("final_graphs_contextual/p_sweep_scenario1.csv")

    fig, (ax, ax2) = plt.subplots(
        2, 1,
        figsize=(7, 8),
        gridspec_kw={'height_ratios': [3, 2.1]}
    )
    fig.patch.set_facecolor("#ffffff")

    # ---------- top panel: regret sweep ----------
    for col, se_col, label, key in [
        ('mean_unpooled', 'se_unpooled', 'Unpooled TS', 'unpooled'),
        ('mean_pooled',   'se_pooled',   'Pooled TS',   'pooled'),
        ('mean_eb',       'se_eb',       'EB',          'eb'),
    ]:
        ax.plot(
            df.p_context, df[col],
            label=label, color=colors[key], linewidth=2.2, alpha=0.95
        )
        ax.fill_between(
            df.p_context,
            df[col] - 1.96 * df[se_col],
            df[col] + 1.96 * df[se_col],
            color=colors[key], alpha=0.15
        )

    ax.set_facecolor("#ffffff")
    ax.set_xlabel(r'$P(c_{i,t}=1)$', fontsize=12)
    ax.set_ylabel('Final cumulative regret', fontsize=12)
    ax.set_title(r'$n=50$, $T=200$, 100 runs per $p$', fontsize=13, pad=18)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    winners = df['winner'].values
    p_vals = df['p_context'].values
    ranges = []
    current = winners[0]
    start = p_vals[0]

    for i in range(1, len(winners)):
        if winners[i] != current:
            ranges.append((current, start, p_vals[i-1]))
            current = winners[i]
            start = p_vals[i]
    ranges.append((current, start, p_vals[-1]))

    for i in range(len(ranges) - 1):
        ax.axvline(
            x=ranges[i+1][1],
            color='gray',
            linestyle=':',
            linewidth=1,
            alpha=0.5
        )

    if ranges:
        range_text = "Best performing:\n"
        for winner, s, e in ranges:
            if s == e:
                range_text += f"{winner}: $p={s:.2f}$\n"
            else:
                range_text += f"{winner}: $p\\in[{s:.2f},{e:.2f}]$\n"

        ax.text(
            0.02, 0.985, range_text.rstrip('\n'),
            transform=ax.transAxes,
            fontsize=9,
            va='top',
            ha='left',
            bbox=dict(
                boxstyle='round',
                facecolor='#f5f5f5',
                alpha=0.65,
                edgecolor='#cccccc'
            )
        )

    ax.legend(
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),
        frameon=True,
        framealpha=0.65,
        facecolor='#f5f5f5',
        edgecolor='#cccccc',
        fontsize=9,
        handlelength=1.5,
        handletextpad=0.5,
        borderpad=0.4,
        labelspacing=0.3
    )

    # ---------- bottom panel: theoretical reward distributions ----------
    # ---------- single source of truth for environment parameters ----------
    mu_a = {
        0: np.array([0.0, 1.0]),
        1: np.array([0.5, -1.0]),
    }
    Sigma_a = {
        0: [0.2, 0.0, 0.0, 0.2],
        1: [0.2, 0.0, 0.0, 0.2],
    }
    sigma_r = 0.5
    mu_prior = np.array([0.0, 0.0])
    lam = 1e-6

    combos = [
        (0, 0, r'$a=0,\ c=0$', 'a0c0'),
        (0, 1, r'$a=0,\ c=1$', 'a0c1'),
        (1, 0, r'$a=1,\ c=0$', 'a1c0'),
        (1, 1, r'$a=1,\ c=1$', 'a1c1'),
    ]

    means = [reward_mean(mu_a[a], c) for a, c, _, _ in combos]
    vars_ = [reward_var(Sigma_a[a], sigma_r, c) for a, c, _, _ in combos]
    sds = [np.sqrt(v) for v in vars_]

    x_min = min(m - 4*s for m, s in zip(means, sds))
    x_max = max(m + 4*s for m, s in zip(means, sds))
    xgrid = np.linspace(x_min, x_max, 600)

    for (a, c, label, key), mu, var in zip(combos, means, vars_):
        y = normal_pdf(xgrid, mu, var)
        ax2.plot(
            xgrid, y,
            color=dist_colors[key],
            linewidth=2.2,
            label=label
        )
        ax2.fill_between(
            xgrid, 0, y,
            color=dist_colors[key],
            alpha=0.22
        )

    ax2.set_facecolor("#ffffff")
    ax2.set_xlabel('Reward', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Theoretical reward distributions', fontsize=12, pad=12)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.legend(
        loc='upper right',
        frameon=True,
        framealpha=0.65,
        facecolor='#f5f5f5',
        edgecolor='#cccccc',
        fontsize=9
    )

    # ---------- auto-generated figure annotations ----------
    mu00, mu01 = mu_a[0]
    mu10, mu11 = mu_a[1]

    s0 = Sigma_a[0]
    s1 = Sigma_a[1]

    er_00 = reward_mean(mu_a[0], 0)
    er_10 = reward_mean(mu_a[1], 0)
    er_01 = reward_mean(mu_a[0], 1)
    er_11 = reward_mean(mu_a[1], 1)

    params_line1 = (
        rf"\textbf{{True environment}}: "
        rf"$\bm{{\mu}}_0=({mu00:.2f},\,{mu01:.2f})^\top$, "
        rf"$\bm{{\mu}}_1=({mu10:.2f},\,{mu11:.2f})^\top$, "
        rf"$\bm{{\Sigma}}_0=\begin{{pmatrix}}{s0[0]:.2f}&{s0[1]:.2f}\\{s0[2]:.2f}&{s0[3]:.2f}\end{{pmatrix}}$, "
        rf"$\bm{{\Sigma}}_1=\begin{{pmatrix}}{s1[0]:.2f}&{s1[1]:.2f}\\{s1[2]:.2f}&{s1[3]:.2f}\end{{pmatrix}}$, "
        rf"$\sigma={sigma_r:.2f}$"
    )

    params_line2 = (
        rf"$c=0$: $\mathbb{{E}}[R_{{i,t}}(0)]={er_00:.2f}$, "
        rf"$\mathbb{{E}}[R_{{i,t}}(1)]={er_10:.2f}$"
    )

    params_line3 = (
        rf"$c=1$: $\mathbb{{E}}[R_{{i,t}}(0)]={er_01:.2f}$, "
        rf"$\mathbb{{E}}[R_{{i,t}}(1)]={er_11:.2f}$"
    )

    params_line4 = (
        rf"\textbf{{Learning prior}}: "
        rf"$\bm{{\mu}}_{{\mathrm{{prior}}}}=({mu_prior[0]:.2f},{mu_prior[1]:.2f})^\top$, "
        rf"$\lambda={lam:.0e}$"
    )

    fig.text(0.5, 0.108, params_line1, fontsize=9, va='bottom', ha='center')
    fig.text(0.5, 0.081, params_line2, fontsize=9, va='bottom', ha='center')
    fig.text(0.5, 0.057, params_line3, fontsize=9, va='bottom', ha='center')
    fig.text(0.5, 0.033, params_line4, fontsize=9, va='bottom', ha='center')

    # more room at bottom so annotations don't get cut off
    plt.tight_layout(rect=[0, 0.16, 1.0, 0.97])

    os.makedirs('final_graphs_contextual', exist_ok=True)
    plt.savefig(
        "final_graphs_contextual/p_sweep_scenario1.png",
        dpi=150,
        bbox_inches='tight'
    )
    print("Saved plot to final_graphs_contextual/p_sweep_scenario1.png")

    print("\n=== Summary ===")
    print(f"Total p values tested: {len(df)}")
    print(f"\nWinner distribution:\n{df['winner'].value_counts().to_string()}")

    if ranges:
        print("\nRanges where each algorithm performs best:")
        for w, s, e in ranges:
            if s == e:
                print(f"  {w}: p = {s:.2f}")
            else:
                print(f"  {w}: p in [{s:.2f}, {e:.2f}]")

    plt.show()

if __name__ == "__main__":
    main()