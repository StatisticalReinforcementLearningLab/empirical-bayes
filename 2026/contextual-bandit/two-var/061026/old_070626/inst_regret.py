# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import numpy as np

# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}'

# def normal_pdf(x, mu, var):
#     return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mu)**2 / (2 * var))

# def reward_mean(mu_vec, c):
#     return mu_vec[0] + mu_vec[1] * c

# def reward_var(Sigma, sigma_r, c):
#     x = np.array([1.0, float(c)])
#     S = np.array([[Sigma[0], Sigma[1]],
#                   [Sigma[2], Sigma[3]]])
#     return float(x @ S @ x + sigma_r**2)

# def main():
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.size'] = 12
#     plt.rcParams['axes.labelsize'] = 12
#     plt.rcParams['axes.titlesize'] = 12
#     plt.rcParams['legend.fontsize'] = 11
#     plt.rcParams['xtick.labelsize'] = 11
#     plt.rcParams['ytick.labelsize'] = 11

#     colors = {
#         'unpooled': '#1f77b4',
#         'pooled':   '#d62728',
#         'eb':       '#ff7f0e',
#     }

#     dist_colors = {
#         'a0c0': '#1f77b4',
#         'a0c1': '#2ca02c',
#         'a1c0': '#d62728',
#         'a1c1': '#9467bd',
#     }

#     df = pd.read_csv("testing_code/INSTANT.csv")

#     # # ==========================================
#     # # --- NEW CODE: Calculate Average Regret ---
#     # # ==========================================
#     # # Divide cumulative regret (and its standard error) by T
#     # df['avg_regret_unpooled'] = df['mean_unpooled'] / df['T']
#     # df['avg_regret_pooled']   = df['mean_pooled'] / df['T']
#     # df['avg_regret_eb']       = df['mean_eb'] / df['T']

#     # df['se_avg_unpooled'] = df['se_unpooled'] / df['T']
#     # df['se_avg_pooled']   = df['se_pooled'] / df['T']
#     # df['se_avg_eb']       = df['se_eb'] / df['T']
#     # # ==========================================

#     fig, (ax, ax2) = plt.subplots(
#         2, 1,
#         figsize=(7, 8),
#         gridspec_kw={'height_ratios': [3, 2.1]}
#     )
#     fig.patch.set_facecolor("#ffffff")

#     # ---------- top panel: average regret sweep ----------
#     # Notice we swapped 'mean_...' for 'avg_regret_...' below
#     # for col, se_col, label, key in [
#     #     ('avg_regret_unpooled', 'se_avg_unpooled', 'Unpooled TS', 'unpooled'),
#     #     ('avg_regret_pooled',   'se_avg_pooled',   'Pooled TS',   'pooled'),
#     #     ('avg_regret_eb',       'se_avg_eb',       'EB',          'eb'),
#     # ]:

#     # ---------- top panel: instantaneous regret sweep ----------
#     for col, se_col, label, key in [
#         ('mean_unpooled', 'se_unpooled', 'Unpooled TS', 'unpooled'),
#         ('mean_pooled',   'se_pooled',   'Pooled TS',   'pooled'),
#         ('mean_eb',       'se_eb',       'EB',          'eb'),
#     ]:

#         ax.plot(
#             df['T'], df[col],
#             label=label, color=colors[key], linewidth=2.2, alpha=0.95
#         )
#         ax.fill_between(
#             df['T'],
#             df[col] - 1.96 * df[se_col],
#             df[col] + 1.96 * df[se_col],
#             color=colors[key], alpha=0.15
#         )

#     ax.set_facecolor("#ffffff")
#     ax.set_xlabel(r'$T$', fontsize=12)
#     ax.set_ylabel('Instantaneous regret at step $T$', fontsize=12)
#     ax.set_title(r'$n=100$, $p=0.5$, 300 runs per $T$', fontsize=13, pad=18)
#     ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

#     winners = df['winner'].values
#     p_vals = df['T'].values
#     ranges = []
#     current = winners[0]
#     start = p_vals[0]

#     for i in range(1, len(winners)):
#         if winners[i] != current:
#             ranges.append((current, start, p_vals[i-1]))
#             current = winners[i]
#             start = p_vals[i]
#     ranges.append((current, start, p_vals[-1]))

#     for i in range(len(ranges) - 1):
#         ax.axvline(
#             x=ranges[i+1][1],
#             color='gray',
#             linestyle=':',
#             linewidth=1,
#             alpha=0.5
#         )

#     if ranges:
#         range_text = "Best performing:\n"
#         for winner, s, e in ranges:
#             if s == e:
#                 range_text += f"{winner}: $T={int(s)}$\n"
#             else:
#                 range_text += f"{winner}: $T\\in[{int(s)},{int(e)}]$\n"

#         ax.text(
#             0.015, 0.96, range_text.rstrip('\n'),
#             transform=ax.transAxes,
#             fontsize=9,
#             va='top',
#             ha='left',
#             bbox=dict(
#                 boxstyle='round',
#                 facecolor='#f5f5f5',
#                 alpha=0.65,
#                 edgecolor='#cccccc'
#             )
#         )

#     # ==========================================
#     # --- Add Secondary Y-Axis for Shrinkage ---
#     # ==========================================
#     ax_twin = ax.twinx()
#     ax_twin.plot(
#         df['T'], df['final_shrinkage'], 
#         label='EB Shrinkage Ratio', 
#         color='black', linestyle='--', linewidth=2, alpha=0.8
#     )
#     ax_twin.set_ylabel('Shrinkage Ratio', fontsize=12, color='black')
#     ax_twin.set_ylim(-0.05, 1.05)
#     ax_twin.tick_params(axis='y', labelcolor='black', labelsize=11)
    
#     # Combined legend
#     h1, l1 = ax.get_legend_handles_labels()
#     h2, l2 = ax_twin.get_legend_handles_labels()
    
#     ax.legend(
#         h1 + h2, l1 + l2,
#         loc='upper right',
#         bbox_to_anchor=(0.999, 1),
#         frameon=True,
#         framealpha=0.65,
#         facecolor='#f5f5f5',
#         edgecolor='#cccccc',
#         fontsize=9,
#         handlelength=1.5,
#         handletextpad=0.5,
#         borderpad=0.4,
#         labelspacing=0.3
#     )
#     # ==========================================

#     # MAKE AXES CONSISTENT
#     ax2.set_xlim(-20, 20)      
#     ax2.set_ylim(0, 1)      

#     # ---------- bottom panel: theoretical reward distributions ----------
#     mu_a = {
#         0: np.array([0.0, 0.00]),
#         1: np.array([0.07, 0.04]),
#     }
#     Sigma_a = {
#         0: [0.50, 0.0, 0.0, 0.50],
#         1: [0.50, 0.0, 0.0, 0.50],
#     }
#     sigma_r = 0.5
#     mu_prior = np.array([0.0, 0.0])
#     lam = 1e-6

#     combos = [
#         (0, 0, r'$a=0,\ c=0$', 'a0c0'),
#         (0, 1, r'$a=0,\ c=1$', 'a0c1'),
#         (1, 0, r'$a=1,\ c=0$', 'a1c0'),
#         (1, 1, r'$a=1,\ c=1$', 'a1c1'),
#     ]

#     means = [reward_mean(mu_a[a], c) for a, c, _, _ in combos]
#     vars_ = [reward_var(Sigma_a[a], sigma_r, c) for a, c, _, _ in combos]
#     sds = [np.sqrt(v) for v in vars_]

#     x_min = min(m - 4*s for m, s in zip(means, sds))
#     x_max = max(m + 4*s for m, s in zip(means, sds))
#     xgrid = np.linspace(x_min, x_max, 600)

#     for (a, c, label, key), mu, var in zip(combos, means, vars_):
#         y = normal_pdf(xgrid, mu, var)
#         ax2.plot(
#             xgrid, y,
#             color=dist_colors[key],
#             linewidth=2.2,
#             label=label
#         )
#         ax2.fill_between(
#             xgrid, 0, y,
#             color=dist_colors[key],
#             alpha=0.22
#         )

#     ax2.set_facecolor("#ffffff")
#     ax2.set_xlabel('Reward', fontsize=12)
#     ax2.set_ylabel('Density', fontsize=12)
#     ax2.set_title('Theoretical reward distributions', fontsize=12, pad=12)
#     ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
#     ax2.legend(
#         loc='upper right',
#         frameon=True,
#         framealpha=0.65,
#         facecolor='#f5f5f5',
#         edgecolor='#cccccc',
#         fontsize=9
#     )

#     # ---------- auto-generated figure annotations ----------
#     mu00, mu01 = mu_a[0]
#     mu10, mu11 = mu_a[1]

#     s0 = Sigma_a[0]
#     s1 = Sigma_a[1]

#     er_00 = reward_mean(mu_a[0], 0)
#     er_10 = reward_mean(mu_a[1], 0)
#     er_01 = reward_mean(mu_a[0], 1)
#     er_11 = reward_mean(mu_a[1], 1)

#     params_line1 = (
#         rf"\textbf{{True environment}}: "
#         rf"$\bm{{\mu}}_0=({mu00:.2f},\,{mu01:.2f})^\top$, "
#         rf"$\bm{{\mu}}_1=({mu10:.2f},\,{mu11:.2f})^\top$, "
#         rf"$\bm{{\Sigma}}_0=\begin{{pmatrix}}{s0[0]:.2f}&{s0[1]:.2f}\\{s0[2]:.2f}&{s0[3]:.2f}\end{{pmatrix}}$, "
#         rf"$\bm{{\Sigma}}_1=\begin{{pmatrix}}{s1[0]:.2f}&{s1[1]:.2f}\\{s1[2]:.2f}&{s1[3]:.2f}\end{{pmatrix}}$, "
#         rf"$\sigma={sigma_r:.2f}$"
#     )

#     params_line2 = (
#         rf"$c=0$: $\mathbb{{E}}[R_{{i,t}}(0)]={er_00:.2f}$, "
#         rf"$\mathbb{{E}}[R_{{i,t}}(1)]={er_10:.2f}$"
#     )

#     params_line3 = (
#         rf"$c=1$: $\mathbb{{E}}[R_{{i,t}}(0)]={er_01:.2f}$, "
#         rf"$\mathbb{{E}}[R_{{i,t}}(1)]={er_11:.2f}$"
#     )

#     params_line4 = (
#         rf"\textbf{{Learning prior}}: "
#         rf"$\bm{{\mu}}_{{\mathrm{{prior}}}}=({mu_prior[0]:.2f},{mu_prior[1]:.2f})^\top$, "
#         rf"$\lambda={lam:.0e}$"
#     )

#     fig.text(0.5, 0.108, params_line1, fontsize=9, va='bottom', ha='center')
#     fig.text(0.5, 0.081, params_line2, fontsize=9, va='bottom', ha='center')
#     fig.text(0.5, 0.057, params_line3, fontsize=9, va='bottom', ha='center')
#     fig.text(0.5, 0.033, params_line4, fontsize=9, va='bottom', ha='center')

#     plt.tight_layout(rect=[0, 0.16, 1.0, 0.97])

#     # os.makedirs('testing_env_betina', exist_ok=True)
#     plt.savefig("testing_code/INSTANT.png", dpi=150, bbox_inches='tight')
#     print("Saved plot to testing_code/INSTANT.png")
    
#     plt.show()

# if __name__ == "__main__":
#     main()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Two-variable T-sweep plotting file.
# This version intentionally removes the theoretical reward-distribution graph.
# It shows only:
#   1. final cumulative regret versus T
#   2. the true-environment / learning-prior text block underneath

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}'


def reward_mean(mu_vec, c, s):
    """E[R_i,t(a) | c_i=c, s_i,t=s] under the population mean mu_a."""
    x = np.array([1.0, float(c), float(s)])
    return float(x @ mu_vec)


def format_vec3(v):
    return rf"({v[0]:.2f},\,{v[1]:.2f},\,{v[2]:.2f})^\top"


def format_mat3(M):
    return (
        rf"\begin{{pmatrix}}"
        rf"{M[0,0]:.2f}&{M[0,1]:.2f}&{M[0,2]:.2f}\\"
        rf"{M[1,0]:.2f}&{M[1,1]:.2f}&{M[1,2]:.2f}\\"
        rf"{M[2,0]:.2f}&{M[2,1]:.2f}&{M[2,2]:.2f}"
        rf"\end{{pmatrix}}"
    )


def best_ranges(df):
    winners = df['winner'].values
    t_vals = df['T'].values

    if len(winners) == 0:
        return []

    ranges = []
    current = winners[0]
    start = t_vals[0]

    for i in range(1, len(winners)):
        if winners[i] != current:
            ranges.append((current, start, t_vals[i - 1]))
            current = winners[i]
            start = t_vals[i]
    ranges.append((current, start, t_vals[-1]))
    return ranges


def main():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11

    # ---------- file paths ----------
    csv_path = "testing_code/INSTANT.csv"
    output_path = "testing_code/INSTANT_a.png"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. Run t_sweep_two_variable.cpp first, "
            f"or change csv_path at the top of main()."
        )

    df = pd.read_csv(csv_path)

    required_cols = [
        'T',
        'mean_unpooled', 'se_unpooled',
        'mean_pooled', 'se_pooled',
        'mean_eb', 'se_eb',
        'winner',
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    colors = {
        'unpooled': '#1f77b4',
        'pooled': '#d62728',
        'eb': '#ff7f0e',
    }

    # ---------- single source of truth for environment parameters ----------
    # Keep these synchronized with t_sweep_two_variable.cpp.
    n = 200
    runs_per_T = 20
    p_context = 0.5
    p_state = 0.5

    mu_a = {
        0: np.array([0.00, 0.00, 0.00]),
        1: np.array([0.40, 0.20, 0.20]),
    }
    Sigma_a = {
        0: np.array([
            [0.50, 0.00, 0.00],
            [0.00, 0.50, 0.00],
            [0.00, 0.00, 0.50],
        ]),
        1: np.array([
            [0.50, 0.00, 0.00],
            [0.00, 0.50, 0.00],
            [0.00, 0.00, 0.50],
        ]),
    }
    sigma_r = 0.5
    lam = 1e-6

    # Individual learner uses x^L=(1,s), so its prior is 2-dimensional.
    mu_prior_L = np.array([0.0, 0.0])

    # Pooled and EB population-level parameter is theta=(intercept,c,s), so this prior is 3-dimensional.
    mu_prior_P = np.array([0.0, 0.0, 0.0])

    # ---------- figure layout ----------
    # Only two panels now: graph + text block.
    fig, (ax, ax_note) = plt.subplots(
        2, 1,
        figsize=(8.8, 6.8),
        gridspec_kw={'height_ratios': [3.6, 1.25]}
    )
    fig.patch.set_facecolor('#ffffff')

    # ---------- top panel: regret sweep ----------
    for col, se_col, label, key in [
        ('mean_unpooled', 'se_unpooled', 'Unpooled TS', 'unpooled'),
        ('mean_pooled', 'se_pooled', 'Pooled TS', 'pooled'),
        ('mean_eb', 'se_eb', 'EB', 'eb'),
    ]:
        ax.plot(
            df['T'], df[col],
            label=label,
            color=colors[key],
            linewidth=2.3,
            alpha=0.95,
        )
        ax.fill_between(
            df['T'],
            df[col] - 1.96 * df[se_col],
            df[col] + 1.96 * df[se_col],
            color=colors[key],
            alpha=0.15,
        )

    ax.set_facecolor('#ffffff')
    ax.set_xlabel(r'$T$', fontsize=13)
    ax.set_ylabel('Instant regret', fontsize=13)
    ax.set_title(
        rf'$n={n}$, $p_c={p_context}$, $p_s={p_state}$, {runs_per_T} runs per $T$',
        fontsize=14,
        pad=16,
    )
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    ranges = best_ranges(df)
    for i in range(len(ranges) - 1):
        ax.axvline(
            x=ranges[i + 1][1],
            color='gray',
            linestyle=':',
            linewidth=1,
            alpha=0.5,
        )

    if ranges:
        range_text = 'Best performing:\n'
        for winner, start, end in ranges:
            if start == end:
                range_text += f'{winner}: $T={int(start)}$\n'
            else:
                range_text += f'{winner}: $T\\in[{int(start)},{int(end)}]$\n'

        ax.text(
            0.015,
            0.96,
            range_text.rstrip('\n'),
            transform=ax.transAxes,
            fontsize=9.5,
            va='top',
            ha='left',
            bbox=dict(
                boxstyle='round',
                facecolor='#f5f5f5',
                alpha=0.72,
                edgecolor='#cccccc',
            ),
        )

    # Keep the EB shrinkage-ratio axis when final_shrinkage exists in the CSV.
    ax_twin = None
    if 'final_shrinkage' in df.columns:
        ax_twin = ax.twinx()
        ax_twin.plot(
            df['T'],
            df['final_shrinkage'],
            label='EB Shrinkage Ratio',
            color='black',
            linestyle='--',
            linewidth=2.0,
            alpha=0.8,
        )
        ax_twin.set_ylabel('Shrinkage Ratio', fontsize=13, color='black')
        ax_twin.set_ylim(-0.05, 1.05)
        ax_twin.tick_params(axis='y', labelcolor='black', labelsize=11)

    h1, l1 = ax.get_legend_handles_labels()
    if ax_twin is not None:
        h2, l2 = ax_twin.get_legend_handles_labels()
        handles, labels = h1 + h2, l1 + l2
    else:
        handles, labels = h1, l1

    ax.legend(
        handles,
        labels,
        loc='upper right',
        bbox_to_anchor=(0.999, 1),
        frameon=True,
        framealpha=0.72,
        facecolor='#f5f5f5',
        edgecolor='#cccccc',
        fontsize=9.5,
        handlelength=1.6,
        handletextpad=0.5,
        borderpad=0.45,
        labelspacing=0.32,
    )

    # ---------- annotation panel ----------
    treatment_effects = {
        (c, s): reward_mean(mu_a[1], c, s) - reward_mean(mu_a[0], c, s)
        for c in [0, 1]
        for s in [0, 1]
    }

    params_line1 = (
        rf"\textbf{{True environment}}: "
        rf"$\bm{{\mu}}_0={format_vec3(mu_a[0])}$, "
        rf"$\bm{{\mu}}_1={format_vec3(mu_a[1])}$, "
        rf"$\sigma={sigma_r:.2f}$, "
        rf"$p_c={p_context}$, $p_s={p_state}$"
    )

    params_sigma0 = rf"$\bm{{\Sigma}}_0={format_mat3(Sigma_a[0])}$"
    params_sigma1 = rf"$\bm{{\Sigma}}_1={format_mat3(Sigma_a[1])}$"

    params_line4 = (
        rf"$\Delta(c,s)=\mathbb{{E}}[R_{{i,t}}(1)-R_{{i,t}}(0)\mid c,s]$: "
        rf"$(0,0)={treatment_effects[(0,0)]:.2f}$, " 
        rf"$(1,0)={treatment_effects[(1,0)]:.2f}$, "
        rf"$(0,1)={treatment_effects[(0,1)]:.2f}$, "
        rf"$(1,1)={treatment_effects[(1,1)]:.2f}$"
    )

    params_line5 = (
        rf"\textbf{{Learning prior}}: "
        rf"$\bm{{\mu}}_{{\mathrm{{prior}}}}^L=({mu_prior_L[0]:.2f},{mu_prior_L[1]:.2f})^\top$, "
        rf"$\bm{{\mu}}_{{\mathrm{{prior}}}}^P={format_vec3(mu_prior_P)}$, "
        rf"$\lambda={lam:.0e}$"
    )

    ax_note.axis('off')
    ax_note.set_facecolor('#ffffff')

    # Larger text and more vertical room than the three-panel version.
    ax_note.text(
        0.5, 0.87, params_line1,
        fontsize=9.8, va='center', ha='center', transform=ax_note.transAxes
    )
    ax_note.text(
        0.28, 0.52, params_sigma0,
        fontsize=9.6, va='center', ha='center', transform=ax_note.transAxes
    )
    ax_note.text(
        0.72, 0.52, params_sigma1,
        fontsize=9.6, va='center', ha='center', transform=ax_note.transAxes
    )
    ax_note.text(
        0.5, 0.23, params_line4,
        fontsize=9.7, va='center', ha='center', transform=ax_note.transAxes
    )
    ax_note.text(
        0.5, 0.05, params_line5,
        fontsize=9.7, va='center', ha='center', transform=ax_note.transAxes
    )

    plt.tight_layout(rect=[0, 0, 1.0, 0.98])
    fig.subplots_adjust(hspace=0.30)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=170, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    print("\n=== Summary ===")
    print(f"Total T values tested: {len(df)}")
    print(f"\nWinner distribution:\n{df['winner'].value_counts().to_string()}")

    if ranges:
        print("\nRanges where each algorithm performs best:")
        for winner, start, end in ranges:
            if start == end:
                print(f"  {winner}: T = {int(start)}")
            else:
                print(f"  {winner}: T in [{int(start)}, {int(end)}]")

    plt.show()


if __name__ == "__main__":
    main()
