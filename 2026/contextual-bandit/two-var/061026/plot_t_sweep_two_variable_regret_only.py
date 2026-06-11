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
    csv_path = "conjectures/two_variable_conj5_a.csv"
    output_path = "conjectures/two_variable_conj5_a.png"

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
    runs_per_T = 300
    p_context = 0.5
    p_state = 0.5

    mu_a = {
        0: np.array([0.00, 0.00, 0.00]),
        1: np.array([0.40, 0.05, 0.30]),
    }
    Sigma_a = {
        0: np.array([
            [0.50, 0.00, 0.00],
            [0.00, 0.25, 0.00],
            [0.00, 0.00, 0.75],
        ]),
        1: np.array([
            [0.50, 0.00, 0.00],
            [0.00, 0.25, 0.00],
            [0.00, 0.00, 0.75],
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
    ax.set_ylabel('Final cumulative regret', fontsize=13)
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
