import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This plotting file is the two-variable version of the previous T-sweep plots.
# Environment: R_i,t(a) = theta_{i,a,0} + theta_{i,a,1} c_i + theta_{i,a,2} s_i,t + epsilon.

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}'


def normal_pdf(x, mu, var):
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mu) ** 2 / (2 * var))


# CHANGE 1 FROM OLD PLOTTING FILES:
# reward_mean now takes both fixed context c and time-varying state s.
def reward_mean(mu_vec, c, s):
    x = np.array([1.0, float(c), float(s)])
    return float(x @ mu_vec)


# CHANGE 2 FROM OLD PLOTTING FILES:
# reward_var now uses a 3x3 Sigma and x=(1,c,s), instead of a 2x2 Sigma and x=(1,c).
def reward_var(Sigma, sigma_r, c, s):
    x = np.array([1.0, float(c), float(s)])
    return float(x @ Sigma @ x + sigma_r ** 2)


def format_vec3(v):
    return rf"({v[0]:.2f},\,{v[1]:.2f},\,{v[2]:.2f})^\top"


def format_mat3(M):
    # Use smallmatrix instead of pmatrix for the bottom caption.
    # A full 3x3 pmatrix is too tall for fig.text lines and causes overlap.
    return (
        rf"\left(\begin{{smallmatrix}}"
        rf"{M[0,0]:.2f}&{M[0,1]:.2f}&{M[0,2]:.2f}\\"
        rf"{M[1,0]:.2f}&{M[1,1]:.2f}&{M[1,2]:.2f}\\"
        rf"{M[2,0]:.2f}&{M[2,1]:.2f}&{M[2,2]:.2f}"
        rf"\end{{smallmatrix}}\right)"
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
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11

    # ---------- file paths ----------
    # CHANGE 3 FROM OLD PLOTTING FILES:
    # This reads the CSV produced by t_sweep_two_variable.cpp.
    csv_path = "conjectures/two_variable_conj1.csv"
    output_path = "conjectures/two_variable_graphs/two_variable_conj1.png"

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

    # CHANGE 4 FROM OLD PLOTTING FILES:
    # There are now four context-state cells: (c,s) in {(0,0),(1,0),(0,1),(1,1)}.
    # Color identifies the (c,s) cell; line style identifies the action.
    cell_colors = {
        (0, 0): '#1f77b4',
        (1, 0): '#2ca02c',
        (0, 1): '#d62728',
        (1, 1): '#9467bd',
    }
    action_linestyle = {
        0: '--',
        1: '-',
    }

    # ---------- single source of truth for environment parameters ----------
    # CHANGE 5 FROM OLD PLOTTING FILES:
    # mu_a is now length 3 and Sigma_a is now 3x3.
    # Keep these synchronized with t_sweep_two_variable.cpp.
    n = 200
    runs_per_T = 300
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
    sigma_r = 1.0
    lam = 1e-6

    # Individual learner uses x^L=(1,s), so its prior is 2-dimensional.
    mu_prior_L = np.array([0.0, 0.0])

    # Pooled and EB population-level parameter is theta=(intercept,c,s), so this prior is 3-dimensional.
    mu_prior_P = np.array([0.0, 0.0, 0.0])

    # CHANGE 8: Add a dedicated annotation panel under the density plot.
    # This keeps the 3x3 covariance matrices out of fig.text() so they do not overlap.
    fig, (ax, ax2, ax_note) = plt.subplots(
        3, 1,
        figsize=(8.5, 10.6),
        gridspec_kw={'height_ratios': [3.1, 2.35, 1.00]}
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
            linewidth=2.2,
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
    ax.set_xlabel(r'$T$', fontsize=12)
    ax.set_ylabel('Final cumulative regret', fontsize=12)
    ax.set_title(
        rf'$n={n}$, $p_c={p_context}$, $p_s={p_state}$, {runs_per_T} runs per $T$',
        fontsize=13,
        pad=18,
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
            fontsize=9,
            va='top',
            ha='left',
            bbox=dict(
                boxstyle='round',
                facecolor='#f5f5f5',
                alpha=0.65,
                edgecolor='#cccccc',
            ),
        )

    # CHANGE 6 FROM OLD PLOTTING FILES:
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
            linewidth=2,
            alpha=0.8,
        )
        ax_twin.set_ylabel('Shrinkage Ratio', fontsize=12, color='black')
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
        framealpha=0.65,
        facecolor='#f5f5f5',
        edgecolor='#cccccc',
        fontsize=9,
        handlelength=1.5,
        handletextpad=0.5,
        borderpad=0.4,
        labelspacing=0.3,
    )

    # ---------- bottom panel: theoretical reward distributions ----------
    # CHANGE 7 FROM OLD PLOTTING FILES:
    # Plot all 8 distributions: two actions x two context values x two state values.
    combos = []
    for a in [0, 1]:
        for c in [0, 1]:
            for s in [0, 1]:
                label = rf'$a={a},\ c={c},\ s={s}$'
                combos.append((a, c, s, label))

    means = [reward_mean(mu_a[a], c, s) for a, c, s, _ in combos]
    vars_ = [reward_var(Sigma_a[a], sigma_r, c, s) for a, c, s, _ in combos]
    sds = [np.sqrt(v) for v in vars_]

    x_min = min(m - 4 * sd for m, sd in zip(means, sds))
    x_max = max(m + 4 * sd for m, sd in zip(means, sds))
    pad = 0.05 * (x_max - x_min)
    xgrid = np.linspace(x_min - pad, x_max + pad, 800)

    for (a, c, s, label), mu, var in zip(combos, means, vars_):
        y = normal_pdf(xgrid, mu, var)
        ax2.plot(
            xgrid,
            y,
            color=cell_colors[(c, s)],
            linestyle=action_linestyle[a],
            linewidth=2.1,
            label=label,
            alpha=0.95,
        )
        ax2.fill_between(
            xgrid,
            0,
            y,
            color=cell_colors[(c, s)],
            alpha=0.045,
        )

    ax2.set_facecolor('#ffffff')
    ax2.set_xlabel('Reward', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Theoretical reward distributions', fontsize=12, pad=12)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_xlim(x_min - pad, x_max + pad)

    ax2.legend(
        loc='upper right',
        ncol=2,
        frameon=True,
        framealpha=0.65,
        facecolor='#f5f5f5',
        edgecolor='#cccccc',
        fontsize=8,
        columnspacing=1.0,
        handlelength=2.0,
    )

    # ---------- auto-generated figure annotations ----------
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

    # CHANGE 9: Put the matrices side-by-side in their own panel.
    # Keeping each matrix centered vertically gives LaTeX enough room to render all rows.
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

    ax_note.text(
        0.5, 0.90, params_line1,
        fontsize=8.8, va='center', ha='center', transform=ax_note.transAxes
    )
    ax_note.text(
        0.30, 0.54, params_sigma0,
        fontsize=8.2, va='center', ha='center', transform=ax_note.transAxes
    )
    ax_note.text(
        0.70, 0.54, params_sigma1,
        fontsize=8.2, va='center', ha='center', transform=ax_note.transAxes
    )
    ax_note.text(
        0.5, 0.20, params_line4,
        fontsize=8.7, va='center', ha='center', transform=ax_note.transAxes
    )
    ax_note.text(
        0.5, 0.04, params_line5,
        fontsize=8.7, va='center', ha='center', transform=ax_note.transAxes
    )

    # CHANGE 10: Since the annotations are now a real subplot, do not reserve
    # bottom space manually. hspace controls the gap between panels.
    plt.tight_layout(rect=[0, 0, 1.0, 0.98])
    fig.subplots_adjust(hspace=0.42)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
