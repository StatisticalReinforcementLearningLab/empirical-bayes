import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    colors = {
        'thompson': '#1f77b4',
        'pooled': '#d62728',
        'eb': '#ff7f0e'
    }

    df = pd.read_csv('mean_regret_over_time.csv')

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('white')

    ax.plot(
        df.t,
        df.mean_regret_thompson,
        label='Unpooled TS (context-blind)',
        color=colors['thompson'],
        linewidth=2,
        alpha=0.95,
    )
    ax.plot(
        df.t,
        df.mean_regret_pooled,
        label='Pooled TS (context-blind)',
        color=colors['pooled'],
        linewidth=2,
        alpha=0.95,
    )
    ax.plot(
        df.t,
        df.mean_regret_eb,
        label='EB (context-blind)',
        color=colors['eb'],
        linewidth=2,
        alpha=0.95,
    )

    ax.set_facecolor('white')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Mean cumulative regret')
    ax.set_title(r'Mean Cumulative Regret Over Time ($n=100$, $T=200$, runs=500)')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(
        loc='best',
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=1.0,
        facecolor='#f5f5f5',
        edgecolor='#cccccc',
    )

    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    out_path = 'plots/mean_regret_over_time.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved plot to {out_path}')


if __name__ == '__main__':
    main()
