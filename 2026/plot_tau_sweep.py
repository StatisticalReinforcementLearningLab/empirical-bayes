import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Enable LaTeX rendering for consistent math fonts
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # Mature, muted color palette for academic publications (same as before)
    colors = {
        'unpooled': '#1f77b4',  # steel blue
        'pooled': '#d62728',    # brick red
        'eb': '#ff7f0e'         # burnt orange
    }
    
    # Read the tau sweep results
    df = pd.read_csv("tau_sweep_results.csv")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot each algorithm with 95% CI bands
    ax.plot(df.tau, df.mean_unpooled, label='Unpooled TS', 
            color=colors['unpooled'], linewidth=2, alpha=0.95)
    ax.fill_between(df.tau,
                    df.mean_unpooled - 1.96 * df.se_unpooled,
                    df.mean_unpooled + 1.96 * df.se_unpooled,
                    color=colors['unpooled'], alpha=0.15)
    
    ax.plot(df.tau, df.mean_pooled, label='Pooled TS', 
            color=colors['pooled'], linewidth=2, alpha=0.95)
    ax.fill_between(df.tau,
                    df.mean_pooled - 1.96 * df.se_pooled,
                    df.mean_pooled + 1.96 * df.se_pooled,
                    color=colors['pooled'], alpha=0.15)
    
    ax.plot(df.tau, df.mean_eb, label='EB', 
            color=colors['eb'], linewidth=2, alpha=0.95)
    ax.fill_between(df.tau,
                    df.mean_eb - 1.96 * df.se_eb,
                    df.mean_eb + 1.96 * df.se_eb,
                    color=colors['eb'], alpha=0.15)
    
    # Labels and title
    ax.set_xlabel(r'Between-person heterogeneity ($\tau$)', fontsize=12)
    ax.set_ylabel('Final cumulative regret', fontsize=12)
    ax.set_title(r'Algorithm Performance vs. Heterogeneity ($\mu_2=1$, $\sigma=1$, $T=100$, $n=50$)', 
                 fontsize=13, pad=15)
    
    # Add grid for readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False,
              framealpha=1.0, facecolor='#f5f5f5', edgecolor='#cccccc')
    
    # Add metadata at top
    fig.text(0.5, 0.96, r"Priors: $\mu_0 = [0, 0]$, $\tau_0 = [1, 1]$,  Runs: 500 per $\tau$ value", 
             fontsize=10, va='top', ha='center')
    
    # Identify transition points (simplified - find where winner changes)
    winners = df['winner'].values
    transitions = []
    for i in range(1, len(winners)):
        if winners[i] != winners[i-1]:
            transitions.append(df.tau.values[i])
    
    # Add vertical lines at major transitions
    for trans_tau in transitions:
        ax.axvline(x=trans_tau, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    os.makedirs('plots', exist_ok=True)
    plt.savefig("plots/tau_sweep.png", dpi=150, bbox_inches='tight')
    print("Saved plot to plots/tau_sweep.png")
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total tau values tested: {len(df)}")
    print(f"\nWinner distribution:")
    print(df['winner'].value_counts())
    
    if transitions:
        print(f"\nTransition points (tau values where winner changes):")
        for i, tau_val in enumerate(transitions, 1):
            idx = df[df.tau >= tau_val].index[0]
            print(f"  {i}. tau ≈ {tau_val:.2f}: {df.loc[idx-1, 'winner']} → {df.loc[idx, 'winner']}")
    
    plt.show()

if __name__ == "__main__":
    main()
