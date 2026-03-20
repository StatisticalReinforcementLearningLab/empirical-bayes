import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Enable LaTeX rendering for consistent math fonts
    # plt.rcParams['text.usetex'] = True
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
    
    # Read the sigma sweep results
    df = pd.read_csv("no_heterogeneity_xaxis.csv")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")  # Set entire figure background
    
    # Plot each algorithm with 95% CI bands
    ax.plot(df.sigma, df.mean_unpooled, label='Unpooled TS', 
            color=colors['unpooled'], linewidth=2, alpha=0.95)
    ax.fill_between(df.sigma,
                    df.mean_unpooled - 1.96 * df.se_unpooled,
                    df.mean_unpooled + 1.96 * df.se_unpooled,
                    color=colors['unpooled'], alpha=0.15)
    
    ax.plot(df.sigma, df.mean_pooled, label='Pooled TS', 
            color=colors['pooled'], linewidth=2, alpha=0.95)
    ax.fill_between(df.sigma,
                    df.mean_pooled - 1.96 * df.se_pooled,
                    df.mean_pooled + 1.96 * df.se_pooled,
                    color=colors['pooled'], alpha=0.15)
    
    ax.plot(df.sigma, df.mean_eb, label='EB', 
            color=colors['eb'], linewidth=2, alpha=0.95)
    ax.fill_between(df.sigma,
                    df.mean_eb - 1.96 * df.se_eb,
                    df.mean_eb + 1.96 * df.se_eb,
                    color=colors['eb'], alpha=0.15)
    
    # Set background color
    ax.set_facecolor("#ffffff")
    
    # Labels and title
    ax.set_xlabel(r'$\sigma$', fontsize=12)
    ax.set_ylabel('Final cumulative regret', fontsize=12)
    ax.set_title(r'Algorithm Performance vs. Noise ($\mu_2=1$, $\tau=0$, $T=100$, $n=50$)', 
                 fontsize=13, pad=15)
    
    # Add grid for readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=False,
              framealpha=1.0, facecolor="#ffffff", edgecolor='#cccccc')
    
    # Find ranges where each algorithm performs best
    winners = df['winner'].values
    sigma_vals = df['sigma'].values
    
    # Group consecutive winners into ranges
    ranges = []
    current_winner = winners[0]
    start_sigma = sigma_vals[0]
    
    for i in range(1, len(winners)):
        if winners[i] != current_winner:
            # End of current range
            ranges.append((current_winner, start_sigma, sigma_vals[i-1]))
            current_winner = winners[i]
            start_sigma = sigma_vals[i]
    # Add last range
    ranges.append((current_winner, start_sigma, sigma_vals[-1]))
    
    # Add vertical lines at transitions
    for i in range(len(ranges) - 1):
        trans_sigma = ranges[i+1][1]  # Start of next range
        ax.axvline(x=trans_sigma, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Display ranges where each algorithm is best
    if ranges:
        range_text = "Best performing:\n"
        for winner, start, end in ranges:
            if start == end:
                range_text += f"{winner}: $\\sigma={start:.2f}$\n"
            else:
                range_text += f"{winner}: $\\sigma\\in[{start:.2f},{end:.2f}]$\n"
        range_text = range_text.rstrip('\n')  # Remove last newline
        ax.text(0.02, 0.98, range_text, transform=ax.transAxes, 
                fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=1.0, edgecolor='#cccccc'))
    
    # Add priors and runs at bottom
    fig.text(0.5, 0.065, r"Priors: $\mu_0 = [0, 0]$, $\tau_0 = [1, 1]$", 
             fontsize=10, va='bottom', ha='center')
    fig.text(0.5, 0.04, r"Runs: 500 per $\sigma$ value", 
             fontsize=10, va='bottom', ha='center')
    
    # Add parameter key at bottom center
    key_text = (r"$\mu_2$ = true difference, "
                r"$\sigma^2$ = within-person variance, " 
                r"$\tau^2$ = between-person variance")
    fig.text(0.5, 0.015, key_text, fontsize=10, va='bottom', ha='center')
    
    plt.tight_layout(rect=[0, 0.10, 1, 1.0])
    
    # Save figure
    os.makedirs('plots', exist_ok=True)
    plt.savefig("plots/no_heterogeneity_xaxis.png", dpi=150, bbox_inches='tight')
    print("Saved plot to plots/no_heterogeneity_xaxis.png")
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total sigma values tested: {len(df)}")
    print(f"\nWinner distribution:")
    print(df['winner'].value_counts())
    
    if ranges:
        print(f"\nRanges where each algorithm performs best:")
        for winner, start, end in ranges:
            if start == end:
                print(f"  {winner}: σ = {start:.2f}")
            else:
                print(f"  {winner}: σ ∈ [{start:.2f}, {end:.2f}]")
    
    plt.show()

if __name__ == "__main__":
    main()
