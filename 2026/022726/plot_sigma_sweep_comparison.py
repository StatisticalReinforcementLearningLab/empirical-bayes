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
    
    # Mature, muted color palette for academic publications
    colors = {
        'unpooled': '#1f77b4',  # steel blue
        'pooled': '#d62728',    # brick red
        'eb': '#ff7f0e'         # burnt orange
    }
    
    # Read dataset
    df_dif = pd.read_csv("sigma_sweep_results.csv")
    
    # Calculate sigma/tau ratio (tau=1 in both simulations)
    tau = 1.0
    df_dif['sigma_tau_ratio'] = df_dif['sigma'] / tau
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    # Plot each algorithm with 95% CI bands
    ax.plot(df_dif.sigma_tau_ratio, df_dif.mean_unpooled, label='Unpooled TS', 
            color=colors['unpooled'], linewidth=2, alpha=0.95)
    ax.fill_between(df_dif.sigma_tau_ratio,
                    df_dif.mean_unpooled - 1.96 * df_dif.se_unpooled,
                    df_dif.mean_unpooled + 1.96 * df_dif.se_unpooled,
                    color=colors['unpooled'], alpha=0.15)
    
    ax.plot(df_dif.sigma_tau_ratio, df_dif.mean_pooled, label='Pooled TS', 
            color=colors['pooled'], linewidth=2, alpha=0.95)
    ax.fill_between(df_dif.sigma_tau_ratio,
                    df_dif.mean_pooled - 1.96 * df_dif.se_pooled,
                    df_dif.mean_pooled + 1.96 * df_dif.se_pooled,
                    color=colors['pooled'], alpha=0.15)
    
    ax.plot(df_dif.sigma_tau_ratio, df_dif.mean_eb, label='EB',
            color=colors['eb'], linewidth=2, alpha=0.95)
    ax.fill_between(df_dif.sigma_tau_ratio,
                    df_dif.mean_eb - 1.96 * df_dif.se_eb,
                    df_dif.mean_eb + 1.96 * df_dif.se_eb,
                    color=colors['eb'], alpha=0.15)
    
    ax.set_facecolor('white')
    ax.set_xlabel(r'Noise-to-heterogeneity ratio ($\sigma/\tau$)', fontsize=12)
    ax.set_ylabel('Final cumulative regret', fontsize=12)
    ax.set_title(r'Algorithm Performance vs. Noise ($\mu_2=1$, $\tau=1$, $T=100$, $n=50$)', 
                 fontsize=13, pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False,
              framealpha=1.0, facecolor='white', edgecolor='#cccccc')
    
    # Add priors and runs at bottom
    fig.text(0.5, 0.065, r"Priors: $\mu_0 = [0, 0]$, $\tau_0 = [1, 1]$", 
             fontsize=10, va='bottom', ha='center')
    fig.text(0.5, 0.04, r"Runs: 500 per $\sigma$ value", 
             fontsize=10, va='bottom', ha='center')
    
    # Add parameter key at bottom center
    key_text = (r"$\mu_2$ = true difference, "
                r"$\sigma$ = within-person noise, " 
                r"$\tau$ = between-person heterogeneity")
    fig.text(0.5, 0.015, key_text, fontsize=10, va='bottom', ha='center')
    
    plt.tight_layout(rect=[0, 0.10, 1, 1.0])
    plt.savefig('sigma_sweep_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved plot to sigma_sweep_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
