import os

import matplotlib.pyplot as plt
import pandas as pd


def format_ranges(xs):
    if not xs:
        return "none"

    vals = sorted(xs)
    spans = []
    start = vals[0]
    prev = vals[0]
    for x in vals[1:]:
        if abs(x - prev) <= 0.251:
            prev = x
            continue
        spans.append((start, prev))
        start = x
        prev = x
    spans.append((start, prev))

    parts = []
    for lo, hi in spans:
        if abs(lo - hi) < 1e-9:
            parts.append(f"[{lo:.2f}]")
        else:
            parts.append(f"[{lo:.2f},{hi:.2f}]")
    return ", ".join(parts)


def add_panel(ax, df, colors, panel_title):
    df = df.sort_values("distance_ratio").copy()

    def winner_row(row):
        vals = {
            "Unpooled TS": row["mean_final_regret_thompson"],
            "Pooled TS": row["mean_final_regret_pooled"],
            "EB": row["mean_final_regret_eb"],
        }
        return min(vals, key=vals.get)

    df["winner_calc"] = df.apply(winner_row, axis=1)
    x_unpooled = df.loc[df["winner_calc"] == "Unpooled TS", "distance_ratio"].tolist()
    x_eb = df.loc[df["winner_calc"] == "EB", "distance_ratio"].tolist()
    x_pooled = df.loc[df["winner_calc"] == "Pooled TS", "distance_ratio"].tolist()

    ax.plot(
        df.distance_ratio,
        df.mean_final_regret_thompson,
        label="Unpooled TS",
        color=colors["thompson"],
        linewidth=2.1,
        alpha=0.95,
    )
    ax.fill_between(
        df.distance_ratio,
        df.mean_final_regret_thompson - df.se_final_regret_thompson,
        df.mean_final_regret_thompson + df.se_final_regret_thompson,
        color=colors["thompson"],
        alpha=0.14,
        linewidth=0,
    )

    ax.plot(
        df.distance_ratio,
        df.mean_final_regret_pooled,
        label="Pooled TS",
        color=colors["pooled"],
        linewidth=2.1,
        alpha=0.95,
    )
    ax.fill_between(
        df.distance_ratio,
        df.mean_final_regret_pooled - df.se_final_regret_pooled,
        df.mean_final_regret_pooled + df.se_final_regret_pooled,
        color=colors["pooled"],
        alpha=0.14,
        linewidth=0,
    )

    ax.plot(
        df.distance_ratio,
        df.mean_final_regret_eb,
        label="EB",
        color=colors["eb"],
        linewidth=2.1,
        alpha=0.95,
    )
    ax.fill_between(
        df.distance_ratio,
        df.mean_final_regret_eb - df.se_final_regret_eb,
        df.mean_final_regret_eb + df.se_final_regret_eb,
        color=colors["eb"],
        alpha=0.14,
        linewidth=0,
    )

    ax.set_xlabel("Ratio of arm differences between contexts")
    ax.set_ylabel("Final cumulative regret")
    ax.set_title(panel_title, pad=10)
    ax.set_xlim(-0.05, 3.05)
    ax.grid(True, alpha=0.9)
    ax.axvline(0.0, color="#b5b5b5", linestyle=":", linewidth=1.2)

    summary = (
        "Best:\n"
        f"Unpooled: $r \\in {format_ranges(x_unpooled)}$\n"
        f"EB: $r \\in {format_ranges(x_eb)}$\n"
        f"Pooled: $r \\in {format_ranges(x_pooled)}$"
    )
    ax.text(
        0.015,
        0.985,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={"facecolor": "#f5f5f5", "edgecolor": "#b8b8b8", "boxstyle": "round,pad=0.30"},
    )


def main():
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["axes.titlesize"] = 17
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13

    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#555555"
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["grid.color"] = "#d0d0d0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.8

    colors = {
        "thompson": "#1f77b4",
        "pooled": "#d62728",
        "eb": "#ff7f0e",
    }

    df_p02 = pd.read_csv("final_regret_vs_distance_ratio_context_aware_p02.csv")
    df_p05 = pd.read_csv("final_regret_vs_distance_ratio_context_aware_p05.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 7.6), sharey=True)
    fig.patch.set_facecolor("white")

    add_panel(axes[0], df_p02, colors, r"$p=0.2$")
    add_panel(axes[1], df_p05, colors, r"$p=0.5$")

    axes[0].legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        shadow=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#bbbbbb",
    )

    fig.suptitle(r"Context-Aware Algorithm Performance vs. Distance Ratio", y=0.9, fontsize=18)

    fig.text(
        0.5,
        0.045,
        "Priors: $\\mu_0=[0,0], \\tau_0=[1,1]$ | Runs: 500 per ratio value | $T=200, n=100$",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#333333",
    )

    plt.tight_layout(rect=[0, 0.09, 1, 0.93])

    os.makedirs("plots", exist_ok=True)
    out_path = "plots/final_regret_vs_distance_ratio_context_aware.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
