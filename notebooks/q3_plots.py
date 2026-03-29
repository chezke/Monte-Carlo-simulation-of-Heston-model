"""
Q3 benchmark visualisation helpers.
Import from the Colab notebook:
    from q3_plots import plot_execution_time, plot_error_analysis, plot_dt_impact
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_execution_time(df, save_path="fig_q3_time.png"):
    """Box-plot + bar-chart of kernel execution times across the parameter grid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    time_cols = {
        "Exact (BK)\n\u0394t=1/1000": "ms_exact",
        "Euler\n\u0394t=1/1000": "ms_euler",
        "Almost-exact\n\u0394t=1/1000": "ms_almost_dt1000",
        "Almost-exact\n\u0394t=1/30": "ms_almost_dt1_30",
    }
    time_data = [df[c].values for c in time_cols.values()]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    # --- Box plot ---
    bp = axes[0].boxplot(time_data, labels=time_cols.keys(), patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Kernel time (ms)")
    axes[0].set_title("Execution time distribution across parameter grid")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.3)

    # --- Bar chart ---
    means = [df[c].mean() for c in time_cols.values()]
    bars = axes[1].bar(time_cols.keys(), means, color=colors, alpha=0.7, edgecolor="black")
    for bar, m in zip(bars, means):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, m + 2,
            f"{m:.1f}", ha="center", va="bottom", fontsize=10,
        )
    axes[1].set_ylabel("Mean kernel time (ms)")
    axes[1].set_title("Average execution time")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    # --- Summary ---
    print("Mean execution time (ms):")
    for name, col in time_cols.items():
        print(f"  {name:30s}  {df[col].mean():8.2f} \u00b1 {df[col].std():6.2f}")
    print(f"\nSpeedup  Euler / Exact (BK):            "
          f"{df['ms_exact'].mean() / df['ms_euler'].mean():.1f}\u00d7")
    print(f"Speedup  Almost \u0394t=1/30 / Almost \u0394t=1/1000: "
          f"{df['ms_almost_dt1000'].mean() / df['ms_almost_dt1_30'].mean():.1f}\u00d7")


def plot_error_analysis(df, save_path="fig_q3_bias.png"):
    """Histogram, scatter vs Feller gap, and box-plot of absolute errors."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    err_cols = {
        "Euler (\u0394t=1/1000)": "err_euler",
        "Almost-exact (\u0394t=1/1000)": "err_almost_1000",
        "Almost-exact (\u0394t=1/30)": "err_almost_30",
    }
    colors_err = ["#dd8452", "#55a868", "#c44e52"]

    # --- (a) Histogram ---
    for i, (label, col) in enumerate(err_cols.items()):
        axes[0].hist(
            df[col].abs().values, bins=30, alpha=0.55,
            label=label, color=colors_err[i], edgecolor="black", linewidth=0.4,
        )
    axes[0].set_xlabel("|bias + MC noise|")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of |err| across parameter grid")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # --- (b) |err| vs feller_gap ---
    for i, (label, col) in enumerate(err_cols.items()):
        axes[1].scatter(
            df["feller_gap"], df[col].abs(), s=8, alpha=0.5,
            label=label, color=colors_err[i],
        )
    axes[1].set_xlabel("Feller gap  (20\u03ba\u03b8 \u2212 \u03c3\u00b2)")
    axes[1].set_ylabel("|err|")
    axes[1].set_title("|Bias| vs Feller margin")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # --- (c) Box plot ---
    abs_data = [df[c].abs().values for c in err_cols.values()]
    bp = axes[2].boxplot(
        abs_data,
        labels=["Euler\n\u0394t=1/1000", "Almost\n\u0394t=1/1000", "Almost\n\u0394t=1/30"],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], colors_err):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[2].set_ylabel("|err|")
    axes[2].set_title("|Bias + MC noise| distribution")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    # --- Summary ---
    print("Absolute error statistics:")
    print(f"  {'Method':<28s}  {'mean |err|':>10s}  {'median':>10s}  {'max':>10s}")
    for label, col in err_cols.items():
        ae = df[col].abs()
        print(f"  {label:<28s}  {ae.mean():10.6f}  {ae.median():10.6f}  {ae.max():10.6f}")


def plot_dt_impact(df, save_path="fig_q3_dt_impact.png"):
    """Fine vs coarse delta-t comparison for the almost-exact scheme."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- (a) |err| fine vs coarse ---
    axes[0].scatter(
        df["err_almost_1000"].abs(), df["err_almost_30"].abs(),
        s=10, alpha=0.5, c=df["feller_gap"], cmap="viridis",
    )
    lim = max(df["err_almost_1000"].abs().max(), df["err_almost_30"].abs().max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x")
    axes[0].set_xlabel("|err|  Almost-exact \u0394t = 1/1000")
    axes[0].set_ylabel("|err|  Almost-exact \u0394t = 1/30")
    axes[0].set_title("Bias: fine vs coarse \u0394t")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- (b) Time fine vs coarse ---
    axes[1].scatter(
        df["ms_almost_dt1000"], df["ms_almost_dt1_30"],
        s=10, alpha=0.5, color="#c44e52",
    )
    axes[1].set_xlabel("Time (ms) \u0394t = 1/1000")
    axes[1].set_ylabel("Time (ms) \u0394t = 1/30")
    axes[1].set_title("Execution time: fine vs coarse \u0394t")
    axes[1].grid(alpha=0.3)

    # --- (c) |err_30| vs sigma ---
    sc = axes[2].scatter(
        df["sigma"], df["err_almost_30"].abs(),
        s=10, alpha=0.5, c=df["kappa"], cmap="coolwarm",
    )
    axes[2].set_xlabel("\u03c3 (vol-of-vol)")
    axes[2].set_ylabel("|err|  Almost-exact \u0394t = 1/30")
    axes[2].set_title("Coarse-\u0394t bias grows with \u03c3")
    cb = plt.colorbar(sc, ax=axes[2])
    cb.set_label("\u03ba")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    # --- Summary ---
    worse = (df["err_almost_30"].abs() > df["err_almost_1000"].abs()).mean()
    print(f"Fraction of cases where |err(\u0394t=1/30)| > |err(\u0394t=1/1000)|: {worse:.1%}")
    print(f"Mean |err|  \u0394t=1/1000: {df['err_almost_1000'].abs().mean():.6f}")
    print(f"Mean |err|  \u0394t=1/30  : {df['err_almost_30'].abs().mean():.6f}")
    print(f"Ratio: {df['err_almost_30'].abs().mean() / df['err_almost_1000'].abs().mean():.2f}\u00d7")
    print(f"\nMean time  \u0394t=1/1000: {df['ms_almost_dt1000'].mean():.1f} ms")
    print(f"Mean time  \u0394t=1/30  : {df['ms_almost_dt1_30'].mean():.1f} ms")
    print(f"Speedup: {df['ms_almost_dt1000'].mean() / df['ms_almost_dt1_30'].mean():.1f}\u00d7")
