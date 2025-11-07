#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Comparison Visualization

This script visualizes the comparison between Biot-Savart and Dipole models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


def load_results(biot_file, dipole_file):
    """Load comparison results from CSV files"""
    biot_df = pd.read_csv(biot_file)
    dipole_df = pd.read_csv(dipole_file)
    return biot_df, dipole_df


def plot_comparison(biot_df, dipole_df, output_file="model_comparison.png"):
    """Create comprehensive comparison plots"""

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Biot-Savart vs Dipole Model Comparison',
                 fontsize=16, fontweight='bold')

    # 1. RMSE Distribution
    ax = axes[0, 0]
    ax.hist(biot_df['rmse'] * 1e6, bins=50, alpha=0.6, label='Biot-Savart', color='blue')
    ax.hist(dipole_df['rmse'] * 1e6, bins=50, alpha=0.6, label='Dipole', color='red')
    ax.set_xlabel('RMSE (uT)')
    ax.set_ylabel('Frequency')
    ax.set_title('RMSE Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add mean lines
    ax.axvline(biot_df['rmse'].mean() * 1e6, color='blue', linestyle='--', linewidth=2,
               label=f'Biot Mean: {biot_df["rmse"].mean() * 1e6:.1f} uT')
    ax.axvline(dipole_df['rmse'].mean() * 1e6, color='red', linestyle='--', linewidth=2,
               label=f'Dipole Mean: {dipole_df["rmse"].mean() * 1e6:.1f} uT')

    # 2. Position Error Distribution
    ax = axes[0, 1]
    ax.hist(biot_df['pos_error'] * 1000, bins=50, alpha=0.6, label='Biot-Savart', color='blue')
    ax.hist(dipole_df['pos_error'] * 1000, bins=50, alpha=0.6, label='Dipole', color='red')
    ax.set_xlabel('Position Error (mm)')
    ax.set_ylabel('Frequency')
    ax.set_title('Position Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Direction Error Distribution
    ax = axes[0, 2]
    ax.hist(biot_df['dir_error'], bins=50, alpha=0.6, label='Biot-Savart', color='blue')
    ax.hist(dipole_df['dir_error'], bins=50, alpha=0.6, label='Dipole', color='red')
    ax.set_xlabel('Direction Error (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Direction Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Scale Factor Distribution
    ax = axes[1, 0]
    ax.hist(biot_df['scale'], bins=50, alpha=0.6, label='Biot-Savart', color='blue')
    ax.hist(dipole_df['scale'], bins=50, alpha=0.6, label='Dipole', color='red')
    ax.set_xlabel('Scale Factor')
    ax.set_ylabel('Frequency')
    ax.set_title('Scale Factor Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Computation Time Distribution
    ax = axes[1, 1]
    ax.hist(biot_df['time_ms'], bins=50, alpha=0.6, label='Biot-Savart', color='blue')
    ax.hist(dipole_df['time_ms'], bins=50, alpha=0.6, label='Dipole', color='red')
    ax.set_xlabel('Computation Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Computation Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. RMSE vs Position Error Scatter
    ax = axes[1, 2]
    ax.scatter(biot_df['pos_error'] * 1000, biot_df['rmse'] * 1e6,
               alpha=0.3, s=10, label='Biot-Savart', color='blue')
    ax.scatter(dipole_df['pos_error'] * 1000, dipole_df['rmse'] * 1e6,
               alpha=0.3, s=10, label='Dipole', color='red')
    ax.set_xlabel('Position Error (mm)')
    ax.set_ylabel('RMSE (uT)')
    ax.set_title('RMSE vs Position Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7. Iteration Count Distribution
    ax = axes[2, 0]
    # Status values represent iteration outcomes
    biot_converged = (biot_df['status'] > 0).sum()
    dipole_converged = (dipole_df['status'] > 0).sum()

    models = ['Biot-Savart', 'Dipole']
    converged = [biot_converged, dipole_converged]
    failed = [len(biot_df) - biot_converged, len(dipole_df) - dipole_converged]

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x, converged, width, label='Converged', color='green', alpha=0.7)
    ax.bar(x, failed, width, bottom=converged, label='Failed', color='red', alpha=0.7)
    ax.set_ylabel('Number of Samples')
    ax.set_title('Convergence Statistics')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 8. Box Plot Comparison
    ax = axes[2, 1]
    data_to_plot = [biot_df['rmse'] * 1e6, dipole_df['rmse'] * 1e6]
    bp = ax.boxplot(data_to_plot, tick_labels=['Biot-Savart', 'Dipole'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    ax.set_ylabel('RMSE (uT)')
    ax.set_title('RMSE Box Plot Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # 9. Summary Statistics Table
    ax = axes[2, 2]
    ax.axis('off')

    # Calculate statistics
    stats = {
        'Metric': ['Mean RMSE (uT)', 'Mean Pos Error (mm)', 'Mean Dir Error (deg)',
                   'Mean Time (ms)', 'Convergence Rate (%)'],
        'Biot-Savart': [
            f'{biot_df["rmse"].mean() * 1e6:.2f}',
            f'{biot_df["pos_error"].mean() * 1000:.2f}',
            f'{biot_df["dir_error"].mean():.2f}',
            f'{biot_df["time_ms"].mean():.2f}',
            f'{100 * biot_converged / len(biot_df):.1f}'
        ],
        'Dipole': [
            f'{dipole_df["rmse"].mean() * 1e6:.2f}',
            f'{dipole_df["pos_error"].mean() * 1000:.2f}',
            f'{dipole_df["dir_error"].mean():.2f}',
            f'{dipole_df["time_ms"].mean():.2f}',
            f'{100 * dipole_converged / len(dipole_df):.1f}'
        ]
    }

    # Create table
    stats_df = pd.DataFrame(stats)
    table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows
    for i in range(1, len(stats['Metric']) + 1):
        if i % 2 == 0:
            for j in range(3):
                table[(i, j)].set_facecolor('#E7E6E6')

    ax.set_title('Summary Statistics', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Comparison plot saved to: {output_file}")

    return fig


def plot_metrics_summary(biot_df, dipole_df, output_file="metrics_summary.png"):
    """Create summary bar charts for key metrics"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Key Metrics Summary', fontsize=16, fontweight='bold')

    # Calculate metrics
    metrics = {
        'RMSE (uT)': [biot_df['rmse'].mean() * 1e6, dipole_df['rmse'].mean() * 1e6],
        'Position Error (mm)': [biot_df['pos_error'].mean() * 1000, dipole_df['pos_error'].mean() * 1000],
        'Direction Error (deg)': [biot_df['dir_error'].mean(), dipole_df['dir_error'].mean()],
        'Computation Time (ms)': [biot_df['time_ms'].mean(), dipole_df['time_ms'].mean()]
    }

    models = ['Biot-Savart', 'Dipole']
    colors = ['blue', 'red']

    for idx, (metric, values) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(models, values, color=colors, alpha=0.7)
        ax.set_ylabel(metric.split('(')[0].strip())
        ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom', fontweight='bold')

        # Highlight winner
        winner_idx = 0 if values[0] < values[1] else 1
        bars[winner_idx].set_edgecolor('gold')
        bars[winner_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Metrics summary saved to: {output_file}")

    return fig


def generate_report(biot_df, dipole_df, output_file="comparison_report.txt"):
    """Generate a text report of the comparison"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("Biot-Savart vs Dipole\n")
        f.write("=" * 80 + "\n\n")

        # Biot-Savart Statistics
        f.write("BIOT-SAVART MODEL\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Samples:         {len(biot_df)}\n")
        f.write(
            f"Converged Samples:     {(biot_df['status'] > 0).sum()} ({100 * (biot_df['status'] > 0).sum() / len(biot_df):.1f}%)\n")
        f.write(f"Mean RMSE:             {biot_df['rmse'].mean() * 1e6:.3f} uT\n")
        f.write(f"Median RMSE:           {biot_df['rmse'].median() * 1e6:.3f} uT\n")
        f.write(f"Std RMSE:              {biot_df['rmse'].std() * 1e6:.3f} uT\n")
        f.write(f"Mean Position Error:   {biot_df['pos_error'].mean() * 1000:.3f} mm\n")
        f.write(f"Mean Direction Error:  {biot_df['dir_error'].mean():.3f} deg\n")
        f.write(f"Mean Scale Factor:     {biot_df['scale'].mean():.3f}\n")
        f.write(f"Mean Computation Time: {biot_df['time_ms'].mean():.3f} ms\n\n")

        # Dipole Statistics
        f.write("DIPOLE MODEL\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Samples:         {len(dipole_df)}\n")
        f.write(
            f"Converged Samples:     {(dipole_df['status'] > 0).sum()} ({100 * (dipole_df['status'] > 0).sum() / len(dipole_df):.1f}%)\n")
        f.write(f"Mean RMSE:             {dipole_df['rmse'].mean() * 1e6:.3f} uT\n")
        f.write(f"Median RMSE:           {dipole_df['rmse'].median() * 1e6:.3f} uT\n")
        f.write(f"Std RMSE:              {dipole_df['rmse'].std() * 1e6:.3f} uT\n")
        f.write(f"Mean Position Error:   {dipole_df['pos_error'].mean() * 1000:.3f} mm\n")
        f.write(f"Mean Direction Error:  {dipole_df['dir_error'].mean():.3f} deg\n")
        f.write(f"Mean Scale Factor:     {dipole_df['scale'].mean():.3f}\n")
        f.write(f"Mean Computation Time: {dipole_df['time_ms'].mean():.3f} ms\n\n")

        # Comparison
        f.write("COMPARISON\n")
        f.write("-" * 80 + "\n")

        rmse_diff = ((biot_df['rmse'].mean() - dipole_df['rmse'].mean()) / dipole_df['rmse'].mean()) * 100
        pos_diff = ((biot_df['pos_error'].mean() - dipole_df['pos_error'].mean()) / dipole_df[
            'pos_error'].mean()) * 100
        time_diff = ((biot_df['time_ms'].mean() - dipole_df['time_ms'].mean()) / dipole_df['time_ms'].mean()) * 100

        f.write(f"RMSE Difference:       {rmse_diff:+.1f}% ({'Biot better' if rmse_diff < 0 else 'Dipole better'})\n")
        f.write(
            f"Position Error Diff:   {pos_diff:+.1f}% ({'Biot better' if pos_diff < 0 else 'Dipole better'})\n")
        f.write(
            f"Computation Time Diff: {time_diff:+.1f}% ({'Biot faster' if time_diff < 0 else 'Dipole faster'})\n\n")

        # Recommendation
        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("=" * 80 + "\n")

        if biot_df['rmse'].mean() < dipole_df['rmse'].mean() and biot_df['pos_error'].mean() < dipole_df[
            'pos_error'].mean():
            f.write("[OK] Use Biot-Savart Model\n")
            f.write("  Reason: Better accuracy in both RMSE and position error\n")
        elif dipole_df['rmse'].mean() < biot_df['rmse'].mean() and dipole_df['pos_error'].mean() < biot_df[
            'pos_error'].mean():
            f.write("[OK] Use Dipole Model\n")
            f.write("  Reason: Better accuracy and potentially faster computation\n")
        else:
            f.write("[!] Trade-off exists between models\n")
            f.write("  Consider application-specific requirements:\n")
            f.write(
                f"  - For best RMSE: {'Biot-Savart' if biot_df['rmse'].mean() < dipole_df['rmse'].mean() else 'Dipole'}\n")
            f.write(
                f"  - For best position: {'Biot-Savart' if biot_df['pos_error'].mean() < dipole_df['pos_error'].mean() else 'Dipole'}\n")
            f.write(
                f"  - For speed: {'Biot-Savart' if biot_df['time_ms'].mean() < dipole_df['time_ms'].mean() else 'Dipole'}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"[OK] Comparison report saved to: {output_file}")


def main():
    """Main function"""

    # File paths
    biot_file = "results/biot_savart_results.csv"
    dipole_file = "results/dipole_results.csv"

    # Check if files exist
    if not Path(biot_file).exists() or not Path(dipole_file).exists():
        print("Error: Result files not found!")
        print(f"  Expected: {biot_file}")
        print(f"  Expected: {dipole_file}")
        print("\nPlease run the model_comparison executable first.")
        return

    print("Loading comparison results...")
    biot_df, dipole_df = load_results(biot_file, dipole_file)

    print(f"Loaded {len(biot_df)} samples from Biot-Savart results")
    print(f"Loaded {len(dipole_df)} samples from Dipole results")

    print("\nGenerating visualizations...")
    plot_comparison(biot_df, dipole_df, "results/model_comparison.png")
    plot_metrics_summary(biot_df, dipole_df, "results/metrics_summary.png")

    print("\nGenerating report...")
    generate_report(biot_df, dipole_df, "results/comparison_report.txt")

    print("\n" + "=" * 80)
    print("[OK] Analysis complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - results/model_comparison.png    (Detailed comparison plots)")
    print("  - results/metrics_summary.png     (Key metrics summary)")
    print("  - results/comparison_report.txt   (Text report)")
    print()


if __name__ == "__main__":
    main()