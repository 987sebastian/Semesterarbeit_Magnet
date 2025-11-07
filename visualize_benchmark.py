#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Core Benchmark Visualization
Visualize parallel performance scaling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_benchmark_results(filename="results/benchmark_parallel.csv"):
    """Load benchmark results from CSV"""
    if not Path(filename).exists():
        print(f"Error: {filename} not found!")
        print("Please run the benchmark first: run_benchmark.bat")
        return None

    df = pd.read_csv(filename)
    return df


def plot_benchmark_results(df, output_file="results/benchmark_visualization.png"):
    """Create comprehensive benchmark visualization"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multi-Core Performance Benchmark Results\nBiot-Savart Magnetic Tracking',
                 fontsize=16, fontweight='bold')

    # 1. Throughput vs Thread Count
    ax = axes[0, 0]
    ax.plot(df['num_threads'], df['throughput_hz'], 'o-', linewidth=2, markersize=8, color='blue')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Throughput (Hz)')
    ax.set_title('Throughput vs Thread Count')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df['num_threads'])

    # Mark optimal point
    optimal_idx = df['throughput_hz'].idxmax()
    ax.plot(df.loc[optimal_idx, 'num_threads'], df.loc[optimal_idx, 'throughput_hz'],
            'r*', markersize=20, label='Optimal')
    ax.legend()

    # 2. Speedup vs Thread Count
    ax = axes[0, 1]
    ax.plot(df['num_threads'], df['speedup'], 'o-', linewidth=2, markersize=8, color='green',
            label='Actual Speedup')

    # Plot ideal linear speedup
    ax.plot(df['num_threads'], df['num_threads'], '--', linewidth=2, color='red',
            alpha=0.5, label='Ideal Linear Speedup')

    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup vs Thread Count')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df['num_threads'])
    ax.legend()

    # 3. Efficiency vs Thread Count
    ax = axes[0, 2]
    ax.plot(df['num_threads'], df['efficiency'] * 100, 'o-', linewidth=2, markersize=8, color='orange')
    ax.axhline(y=100, linestyle='--', color='red', alpha=0.5, label='100% Efficiency')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Parallel Efficiency')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df['num_threads'])
    ax.set_ylim([0, 110])
    ax.legend()

    # 4. Computation Time vs Thread Count
    ax = axes[1, 0]
    ax.plot(df['num_threads'], df['total_time_s'], 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Total Time (s)')
    ax.set_title('Computation Time vs Thread Count')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df['num_threads'])

    # 5. Average Time per Sample
    ax = axes[1, 1]
    ax.plot(df['num_threads'], df['avg_time_ms'], 'o-', linewidth=2, markersize=8, color='brown')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Avg Time per Sample (ms)')
    ax.set_title('Average Processing Time')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df['num_threads'])

    # 6. Summary Statistics Table
    ax = axes[1, 2]
    ax.axis('off')

    # Find optimal and key configurations
    optimal_idx = df['throughput_hz'].idxmax()
    opt_threads = df.loc[optimal_idx, 'num_threads']
    opt_throughput = df.loc[optimal_idx, 'throughput_hz']
    opt_speedup = df.loc[optimal_idx, 'speedup']
    opt_efficiency = df.loc[optimal_idx, 'efficiency'] * 100

    # Get 1-thread baseline
    baseline_throughput = df.loc[0, 'throughput_hz']

    stats = {
        'Metric': [
            'Single-Thread',
            'Optimal Threads',
            'Optimal Throughput',
            'Max Speedup',
            'Efficiency at Optimal',
            'Total Samples'
        ],
        'Value': [
            f'{baseline_throughput:.1f} Hz',
            f'{int(opt_threads)}',
            f'{opt_throughput:.1f} Hz',
            f'{opt_speedup:.2f}x',
            f'{opt_efficiency:.1f}%',
            f'{df.loc[0, "total_rows"]}'
        ]
    }

    stats_df = pd.DataFrame(stats)
    table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns,
                     cellLoc='center', loc='center', bbox=[0, 0.2, 1, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Color header
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows
    for i in range(1, len(stats['Metric']) + 1):
        if i % 2 == 0:
            for j in range(2):
                table[(i, j)].set_facecolor('#E7E6E6')

    ax.set_title('Performance Summary', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Benchmark visualization saved to: {output_file}")

    return fig


def generate_benchmark_report(df, output_file="results/benchmark_report.txt"):
    """Generate text report of benchmark results"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-CORE PERFORMANCE BENCHMARK REPORT\n")
        f.write("Biot-Savart Magnetic Tracking System\n")
        f.write("=" * 80 + "\n\n")

        # System information
        f.write("SYSTEM CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Test Date:          {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Samples:      {df.loc[0, 'total_rows']}\n")
        f.write(f"Converged Samples:  {df.loc[0, 'converged']}\n")
        f.write(f"Thread Range:       {df['num_threads'].min()} - {df['num_threads'].max()}\n\n")

        # Performance metrics
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Threads':<10} {'Time (s)':<12} {'Avg (ms)':<12} {'Throughput':<15} "
                f"{'Speedup':<12} {'Efficiency':<12}\n")
        f.write("-" * 80 + "\n")

        for idx, row in df.iterrows():
            f.write(f"{row['num_threads']:<10} "
                    f"{row['total_time_s']:<12.2f} "
                    f"{row['avg_time_ms']:<12.3f} "
                    f"{row['throughput_hz']:<15.1f} "
                    f"{row['speedup']:<12.2f} "
                    f"{row['efficiency'] * 100:<12.1f}%\n")

        f.write("\n")

        # Optimal configuration
        optimal_idx = df['throughput_hz'].idxmax()
        f.write("OPTIMAL CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Optimal Threads:    {int(df.loc[optimal_idx, 'num_threads'])}\n")
        f.write(f"Throughput:         {df.loc[optimal_idx, 'throughput_hz']:.1f} Hz\n")
        f.write(f"Speedup:            {df.loc[optimal_idx, 'speedup']:.2f}x\n")
        f.write(f"Efficiency:         {df.loc[optimal_idx, 'efficiency'] * 100:.1f}%\n")
        f.write(f"Total Time:         {df.loc[optimal_idx, 'total_time_s']:.2f} s\n")
        f.write(f"Avg Time/Sample:    {df.loc[optimal_idx, 'avg_time_ms']:.3f} ms\n\n")

        # Scaling analysis
        f.write("SCALING ANALYSIS\n")
        f.write("-" * 80 + "\n")

        # Amdahl's law analysis
        if len(df) >= 2:
            speedup_2 = df.loc[1, 'speedup']
            f.write(f"2-thread speedup:   {speedup_2:.2f}x ({speedup_2 / 2 * 100:.0f}% efficient)\n")

        if len(df) >= 4:
            speedup_4 = df.loc[3, 'speedup']
            f.write(f"4-thread speedup:   {speedup_4:.2f}x ({speedup_4 / 4 * 100:.0f}% efficient)\n")

        if len(df) >= 8:
            speedup_8 = df.loc[7, 'speedup']
            f.write(f"8-thread speedup:   {speedup_8:.2f}x ({speedup_8 / 8 * 100:.0f}% efficient)\n")

        # Estimate parallel fraction using Amdahl's law
        if len(df) >= 2:
            p = df.loc[optimal_idx, 'num_threads']
            s = df.loc[optimal_idx, 'speedup']
            # s = 1 / ((1-f) + f/p) => f ≈ (s*(p-1))/(p*(s-1))
            if s > 1:
                parallel_fraction = (s * (p - 1)) / (p * (s - 1))
                parallel_fraction = min(max(parallel_fraction, 0), 1)  # Clamp to [0,1]
                f.write(f"\nEstimated parallel fraction: {parallel_fraction * 100:.1f}%\n")
                f.write(f"Estimated serial fraction:   {(1 - parallel_fraction) * 100:.1f}%\n")

        f.write("\n")

        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")

        opt_threads = int(df.loc[optimal_idx, 'num_threads'])
        opt_efficiency = df.loc[optimal_idx, 'efficiency'] * 100

        if opt_efficiency > 80:
            f.write(f"[EXCELLENT] Use {opt_threads} threads for optimal performance.\n")
            f.write(f"Efficiency is high ({opt_efficiency:.1f}%), indicating good parallelization.\n")
        elif opt_efficiency > 60:
            f.write(f"[GOOD] Use {opt_threads} threads for good performance.\n")
            f.write(f"Efficiency is reasonable ({opt_efficiency:.1f}%).\n")
        else:
            f.write(f"[MODERATE] Use {opt_threads} threads for best throughput.\n")
            f.write(f"Efficiency is moderate ({opt_efficiency:.1f}%), consider optimizing parallel code.\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"[OK] Benchmark report saved to: {output_file}")


def main():
    """Main function"""

    # Load results
    df = load_benchmark_results()
    if df is None:
        return

    print(f"Loaded benchmark results: {len(df)} thread configurations tested")
    print()

    # Generate visualizations
    print("Generating visualizations...")
    plot_benchmark_results(df)

    # Generate report
    print("\nGenerating report...")
    generate_benchmark_report(df)

    print("\n" + "=" * 80)
    print("[OK] Visualization complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - results/benchmark_visualization.png")
    print("  - results/benchmark_report.txt")
    print()


if __name__ == "__main__":
    main()