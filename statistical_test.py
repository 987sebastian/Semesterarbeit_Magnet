#!/usr/bin/env python3
# statistical_test.py
# Statistical comparison between Biot-Savart and Dipole models

import pandas as pd
import numpy as np
from scipy import stats
import os
import sys


def statistical_comparison(biot_csv, dipole_csv, output_txt):
    """
    Statistical comparison between Biot-Savart and Dipole models

    Args:
        biot_csv: Path to Biot-Savart results CSV
        dipole_csv: Path to Dipole results CSV
        output_txt: Output report file path
    """
    # Check if files exist
    if not os.path.exists(biot_csv):
        print(f"[ERROR] File not found: {biot_csv}")
        return None

    if not os.path.exists(dipole_csv):
        print(f"[ERROR] File not found: {dipole_csv}")
        return None

    print(f"Loading data from:")
    print(f"  Biot-Savart: {biot_csv}")
    print(f"  Dipole:      {dipole_csv}")
    print()

    try:
        df_biot = pd.read_csv(biot_csv)
        df_dipole = pd.read_csv(dipole_csv)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV files: {e}")
        return None

    print(f"[OK] Loaded {len(df_biot)} samples from Biot-Savart")
    print(f"[OK] Loaded {len(df_dipole)} samples from Dipole")
    print()

    # Check if required columns exist
    required_cols = ['pos_error', 'dir_error']
    for col in required_cols:
        if col not in df_biot.columns:
            print(f"[ERROR] Column '{col}' not found in Biot-Savart CSV")
            print(f"Available columns: {list(df_biot.columns)}")
            return None
        if col not in df_dipole.columns:
            print(f"[ERROR] Column '{col}' not found in Dipole CSV")
            print(f"Available columns: {list(df_dipole.columns)}")
            return None

    # Extract error data
    biot_pos = df_biot['pos_error'].values
    biot_dir = df_biot['dir_error'].values
    dipole_pos = df_dipole['pos_error'].values
    dipole_dir = df_dipole['dir_error'].values

    # Paired t-test (same samples tracked by both models)
    # 配对t检验(两个模型追踪相同样本)
    print("Performing paired t-tests...")
    t_stat_pos, p_val_pos = stats.ttest_rel(biot_pos, dipole_pos)
    t_stat_dir, p_val_dir = stats.ttest_rel(biot_dir, dipole_dir)

    # Calculate Cohen's d (effect size)
    mean_diff_pos = np.mean(biot_pos - dipole_pos)
    std_diff_pos = np.std(biot_pos - dipole_pos, ddof=1)
    cohen_d_pos = mean_diff_pos / std_diff_pos if std_diff_pos > 0 else 0

    mean_diff_dir = np.mean(biot_dir - dipole_dir)
    std_diff_dir = np.std(biot_dir - dipole_dir, ddof=1)
    cohen_d_dir = mean_diff_dir / std_diff_dir if std_diff_dir > 0 else 0

    # Generate report
    report = []
    report.append("=" * 80)
    report.append("STATISTICAL COMPARISON: Biot-Savart vs Dipole Model")
    report.append("=" * 80)
    report.append("")
    report.append(f"Sample size: {len(biot_pos)} paired observations")
    report.append("")

    report.append("POSITION ERROR ANALYSIS")
    report.append("-" * 80)
    report.append(f"Biot-Savart:  {np.mean(biot_pos) * 1000:.2f} ± {np.std(biot_pos) * 1000:.2f} mm")
    report.append(f"Dipole:       {np.mean(dipole_pos) * 1000:.2f} ± {np.std(dipole_pos) * 1000:.2f} mm")
    report.append(f"Difference:   {mean_diff_pos * 1000:.2f} mm (Biot - Dipole)")
    report.append("")
    report.append(f"Paired t-test results:")
    report.append(f"  t-statistic = {t_stat_pos:.3f}")
    report.append(f"  p-value     = {p_val_pos:.6f}")
    report.append(f"  Cohen's d   = {cohen_d_pos:.3f}")
    report.append("")

    # Significance level
    if p_val_pos < 0.001:
        report.append("  Result: HIGHLY SIGNIFICANT difference (p < 0.001) ***")
    elif p_val_pos < 0.01:
        report.append("  Result: VERY SIGNIFICANT difference (p < 0.01) **")
    elif p_val_pos < 0.05:
        report.append("  Result: SIGNIFICANT difference (p < 0.05) *")
    else:
        report.append("  Result: NO significant difference (p >= 0.05)")

    report.append("")

    # Effect size interpretation
    if abs(cohen_d_pos) < 0.2:
        effect_size = "negligible"
    elif abs(cohen_d_pos) < 0.5:
        effect_size = "small"
    elif abs(cohen_d_pos) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    report.append(f"  Effect size: {effect_size}")
    report.append("")

    report.append("DIRECTION ERROR ANALYSIS")
    report.append("-" * 80)
    report.append(f"Biot-Savart:  {np.mean(biot_dir):.2f} ± {np.std(biot_dir):.2f} deg")
    report.append(f"Dipole:       {np.mean(dipole_dir):.2f} ± {np.std(dipole_dir):.2f} deg")
    report.append(f"Difference:   {mean_diff_dir:.2f} deg (Biot - Dipole)")
    report.append("")
    report.append(f"Paired t-test results:")
    report.append(f"  t-statistic = {t_stat_dir:.3f}")
    report.append(f"  p-value     = {p_val_dir:.6f}")
    report.append(f"  Cohen's d   = {cohen_d_dir:.3f}")
    report.append("")

    if p_val_dir < 0.001:
        report.append("  Result: HIGHLY SIGNIFICANT difference (p < 0.001) ***")
    elif p_val_dir < 0.01:
        report.append("  Result: VERY SIGNIFICANT difference (p < 0.01) **")
    elif p_val_dir < 0.05:
        report.append("  Result: SIGNIFICANT difference (p < 0.05) *")
    else:
        report.append("  Result: NO significant difference (p >= 0.05)")

    report.append("")

    # Direction effect size
    if abs(cohen_d_dir) < 0.2:
        effect_size_dir = "negligible"
    elif abs(cohen_d_dir) < 0.5:
        effect_size_dir = "small"
    elif abs(cohen_d_dir) < 0.8:
        effect_size_dir = "medium"
    else:
        effect_size_dir = "large"
    report.append(f"  Effect size: {effect_size_dir}")
    report.append("")

    report.append("=" * 80)
    report.append("INTERPRETATION GUIDE")
    report.append("=" * 80)
    report.append("Statistical Significance (p-value):")
    report.append("  p < 0.001: Highly significant (***)")
    report.append("  p < 0.01:  Very significant (**)")
    report.append("  p < 0.05:  Significant (*)")
    report.append("  p >= 0.05: Not significant")
    report.append("")
    report.append("Effect Size (Cohen's d):")
    report.append("  |d| < 0.2:  Negligible")
    report.append("  0.2 <= |d| < 0.5: Small")
    report.append("  0.5 <= |d| < 0.8: Medium")
    report.append("  |d| >= 0.8: Large")
    report.append("")
    report.append("NOTE: Statistical significance ≠ Practical significance")
    report.append("Consider both p-value and effect size when interpreting results.")
    report.append("=" * 80)

    # Print to console
    for line in report:
        print(line)

    # Save to file
    try:
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print(f"\n[OK] Report saved to: {output_txt}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save report: {e}")

    return {
        'pos_t': t_stat_pos,
        'pos_p': p_val_pos,
        'pos_d': cohen_d_pos,
        'dir_t': t_stat_dir,
        'dir_p': p_val_dir,
        'dir_d': cohen_d_dir
    }


if __name__ == "__main__":
    print("=" * 80)
    print("Statistical Test: Biot-Savart vs Dipole Model")
    print("=" * 80)
    print()

    # File paths
    biot_file = "results/biot_savart_results.csv"
    dipole_file = "results/dipole_results.csv"
    output_file = "results/statistical_test_report.txt"

    # Check if results directory exists
    if not os.path.exists("results"):
        print("[ERROR] 'results' directory not found!")
        print("Please run model_comparison.exe first to generate results.")
        sys.exit(1)

    results = statistical_comparison(biot_file, dipole_file, output_file)

    if results is None:
        print("\n[FAILED] Statistical test could not be completed.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Statistical test completed successfully!")
        sys.exit(0)