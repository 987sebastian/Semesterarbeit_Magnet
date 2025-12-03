"""
3D Visualization: Point Size = Error, Color = Height
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def plot_error_as_size(results_csv, output_file, model_name="Model"):
    """
    Plot 3D results with:
    - Point size proportional to position error (larger = bigger error)
    - Color representing height (purple/blue = low, green/yellow = high)

    """

    print("=" * 80)
    print(f"{model_name} - Error as Point Size, Height as Color")
    print("=" * 80)

    # Load data / 加载数据
    df = pd.read_csv(results_csv)

    # Extract positions in mm / 提取位置(mm)
    x_mm = df['pos_x'].values * 1000  # m -> mm
    y_mm = df['pos_y'].values * 1000
    z_mm = df['pos_z'].values * 1000

    # Extract errors / 提取误差
    pos_error_mm = df['pos_error'].values * 1000  # m -> mm
    rmse_uT = df['rmse'].values * 1e6  # T -> µT

    print(f"\nLoaded {len(df)} data points")
    print(f"\nPosition range (mm):")
    print(f"  X: [{np.min(x_mm):.1f}, {np.max(x_mm):.1f}]")
    print(f"  Y: [{np.min(y_mm):.1f}, {np.max(y_mm):.1f}]")
    print(f"  Z: [{np.min(z_mm):.1f}, {np.max(z_mm):.1f}]")

    print(f"\nPosition error range: {np.min(pos_error_mm):.2f} - {np.max(pos_error_mm):.2f} mm")
    print(f"Position error mean:  {np.mean(pos_error_mm):.2f} mm")
    print(f"\nRMSE range: {np.min(rmse_uT):.2f} - {np.max(rmse_uT):.2f} µT")
    print(f"RMSE mean:  {np.mean(rmse_uT):.2f} µT")

    # Classify by 150mm boundary / 150mm分界
    valid_mask = z_mm <= 150.0
    beyond_mask = z_mm > 150.0

    n_valid = np.sum(valid_mask)
    n_beyond = np.sum(beyond_mask)

    print(f"\nHeight distribution:")
    print(f"  Z ≤ 150mm:  {n_valid:4d} points ({n_valid / len(df) * 100:5.1f}%)")
    print(f"  Z > 150mm:  {n_beyond:4d} points ({n_beyond / len(df) * 100:5.1f}%)")

    if n_valid > 0:
        print(f"\nErrors at Z ≤ 150mm:")
        print(f"  Mean: {np.mean(pos_error_mm[valid_mask]):.2f} mm")
        print(f"  Max:  {np.max(pos_error_mm[valid_mask]):.2f} mm")

    if n_beyond > 0:
        print(f"\nErrors at Z > 150mm:")
        print(f"  Mean: {np.mean(pos_error_mm[beyond_mask]):.2f} mm")
        print(f"  Max:  {np.max(pos_error_mm[beyond_mask]):.2f} mm")

        if n_valid > 0:
            error_increase = (np.mean(pos_error_mm[beyond_mask]) - np.mean(pos_error_mm[valid_mask])) / np.mean(
                pos_error_mm[valid_mask]) * 100
            print(f"\n  Error increase: {error_increase:+.1f}%")

    # Create figure / 创建图形
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    # Point sizes proportional to position error / 圆点大小正比于位置误差
    # Use exponential scaling to emphasize large errors / 使用指数缩放强调大误差
    min_error = np.min(pos_error_mm)
    max_error = np.max(pos_error_mm)

    # Normalize errors to 0-1 range / 归一化误差到0-1
    error_normalized = (pos_error_mm - min_error) / (max_error - min_error)

    # Apply power scaling for visual emphasis (power > 1 makes differences more obvious)
    # 应用幂次缩放增强视觉对比（幂次>1使差异更明显）
    error_scaled = np.power(error_normalized, 0.7)  # 0.7 power emphasizes differences

    # Map to point sizes: small error -> small points, large error -> MUCH larger points
    # 映射到点大小：小误差->小点，大误差->显著更大的点
    point_sizes = 30 + 300 * error_scaled  # 30-330 range (wider range for more contrast)

    print(f"\nPoint size range: {np.min(point_sizes):.1f} - {np.max(point_sizes):.1f}")

    # Plot with color = height (viridis: purple/blue=low, green/yellow=high)
    # 绘图：颜色=高度 (viridis色图: 紫色/深蓝=低，浅绿/黄色=高)
    scatter = ax.scatter(x_mm, y_mm, z_mm,
                         c=z_mm,  # Color by height / 按高度着色
                         cmap='viridis',  # Purple->Blue->Green->Yellow
                         s=point_sizes,  # Size by error / 按误差定大小
                         alpha=0.7,
                         edgecolors='black',
                         linewidths=0.3)

    # Add 150mm reference plane / 添加150mm参考平面
    x_range = [np.min(x_mm) - 10, np.max(x_mm) + 10]
    y_range = [np.min(y_mm) - 10, np.max(y_mm) + 10]
    xx, yy = np.meshgrid(x_range, y_range)
    zz = np.ones_like(xx) * 150.0  # 150mm

    ax.plot_surface(xx, yy, zz, alpha=0.1, color='red',
                    edgecolor='red', linewidth=1.5, linestyle='--')

    # Add text label for 150mm plane / 添加150mm平面标签
    ax.text(np.mean(x_range), np.mean(y_range), 150.0,
            '150mm boundary', fontsize=11, color='red',
            weight='bold', ha='center')

    # Add colorbar for height / 添加高度色标
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6, aspect=15)
    cbar.set_label('Z-position [mm]', rotation=270, labelpad=20, fontsize=12)

    # Add horizontal line at 150mm on colorbar / 在色标上标记150mm
    cbar.ax.axhline(y=150.0, color='red', linewidth=2, linestyle='--')
    cbar.ax.text(1.5, 150.0, '150mm', va='center', ha='left',
                 fontsize=9, color='red', weight='bold')

    # Add legend for point sizes / 添加点大小图例
    # Create dummy scatter plots for legend / 创建图例用的虚拟散点
    # Use more spaced out error levels for clarity / 使用间距更大的误差级别
    error_levels = [2.0, 7.0, 15.0]  # Representative error values in mm
    size_levels = []
    for e in error_levels:
        e_norm = (e - min_error) / (max_error - min_error)
        e_scaled = np.power(e_norm, 0.7)
        s = 30 + 300 * e_scaled
        size_levels.append(s)

    legend_labels = [f'{e:.1f} mm' for e in error_levels]

    for size, label in zip(size_levels, legend_labels):
        ax.scatter([], [], s=size, c='gray', alpha=0.6,
                   edgecolors='black', linewidths=0.3, label=label)

    ax.legend(title='Position Error', loc='upper left', fontsize=10,
              framealpha=0.9, edgecolor='black')

    # Set labels / 设置轴标签
    ax.set_xlabel('X-position [mm]', fontsize=13, labelpad=10)
    ax.set_ylabel('Y-position [mm]', fontsize=13, labelpad=10)
    ax.set_zlabel('Z-position [mm]', fontsize=13, labelpad=10)

    # Set title / 设置标题
    ax.set_title(f'{model_name} Tracking Results\n'
                 f'Point Size = Position Error | Color = Height\n'
                 f'(Valid Range: Z ≤ 150mm)',
                 fontsize=14, fontweight='bold', pad=15)

    # Set viewing angle (matching supervisor's figure) / 设置视角
    ax.view_init(elev=25, azim=225)

    # Grid / 网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='gray')

    # Auto-adjust axis limits / 自动调整坐标轴范围
    x_pad = (np.max(x_mm) - np.min(x_mm)) * 0.1
    y_pad = (np.max(y_mm) - np.min(y_mm)) * 0.1
    z_pad = (np.max(z_mm) - np.min(z_mm)) * 0.1

    ax.set_xlim(np.min(x_mm) - x_pad, np.max(x_mm) + x_pad)
    ax.set_ylim(np.min(y_mm) - y_pad, np.max(y_mm) + y_pad)
    ax.set_zlim(np.min(z_mm) - z_pad, np.max(z_mm) + z_pad)

    # Make panes semi-transparent / 使坐标平面半透明
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.2)
    ax.yaxis.pane.set_alpha(0.2)
    ax.zaxis.pane.set_alpha(0.2)

    # Save figure / 保存图形
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[OK] Figure saved to: {output_file}")
    plt.close()

    return fig


def main():
    """Main function / 主函数"""

    print("=" * 80)
    print("3D Visualization: Error as Point Size, Height as Color")
    print("=" * 80)
    print()

    # Create results directory / 创建结果目录
    Path("results").mkdir(exist_ok=True)

    # File paths / 文件路径
    biot_csv = "results/biot_savart_results.csv"
    dipole_csv = "results/dipole_results.csv"

    # Try uploaded location if not found / 如果找不到则尝试上传位置
    if not Path(biot_csv).exists():
        biot_csv = "/mnt/user-data/uploads/biot_savart_results.csv"
    if not Path(dipole_csv).exists():
        dipole_csv = "/mnt/user-data/uploads/dipole_results.csv"

    # Plot Biot-Savart / 绘制Biot-Savart结果
    if Path(biot_csv).exists():
        print("\n" + "=" * 80)
        print("Plotting Biot-Savart Results:")
        print("=" * 80)
        plot_error_as_size(
            biot_csv,
            "results/3d_biot_error_size.png",
            model_name="Biot-Savart"
        )
    else:
        print(f"\n[WARNING] {biot_csv} not found, skipping...")

    # Plot Dipole / 绘制Dipole结果
    if Path(dipole_csv).exists():
        print("\n" + "=" * 80)
        print("Plotting Dipole Results:")
        print("=" * 80)
        plot_error_as_size(
            dipole_csv,
            "results/3d_dipole_error_size.png",
            model_name="Dipole"
        )
    else:
        print(f"\n[WARNING] {dipole_csv} not found, skipping...")

    print("\n" + "=" * 80)
    print("Done! Generated files:")
    print("  - results/3d_biot_error_size.png")
    print("  - results/3d_dipole_error_size.png")
    print("\nVisualization features:")
    print("  • Point size = Position error (exponential scaling for emphasis)")
    print("  • Color = Height (purple/blue = low, green/yellow = high)")
    print("  • 150mm boundary plane shown in red")
    print("  • Units in millimeters")
    print("=" * 80)


if __name__ == "__main__":
    main()