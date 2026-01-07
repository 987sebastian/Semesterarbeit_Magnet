import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =============== Path Configuration ===============
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "B-H Kurve"

# Uncomment if relative path doesn't work:
# DATA_DIR = Path(r"C:\Users\Luo Yi\Desktop\Semesterarbeit\Teil 2\B-H Kurve")

if not DATA_DIR.exists():
    print(f"Warning: Data directory not found: {DATA_DIR}")
    exit(1)

# =============== Global Plotting Parameters ===============
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2.5,
    'lines.markersize': 4,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
})


# =============== Data Loading Function ===============
def load_path_data(path_name):
    """Load three types of data for a specified path"""
    try:
        biot_file = DATA_DIR / f"{path_name}_Biot.csv"
        dipole_file = DATA_DIR / f"{path_name}_Dipole.csv"
        mag_file = DATA_DIR / f"{path_name}_Mag.csv"

        df_biot = pd.read_csv(biot_file)
        df_dipole = pd.read_csv(dipole_file)
        df_mag = pd.read_csv(mag_file)

        dist_biot = df_biot['Distance [mm]'].values
        B_biot = df_biot['Mag_B [mTesla]'].values
        dist_dipole = df_dipole['Distance [mm]'].values
        B_dipole = df_dipole['Mag_B [mTesla]'].values
        dist_maxwell = df_mag['Distance [mm]'].values
        B_maxwell = df_mag['Mag_B [mTesla]'].values

        # Truncate to minimum length
        min_len = min(len(B_biot), len(B_dipole), len(B_maxwell))

        return {
            'distance': dist_biot[:min_len],
            'biot': B_biot[:min_len],
            'dipole': B_dipole[:min_len],
            'maxwell': B_maxwell[:min_len]
        }

    except Exception as e:
        print(f"Error loading {path_name}: {e}")
        return None


# =============== Main Plotting Function ===============
def plot_model_comparison():
    """Generate Figure 3.3 in large format for manual scaling"""

    paths = ['X0', 'X5', 'Z0', 'Z5', 'Random', 'Y5']

    # Large figure size for clarity
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)

    fig.suptitle('Magnetfeldstärke: Modellvergleich auf sechs Messpfaden',
                 fontsize=20, fontweight='bold', y=0.995)

    # Color scheme
    colors = {
        'maxwell': '#2ca02c',  # Green
        'biot': '#1f77b4',  # Blue
        'dipole': '#d62728'  # Red
    }

    # Plot each path
    for idx, path_name in enumerate(paths):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        data = load_path_data(path_name)
        if data is None:
            ax.text(0.5, 0.5, f'Data not found for {path_name}',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        dist = data['distance']
        B_maxwell = data['maxwell']
        B_biot = data['biot']
        B_dipole = data['dipole']

        # Plot curves
        ax.plot(dist, B_maxwell, 'o', color=colors['maxwell'],
                label='Ansys Maxwell FEM', markersize=4, alpha=0.6,
                markeredgewidth=0, zorder=1)
        ax.plot(dist, B_biot, '-', color=colors['biot'],
                label='Biot-Savart', linewidth=2.5, zorder=3)
        ax.plot(dist, B_dipole, '--', color=colors['dipole'],
                label='Dipol', linewidth=2.5, dashes=(5, 3), zorder=2)

        # Axis labels
        ax.set_xlabel('Abstand entlang Pfad [mm]', fontsize=14)
        ax.set_ylabel('|B| [mT]', fontsize=14)
        ax.set_title(f'Pfad {path_name}', fontweight='bold', pad=10, fontsize=16)

        # Legend only on first subplot
        if idx == 0:
            ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray',
                      fontsize=11, shadow=True)

        # Grid
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        ax.set_facecolor('#fafafa')

        # Tick styling
        ax.tick_params(direction='in', which='both', top=True, right=True)

    # Layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save PNG only
    output_path = Path('.') / 'figure_3_3.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')

    print(f"\nSuccessfully saved: {output_path.absolute()}")
    print(f"Image size: 5400 x 3600 pixels (300 DPI)")
    print(f"Original size: 18 x 12 inches (45.7 x 30.5 cm)")
    print(f"\nRecommendation: Scale to 14.64 cm width in Word")

    plt.show()
    return fig


# =============== Main Execution ===============
if __name__ == "__main__":
    print("=" * 60)
    print("Figure 3.3 Generation")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR.absolute()}")

    if DATA_DIR.exists():
        csv_files = list(DATA_DIR.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files")

    print("\nGenerating figure...")
    fig = plot_model_comparison()

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)