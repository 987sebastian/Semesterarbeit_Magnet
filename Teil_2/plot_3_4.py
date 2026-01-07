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
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2.5,
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


# =============== Calculate Relative Error ===============
def calculate_relative_error(model_values, reference_values):
    """
    Calculate relative error: |model - reference| / reference * 100%

    Args:
        model_values: array of model predictions
        reference_values: array of reference (Ansys) values

    Returns:
        relative_error: array of relative errors in percentage
    """
    # Avoid division by zero
    epsilon = 1e-20
    reference_safe = np.where(np.abs(reference_values) < epsilon, epsilon, reference_values)

    error = np.abs(model_values - reference_values) / np.abs(reference_safe) * 100.0

    return error


# =============== Main Plotting Function ===============
def plot_relative_error():
    """Generate Figure 3.4: Relative error vs distance"""

    paths = ['X0', 'X5', 'Z0', 'Z5', 'Random', 'Y5']

    # Large figure size
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)

    fig.suptitle('Relativer Fehler der beiden analytischen Modelle gegenüber Ansys Maxwell',
                 fontsize=18, fontweight='bold', y=0.995)

    # Color scheme
    colors = {
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

        # Calculate relative errors
        error_biot = calculate_relative_error(B_biot, B_maxwell)
        error_dipole = calculate_relative_error(B_dipole, B_maxwell)

        # Plot error curves (log scale)
        ax.plot(dist, error_biot, '-', color=colors['biot'],
                label='Biot-Savart vs Ansys', linewidth=2.5, zorder=2)
        ax.plot(dist, error_dipole, '-', color=colors['dipole'],
                label='Dipole vs Ansys', linewidth=2.5, zorder=1)

        # Axis labels
        ax.set_xlabel('Distance along path [mm]', fontsize=13)
        ax.set_ylabel('Relative Error [%]', fontsize=13)
        ax.set_title(f'Path: {path_name}', fontweight='bold', pad=8, fontsize=15)

        # Log scale for y-axis
        ax.set_yscale('log')

        # Set y-axis limits for better visibility
        ax.set_ylim([1e-2, 1e3])

        # Legend
        if idx == 0:
            ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray',
                      fontsize=11, shadow=False)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')
        ax.set_facecolor('white')

        # Tick styling
        ax.tick_params(direction='in', which='both', top=True, right=True)

        # Print statistics
        print(f"\n{path_name} Error Statistics:")
        print(f"  Biot-Savart: mean={np.mean(error_biot):.2f}%, "
              f"max={np.max(error_biot):.2f}%")
        print(f"  Dipole: mean={np.mean(error_dipole):.2f}%, "
              f"max={np.max(error_dipole):.2f}%")

    # Layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save PNG only
    output_path = Path('.') / 'figure_3_4.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')

    print(f"\n{'=' * 60}")
    print(f"Successfully saved: {output_path.absolute()}")
    print(f"Image size: 5400 x 3600 pixels (300 DPI)")
    print(f"Original size: 18 x 12 inches (45.7 x 30.5 cm)")
    print(f"\nRecommendation: Scale to 14.64 cm width in Word")
    print(f"{'=' * 60}")

    plt.show()
    return fig


# =============== Main Execution ===============
if __name__ == "__main__":
    print("=" * 60)
    print("Figure 3.4 Generation - Relative Error Analysis")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR.absolute()}")

    if DATA_DIR.exists():
        csv_files = list(DATA_DIR.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files")

    print("\nGenerating relative error figure...")
    fig = plot_relative_error()

    print("\nGeneration Complete!")