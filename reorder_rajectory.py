"""
Reorder observation data to create a continuous trajectory.
Uses nearest-neighbor algorithm to minimize frame-to-frame jumps.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path


def calculate_transition_cost(pos1, dir1, pos2, dir2, w_pos=1.0, w_dir=0.5):
    """
    Calculate cost of transitioning from state 1 to state 2.

    Args:
        pos1, pos2: 3D positions [m]
        dir1, dir2: 3D directions (normalized)
        w_pos: weight for position distance [m]
        w_dir: weight for direction change [rad]

    Returns:
        Combined cost
    """
    # Position distance
    pos_dist = np.linalg.norm(pos2 - pos1)

    # Direction angle (ensure normalized)
    d1 = dir1 / np.linalg.norm(dir1)
    d2 = dir2 / np.linalg.norm(dir2)
    cos_angle = np.clip(np.dot(d1, d2), -1.0, 1.0)
    angle_diff = np.arccos(cos_angle)

    # Combined cost (position in meters, angle in radians)
    cost = w_pos * pos_dist + w_dir * angle_diff
    return cost


def greedy_path_ordering(positions, directions, start_idx=0):
    """
    Greedy nearest-neighbor algorithm to create smooth trajectory.

    Args:
        positions: (N, 3) array of positions
        directions: (N, 3) array of directions
        start_idx: index to start from (default: 0)

    Returns:
        sorted_indices: list of indices creating smooth path
    """
    n_points = len(positions)
    sorted_indices = [start_idx]
    remaining = set(range(n_points)) - {start_idx}

    print(f"Reordering {n_points} frames into continuous trajectory...")
    print(f"Starting from frame {start_idx}")

    while remaining:
        current_idx = sorted_indices[-1]
        current_pos = positions[current_idx]
        current_dir = directions[current_idx]

        # Find nearest neighbor among remaining points
        min_cost = float('inf')
        best_idx = None

        for idx in remaining:
            cost = calculate_transition_cost(
                current_pos, current_dir,
                positions[idx], directions[idx]
            )
            if cost < min_cost:
                min_cost = cost
                best_idx = idx

        sorted_indices.append(best_idx)
        remaining.remove(best_idx)

        if len(sorted_indices) % 50 == 0:
            print(f"  Progress: {len(sorted_indices)}/{n_points} frames ordered")

    return sorted_indices


def analyze_trajectory(positions, directions, indices, name="Trajectory"):
    """Analyze and print trajectory statistics."""
    print(f"\n=== {name} Statistics ===")

    position_jumps = []
    angle_changes = []

    for i in range(1, len(indices)):
        idx_prev = indices[i - 1]
        idx_curr = indices[i]

        # Position jump
        dp = np.linalg.norm(positions[idx_curr] - positions[idx_prev]) * 1000  # to mm
        position_jumps.append(dp)

        # Angle change
        d1 = directions[idx_prev] / np.linalg.norm(directions[idx_prev])
        d2 = directions[idx_curr] / np.linalg.norm(directions[idx_curr])
        cos_angle = np.clip(np.dot(d1, d2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        angle_changes.append(angle_deg)

    position_jumps = np.array(position_jumps)
    angle_changes = np.array(angle_changes)

    print(f"Position jumps (mm):")
    print(f"  Mean:   {np.mean(position_jumps):8.2f} mm")
    print(f"  Median: {np.median(position_jumps):8.2f} mm")
    print(f"  Min:    {np.min(position_jumps):8.2f} mm")
    print(f"  Max:    {np.max(position_jumps):8.2f} mm")
    print(f"  Std:    {np.std(position_jumps):8.2f} mm")
    print(f"  P95:    {np.percentile(position_jumps, 95):8.2f} mm")

    print(f"\nAngle changes (degrees):")
    print(f"  Mean:   {np.mean(angle_changes):8.2f}°")
    print(f"  Median: {np.median(angle_changes):8.2f}°")
    print(f"  Min:    {np.min(angle_changes):8.2f}°")
    print(f"  Max:    {np.max(angle_changes):8.2f}°")
    print(f"  Std:    {np.std(angle_changes):8.2f}°")
    print(f"  P95:    {np.percentile(angle_changes, 95):8.2f}°")

    # Count frames within tracking limits
    good_pos = np.sum(position_jumps <= 10.0)
    good_angle = np.sum(angle_changes <= 5.0)

    print(f"\nFrames within tracking limits:")
    print(f"  Position ≤10mm: {good_pos}/{len(position_jumps)} ({100 * good_pos / len(position_jumps):.1f}%)")
    print(f"  Angle ≤5°:      {good_angle}/{len(angle_changes)} ({100 * good_angle / len(angle_changes):.1f}%)")

    return position_jumps, angle_changes


def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # File paths - try common locations
        candidates = [
            "data/Observational_data_new.csv",
            "Observational_data_new.csv",
            "../data/Observational_data_new.csv",
        ]
        input_file = None
        for candidate in candidates:
            if Path(candidate).exists():
                input_file = candidate
                break

        if input_file is None:
            print(f"ERROR: Cannot find input file!")
            print("Tried locations:")
            for c in candidates:
                print(f"  - {c}")
            print("\nUsage: python3 reorder_trajectory.py <input_file>")
            sys.exit(1)

    if not Path(input_file).exists():
        print(f"ERROR: File not found: {input_file}")
        sys.exit(1)

    # Output file in same directory as input
    input_path = Path(input_file)
    output_file = str(input_path.parent / "Observational_data_sequential.csv")

    print("=" * 70)
    print("  Trajectory Reordering for Real-Time Tracking Simulation")
    print("=" * 70)
    print(f"\nInput:  {input_file}")
    print(f"Output: {output_file}")

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} frames with {len(df.columns)} columns")

    # Extract positions and directions (last 6 columns)
    positions = df.iloc[:, -6:-3].values  # mag_x0, mag_y0, mag_z0
    directions = df.iloc[:, -3:].values  # mag_mx0, mag_my0, mag_mz0

    print(f"\nPosition range:")
    print(f"  X: [{np.min(positions[:, 0]):.4f}, {np.max(positions[:, 0]):.4f}] m")
    print(f"  Y: [{np.min(positions[:, 1]):.4f}, {np.max(positions[:, 1]):.4f}] m")
    print(f"  Z: [{np.min(positions[:, 2]):.4f}, {np.max(positions[:, 2]):.4f}] m")

    # Analyze original trajectory
    original_indices = list(range(len(df)))
    analyze_trajectory(positions, directions, original_indices, "Original Order")

    # Reorder using greedy nearest-neighbor
    print("\n" + "=" * 70)
    sorted_indices = greedy_path_ordering(positions, directions, start_idx=0)

    # Analyze reordered trajectory
    analyze_trajectory(positions, directions, sorted_indices, "Reordered Trajectory")

    # Save reordered data
    print("\n" + "=" * 70)
    print("Saving reordered data...")
    df_sorted = df.iloc[sorted_indices].reset_index(drop=True)

    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_sorted.to_csv(output_file, index=False)
    print(f"✓ Saved to: {output_file}")

    # Save mapping file (original_index -> new_index)
    mapping_file = output_file.replace('.csv', '_mapping.txt')
    with open(mapping_file, 'w') as f:
        f.write("# Mapping: new_row_index -> original_row_index\n")
        for new_idx, orig_idx in enumerate(sorted_indices):
            f.write(f"{new_idx},{orig_idx}\n")
    print(f"✓ Saved mapping to: {mapping_file}")

    print("\n" + "=" * 70)
    print("SUCCESS! Trajectory reordering complete.")
    print("\nNext steps:")
    print(f"1. Update config.hpp to use: {output_file}")
    print("2. Run your tracking algorithm with motion prediction enabled")
    print("3. Expect much better results with continuous trajectory!")
    print("=" * 70)


if __name__ == "__main__":

    main()
