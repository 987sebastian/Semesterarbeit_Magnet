# Method / 方法

1. **Pose**: position `p∈R^3`, direction vector `u∈R^3` (normalized each evaluation), scalar strength `k`.
2. **Calibration**: for sensor i, measured `B_i` rotated by `R_i`, sensor position set to translation `t_i`.
3. **Models**:
   - Biot–Savart cylinder: insert your closed-form value and Jacobians.
   - Dipole: provided analytic B and Jacobians.
4. **Objective**: minimize `Σ_i || B_meas_i - B_model_i(p,u,k) ||^2` per row.
5. **Initialization**: `(p0,u0)` from columns `mag_*`, `k0` from median |B|.
6. **Outputs**: per-row estimates and residual statistics.
