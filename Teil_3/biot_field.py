# -*- coding: utf-8 -*-
"""
Biot–Savart field model for an axially magnetized cylinder.
圆柱永磁体基于Biot–Savart定律的解析磁场模型。
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
import sys

# ==== load your Biot–Savart forward model from Punkte_N ====
base = Path(__file__).resolve().parent
punkte_path = base / "1_Punkte_N.txt"

if not punkte_path.exists():
    # try parent dir
    punkte_path = base.parent / "1_Punkte_N.txt"

if not punkte_path.exists():
    raise FileNotFoundError(f"Cannot find 1_Punkte_N.txt in {base} or {base.parent}")

ns = {}
with open(punkte_path, "r", encoding="utf-8") as f:
    exec(f.read(), ns, ns)

CylinderGeom = ns["CylinderGeom"]
Discretization = ns["Discretization"]
ModelConfig = ns["ModelConfig"]
ForwardModel = ns["ForwardModel"]

# global constant geometry for your magnet
GEOM_DEFAULT = CylinderGeom(Br=1.35, R=0.004, L=0.005)


def field_and_jacobian(r: np.ndarray, p: np.ndarray, u: np.ndarray, k: float, geom: dict | None = None):
    """
    Compute magnetic flux density B and analytic Jacobians using your Biot–Savart model.
    计算圆柱磁体在观测点r的磁感应强度B及解析雅可比。
    Parameters:
        r : ndarray (3,)       observation point [m]
        p : ndarray (3,)       magnet center [m]
        u : ndarray (3,)       direction vector (not necessarily normalized)
        k : float              strength scaling factor
        geom : dict|None       geometry dictionary, optional
    Returns:
        B  : ndarray (3,)      magnetic flux density [T]
        dBp: ndarray (3,3)     Jacobian ∂B/∂p
        dBu: ndarray (3,3)     Jacobian ∂B/∂u
        dBk: ndarray (3,1)     Jacobian ∂B/∂k
    """
    # Build ForwardModel each call (small overhead)
    geom_obj = GEOM_DEFAULT
    cfg = ModelConfig(geom=geom_obj, disc=Discretization())
    fwd = ForwardModel(cfg)

    # ForwardModel expects multiple sensors, so wrap single r
    sensors = np.atleast_2d(r)
    B_all, J_all = fwd.compute_B_and_J(sensors, p, u, k)

    # Extract first sensor only
    B = B_all[0]
    J = J_all[0]  # shape (3,7): [p(3), u(3), k(1)]
    dBp = J[:, 0:3]
    dBu = J[:, 3:6]
    dBk = J[:, 6:7]
    return B, dBp, dBu, dBk


# ==================== Verification / 验证 ====================
def verify_jacobians(r, p, u, k, eps=1e-7):
    """
    Verify analytic Jacobians against numerical finite differences.
    验证解析雅可比与数值有限差分。
    """
    print("=== Biot-Savart Jacobian Verification / Biot-Savart雅可比验证 ===\n")

    B, dB_dp_a, dB_du_a, dB_dk_a = field_and_jacobian(r, p, u, k)

    print(f"B field: {B}")
    print(f"||B||: {np.linalg.norm(B):.6e}\n")

    # Numerical dB/dp
    dB_dp_n = np.zeros((3, 3))
    for i in range(3):
        p_plus = p.copy()
        p_plus[i] += eps
        B_plus, _, _, _ = field_and_jacobian(r, p_plus, u, k)

        p_minus = p.copy()
        p_minus[i] -= eps
        B_minus, _, _, _ = field_and_jacobian(r, p_minus, u, k)

        dB_dp_n[:, i] = (B_plus - B_minus) / (2 * eps)

    # Numerical dB/du
    dB_du_n = np.zeros((3, 3))
    for i in range(3):
        u_plus = u.copy()
        u_plus[i] += eps
        B_plus, _, _, _ = field_and_jacobian(r, p, u_plus, k)

        u_minus = u.copy()
        u_minus[i] -= eps
        B_minus, _, _, _ = field_and_jacobian(r, p, u_minus, k)

        dB_du_n[:, i] = (B_plus - B_minus) / (2 * eps)

    # Numerical dB/dk
    B_kplus, _, _, _ = field_and_jacobian(r, p, u, k + eps)
    B_kminus, _, _, _ = field_and_jacobian(r, p, u, k - eps)
    dB_dk_n = ((B_kplus - B_kminus) / (2 * eps)).reshape(3, 1)

    # Compare
    print("dB/dp error (analytic vs numerical):")
    print(f"  Max abs diff: {np.max(np.abs(dB_dp_a - dB_dp_n)):.3e}")
    print(f"  Relative error: {np.linalg.norm(dB_dp_a - dB_dp_n) / (np.linalg.norm(dB_dp_n) + 1e-12):.3e}\n")

    print("dB/du error:")
    print(f"  Max abs diff: {np.max(np.abs(dB_du_a - dB_du_n)):.3e}")
    print(f"  Relative error: {np.linalg.norm(dB_du_a - dB_du_n) / (np.linalg.norm(dB_du_n) + 1e-12):.3e}\n")

    print("dB/dk error:")
    print(f"  Max abs diff: {np.max(np.abs(dB_dk_a - dB_dk_n)):.3e}")
    print(f"  Relative error: {np.linalg.norm(dB_dk_a - dB_dk_n) / (np.linalg.norm(dB_dk_n) + 1e-12):.3e}\n")


if __name__ == "__main__":
    r_test = np.array([0.05, 0.03, 0.04])
    p_test = np.array([0.01, 0.01, 0.012])
    u_test = np.array([0.1, 0.2, 0.9])
    k_test = 1.0  # Magnetization scaling factor

    verify_jacobians(r_test, p_test, u_test, k_test)
