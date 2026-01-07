# -*- coding: utf-8 -*-
"""
Magnetic dipole model with unified k parameter definition.
"""
from __future__ import annotations
import numpy as np

# Unified definition of constants
MU0 = 4 * np.pi * 1e-7  # [H/m]

# Unified geometric parameters
BR_DEFAULT = 1.35   # [T] 剩磁
R_DEFAULT = 0.004   # [m] 半径
L_DEFAULT = 0.005   # [m] 长度


def unit(v: np.ndarray) -> np.ndarray:
    """Normalize vector / 归一化向量"""
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError(f"Vector too small to normalize: ||v|| = {n:.3e}")
    return v / n


def dipole_B(r: np.ndarray, p: np.ndarray, u: np.ndarray, k: float,
             Br: float = BR_DEFAULT, R_mag: float = R_DEFAULT, L_mag: float = L_DEFAULT) -> np.ndarray:
    """
    Magnetic field of a dipole at observation point r.
    偶极子在观测点r的磁场。

    Parameters:
        r : (3,) observation point [m] / 观测点
        p : (3,) dipole center [m] / 偶极子中心
        u : (3,) direction vector (will be normalized) / 方向向量（会被归一化）
        k : float, dimensionless scaling factor (0-1)
              k=1.0 means full strength magnetization
              无量纲缩放因子，k=1.0表示全强度磁化
        Br : float, remanence [T] / 剩磁
        R_mag : float, magnet radius [m] / 磁体半径
        L_mag : float, magnet length [m] / 磁体长度

    Returns:
        B : (3,) magnetic flux density [T] / 磁通密度

    Formula: B = (μ₀/(4π)) * m * [3(û·R)R/r⁵ - û/r³]
    where m = (Br/μ₀) * V * k, V = π*R²*L
    """
    uhat = unit(u)
    R = r - p
    r2 = np.dot(R, R)

    if r2 < 1e-20:
        return np.zeros(3)

    r = np.sqrt(r2)
    r3 = r2 * r
    r5 = r2 * r3

    uhR = np.dot(uhat, R)

    # Calculate magnetic moment：m = M * V = (Br/μ₀) * V * k
    V = np.pi * R_mag**2 * L_mag
    M = (Br / MU0) * k  # 磁化强度
    m = M * V           # 磁矩

    # Dipole formula
    term1 = 3.0 * uhR * R / r5
    term2 = uhat / r3

    return (MU0 / (4 * np.pi)) * m * (term1 - term2)


def jacobians(r: np.ndarray, p: np.ndarray, u: np.ndarray, k: float,
              Br: float = BR_DEFAULT, R_mag: float = R_DEFAULT, L_mag: float = L_DEFAULT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytic Jacobians of dipole field.

    Returns:
        dB_dp : (3,3) Jacobian ∂B/∂p
        dB_du : (3,3) Jacobian ∂B/∂u
        dB_dk : (3,1) Jacobian ∂B/∂k

    Note: u will be normalized, Jacobian accounts for this.
    """
    uhat = unit(u)
    R = r - p
    r2 = np.dot(R, R)

    if r2 < 1e-20:
        return np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 1))

    r = np.sqrt(r2)
    r3 = r2 * r
    r5 = r2 * r3
    r7 = r2 * r5

    uhR = np.dot(uhat, R)

    # Calculate the effective intensity factor
    V = np.pi * R_mag**2 * L_mag
    M = (Br / MU0) * k
    m = M * V
    coeff = (MU0 / (4 * np.pi)) * m

    # --- dB/dk: linear B ∝ k ---
    B = dipole_B(r, p, u, k, Br, R_mag, L_mag)
    if abs(k) > 1e-20:
        dB_dk = (B / k).reshape(3, 1)
    else:
        dB_dk = np.zeros((3, 1))

    # --- dB/dp ---
    I = np.eye(3)
    RRt = np.outer(R, R)

    d_term1_dR = (3.0 / r5) * (np.outer(R, uhat) + np.outer(uhat, R)) \
                 + (3.0 * uhR / r5) * I \
                 - (15.0 * uhR / r7) * RRt

    d_term2_dR = (3.0 / r5) * np.outer(uhat, R)

    dB_dR = coeff * (d_term1_dR - d_term2_dR)
    dB_dp = -dB_dR

    # --- dB/du: Normalization u -> uhat ---
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-12:
        dB_du = np.zeros((3, 3))
    else:
        dB_duhat = coeff * ((3.0 / r5) * np.outer(R, R) - (1.0 / r3) * I)
        P_proj = (I - np.outer(uhat, uhat)) / u_norm
        dB_du = dB_duhat @ P_proj

    return dB_dp, dB_du, dB_dk


# ==================== Verification / 验证 ====================
def verify_jacobians(r, p, u, k, eps=1e-7,
                    Br=BR_DEFAULT, R_mag=R_DEFAULT, L_mag=L_DEFAULT):
    """
    Verify analytic Jacobians against numerical finite differences.
    验证解析雅可比与数值有限差分。
    """
    print("=== Dipole Jacobian Verification (Unified k) ===\n")

    dB_dp_a, dB_du_a, dB_dk_a = jacobians(r, p, u, k, Br, R_mag, L_mag)

    # Numerical dB/dp
    dB_dp_n = np.zeros((3, 3))
    for i in range(3):
        p_plus = p.copy()
        p_plus[i] += eps
        B_plus = dipole_B(r, p_plus, u, k, Br, R_mag, L_mag)

        p_minus = p.copy()
        p_minus[i] -= eps
        B_minus = dipole_B(r, p_minus, u, k, Br, R_mag, L_mag)

        dB_dp_n[:, i] = (B_plus - B_minus) / (2 * eps)

    # Numerical dB/du
    dB_du_n = np.zeros((3, 3))
    for i in range(3):
        u_plus = u.copy()
        u_plus[i] += eps
        B_plus = dipole_B(r, p, u_plus, k, Br, R_mag, L_mag)

        u_minus = u.copy()
        u_minus[i] -= eps
        B_minus = dipole_B(r, p, u_minus, k, Br, R_mag, L_mag)

        dB_du_n[:, i] = (B_plus - B_minus) / (2 * eps)

    # Numerical dB/dk
    B_kplus = dipole_B(r, p, u, k + eps, Br, R_mag, L_mag)
    B_kminus = dipole_B(r, p, u, k - eps, Br, R_mag, L_mag)
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
    k_test = 1.0

    B = dipole_B(r_test, p_test, u_test, k_test)
    print(f"B field: {B}")
    print(f"||B||: {np.linalg.norm(B):.6e}\n")

    verify_jacobians(r_test, p_test, u_test, k_test)