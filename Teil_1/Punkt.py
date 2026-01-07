from __future__ import annotations
import numpy as np
from dataclasses import dataclass

MU0 = 4 * np.pi * 1e-7  # Vacuum permeability 真空磁导率 [H/m]

@dataclass
class CylinderParams:
    Br: float                 # Remanence flux density [T] 剩磁 [T]
    R: float                  # Cylinder radius [m] 圆柱半径 [m]
    l: float                  # Cylinder height/length [m] 圆柱高度/长度 [m]
    center: np.ndarray        # Magnet center position [m] 磁体中心位置 [m]
    u_dir: np.ndarray         # Magnet axis direction vector (will be normalized) 磁化方向向量（自动归一化）
    pos: np.ndarray           # Measurement point position [m] 测量点位置 [m]
    N_theta: int = 180        # Discretization along circumference 圆周离散点数
    N_z: int = 90             # Discretization along axis 轴向离散点数
    r2_eps: float = 1e-18     # Singularity protection threshold 奇异性保护阈值

def _unit(v: np.ndarray) -> np.ndarray:
    """Normalize a vector 单位化向量"""
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Direction vector cannot be zero. 方向向量不可为零")
    return v / n

def _orthonormal_basis(z_axis: np.ndarray) -> np.ndarray:
    """
    Build orthonormal basis from z_axis (columns: x, y, z) 从 z_axis 构造正交单位基（列向量：[x | y | z]）
    Handles parallel and anti-parallel cases with global Z 处理与全局 Z 轴平行/反平行的情况
    """
    z = _unit(z_axis)
    # If parallel or anti-parallel to global Z, choose x = [1,0,0] 若与全局Z平行或反平行，直接取x=[1,0,0]
    if np.allclose(z, [0.0, 0.0, 1.0]) or np.allclose(z, [0.0, 0.0, -1.0]):
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = _unit(np.cross([0.0, 0.0, 1.0], z))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)  # 3x3 matrix 三列分别是 x, y, z 基向量

def magnetic_field_cylinder_point(params: CylinderParams) -> tuple[np.ndarray, float]:
    """
    Compute B-field at a single point for an axially magnetized cylinder using Biot–Savart
    使用 Biot–Savart 定律计算轴向均匀磁化圆柱在单个测点的磁场
    Returns: (B_vec_T, B_mag_mT)
        B_vec_T: np.ndarray(3,) -> [Bx, By, Bz] in Tesla 磁场三分量 (T)
        B_mag_mT: float -> |B| in milliTesla 磁场模长 (mT)
    """
    # Parameters and preparation 参数与准备
    Br, R, l = float(params.Br), float(params.R), float(params.l)  # Extract geometry and Br 提取几何与剩磁
    center = np.asarray(params.center, dtype=float).reshape(3)     # Magnet center 磁体中心
    pos = np.asarray(params.pos, dtype=float).reshape(3)           # Measurement point 测量点
    Rmat = _orthonormal_basis(np.asarray(params.u_dir, dtype=float))  # Rotation matrix 旋转矩阵
    Js = Br / MU0  # Magnetization [A/m] 磁化强度 [A/m]

    # Discretization grid 离散积分网格
    theta_vals = np.linspace(0.0, 2.0*np.pi, params.N_theta, endpoint=False)
    z_vals = np.linspace(-l/2.0, l/2.0, params.N_z)
    dtheta = (theta_vals[1] - theta_vals[0]) if params.N_theta > 1 else 2*np.pi
    dz = (z_vals[1] - z_vals[0]) if params.N_z > 1 else l

    # Local coordinates of surface points 圆柱侧壁表面局部坐标
    Theta, Z0 = np.meshgrid(theta_vals, z_vals, indexing="ij")
    x_s = R * np.cos(Theta)
    y_s = R * np.sin(Theta)
    z_s = Z0

    # Local current elements (along +θ) 局部电流元（沿 +θ 方向）
    dl_local = R * dtheta * np.stack(
        (-np.sin(Theta), np.cos(Theta), np.zeros_like(Theta)), axis=-1
    )  # Shape: (Nθ, Nz, 3)

    # Transform to global coordinates 转换到全局坐标
    r_local = np.stack((x_s, y_s, z_s), axis=-1)             # Local positions 局部位置
    r_global = center + r_local @ Rmat.T                      # Global positions 全局位置
    dl_global = dl_local @ Rmat.T                             # Global current elements 全局电流元

    # === Apply Biot–Savart law ===
    # dB = μ0 * Js / (4π) * (dl × r_vec) / |r_vec|^3
    B = np.zeros(3, dtype=float)
    coeff = MU0 * Js / (4.0 * np.pi)

    for i in range(theta_vals.size):
        for j in range(z_vals.size):
            r_vec = pos - r_global[i, j]      # Vector from element to point 元素到测点的向量
            r2 = float(np.dot(r_vec, r_vec))
            if r2 < params.r2_eps:            # Skip near singularity 奇异性保护
                continue
            r3 = r2 * np.sqrt(r2)
            dB = coeff * np.cross(dl_global[i, j], r_vec) / r3  # Biot–Savart formula 毕奥-萨伐尔公式
            B += dB * dz

    B_mag_mT = float(np.linalg.norm(B) * 1e3)  # Magnitude in mT 磁场模长 [mT]
    return B, B_mag_mT

# ===================== Parameters & Output =====================

PARAMETERS = {
    "Br": 1.35,                                          # Remanence [T] 剩磁
    "R": 4e-3,                                           # Radius [m] 半径
    "l": 5e-3,                                           # Length [m] 高度/长度
    "center": np.array([10e-3, 10e-3, 12e-3]),           # Magnet center [m] 磁体中心
    "u_dir": np.array([0.0, 0.0, 1.0]),                # Axis direction (anti-parallel ok) 磁化方向（反向可）
    "pos": np.array([19e-3, 10e-3, 14e-3]),              # Measurement point [m] 测量点
    "N_theta": 180,                                      # Circumference divisions 圆周离散数
    "N_z": 90,                                           # Axial divisions 轴向离散数
}

def _run_and_print():
    p = CylinderParams(
        Br=PARAMETERS["Br"],
        R=PARAMETERS["R"],
        l=PARAMETERS["l"],
        center=PARAMETERS["center"],
        u_dir=PARAMETERS["u_dir"],
        pos=PARAMETERS["pos"],
        N_theta=PARAMETERS["N_theta"],
        N_z=PARAMETERS["N_z"],
    )
    B_vec_T, B_mag_mT = magnetic_field_cylinder_point(p)

    print("=== Result ===")
    print(f"B (T)        : [{B_vec_T[0]:.6e}, {B_vec_T[1]:.6e}, {B_vec_T[2]:.6e}]")
    print(f"|B| (mT)     : {B_mag_mT:.6f}")

if __name__ == "__main__":
    _run_and_print()
