from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Vacuum permeability [H/m]
# 真空磁导率 [H/m]
MU0 = 4 * np.pi * 1e-7


@dataclass
class CylinderParams:
    # Magnet parameters and query settings
    # 磁体参数与计算设置
    Br: float                 # remanent flux density [T] / 剩磁 [T]
    R: float                  # cylinder radius [m]     / 半径 [m]
    l: float                  # cylinder length [m]     / 高度/长度 [m]
    center: np.ndarray        # magnet center [m]       / 磁体中心 [m]
    u_dir: np.ndarray         # magnet axis direction   / 磁化方向
    N_theta: int = 180        # theta divisions         / 圆周离散数
    N_z: int = 90             # axial divisions         / 轴向离散数
    r2_eps: float = 1e-18     # singularity guard       / 奇异性阈值


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalize 3D vector. / 单位化三维向量。"""
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Direction vector cannot be zero.")
    return v / n


def _orthonormal_basis(z_axis: np.ndarray) -> np.ndarray:
    """Build orthonormal basis with given z-axis; columns are x,y,z.
    用给定z轴构造正交基；按列返回x,y,z。"""
    z = _unit(z_axis)
    if np.allclose(z, [0.0, 0.0, 1.0]) or np.allclose(z, [0.0, 0.0, -1.0]):
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = _unit(np.cross([0.0, 0.0, 1.0], z))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def magnetic_field_cylinder_points(params: CylinderParams,
                                   pos_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute B at multiple points on an axially magnetized cylinder via Biot–Savart.
    使用Biot–Savart计算轴向均匀磁化圆柱在多测点处的磁场。

    Args:
        params: cylinder and discretization parameters / 圆柱与离散参数
        pos_all: (N,3) query positions [m] / 待计算的N个点坐标[m]

    Returns:
        B_all_T: (N,3) Tesla components / 三分量结果(T)
        B_abs_mT: (N,) magnitudes in mT / 模长(mT)
    """
    Br, R, l = float(params.Br), float(params.R), float(params.l)
    center = np.asarray(params.center, float).reshape(3)
    Rmat = _orthonormal_basis(np.asarray(params.u_dir, float))
    Js = Br / MU0

    # Discretization / 网格离散
    theta_vals = np.linspace(0.0, 2.0*np.pi, params.N_theta, endpoint=False)
    z_vals = np.linspace(-l/2.0, l/2.0, params.N_z)
    dtheta = (theta_vals[1] - theta_vals[0]) if params.N_theta > 1 else 2*np.pi
    dz = (z_vals[1] - z_vals[0]) if params.N_z > 1 else l

    # Side surface parameterization / 侧壁参数化
    Theta, Z0 = np.meshgrid(theta_vals, z_vals, indexing="ij")
    x_s = R * np.cos(Theta)
    y_s = R * np.sin(Theta)
    z_s = Z0

    # dl along +theta / 电流元沿 +θ
    dl_local = R * dtheta * np.stack(
        (-np.sin(Theta), np.cos(Theta), np.zeros_like(Theta)), axis=-1
    )  # (Nθ,Nz,3)

    # Transform to global / 转到全局坐标
    r_local = np.stack((x_s, y_s, z_s), axis=-1)           # (Nθ,Nz,3)
    r_global = center + r_local @ Rmat.T                   # (Nθ,Nz,3)
    dl_global = dl_local @ Rmat.T                          # (Nθ,Nz,3)

    coeff = MU0 * Js / (4.0 * np.pi)

    P = np.atleast_2d(np.asarray(pos_all, float))          # (N,3)
    Np = P.shape[0]
    B = np.zeros((Np, 3), dtype=float)

    # Numerical integration over side surface / 在侧壁做数值积分
    for i in range(theta_vals.size):
        for j in range(z_vals.size):
            r_vecs = P - r_global[i, j]                    # (N,3)
            r2 = np.einsum("ij,ij->i", r_vecs, r_vecs)     # (N,)
            mask = r2 >= params.r2_eps
            if not np.any(mask):
                continue
            r3 = r2[mask] * np.sqrt(r2[mask])
            dB = coeff * np.cross(dl_global[i, j], r_vecs[mask]) / r3[:, None]
            B[mask] += dB * dz

    B_abs_mT = np.linalg.norm(B, axis=1) * 1e3
    return B, B_abs_mT


# ===================== Demo with 41 sensors / 用41个点演示 =====================

SENSOR_POSITIONS = np.array([
    [0.005 , 0.005 , 0.    ],
    [0.04  , 0.005 , 0.    ],
    [0.075 , 0.005 , 0.    ],
    [0.11  , 0.005 , 0.    ],
    [0.145 , 0.005 , 0.    ],
    [0.005 , 0.04  , 0.    ],
    [0.04  , 0.04  , 0.    ],
    [0.075 , 0.04  , 0.    ],
    [0.11  , 0.04  , 0.    ],
    [0.145 , 0.04  , 0.    ],
    [0.005 , 0.075 , 0.    ],
    [0.04  , 0.075 , 0.    ],
    [0.075 , 0.075 , 0.    ],
    [0.11  , 0.075 , 0.    ],
    [0.145 , 0.075 , 0.    ],
    [0.005 , 0.11  , 0.    ],
    [0.04  , 0.11  , 0.    ],
    [0.075 , 0.11  , 0.    ],
    [0.11  , 0.11  , 0.    ],
    [0.145 , 0.11  , 0.    ],
    [0.005 , 0.145 , 0.    ],
    [0.04  , 0.145 , 0.    ],
    [0.075 , 0.145 , 0.    ],
    [0.11  , 0.145 , 0.    ],
    [0.145 , 0.145 , 0.    ],
    [0.0225, 0.0225, 0.    ],
    [0.0575, 0.0225, 0.    ],
    [0.0925, 0.0225, 0.    ],
    [0.1275, 0.0225, 0.    ],
    [0.0225, 0.0575, 0.    ],
    [0.0575, 0.0575, 0.    ],
    [0.0925, 0.0575, 0.    ],
    [0.1275, 0.0575, 0.    ],
    [0.0225, 0.0925, 0.    ],
    [0.0575, 0.0925, 0.    ],
    [0.0925, 0.0925, 0.    ],
    [0.1275, 0.0925, 0.    ],
    [0.0225, 0.1275, 0.    ],
    [0.0575, 0.1275, 0.    ],
    [0.0925, 0.1275, 0.    ],
    [0.1275, 0.1275, 0.    ],
])


def _run_and_print():
    # Example magnet setup (same as your single-point code)
    # 示例磁体设置（与单点版本一致）
    params = CylinderParams(
        Br=1.35,
        R=4e-3,
        l=5e-3,
        center=np.array([10e-3, 10e-3, 12e-3]),
        u_dir=np.array([0.0, 0.0, 1.0]),
        N_theta=180,
        N_z=90,
    )

    B_vecs_T, B_abs_mT = magnetic_field_cylinder_points(params, SENSOR_POSITIONS)

    # Print per-sensor results / 逐点打印
    for i, (b, mag) in enumerate(zip(B_vecs_T, B_abs_mT), start=1):
        print(f"Sensor {i:2d}: B(T)=[{b[0]:.3e}, {b[1]:.3e}, {b[2]:.3e}]  |B|(mT)={mag:.3f}")

    # Optional: aggregated outputs can be computed outside if needed
    # 可选：聚合输出可在外部按需计算（简单平均/加权/核聚合）


if __name__ == "__main__":
    _run_and_print()
