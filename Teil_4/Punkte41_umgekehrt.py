# Basic Inverse Problem Solver
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# ===== Constants / 常量 =====
MU0 = 4 * np.pi * 1e-7  # Vacuum permeability [H/m] / 真空磁导率

# ===== Sensors / 传感器坐标（米）=====
SENSOR_POSITIONS = np.array([
    [0.005 , 0.005 , 0.    ], [0.04  , 0.005 , 0.    ], [0.075 , 0.005 , 0.    ],
    [0.11  , 0.005 , 0.    ], [0.145 , 0.005 , 0.    ], [0.005 , 0.04  , 0.    ],
    [0.04  , 0.04  , 0.    ], [0.075 , 0.04  , 0.    ], [0.11  , 0.04  , 0.    ],
    [0.145 , 0.04  , 0.    ], [0.005 , 0.075 , 0.    ], [0.04  , 0.075 , 0.    ],
    [0.075 , 0.075 , 0.    ], [0.11  , 0.075 , 0.    ], [0.145 , 0.075 , 0.    ],
    [0.005 , 0.11  , 0.    ], [0.04  , 0.11  , 0.    ], [0.075 , 0.11  , 0.    ],
    [0.11  , 0.11  , 0.    ], [0.145 , 0.11  , 0.    ], [0.005 , 0.145 , 0.    ],
    [0.04  , 0.145 , 0.    ], [0.075 , 0.145 , 0.    ], [0.11  , 0.145 , 0.    ],
    [0.145 , 0.145 , 0.    ], [0.0225, 0.0225, 0.    ], [0.0575, 0.0225, 0.    ],
    [0.0925, 0.0225, 0.    ], [0.1275, 0.0225, 0.    ], [0.0225, 0.0575, 0.    ],
    [0.0575, 0.0575, 0.    ], [0.0925, 0.0575, 0.    ], [0.1275, 0.0575, 0.    ],
    [0.0225, 0.0925, 0.    ], [0.0575, 0.0925, 0.    ], [0.0925, 0.0925, 0.    ],
    [0.1275, 0.0925, 0.    ], [0.0225, 0.1275, 0.    ], [0.0575, 0.1275, 0.    ],
    [0.0925, 0.1275, 0.    ], [0.1275, 0.1275, 0.    ],
])

# ===== Example measured B (Tesla) / 示例观测（特斯拉）=====
B_MEAS_T = np.array([
 [8.410e-03, 8.410e-03, 1.275e-02], [-8.378e-04, 1.396e-04, -4.803e-04],
 [-5.217e-05, 4.013e-06, -8.799e-05], [-9.809e-06, 4.905e-07, -2.649e-05],
 [-3.005e-06, 1.113e-07, -1.110e-05], [1.396e-04, -8.378e-04, -4.803e-04],
 [-1.864e-04, -1.864e-04, -2.581e-04], [-3.309e-05, -1.527e-05, -6.819e-05],
 [-7.978e-06, -2.393e-06, -2.348e-05], [-2.675e-06, -5.944e-07, -1.036e-05],
 [4.013e-06, -5.217e-05, -8.799e-05], [-1.527e-05, -3.309e-05, -6.819e-05],
 [-9.717e-06, -9.717e-06, -3.383e-05], [-4.129e-06, -2.684e-06, -1.597e-05],
 [-1.796e-06, -8.648e-07, -8.185e-06], [4.905e-07, -9.809e-06, -2.649e-05],
 [-2.393e-06, -7.978e-06, -2.348e-05], [-2.684e-06, -4.129e-06, -1.597e-05],
 [-1.773e-06, -1.773e-06, -9.701e-06], [-1.016e-06, -7.529e-07, -5.840e-06],
 [1.113e-07, -3.005e-06, -1.110e-05], [-5.944e-07, -2.675e-06, -1.036e-05],
 [-8.648e-07, -1.796e-06, -8.185e-06], [-7.529e-07, -1.016e-06, -5.840e-06],
 [-5.377e-07, -5.377e-07, -3.999e-06], [-2.920e-03, -2.920e-03, -1.364e-05],
 [-1.484e-04, -3.904e-05, -1.829e-04], [-1.985e-05, -3.008e-06, -4.451e-05],
 [-5.080e-06, -5.405e-07, -1.641e-05], [-3.904e-05, -1.484e-04, -1.829e-04],
 [-3.296e-05, -3.296e-05, -8.114e-05], [-1.038e-05, -5.976e-06, -3.061e-05],
 [-3.590e-06, -1.451e-06, -1.338e-05], [-3.008e-06, -1.985e-05, -4.451e-05],
 [-5.976e-06, -1.038e-05, -3.061e-05], [-3.799e-06, -3.799e-06, -1.702e-05],
 [-1.933e-06, -1.357e-06, -9.281e-06], [-5.405e-07, -5.080e-06, -1.641e-05],
 [-1.451e-06, -3.590e-06, -1.338e-05], [-1.357e-06, -1.933e-06, -9.281e-06],
 [-9.343e-07, -9.343e-07, -6.032e-06],
])

# ===== Geometry & discretization / 几何与离散 =====
@dataclass
class CylinderGeom:
    Br: float  # remanence [T] / 剩磁
    R: float   # radius [m]   / 半径
    l: float   # length [m]   / 高度

@dataclass
class Discretization:
    N_theta: int = 180
    N_z: int = 90
    r2_eps: float = 1e-18

# ===== Utils / 工具 =====
def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).reshape(3)
    n = np.linalg.norm(v)
    if n == 0: raise ValueError("zero vector")
    return v / n

def udir_from_theta_phi(theta: float, phi: float) -> np.ndarray:
    # θ: polar from +Z, φ: azimuth from +X / θ自+Z极角, φ自+X方位角
    st, ct = np.sin(theta), np.cos(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    return np.array([cp*st, sp*st, ct], dtype=float)

def _basis_from_z(z_axis: np.ndarray) -> np.ndarray:
    # ONB with given z; columns x,y,z / 用z轴构造正交基; 按列x,y,z
    z = _unit(z_axis)
    if np.allclose(z, [0,0,1]) or np.allclose(z, [0,0,-1]):
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = _unit(np.cross([0,0,1], z))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)

# ===== Precompute local surface grid / 预计算局部侧壁网格 =====
# 保持与原网格一致（不降采样）
DISC = Discretization()  # 可按需修改全局离散，但此处不变
theta = np.linspace(0, 2*np.pi, DISC.N_theta, endpoint=False)
zvals = np.linspace(-0.004/2.0, 0.004/2.0, DISC.N_z)  # 仅占位初始化，真实由几何给出l
dtheta = (theta[1]-theta[0]) if DISC.N_theta>1 else 2*np.pi
dz = (zvals[1]-zvals[0]) if DISC.N_z>1 else 0.004

Theta, Z0 = np.meshgrid(theta, zvals, indexing='ij')
# r_local, dl_local 将在 set_geom_cache 时依几何参数更新
_R_LOCAL = None   # (M,3)
_DL_LOCAL = None  # (M,3)

def set_geom_cache(geom: CylinderGeom):
    """Update local surface cache with current geometry. / 依据几何更新侧壁局部缓存。"""
    global _R_LOCAL, _DL_LOCAL, zvals, dz, Theta
    # 以当前全局 Theta、zvals 生成侧壁参数
    x_s = geom.R * np.cos(Theta)
    y_s = geom.R * np.sin(Theta)
    zvals = np.linspace(-geom.l/2.0, geom.l/2.0, DISC.N_z)
    Z0 = np.meshgrid(theta, zvals, indexing='ij')[1]
    dz = (zvals[1]-zvals[0]) if DISC.N_z>1 else geom.l
    # 局部坐标与电流元
    r_local = np.stack((x_s, y_s, Z0), axis=-1)                 # (Nθ,Nz,3)
    dl_local = geom.R * dtheta * np.stack(
        (-np.sin(Theta), np.cos(Theta), np.zeros_like(Theta)), axis=-1
    )                                                           # (Nθ,Nz,3)
    _R_LOCAL = r_local.reshape(-1, 3).copy()                    # (M,3)
    _DL_LOCAL = dl_local.reshape(-1, 3).copy()                  # (M,3)

# ===== Vectorized forward model / 矢量化前向模型 =====
def B_cylinder_side(center: np.ndarray,
                    u_dir: np.ndarray,
                    geom: CylinderGeom,
                    query_pts: np.ndarray,
                    Br_scale: float = 1.0) -> np.ndarray:
    """Biot–Savart on cylinder side; vectorized. 返回(N,3) Tesla。"""
    assert _R_LOCAL is not None and _DL_LOCAL is not None, "call set_geom_cache(geom) first"
    Br_eff = geom.Br * Br_scale
    Js = Br_eff / MU0
    Rmat = _basis_from_z(u_dir)

    el_pos = (_R_LOCAL @ Rmat.T) + center.reshape(1,3)  # (M,3)
    el_dl  =  _DL_LOCAL @ Rmat.T                       # (M,3)

    P = np.atleast_2d(np.asarray(query_pts, float))    # (N,3)
    # r_vec: from element to sensor
    r_vec = P[None, :, :] - el_pos[:, None, :]         # (M,N,3)
    r2 = np.einsum('mni,mni->mn', r_vec, r_vec)        # (M,N)
    # avoid singularities
    mask = r2 >= DISC.r2_eps
    r3 = np.where(mask, r2 * np.sqrt(r2), np.inf)      # (M,N)

    cross = np.cross(el_dl[:, None, :], r_vec, axis=-1)  # (M,N,3)
    coeff = MU0 * Js / (4*np.pi)
    dB = coeff * (cross / r3[..., None])               # (M,N,3)
    B = dz * np.sum(dB, axis=0)                        # (N,3)
    return B

# ===== LM inverse (6 params) / 反演LM（6参数）=====
@dataclass
class EstimationResult:
    center: np.ndarray
    u_dir: np.ndarray
    scale: float
    iters: int
    success: bool
    rmse: float

def lm_fit_6d(B_meas: np.ndarray,
              geom: CylinderGeom,
              sensors: np.ndarray,
              p0_center: np.ndarray,
              p0_theta_phi: Tuple[float,float],
              p0_scale: float = 1.0,
              max_iter: int = 10,
              lambda0: float = 1e-2) -> EstimationResult:
    """Levenberg–Marquardt for [cx,cy,cz, theta, phi, scale] with tuned steps.
    稳健LM：拟合[中心xyz, 方向θφ, Br缩放]。"""
    sensors = np.asarray(sensors, float)
    N = sensors.shape[0]
    assert B_meas.shape == (N,3)

    p = np.zeros(6, float)
    p[:3] = p0_center.reshape(3)
    p[3:5] = np.array(p0_theta_phi, float)
    p[5] = float(p0_scale)

    CENTER_H = 1e-3  # m
    ANGLE_H  = 1e-2  # rad
    SCALE_H  = 1e-3
    TH_EPS   = 1e-6  # clamp θ

    def resid(pv: np.ndarray) -> np.ndarray:
        c  = pv[:3]
        th = float(np.clip(pv[3], TH_EPS, np.pi-TH_EPS))
        ph = float(pv[4])
        s  = float(pv[5])
        u  = udir_from_theta_phi(th, ph)
        Bp = B_cylinder_side(c, u, geom, sensors, Br_scale=s)
        return (Bp - B_meas).reshape(-1)  # (3N,)

    r = resid(p)
    cost = float(np.sqrt(np.mean(r**2)))
    lam = lambda0; last_cost = np.inf; success = False

    for it in range(1, max_iter+1):
        # Numerical Jacobian
        J = np.zeros((3*N, 6), float)
        for j in range(6):
            dp = np.zeros_like(p)
            step = CENTER_H if j in (0,1,2) else (ANGLE_H if j in (3,4) else SCALE_H)
            dp[j] = step
            rp = resid(p + dp)
            J[:, j] = (rp - r) / step

        JTJ = J.T @ J
        g   = J.T @ r
        H   = JTJ + lam * np.diag(np.diag(JTJ))

        try:
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        p_trial = p + step
        p_trial[3] = float(np.clip(p_trial[3], TH_EPS, np.pi-TH_EPS))

        r_trial = resid(p_trial)
        cost_trial = float(np.sqrt(np.mean(r_trial**2)))

        if cost_trial < cost:
            p, r, cost = p_trial, r_trial, cost_trial
            lam *= 0.3
            # early stop: tiny improvement
            if abs(last_cost - cost) < 1e-6 * (1.0 + cost):
                success = True
                break
            last_cost = cost
        else:
            lam *= 3.0

    th = float(np.clip(p[3], TH_EPS, np.pi-TH_EPS))
    ph = float(p[4])
    u_est = udir_from_theta_phi(th, ph)
    return EstimationResult(center=p[:3], u_dir=u_est, scale=float(p[5]),
                            iters=it, success=success, rmse=cost)

# ===== Main / 主程序 =====
if __name__ == "__main__":
    # Geometry (unchanged sampling grid) / 几何（采样网格不变）
    geom = CylinderGeom(Br=1.12, R=4e-3, l=5e-3)
    set_geom_cache(geom)  # 预计算局部侧壁

    # Initial guess (reuse previous frame in real-time) / 初值（实时用上一帧）
    p0_center = np.array([10e-3, 10e-3, 12e-3])
    p0_theta_phi = (1.0e-2, 0.0)
    p0_scale = 1.0

    est = lm_fit_6d(B_MEAS_T, geom, SENSOR_POSITIONS,
                    p0_center, p0_theta_phi, p0_scale,
                    max_iter=10, lambda0=1e-2)

    # ===== Formatted output (3 decimals) / 三位小数输出 =====
    print(f"Success: {est.success}, iters: {est.iters}, RMSE (mT): {est.rmse*1e3:.3f}")
    print("Center [m]: [{:.3f}, {:.3f}, {:.3f}]".format(est.center[0], est.center[1], est.center[2]))
    print("u_dir     : [{:.3f}, {:.3f}, {:.3f}]".format(est.u_dir[0], est.u_dir[1], est.u_dir[2]))
    print("Br_scale  : {:.3f}".format(est.scale))
