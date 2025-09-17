from dataclasses import dataclass
import numpy as np
from .Discretization import Discretization
from .ForwardModel import ModelConfig, ForwardModel


@dataclass
class Pose:
    center: np.ndarray
    u_dir: np.ndarray
    scale: float
    rmse: float


def estimate_pose(
        sensors_m: np.ndarray,
        B_meas_T: np.ndarray,
        geom: CylinderGeom,
        u_init: np.ndarray,
        c_init: np.ndarray,
        scale_init: float = 1.0,
        max_iter: int = 15,
        tol_T: float = 4e-8
) -> Pose:
    """
    Estimate pose by Gauss–Newton with analytic Jacobian.
    使用解析雅可比的高斯–牛顿估计位姿。
    Inputs:
      sensors_m: (N,3) sensor positions [m]
      B_meas_T: (N,3) measured B [T]
      u_init: (3,) initial direction vector (need not be unit) / 初始方向
      c_init: (3,) initial center / 初始中心
    """
    cfg = ModelConfig(geom=geom, disc=Discretization())
    fwd = ForwardModel(cfg)

    p = np.zeros(7)
    p[0:3] = c_init
    p[3:6] = u_init
    p[6] = scale_init

    for it in range(max_iter):
        c = p[0:3]
        u_raw = p[3:6]
        scale = p[6]
        B_pred, J = fwd.compute_B_and_J(sensors_m, c, u_raw, scale)  # B:(N,3), J:(N,3,7)
        r = (B_pred - B_meas_T).reshape(-1)  # residual / 残差
        # assemble big Jacobian / 堆叠雅可比
        Jbig = J.reshape(-1, 7)
        # solve normal equations / 解法向方程
        # regularize lightly on u_raw to keep scale of components / 对u加轻微正则
        lam = 1e-12
        H = Jbig.T @ Jbig + lam * np.diag([0, 0, 0, 1, 1, 1, 0])
        g = Jbig.T @ r
        dp = -np.linalg.solve(H, g)
        p += dp
        rmse = float(np.sqrt(np.mean(r * r)))
        if rmse < tol_T:
            break

    # pack result
    u_hat = p[3:6] / np.linalg.norm(p[3:6])
    return Pose(center=p[0:3].copy(), u_dir=u_hat, scale=float(p[6]), rmse=rmse)
