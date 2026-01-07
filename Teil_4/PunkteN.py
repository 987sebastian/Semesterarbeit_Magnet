# Inverse problem of a general N-sensor (with analytical Jacobian)
"""
Magnet tracking with Biot–Savart surface-charge model and analytic Jacobian.
基于表面磁荷的Biot–Savart模型，提供解析雅可比。方向由单位向量 u 表示。
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

MU0 = 4*np.pi*1e-7  # vacuum permeability / 真空磁导率

# -------------------- Geometry and discretization / 几何与离散 --------------------
@dataclass
class CylinderGeom:
    Br: float  # remanence [T] / 剩磁
    R: float   # radius [m] / 半径
    l: float   # length [m] / 长度

@dataclass
class DiscGrid:
    Nr: int = 16      # radial divisions / 径向剖分
    Nth: int = 48     # angular divisions / 角向剖分

@dataclass
class Discretization:
    disc: DiscGrid = None
    def __post_init__(self):
        if self.disc is None:
            self.disc = DiscGrid()

# Precomputed local mesh for the two end disks / 两端圆盘的本地网格
class EndDisks:
    def __init__(self, geom: CylinderGeom, disc: DiscGrid):
        self.geom = geom
        self.disc = disc
        self._build()

    def _build(self):
        R = self.geom.R
        Nr, Nth = self.disc.Nr, self.disc.Nth
        # Polar cell centers and areas / 极坐标单元中心与面积
        r_edges = np.linspace(0.0, R, Nr+1)
        th_edges = np.linspace(0.0, 2*np.pi, Nth+1)
        rc = 0.5*(r_edges[:-1] + r_edges[1:])
        thc = 0.5*(th_edges[:-1] + th_edges[1:])
        dr = np.diff(r_edges)[:,None]
        dth = np.diff(th_edges)[None,:]
        # cell area in polar: dS = r dr dθ / 单元面积
        area = (rc[:,None] * dr) @ (np.ones((1,Nth))) * dth
        # centers in Cartesian local frame / 局部笛卡尔坐标的圆盘网格点
        x = (rc[:,None] * np.cos(thc)[None,:]).reshape(-1)
        y = (rc[:,None] * np.sin(thc)[None,:]).reshape(-1)
        dS = area.reshape(-1)
        # Two disks at z = ±l/2 / 两端圆盘
        z_plus = np.full_like(x, +self.geom.l/2)
        z_minus = np.full_like(x, -self.geom.l/2)
        # Stack: (Npts,3) for each end / 每端的点
        self.p_local_plus = np.stack([x, y, z_plus], axis=1)
        self.p_local_minus = np.stack([x, y, z_minus], axis=1)
        self.dS = dS
        # Local normals / 局部法向
        self.n_local_plus = np.tile(np.array([0.0,0.0, 1.0]), (x.size,1))
        self.n_local_minus= np.tile(np.array([0.0,0.0,-1.0]), (x.size,1))
        # Surface magnetic charge densities at ends: ±M / 端面磁荷密度 ±M
        # In local frame, magnetization is along +z (unit), magnitude M = Br/μ0*scale / 当地坐标下磁化沿+z
        # σ_m = M · n_local = ± M / 磁荷密度
        # scale will multiply later / 缩放稍后乘
        self.sigma_sign_plus = +1.0
        self.sigma_sign_minus= -1.0

# -------------------- Orientation frame utilities / 方向正交基工具 --------------------
def sph_basis_from_u(u_hat: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray,float,float]:
    """
    Build an orthonormal frame (e_phi, -e_theta, u_hat), plus spherical angles (alpha, beta).
    用单位方向向量构造正交基，并返回球坐标角 α(极角), β(方位角)。
    """
    ux, uy, uz = u_hat
    # angles from unit vector / 由单位向量求角
    alpha = np.arccos(np.clip(uz, -1.0, 1.0))  # polar angle from +z / 与+z的极角
    beta = np.arctan2(uy, ux)                  # azimuth / 方位角
    # spherical basis / 球坐标基
    e_r = u_hat
    e_theta = np.array([np.cos(alpha)*np.cos(beta),
                        np.cos(alpha)*np.sin(beta),
                        -np.sin(alpha)])
    e_phi = np.array([-np.sin(beta), np.cos(beta), 0.0])
    # Orthonormal frame for disk local xyz -> world / 将磁体局部xyz映射到世界
    # R = [e1 e2 e3] with e1=e_phi, e2=-e_theta, e3=e_r
    return e_phi, -e_theta, e_r, alpha, beta

def d_basis_dalpha(alpha: float, beta: float) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Derivatives of (e1,e2,e3) wrt alpha. / 对α的导数"""
    # from spherical basis derivatives / 基于球基导数
    e_r = np.array([np.sin(alpha)*np.cos(beta),
                    np.sin(alpha)*np.sin(beta),
                    np.cos(alpha)])
    e_theta = np.array([np.cos(alpha)*np.cos(beta),
                        np.cos(alpha)*np.sin(beta),
                        -np.sin(alpha)])
    e_phi = np.array([-np.sin(beta), np.cos(beta), 0.0])
    # derivatives:
    de_r_dalpha = e_theta
    de_theta_dalpha = -e_r
    de_phi_dalpha = np.array([0.0,0.0,0.0])
    # map to (e1,e2,e3) = (e_phi, -e_theta, e_r)
    de1 = de_phi_dalpha
    de2 = -de_theta_dalpha
    de3 = de_r_dalpha
    return de1, de2, de3

def d_basis_dbeta(alpha: float, beta: float) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Derivatives of (e1,e2,e3) wrt beta. / 对β的导数"""
    e_r = np.array([np.sin(alpha)*np.cos(beta),
                    np.sin(alpha)*np.sin(beta),
                    np.cos(alpha)])
    e_theta = np.array([np.cos(alpha)*np.cos(beta),
                        np.cos(alpha)*np.sin(beta),
                        -np.sin(alpha)])
    e_phi = np.array([-np.sin(beta), np.cos(beta), 0.0])
    # derivatives:
    de_r_dbeta = np.sin(alpha)*e_phi
    de_theta_dbeta = np.cos(alpha)*e_phi
    de_phi_dbeta = -np.sin(alpha)*e_r - np.cos(alpha)*e_theta
    # map to (e1,e2,e3)
    de1 = de_phi_dbeta
    de2 = -de_theta_dbeta
    de3 = de_r_dbeta
    return de1, de2, de3

def uhat_and_chain(u_raw: np.ndarray) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """
    Normalize u_raw to u_hat and return alpha,beta and Jacobians dα/du_raw, dβ/du_raw.
    将原始方向向量归一化，并给出 α,β 以及对 u_raw 的链式导数。
    """
    nr = np.linalg.norm(u_raw)
    if nr < 1e-12:
        raise ValueError("u vector too small / 方向向量范数过小")
    u_hat = u_raw / nr
    ux, uy, uz = u_hat
    alpha = np.arccos(np.clip(uz, -1.0, 1.0))
    beta = np.arctan2(uy, ux)
    # d u_hat / d u_raw = (I - u_hat u_hat^T)/||u_raw||
    I = np.eye(3)
    d_uhat_duraw = (I - np.outer(u_hat, u_hat))/nr
    # ∂α/∂u_hat = -1/sinα * ∂uz/∂u_hat = [0,0,-1]/sinα
    sa = max(np.sin(alpha), 1e-12)
    dalpha_duhat = np.array([0.0, 0.0, -1.0]) / sa
    # ∂β/∂u_hat = 1/sin^2α * [ -uy, ux, 0 ]
    s2 = max(sa*sa, 1e-12)
    dbeta_duhat = np.array([-uy, ux, 0.0]) / s2
    # chain to u_raw
    dalpha_duraw = dalpha_duhat @ d_uhat_duraw
    dbeta_duraw = dbeta_duhat @ d_uhat_duraw
    return u_hat, alpha, beta, dalpha_duraw, dbeta_duraw

# kernel and its Jacobian wrt r / 核函数及对r的雅可比
def kernel_and_J(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    f(r) = r / |r|^3 ;  df/dr = (I*|r|^2 - 3 r r^T)/|r|^5
    """
    rx, ry, rz = r
    r2 = rx*rx + ry*ry + rz*rz
    inv_r = 1.0/np.sqrt(r2 + 1e-30)
    inv_r3 = inv_r**3
    f = r * inv_r3
    # Jacobian
    I = np.eye(3)
    rrT = np.outer(r, r)
    J = (I * r2 - 3.0 * rrT) * (inv_r**5)
    return f, J

# -------------------- Forward model / 正演模型 --------------------
@dataclass
class ModelConfig:
    geom: CylinderGeom
    disc: Discretization

class ForwardModel:
    """
    B field at sensors from a uniformly magnetized cylinder using surface magnetic charges on end disks.
    采用端面表面磁荷的模型计算传感器处磁感应强度。
    """
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.mesh = EndDisks(cfg.geom, cfg.disc.disc)

    def compute_B_and_J(self,
                        sensors_m: np.ndarray,
                        center_m: np.ndarray,
                        u_raw: np.ndarray,
                        scale: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return B (N,3) and analytic Jacobian J wrt params [cx,cy,cz, ux,uy,uz, scale].
        计算B与参数[cx,cy,cz, ux,uy,uz, scale]的解析雅可比。
        """
        geom = self.cfg.geom
        M0 = geom.Br/MU0  # magnetization magnitude at scale=1 / 标称磁化强度
        # orientation frame
        u_hat, alpha, beta, dalpha_duraw, dbeta_duraw = uhat_and_chain(u_raw)
        e1, e2, e3, alpha, beta = sph_basis_from_u(u_hat)
        # basis derivatives
        de1_da, de2_da, de3_da = d_basis_dalpha(alpha, beta)
        de1_db, de2_db, de3_db = d_basis_dbeta(alpha, beta)
        # assemble world positions of mesh points for both ends
        P_plus  = center_m + (self.mesh.p_local_plus @ np.stack([e1,e2,e3],axis=1))
        P_minus = center_m + (self.mesh.p_local_minus @ np.stack([e1,e2,e3],axis=1))
        # sigma at ends
        sigma_plus  =  M0*scale * self.mesh.sigma_sign_plus
        sigma_minus =  M0*scale * self.mesh.sigma_sign_minus
        dS = self.mesh.dS
        Npts = dS.size

        N = sensors_m.shape[0]
        B = np.zeros((N,3))
        # Jacobian per sensor wrt 7 params / 每个传感器对7参的雅可比
        Jall = np.zeros((N,3,7))

        # helper to accumulate one end / 处理单个端面
        def acc_end(P_end: np.ndarray, sigma_sign: float):
            nonlocal B, Jall
            for i in range(N):
                s = sensors_m[i]
                # center derivatives enters through r only / 中心参数只通过r
                JB_c = np.zeros((3,3))
                JB_da = np.zeros(3)  # wrt alpha
                JB_db = np.zeros(3)  # wrt beta
                Bi = np.zeros(3)
                for k in range(Npts):
                    r = s - P_end[k]
                    f, Jr = kernel_and_J(r)
                    w = MU0/(4*np.pi) * (M0*scale*sigma_sign) * dS[k]
                    Bi += w * f
                    # center derivative: dr/dc = -I => dB/dc = -Jr * w
                    JB_c += - w * Jr
                    # orientation: P_end depends on alpha,beta via e1,e2,e3
                    xL, yL, zL = (self.mesh.p_local_plus if sigma_sign>0 else self.mesh.p_local_minus)[k]
                    # dP/dα = de1*x + de2*y + de3*z
                    dP_da = de1_da*xL + de2_da*yL + de3_da*zL
                    dP_db = de1_db*xL + de2_db*yL + de3_db*zL
                    # dr/dα = -dP/dα ; dB/dα = Jr @ dr/dα * w
                    JB_da += w * (Jr @ (-dP_da))
                    JB_db += w * (Jr @ (-dP_db))
                B[i] += Bi
                # fill Jacobian blocks
                Jall[i,:,0:3] += JB_c
                # chain rule to u_raw via α,β
                J_u = np.outer(JB_da, dalpha_duraw) + np.outer(JB_db, dbeta_duraw)
                Jall[i,:,3:6] += J_u
                # scale derivative is linear / 对scale线性
                Jall[i,:,6] += (sigma_sign * MU0/(4*np.pi) * M0) * \
                               np.sum([kernel_and_J(s - P_end[k])[0]*dS[k] for k in range(Npts)], axis=0)

        acc_end(P_plus, +1.0)
        acc_end(P_minus, -1.0)
        return B, Jall

# -------------------- Sensors IO / 传感器输入 --------------------
def load_sensors_from_csv(path: str) -> np.ndarray:
    """CSV with columns x,y,z in meters. / 以米为单位的x,y,z三列"""
    import pandas as pd
    df = pd.read_csv(path)
    return df[['x','y','z']].to_numpy(dtype=float)

def sensors_from_list(lst: List[Tuple[float,float,float]]) -> np.ndarray:
    return np.array(lst, dtype=float)

# -------------------- Gauss-Newton pose estimation / 位姿估计 --------------------
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
        H = Jbig.T@Jbig + lam*np.diag([0,0,0,1,1,1,0])
        g = Jbig.T@r
        dp = -np.linalg.solve(H, g)
        p += dp
        rmse = float(np.sqrt(np.mean(r*r)))
        if rmse < tol_T:
            break

    # pack result
    u_hat = p[3:6]/np.linalg.norm(p[3:6])
    return Pose(center=p[0:3].copy(), u_dir=u_hat, scale=float(p[6]), rmse=rmse)

# -------------------- Example run / 演示 --------------------
if __name__ == "__main__":
    # geometry
    geom = CylinderGeom(Br=1.35, R=0.004, l=0.005)  # example / 示例
    # sensors: arbitrary positions / 任意传感器
    sensors = sensors_from_list([
        [0.03, 0.01, 0.02],
        [0.04, 0.01, 0.02],
        [0.05, 0.01, 0.02],
        [0.03, 0.02, 0.03],
        [0.04, 0.02, 0.03],
        [0.05, 0.02, 0.03],
    ])
    # synthetic measurement from true pose / 用真值生成合成观测
    true_center = np.array([0.01, 0.01, 0.012])
    true_u = np.array([0.0, 0.0, 1.0])
    true_scale = 1.0
    cfg = ModelConfig(geom=geom, disc=Discretization())
    fwd = ForwardModel(cfg)
    B_true, _ = fwd.compute_B_and_J(sensors, true_center, true_u, true_scale)
    # add small noise / 加微噪声
    B_meas = B_true + 1e-8*np.random.randn(*B_true.shape)

    # initial guess / 初值
    est = estimate_pose(
        sensors, B_meas, geom,
        u_init=np.array([0.0,0.0,0.9]),
        c_init=np.array([0.009,0.010,0.011]),
        scale_init=0.9,
        max_iter=12,
        tol_T=4e-8
    )
    print("center_m:", est.center.tolist())
    print("u_dir:", est.u_dir.tolist())
    print("scale:", float(est.scale))
    print("rmse_T:", float(est.rmse))



