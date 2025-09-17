from dataclasses import dataclass
from .Geometry import CylinderGeom
from .Discretization import Discretization, EndDisks
from .OrientationFrameUtilities import uhat_and_chain, sph_basis_from_u, d_basis_dalpha, d_basis_dbeta, kernel_and_J
import numpy as np
from typing import Tuple


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
        MU0 = 4 * np.pi * 1e-7  # vacuum permeability / 真空磁导率

        geom = self.cfg.geom
        M0 = geom.Br / MU0  # magnetization magnitude at scale=1 / 标称磁化强度
        # orientation frame
        u_hat, alpha, beta, dalpha_duraw, dbeta_duraw = uhat_and_chain(u_raw)
        e1, e2, e3, alpha, beta = sph_basis_from_u(u_hat)
        # basis derivatives
        de1_da, de2_da, de3_da = d_basis_dalpha(alpha, beta)
        de1_db, de2_db, de3_db = d_basis_dbeta(alpha, beta)
        # assemble world positions of mesh points for both ends
        P_plus = center_m + (self.mesh.p_local_plus @ np.stack([e1, e2, e3], axis=1))
        P_minus = center_m + (self.mesh.p_local_minus @ np.stack([e1, e2, e3], axis=1))
        # sigma at ends
        sigma_plus = M0 * scale * self.mesh.sigma_sign_plus
        sigma_minus = M0 * scale * self.mesh.sigma_sign_minus
        dS = self.mesh.dS
        Npts = dS.size

        N = sensors_m.shape[0]
        B = np.zeros((N, 3))
        # Jacobian per sensor wrt 7 params / 每个传感器对7参的雅可比
        Jall = np.zeros((N, 3, 7))

        # helper to accumulate one end / 处理单个端面
        def acc_end(P_end: np.ndarray, sigma_sign: float):
            nonlocal B, Jall
            for i in range(N):
                s = sensors_m[i]
                # center derivatives enters through r only / 中心参数只通过r
                JB_c = np.zeros((3, 3))
                JB_da = np.zeros(3)  # wrt alpha
                JB_db = np.zeros(3)  # wrt beta
                Bi = np.zeros(3)
                for k in range(Npts):
                    r = s - P_end[k]
                    f, Jr = kernel_and_J(r)
                    w = MU0 / (4 * np.pi) * (M0 * scale * sigma_sign) * dS[k]
                    Bi += w * f
                    # center derivative: dr/dc = -I => dB/dc = -Jr * w
                    JB_c += - w * Jr
                    # orientation: P_end depends on alpha,beta via e1,e2,e3
                    xL, yL, zL = (self.mesh.p_local_plus if sigma_sign > 0 else self.mesh.p_local_minus)[k]
                    # dP/dα = de1*x + de2*y + de3*z
                    dP_da = de1_da * xL + de2_da * yL + de3_da * zL
                    dP_db = de1_db * xL + de2_db * yL + de3_db * zL
                    # dr/dα = -dP/dα ; dB/dα = Jr @ dr/dα * w
                    JB_da += w * (Jr @ (-dP_da))
                    JB_db += w * (Jr @ (-dP_db))
                B[i] += Bi
                # fill Jacobian blocks
                Jall[i, :, 0:3] += JB_c
                # chain rule to u_raw via α,β
                J_u = np.outer(JB_da, dalpha_duraw) + np.outer(JB_db, dbeta_duraw)
                Jall[i, :, 3:6] += J_u
                # scale derivative is linear / 对scale线性
                Jall[i, :, 6] += (sigma_sign * MU0 / (4 * np.pi) * M0) * \
                                 np.sum([kernel_and_J(s - P_end[k])[0] * dS[k] for k in range(Npts)], axis=0)

        acc_end(P_plus, +1.0)
        acc_end(P_minus, -1.0)
        return B, Jall
