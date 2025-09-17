from dataclasses import dataclass
from .Geometry import CylinderGeom
import numpy as np


@dataclass
class DiscGrid:
    Nr: int = 16  # radial divisions / 径向剖分
    Nth: int = 48  # angular divisions / 角向剖分


@dataclass
class Discretization:
    disc: DiscGrid = None

    def __post_init__(self):
        if self.disc is None:
            self.disc = DiscGrid()


def build_end_disks(geom: CylinderGeom, disc: DiscGrid):
    R = geom.R
    Nr, Nth = disc.Nr, disc.Nth
    r_edges = np.linspace(0.0, R, Nr + 1)
    th_edges = np.linspace(0.0, 2 * np.pi, Nth + 1)
    rc = 0.5 * (r_edges[:-1] + r_edges[1:])
    thc = 0.5 * (th_edges[:-1] + th_edges[1:])
    dr = np.diff(r_edges)[:, None]
    dth = np.diff(th_edges)[None, :]
    area = (rc[:, None] * dr) @ (np.ones((1, Nth))) * dth
    x = (rc[:, None] * np.cos(thc)[None, :]).reshape(-1)
    y = (rc[:, None] * np.sin(thc)[None, :]).reshape(-1)
    dS = area.reshape(-1)
    z_plus = np.full_like(x, +geom.l / 2)
    z_minus = np.full_like(x, -geom.l / 2)
    p_plus = np.stack([x, y, z_plus], axis=1)
    p_minus = np.stack([x, y, z_minus], axis=1)
    return p_plus, p_minus, dS


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
        r_edges = np.linspace(0.0, R, Nr + 1)
        th_edges = np.linspace(0.0, 2 * np.pi, Nth + 1)
        rc = 0.5 * (r_edges[:-1] + r_edges[1:])
        thc = 0.5 * (th_edges[:-1] + th_edges[1:])
        dr = np.diff(r_edges)[:, None]
        dth = np.diff(th_edges)[None, :]
        # cell area in polar: dS = r dr dθ / 单元面积
        area = (rc[:, None] * dr) @ (np.ones((1, Nth))) * dth
        # centers in Cartesian local frame / 局部笛卡尔坐标的圆盘网格点
        x = (rc[:, None] * np.cos(thc)[None, :]).reshape(-1)
        y = (rc[:, None] * np.sin(thc)[None, :]).reshape(-1)
        dS = area.reshape(-1)
        # Two disks at z = ±l/2 / 两端圆盘
        z_plus = np.full_like(x, +self.geom.l / 2)
        z_minus = np.full_like(x, -self.geom.l / 2)
        # Stack: (Npts,3) for each end / 每端的点
        self.p_local_plus = np.stack([x, y, z_plus], axis=1)
        self.p_local_minus = np.stack([x, y, z_minus], axis=1)
        self.dS = dS
        # Local normals / 局部法向
        self.n_local_plus = np.tile(np.array([0.0, 0.0, 1.0]), (x.size, 1))
        self.n_local_minus = np.tile(np.array([0.0, 0.0, -1.0]), (x.size, 1))
        # Surface magnetic charge densities at ends: ±M / 端面磁荷密度 ±M
        # In local frame, magnetization is along +z (unit), magnitude M = Br/μ0*scale / 当地坐标下磁化沿+z
        # σ_m = M · n_local = ± M / 磁荷密度
        # scale will multiply later / 缩放稍后乘
        self.sigma_sign_plus = +1.0
        self.sigma_sign_minus = -1.0
