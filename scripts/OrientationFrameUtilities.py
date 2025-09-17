import numpy as np
from typing import Tuple


def sph_basis_from_u(u_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Build an orthonormal frame (e_phi, -e_theta, u_hat), plus spherical angles (alpha, beta).
    用单位方向向量构造正交基，并返回球坐标角 α(极角), β(方位角)。
    """
    ux, uy, uz = u_hat
    # angles from unit vector / 由单位向量求角
    alpha = np.arccos(np.clip(uz, -1.0, 1.0))  # polar angle from +z / 与+z的极角
    beta = np.arctan2(uy, ux)  # azimuth / 方位角
    # spherical basis / 球坐标基
    e_r = u_hat
    e_theta = np.array([np.cos(alpha) * np.cos(beta),
                        np.cos(alpha) * np.sin(beta),
                        -np.sin(alpha)])
    e_phi = np.array([-np.sin(beta), np.cos(beta), 0.0])
    # Orthonormal frame for disk local xyz -> world / 将磁体局部xyz映射到世界
    # R = [e1 e2 e3] with e1=e_phi, e2=-e_theta, e3=e_r
    return e_phi, -e_theta, e_r, alpha, beta


def d_basis_dalpha(alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derivatives of (e1,e2,e3) wrt alpha. / 对α的导数"""
    # from spherical basis derivatives / 基于球基导数
    e_r = np.array([np.sin(alpha) * np.cos(beta),
                    np.sin(alpha) * np.sin(beta),
                    np.cos(alpha)])
    e_theta = np.array([np.cos(alpha) * np.cos(beta),
                        np.cos(alpha) * np.sin(beta),
                        -np.sin(alpha)])
    e_phi = np.array([-np.sin(beta), np.cos(beta), 0.0])
    # derivatives:
    de_r_dalpha = e_theta
    de_theta_dalpha = -e_r
    de_phi_dalpha = np.array([0.0, 0.0, 0.0])
    # map to (e1,e2,e3) = (e_phi, -e_theta, e_r)
    de1 = de_phi_dalpha
    de2 = -de_theta_dalpha
    de3 = de_r_dalpha
    return de1, de2, de3


def d_basis_dbeta(alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derivatives of (e1,e2,e3) wrt beta. / 对β的导数"""
    e_r = np.array([np.sin(alpha) * np.cos(beta),
                    np.sin(alpha) * np.sin(beta),
                    np.cos(alpha)])
    e_theta = np.array([np.cos(alpha) * np.cos(beta),
                        np.cos(alpha) * np.sin(beta),
                        -np.sin(alpha)])
    e_phi = np.array([-np.sin(beta), np.cos(beta), 0.0])
    # derivatives:
    de_r_dbeta = np.sin(alpha) * e_phi
    de_theta_dbeta = np.cos(alpha) * e_phi
    de_phi_dbeta = -np.sin(alpha) * e_r - np.cos(alpha) * e_theta
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
    d_uhat_duraw = (I - np.outer(u_hat, u_hat)) / nr
    # ∂α/∂u_hat = -1/sinα * ∂uz/∂u_hat = [0,0,-1]/sinα
    sa = max(np.sin(alpha), 1e-12)
    dalpha_duhat = np.array([0.0, 0.0, -1.0]) / sa
    # ∂β/∂u_hat = 1/sin^2α * [ -uy, ux, 0 ]
    s2 = max(sa * sa, 1e-12)
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
    r2 = rx * rx + ry * ry + rz * rz
    inv_r = 1.0 / np.sqrt(r2 + 1e-30)
    inv_r3 = inv_r ** 3
    f = r * inv_r3
    # Jacobian
    I = np.eye(3)
    rrT = np.outer(r, r)
    J = (I * r2 - 3.0 * rrT) * (inv_r ** 5)
    return f, J
