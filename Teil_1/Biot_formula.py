import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# ===================== SYMPY PART: Symbolic derivation ===================== 符号推导部分
Js, mu0, R, l = sp.symbols('Js mu0 R l')  # Magnetization [A/m], vacuum permeability, radius [m], length [m] 磁化强度、真空磁导率、半径、长度
x, y, z = sp.symbols('x y z')  # Observation point 观测点
xc, yc, zc = sp.symbols('xc yc zc')  # Magnet center 磁体中心
a, b, c = sp.symbols('a b c')  # Magnetization direction components 磁化方向分量
theta, z0 = sp.symbols('theta z0')  # Parameters 参数变量

# Unit vectors and coordinate transformation 单位向量与坐标变换
r_obs = sp.Matrix([x, y, z])  # Observation point vector 观测点向量
r_center = sp.Matrix([xc, yc, zc])  # Center vector 中心向量
z_axis = sp.Matrix([a, b, c]) / sp.sqrt(a**2 + b**2 + c**2)  # Normalized magnetization direction 归一化磁化方向

temp = sp.Matrix([0, 0, 1])  # Reference z-axis 参考z轴
x_axis = (temp.cross(z_axis)).normalized() if not z_axis.equals(temp) else sp.Matrix([1, 0, 0])  # Orthogonal x-axis 正交x轴
y_axis = z_axis.cross(x_axis)  # Orthogonal y-axis 正交y轴
Rmat = sp.Matrix.hstack(x_axis, y_axis, z_axis)  # Rotation matrix 旋转矩阵

# Surface element position and current element 面元位置和电流元
r_local = sp.Matrix([R * sp.cos(theta), R * sp.sin(theta), z0])  # Local surface point 局部面元坐标
dl_local = R * sp.diff(theta) * sp.Matrix([-sp.sin(theta), sp.cos(theta), 0])  # Local current element 局部电流元
r_global = r_center + Rmat @ r_local  # Transform to global coords 转换到全局坐标
dl_global = Rmat @ dl_local  # Transform current element to global coords 电流元转换到全局

# Biot–Savart law
r_vec = r_obs - r_global  # Vector from element to observation point 元素到观测点的向量
r_norm = sp.sqrt(r_vec.dot(r_vec))  # Distance 距离
dB = mu0 * Js / (4 * sp.pi) * (dl_global.cross(r_vec)) / r_norm**3  # Differential magnetic field 微分磁场公式

print("\nSymbolic expression for differential magnetic field dB:")  # 输出微分磁场符号表达式
sp.pprint(dB)

# ===================== NUMERICAL PART: Vectorized computation ===================== 数值计算部分
μ0 = 4 * np.pi * 1e-7  # Vacuum permeability 真空磁导率
R = 4e-3  # Radius [m] 半径
l = 5e-3  # Length [m] 长度
Js = 1.35 / (4 * np.pi * 1e-7)  # Magnetization [A/m] 磁化强度
center = np.array([10e-3, 10e-3, 12e-3])  # Magnet center 磁体中心

def unit_vector(v):  # Normalize vector 单位化向量
    return v / np.linalg.norm(v)

def orthonormal_basis(z_axis):  # Create orthonormal basis from z-axis 从z轴生成正交归一基
    z = unit_vector(z_axis)
    if np.allclose(z, [0, 0, 1]):
        x = np.array([1, 0, 0])
    else:
        x = unit_vector(np.cross([0, 0, 1], z))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)

def magnetic_field_cylinder_vectorized(R, l, Js, x, y, z, alpha, beta, gamma, N_theta=30, N_z=20):  # Compute B-field via Biot–Savart (vectorized) 使用毕奥-萨伐尔定律计算圆柱磁场（向量化）
    pts = np.stack([x, y, z], axis=-1)  # Observation points 观测点集合
    Rmat = orthonormal_basis(np.array([alpha, beta, gamma]))  # Rotation matrix 旋转矩阵
    theta_vals = np.linspace(0, 2*np.pi, N_theta)  # Azimuth angles 方位角
    z_vals = np.linspace(-l/2, l/2, N_z)  # z positions z方向坐标
    dtheta = theta_vals[1] - theta_vals[0]  # Δtheta 步长
    dz = z_vals[1] - z_vals[0]  # Δz 步长
    Theta, Z0 = np.meshgrid(theta_vals, z_vals, indexing='ij')  # Mesh grid 网格
    x_surf = R * np.cos(Theta)  # Surface x 坐标
    y_surf = R * np.sin(Theta)  # Surface y 坐标
    z_surf = Z0  # Surface z 坐标
    r_local = np.stack([x_surf, y_surf, z_surf], axis=-1)  # Local coords 局部坐标
    dl_local = R * dtheta * np.stack([
        -np.sin(Theta), np.cos(Theta), np.zeros_like(Theta)
    ], axis=-1)  # Local current elements 局部电流元
    r_global = center + np.tensordot(r_local, Rmat.T, axes=1)  # Transform to global 全局转换
    dl_global = np.tensordot(dl_local, Rmat.T, axes=1)  # Current elements in global coords 全局电流元
    B_total = np.zeros_like(pts)  # Initialize B field 磁场初始化
    for i in range(r_global.shape[0]):
        for j in range(r_global.shape[1]):
            r_vec = pts - r_global[i, j]  # Vector from element to obs point 元素到观测点的向量
            r_norm = np.linalg.norm(r_vec, axis=1)  # Distance 距离
            dB = μ0 * Js / (4 * np.pi) * np.cross(
                np.broadcast_to(dl_global[i, j], r_vec.shape), r_vec) / r_norm[:, None]**3  # Biot–Savart formula 毕奥-萨伐尔公式
            B_total += dB * dz  # Integrate along z 方向积分
    return B_total[:, 0], B_total[:, 1], B_total[:, 2]  # Return Bx, By, Bz 返回分量

def make_line(start, end, N=200):  # Create line points 创建路径点
    return np.linspace(start, end, N)

paths = {  # Measurement paths 测量路径
    'X0':  (np.array([19, 10, 14]), np.array([49, 10, 14])),
    'X5':  (np.array([19, 10, 19]), np.array([49, 10, 19])),
    'Z0':  (np.array([10, 10, 18]), np.array([10, 10, 48])),
    'Z5':  (np.array([15, 10, 18]), np.array([15, 10, 48])),
    'Random': (np.array([15, 15, 0]), np.array([25, 25, 20])),
    'Y5':  (np.array([10, 19, 19]), np.array([10, 49, 19]))
}

results = {}  # Store results 存储结果
for name, (start_mm, end_mm) in paths.items():
    start = start_mm * 1e-3  # Convert mm to m 毫米转米
    end = end_mm * 1e-3
    line = make_line(start, end)  # Generate path points 生成路径点
    x, y, z = line[:, 0], line[:, 1], line[:, 2]
    Bx, By, Bz = magnetic_field_cylinder_vectorized(R, l, Js, x, y, z, 0, 0, 1)  # Compute B 计算磁场
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2) * 1e3  # Magnitude in mT 磁场强度[mT]
    dist = np.linalg.norm(line - start, axis=1) * 1e3  # Distance in mm 距离[mm]
    results[name] = (dist, B_mag)

plt.figure(figsize=(10, 6))  # Create figure 创建图形
for name, (dist, Bmag) in results.items():
    plt.plot(dist, Bmag, label=name)  # Plot path 绘制路径曲线
plt.xlabel('Distance along path [mm]')  # 路径距离[毫米]
plt.ylabel('|B| [mT]')  # 磁场强度 [mT]
plt.title('Biot Model: |B| along 6 paths')  # 六条路径的磁场强度
plt.grid(True)  # Show grid 显示网格
plt.legend()  # Show legend 显示图例
plt.tight_layout()  # Adjust layout 调整布局
plt.show()  # Display plot 显示图形
