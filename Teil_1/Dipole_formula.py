import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ===================== SYMPY PART: Dipole Model ===================== 偶极子模型部分
# Define symbols 定义符号
mu0 = sp.symbols('mu0')                  # vacuum permeability 真空磁导率
mbar = sp.symbols('mbar')                # scaled magnetic moment strength (μ0/(4π)*m) 吸收 μ0/(4π) 常数后的磁矩强度
x, y, z = sp.symbols('x y z')            # observation point coordinates 观测点坐标
xc, yc, zc = sp.symbols('xc yc zc')      # magnet center 磁体中心
a, b, c = sp.symbols('a b c')            # magnetization direction components 磁化方向分量

# Define vectors 定义向量
r_obs = sp.Matrix([x, y, z])             # observation point vector 观测点位置向量
r_center = sp.Matrix([xc, yc, zc])       # magnet center vector 磁体中心位置向量
r_vec = r_obs - r_center                 # displacement from magnet center 位移向量（磁体中心→观测点）
r = sp.sqrt(r_vec.dot(r_vec))            # distance magnitude 位移向量的模

# Normalize magnetization direction 单位化磁化方向
u = sp.Matrix([a, b, c])                 # raw direction vector 原始方向向量
u_hat = u / sp.sqrt(u.dot(u))            # normalized direction vector 归一化方向向量

# Dipole magnetic field formula (mbar form) 偶极子磁场公式（mbar形式）
# B = 3 r (mbar_vec · r) / r^5 - mbar_vec / r^3
mbar_vec = mbar * u_hat                  # scaled magnetic moment vector 吸收常数后的磁矩向量
B_dipole = 3 * r_vec * (mbar_vec.dot(r_vec)) / r**5 - mbar_vec / r**3

# Simplify expression 简化表达式
B_dipole = sp.simplify(B_dipole)

# Print symbolic result 打印符号结果
print("\nSymbolic expression for magnetic dipole field B (mbar form):")  # 输出偶极子磁场公式（mbar形式）的符号表达式
sp.pprint(B_dipole, use_unicode=True)

# -------------------- Constants and magnet parameters -------------------- 常量与磁体参数
μ0 = 4 * np.pi * 1e-7               # vacuum permeability 真空磁导 [H/m]
R = 4e-3                            # cylinder radius 圆柱半径 [m]
l = 5e-3                            # cylinder length 圆柱长度 [m]
Js = 1.35 / (4 * np.pi * 1e-7)      # magnetization strength 磁化强度近似 [A/m]
center = np.array([10e-3, 10e-3, 12e-3])  # magnet center 磁体中心 [m]
u_dir = np.array([0.0, 0.0, 1.0])         # magnetization unit direction 磁化方向单位向量（此处沿 +z）

# Magnetic moment m = Js * V * û 偶极矩 m = Js * V * û
V = np.pi * R**2 * l  # magnet volume 磁体体积
m_vec = Js * V * (u_dir / np.linalg.norm(u_dir))  # magnetic moment vector 偶极矩向量

# -------------------- Dipole model: vectorized B computation -------------------- 偶极子模型：向量化计算 B
def B_dipole_at_points(points_m, center_m, m_vec):
    """
    points_m: (N,3) observation points [m] 观测点（米）
    center_m: (3,)  magnet center 磁体中心
    m_vec:    (3,)  magnetic moment [A·m^2] 偶极矩 [A·m^2]
    return:   (N,3) B vector [T] 磁感应强度矢量 [T]
    """
    r = points_m - center_m[None, :]  # displacement vectors 位移向量
    r_norm = np.linalg.norm(r, axis=1)  # distances 距离
    # numerical stability 数值稳定
    eps = 1e-20
    r_norm = np.maximum(r_norm, eps)
    r3 = r_norm**3
    r5 = r_norm**5

    mdotr = r @ m_vec                  # dot product m·r 内积 (N,)
    term1 = 3.0 * r * (mdotr / r5)[:, None]  # first term 第一项
    term2 = m_vec[None, :] / r3[:, None]     # second term 第二项
    B = (μ0 / (4.0 * np.pi)) * (term1 - term2)  # dipole formula 偶极子公式
    return B

# -------------------- Construct 6 paths -------------------- 构造6条路径
def make_line(start_mm, end_mm, N=200):
    start = np.asarray(start_mm, dtype=float) * 1e-3  # convert to m 转换为米
    end = np.asarray(end_mm, dtype=float) * 1e-3
    t = np.linspace(0.0, 1.0, N)[:, None]  # interpolation parameter 插值参数
    return start[None, :] * (1 - t) + end[None, :] * t  # return (N,3) m 返回路径点（米）

paths_mm = {  # path definitions 路径定义
    'X0':    (np.array([19, 10, 14]), np.array([49, 10, 14])),
    'X5':    (np.array([19, 10, 19]), np.array([49, 10, 19])),
    'Z0':    (np.array([10, 10, 18]), np.array([10, 10, 48])),
    'Z5':    (np.array([15, 10, 18]), np.array([15, 10, 48])),
    'Random':(np.array([15, 15,  0]), np.array([25, 25, 20])),
    'Y5':    (np.array([10, 19, 19]), np.array([10, 49, 19])),
}

# -------------------- Compute and plot -------------------- 计算并绘图
plt.figure(figsize=(10, 6))
for name, (p0, p1) in paths_mm.items():
    line_pts = make_line(p0, p1, N=400)             # generate path points 生成路径点
    B = B_dipole_at_points(line_pts, center, m_vec) # compute B field 计算磁场
    Bmag_mT = np.linalg.norm(B, axis=1) * 1e3       # magnitude in mT 磁场强度 [mT]
    dist_mm = np.linalg.norm(line_pts - line_pts[0], axis=1) * 1e3  # distance in mm 路径距离 [mm]
    if name == 'Y5':
        plt.plot(dist_mm, Bmag_mT, label=name, linestyle='--', linewidth=3, color='brown')
    else:
        plt.plot(dist_mm, Bmag_mT, label=name)  # plot curve 绘制曲线

plt.xlabel('Distance along path [mm]')  # 路径距离 [毫米]
plt.ylabel('|B| [mT]')                  # 磁场强度 [mT]
plt.title('Dipole Model: |B| along 6 paths')  # 偶极子模型：6条路径的磁场强度
plt.grid(True)  # show grid 显示网格
plt.legend()    # show legend 显示图例
plt.tight_layout()  # adjust layout 调整布局
plt.show()  # display plot 显示图形

