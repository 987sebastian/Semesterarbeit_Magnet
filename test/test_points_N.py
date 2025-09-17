from scripts.Geometry import CylinderGeom
from scripts.Discretization import Discretization
from scripts.ForwardModel import ForwardModel, ModelConfig
from scripts.PoseEstimation import estimate_pose
import numpy as np

if __name__ == "__main__":
    # geometry
    geom = CylinderGeom(Br=1.2, R=0.005, l=0.004)  # example / 示例
    # sensors: arbitrary positions / 任意传感器
    sensors = np.array([
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
    B_meas = B_true + 1e-8 * np.random.randn(*B_true.shape)

    # initial guess / 初值
    est = estimate_pose(
        sensors, B_meas, geom,
        u_init=np.array([0.0, 0.0, 0.9]),
        c_init=np.array([0.009, 0.010, 0.011]),
        scale_init=0.9,
        max_iter=12,
        tol_T=4e-8
    )
    print("center_m:", est.center.tolist())
    print("u_dir:", est.u_dir.tolist())
    print("scale:", float(est.scale))
    print("rmse_T:", float(est.rmse))
