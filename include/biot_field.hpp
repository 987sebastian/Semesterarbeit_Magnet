#pragma once
// biot_field.hpp

#include <Eigen/Dense>
#include <vector>
#include "geometry.hpp"

namespace biot {

    // Structure for field and gradients / 磁场和梯度结果
    struct BFieldResult {
        Eigen::Vector3d B;           // Magnetic field / 磁场
        Eigen::Matrix3d dB_dp;       // ∂B/∂p: gradient w.r.t. position / 位置梯度
        Eigen::Matrix3d dB_du;       // ∂B/∂u: gradient w.r.t. direction / 方向梯度
        Eigen::Vector3d dB_dscale;   // ∂B/∂scale: gradient w.r.t. scale / 尺度梯度
    };

    // End disk mesh data / 端面网格数据
    struct EndDisks {
        std::vector<Eigen::Vector3d> p_local_plus;   // Local positions on +z disk / +z端面局部坐标
        std::vector<Eigen::Vector3d> p_local_minus;  // Local positions on -z disk / -z端面局部坐标
        std::vector<double> dS;                       // Area elements / 面积元

        // Constructor: build mesh / 构造函数：构建网格
        EndDisks(const CylinderGeom& geom, const DiscGrid& disc);

    private:
        void build(const CylinderGeom& geom, const DiscGrid& disc);
    };

    // Biot-Savart forward model with analytical gradients / 带解析梯度的Biot-Savart正演模型
    class BiotSavartModel {
    public:
        BiotSavartModel(const CylinderGeom& geom, const DiscGrid& disc);

        // Compute B field at sensor position / 计算传感器位置的磁场
        Eigen::Vector3d compute_B(
            const Eigen::Vector3d& sensor,
            const Eigen::Vector3d& center,
            const Eigen::Vector3d& u_hat,
            double scale = 1.0
        ) const;

        // Compute B field for multiple sensors / 为多个传感器计算磁场
        std::vector<Eigen::Vector3d> compute_B_batch(
            const std::vector<Eigen::Vector3d>& sensors,
            const Eigen::Vector3d& center,
            const Eigen::Vector3d& u_hat,
            double scale = 1.0
        ) const;

        // Compute B field with analytical gradients / 计算磁场及解析梯度
        BFieldResult compute_B_with_gradients(
            const Eigen::Vector3d& sensor,
            const Eigen::Vector3d& center,
            const Eigen::Vector3d& u_hat,
            double scale = 1.0
        ) const;

        // Batch computation with gradients / 批量计算带梯度
        std::vector<BFieldResult> compute_B_batch_with_gradients(
            const std::vector<Eigen::Vector3d>& sensors,
            const Eigen::Vector3d& center,
            const Eigen::Vector3d& u_hat,
            double scale = 1.0
        ) const;

    private:
        CylinderGeom geom_;
        EndDisks mesh_;

        void build_frame(
            const Eigen::Vector3d& u_hat,
            Eigen::Vector3d& e1,
            Eigen::Vector3d& e2,
            Eigen::Vector3d& e3
        ) const;

        Eigen::Vector3d kernel(const Eigen::Vector3d& r) const;

        // Analytical gradient of kernel / 核函数的解析梯度
        Eigen::Matrix3d kernel_gradient(const Eigen::Vector3d& r) const;

        // Rotation matrix gradients / 旋转矩阵梯度
        void compute_rotation_gradients(
            const Eigen::Vector3d& u_hat,
            const Eigen::Vector3d& e1,
            const Eigen::Vector3d& e2,
            Eigen::Matrix3d dR_du[3]
        ) const;
    };

} // namespace biot