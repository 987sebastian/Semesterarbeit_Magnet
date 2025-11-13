#pragma once
// dipole_field.hpp
// Magnetic Dipole Model for comparison with Biot-Savart

#include <Eigen/Dense>
#include <vector>
#include "geometry.hpp"

namespace dipole {

    // Dipole field computation result with gradients
    // 偶极子磁场计算结果（带梯度）
    struct DipoleResult {
        Eigen::Vector3d B;               // Magnetic field / 磁场
        Eigen::Matrix3d dB_dp;           // ∂B/∂position / 位置梯度
        Eigen::Matrix3d dB_du;           // ∂B/∂direction / 方向梯度
        Eigen::Vector3d dB_dm;           // ∂B/∂moment magnitude / 磁矩大小梯度
    };

    // Magnetic dipole model
    // 磁偶极子模型
    class DipoleModel {
    public:
        DipoleModel(const biot::CylinderGeom& geom);

        // Compute dipole moment from cylinder geometry
        // 从圆柱几何计算磁矩
        double compute_moment() const;

        // Compute B field at sensor position
        // 计算传感器位置的磁场
        Eigen::Vector3d compute_B(
            const Eigen::Vector3d& sensor,
            const Eigen::Vector3d& center,
            const Eigen::Vector3d& u_hat,
            double scale = 1.0
        ) const;

        // Batch computation
        // 批量计算
        std::vector<Eigen::Vector3d> compute_B_batch(
            const std::vector<Eigen::Vector3d>& sensors,
            const Eigen::Vector3d& center,
            const Eigen::Vector3d& u_hat,
            double scale = 1.0
        ) const;

        // Compute B field with analytical gradients
        // 计算磁场及解析梯度
        DipoleResult compute_B_with_gradients(
            const Eigen::Vector3d& sensor,
            const Eigen::Vector3d& center,
            const Eigen::Vector3d& u_hat,
            double scale = 1.0
        ) const;

        // Batch computation with gradients
        // 批量计算带梯度
        std::vector<DipoleResult> compute_B_batch_with_gradients(
            const std::vector<Eigen::Vector3d>& sensors,
            const Eigen::Vector3d& center,
            const Eigen::Vector3d& u_hat,
            double scale = 1.0
        ) const;

    private:
        biot::CylinderGeom geom_;
        double m_base_;  // Base dipole moment / 基准磁矩

        // Dipole field kernel: B(r) = (μ₀/4π) * [3(m·r̂)r̂ - m] / r³
        // 偶极子场核函数
        Eigen::Vector3d dipole_kernel(
            const Eigen::Vector3d& r,
            const Eigen::Vector3d& m
        ) const;

        // Gradient of dipole field w.r.t. position
        // 偶极子场相对于位置的梯度
        Eigen::Matrix3d compute_dB_dp(
            const Eigen::Vector3d& r,
            const Eigen::Vector3d& m
        ) const;

        // Gradient of dipole field w.r.t. direction
        // 偶极子场相对于方向的梯度
        Eigen::Matrix3d compute_dB_du(
            const Eigen::Vector3d& r,
            const Eigen::Vector3d& u_hat
        ) const;

        // Gradient of dipole field w.r.t. moment magnitude
        // 偶极子场相对于磁矩大小的梯度
        Eigen::Vector3d compute_dB_dm(
            const Eigen::Vector3d& r,
            const Eigen::Vector3d& u_hat
        ) const;
    };

} // namespace dipole