// dipole_field.cpp
// Magnetic Dipole Model Implementation
// 磁偶极子模型实现

#include "dipole_field.hpp"
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dipole {

    // Magnetic constant μ₀ / 磁导率常数
    static constexpr double MU_0 = 4.0 * M_PI * 1e-7;  // T·m/A
    static constexpr double MU_0_OVER_4PI = 1e-7;      // T·m/A

    DipoleModel::DipoleModel(const biot::CylinderGeom& geom)
        : geom_(geom)
    {
        // Compute base dipole moment: m = V * Br / μ₀
        // 计算基准磁矩：m = 体积 × 剩磁 / μ₀
        double volume = M_PI * geom_.R * geom_.R * geom_.L;
        m_base_ = volume * geom_.Br / MU_0;
    }

    double DipoleModel::compute_moment() const {
        return m_base_;
    }

    Eigen::Vector3d DipoleModel::dipole_kernel(
        const Eigen::Vector3d& r,
        const Eigen::Vector3d& m
    ) const {
        // B(r) = (μ₀/4π) * [3(m·r̂)r̂ - m] / r³
        // 偶极子磁场公式

        double r_mag = r.norm();
        if (r_mag < 1e-10) {
            return Eigen::Vector3d::Zero();  // Avoid singularity / 避免奇点
        }

        Eigen::Vector3d r_hat = r / r_mag;
        double m_dot_r_hat = m.dot(r_hat);
        double r3 = r_mag * r_mag * r_mag;

        // B = (μ₀/4π) * [3(m·r̂)r̂ - m] / r³
        Eigen::Vector3d B = MU_0_OVER_4PI * (3.0 * m_dot_r_hat * r_hat - m) / r3;

        return B;
    }

    Eigen::Vector3d DipoleModel::compute_B(
        const Eigen::Vector3d& sensor,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& u_hat,
        double scale
    ) const {
        // Position vector from dipole center to sensor
        // 从偶极子中心到传感器的位置矢量
        Eigen::Vector3d r = sensor - center;

        // Magnetic moment vector: m = m_base * scale * u_hat
        // 磁矩矢量：m = 基准磁矩 × 尺度 × 方向
        Eigen::Vector3d m = m_base_ * scale * u_hat;

        return dipole_kernel(r, m);
    }

    std::vector<Eigen::Vector3d> DipoleModel::compute_B_batch(
        const std::vector<Eigen::Vector3d>& sensors,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& u_hat,
        double scale
    ) const {
        std::vector<Eigen::Vector3d> B_fields(sensors.size());

        Eigen::Vector3d m = m_base_ * scale * u_hat;

        for (size_t i = 0; i < sensors.size(); ++i) {
            Eigen::Vector3d r = sensors[i] - center;
            B_fields[i] = dipole_kernel(r, m);
        }

        return B_fields;
    }

    Eigen::Matrix3d DipoleModel::compute_dB_dp(
        const Eigen::Vector3d& r,
        const Eigen::Vector3d& m
    ) const {
        // ∂B/∂p = -∂B/∂r (因为 r = sensor - center)
        // Gradient of dipole field w.r.t. position
        // 偶极子场相对于位置的梯度

        double r_mag = r.norm();
        if (r_mag < 1e-10) {
            return Eigen::Matrix3d::Zero();
        }

        Eigen::Vector3d r_hat = r / r_mag;
        double m_dot_r_hat = m.dot(r_hat);
        double r3 = r_mag * r_mag * r_mag;
        double r5 = r3 * r_mag * r_mag;

        // ∂B/∂r = (μ₀/4π) * ∂/∂r [3(m·r̂)r̂ - m] / r³
        // Using product rule and chain rule
        // 使用乘法法则和链式法则

        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

        // ∂r̂/∂r = (I - r̂⊗r̂) / r
        Eigen::Matrix3d dr_hat_dr = (I - r_hat * r_hat.transpose()) / r_mag;

        // ∂(m·r̂)/∂r = m^T * ∂r̂/∂r
        Eigen::Vector3d d_mdot_dr = dr_hat_dr.transpose() * m;

        // ∂B/∂r = (μ₀/4π) * {[3·∂(m·r̂)/∂r·r̂ + 3(m·r̂)·∂r̂/∂r] / r³ - [3(m·r̂)r̂ - m]·3/r⁴·∂r/∂r}

        Eigen::Matrix3d term1 = 3.0 * (d_mdot_dr * r_hat.transpose() + m_dot_r_hat * dr_hat_dr) / r3;
        Eigen::Matrix3d term2 = -3.0 * (3.0 * m_dot_r_hat * r_hat - m) * r_hat.transpose() / r5;

        Eigen::Matrix3d dB_dr = MU_0_OVER_4PI * (term1 + term2);

        // ∂B/∂p = -∂B/∂r (because r = sensor - center)
        return -dB_dr;
    }

    Eigen::Matrix3d DipoleModel::compute_dB_du(
        const Eigen::Vector3d& r,
        const Eigen::Vector3d& u_hat
    ) const {
        // ∂B/∂u for unit direction vector
        // B = (μ₀/4π) * [3(m·r̂)r̂ - m] / r³, where m = m_base * scale * u_hat
        // ∂B/∂u = (μ₀/4π) * m_base * scale * [3r̂⊗r̂ - I] / r³

        double r_mag = r.norm();
        if (r_mag < 1e-10) {
            return Eigen::Matrix3d::Zero();
        }

        Eigen::Vector3d r_hat = r / r_mag;
        double r3 = r_mag * r_mag * r_mag;

        // ∂B/∂u = (μ₀/4π) * m_base * scale * [3r̂⊗r̂ - I] / r³
        Eigen::Matrix3d dB_du = MU_0_OVER_4PI * m_base_ *
            (3.0 * r_hat * r_hat.transpose() - Eigen::Matrix3d::Identity()) / r3;

        return dB_du;
    }

    Eigen::Vector3d DipoleModel::compute_dB_dm(
        const Eigen::Vector3d& r,
        const Eigen::Vector3d& u_hat
    ) const {
        // ∂B/∂m_magnitude where m = m_magnitude * u_hat
        // B = (μ₀/4π) * [3(m·r̂)r̂ - m] / r³
        // ∂B/∂m_magnitude = (μ₀/4π) * [3(u_hat·r̂)r̂ - u_hat] / r³

        double r_mag = r.norm();
        if (r_mag < 1e-10) {
            return Eigen::Vector3d::Zero();
        }

        Eigen::Vector3d r_hat = r / r_mag;
        double u_dot_r_hat = u_hat.dot(r_hat);
        double r3 = r_mag * r_mag * r_mag;

        Eigen::Vector3d dB_dm = MU_0_OVER_4PI * (3.0 * u_dot_r_hat * r_hat - u_hat) / r3;

        return dB_dm;
    }

    DipoleResult DipoleModel::compute_B_with_gradients(
        const Eigen::Vector3d& sensor,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& u_hat,
        double scale
    ) const {
        DipoleResult result;

        Eigen::Vector3d r = sensor - center;
        Eigen::Vector3d m = m_base_ * scale * u_hat;

        // Compute field / 计算磁场
        result.B = dipole_kernel(r, m);

        // Compute gradients / 计算梯度
        result.dB_dp = compute_dB_dp(r, m);

        // For direction gradient, need to account for scale
        // ∂B/∂u需要考虑scale因子
        result.dB_du = scale * compute_dB_du(r, u_hat);

        // For moment magnitude gradient
        // 对于磁矩大小的梯度
        result.dB_dm = m_base_ * compute_dB_dm(r, u_hat);

        return result;
    }

    std::vector<DipoleResult> DipoleModel::compute_B_batch_with_gradients(
        const std::vector<Eigen::Vector3d>& sensors,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& u_hat,
        double scale
    ) const {
        std::vector<DipoleResult> results(sensors.size());

        for (size_t i = 0; i < sensors.size(); ++i) {
            results[i] = compute_B_with_gradients(sensors[i], center, u_hat, scale);
        }

        return results;
    }

} // namespace dipole