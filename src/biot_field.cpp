// biot_field.cpp
// Biot-Savart model with analytical gradient computation
#include "biot_field.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace biot {

    // Magnetic constant
    static constexpr double MU_0 = 4.0 * M_PI * 1e-7;
    static constexpr double MU_0_OVER_4PI = 1e-7;

    // NEW HELPER: Skew-symmetric matrix operator for cross product
    Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d& v) {
        Eigen::Matrix3d m;
        m << 0, -v(2), v(1),
            v(2), 0, -v(0),
            -v(1), v(0), 0;
        return m;
    }

    // Build mesh
    EndDisks::EndDisks(const CylinderGeom& geom, const DiscGrid& disc) {
        build(geom, disc);
    }

    void EndDisks::build(const CylinderGeom& geom, const DiscGrid& disc) {
        const double R = geom.R;
        const double L = geom.L;
        const double h = L / 2.0;

        p_local_plus.clear();
        p_local_minus.clear();
        dS.clear();

        for (int ir = 0; ir < disc.Nr; ++ir) {
            double r_inner = (ir == 0) ? 0.0 : R * ir / disc.Nr;
            double r_outer = R * (ir + 1) / disc.Nr;
            double r_mid = 0.5 * (r_inner + r_outer);
            double dA = M_PI * (r_outer * r_outer - r_inner * r_inner) / disc.Nth;

            for (int ith = 0; ith < disc.Nth; ++ith) {
                double theta = 2.0 * M_PI * ith / disc.Nth;
                double x = r_mid * std::cos(theta);
                double y = r_mid * std::sin(theta);

                p_local_plus.push_back(Eigen::Vector3d(x, y, h));
                p_local_minus.push_back(Eigen::Vector3d(x, y, -h));
                dS.push_back(dA);
            }
        }
    }

    BiotSavartModel::BiotSavartModel(const CylinderGeom& geom, const DiscGrid& disc)
        : geom_(geom), mesh_(geom, disc)
    {
    }

    void BiotSavartModel::build_frame(
        const Eigen::Vector3d& u_hat,
        Eigen::Vector3d& e1,
        Eigen::Vector3d& e2,
        Eigen::Vector3d& e3
    ) const {
        e3 = u_hat;
        e1 = (std::abs(e3(0)) < 0.9) ? Eigen::Vector3d(1, 0, 0) : Eigen::Vector3d(0, 1, 0);
        e2 = e3.cross(e1).normalized();
        e1 = e2.cross(e3).normalized();
    }

    // Kernel: K(r) = r / |r|^3
    Eigen::Vector3d BiotSavartModel::kernel(const Eigen::Vector3d& r) const {
        double r_norm = r.norm();
        if (r_norm < 1e-12) return Eigen::Vector3d::Zero();
        return r / (r_norm * r_norm * r_norm);
    }

    // NEW: Kernel gradient ∂K/∂r = I/|r|^3 - 3(r⊗r^T)/|r|^5
    Eigen::Matrix3d BiotSavartModel::kernel_gradient(const Eigen::Vector3d& r) const {
        double r_norm = r.norm();
        if (r_norm < 1e-12) return Eigen::Matrix3d::Zero();

        double r3 = r_norm * r_norm * r_norm;
        double r5 = r3 * r_norm * r_norm;

        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d rrt = r * r.transpose();

        return I / r3 - 3.0 * rrt / r5;
    }

    // 单点计算 B 场（保留原实现）
    Eigen::Vector3d BiotSavartModel::compute_B(
        const Eigen::Vector3d& sensor,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& u_hat,
        double scale
    ) const {
        Eigen::Vector3d e1, e2, e3;
        build_frame(u_hat, e1, e2, e3);

        Eigen::Matrix3d R;
        R.col(0) = e1;
        R.col(1) = e2;
        R.col(2) = e3;

        const double sigma = scale * geom_.Br / MU_0;
        Eigen::Vector3d B = Eigen::Vector3d::Zero();

        for (size_t i = 0; i < mesh_.dS.size(); ++i) {
            Eigen::Vector3d p_plus_world = center + R * mesh_.p_local_plus[i];
            Eigen::Vector3d p_minus_world = center + R * mesh_.p_local_minus[i];

            Eigen::Vector3d r_plus = sensor - p_plus_world;
            Eigen::Vector3d r_minus = sensor - p_minus_world;

            B += MU_0_OVER_4PI * sigma * mesh_.dS[i] * (kernel(r_plus) - kernel(r_minus));
        }

        return B;
    }

    // 批量计算 B 场（优化实现：将 R 和 sigma 提到循环外）
    std::vector<Eigen::Vector3d> BiotSavartModel::compute_B_batch(
        const std::vector<Eigen::Vector3d>& sensors,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& u_hat,
        double scale
    ) const {
        std::vector<Eigen::Vector3d> result(sensors.size());

        // --- 优化点：将坐标系和常数构建移到循环外 ---
        Eigen::Vector3d e1, e2, e3;
        build_frame(u_hat, e1, e2, e3);
        Eigen::Matrix3d R;
        R.col(0) = e1;
        R.col(1) = e2;
        R.col(2) = e3;
        const double sigma = scale * geom_.Br / MU_0;
        // ----------------------------------------

#ifdef USE_OPENMP
#pragma omp parallel for if(sensors.size() > 4)
#endif
        for (int i = 0; i < static_cast<int>(sensors.size()); ++i) {
            Eigen::Vector3d B = Eigen::Vector3d::Zero();
            const Eigen::Vector3d& sensor = sensors[i]; // 当前传感器

            for (size_t j = 0; j < mesh_.dS.size(); ++j) {
                Eigen::Vector3d p_plus_world = center + R * mesh_.p_local_plus[j];
                Eigen::Vector3d p_minus_world = center + R * mesh_.p_local_minus[j];

                Eigen::Vector3d r_plus = sensor - p_plus_world;
                Eigen::Vector3d r_minus = sensor - p_minus_world;

                B += MU_0_OVER_4PI * sigma * mesh_.dS[j] * (kernel(r_plus) - kernel(r_minus));
            }
            result[i] = B;
        }

        return result;
    }

    // 单点计算 B 场和梯度（保留原实现）
    BiotResult BiotSavartModel::compute_B_with_gradient(
        const Eigen::Vector3d& sensor,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& u_hat,
        double scale
    ) const {
        BiotResult result;

        Eigen::Vector3d e1, e2, e3;
        build_frame(u_hat, e1, e2, e3);

        Eigen::Matrix3d R;
        R.col(0) = e1;
        R.col(1) = e2;
        R.col(2) = e3;

        const double sigma = scale * geom_.Br / MU_0;
        result.B = Eigen::Vector3d::Zero();
        result.dB_dp = Eigen::Matrix3d::Zero();

        // Initialize the analytical rotation gradient
        result.dB_du = Eigen::Matrix3d::Zero();

        for (size_t i = 0; i < mesh_.dS.size(); ++i) {
            Eigen::Vector3d p_plus_world = center + R * mesh_.p_local_plus[i];
            Eigen::Vector3d p_minus_world = center + R * mesh_.p_local_minus[i];

            Eigen::Vector3d r_plus = sensor - p_plus_world;
            Eigen::Vector3d r_minus = sensor - p_minus_world;

            // Field
            result.B += MU_0_OVER_4PI * sigma * mesh_.dS[i] *
                (kernel(r_plus) - kernel(r_minus));

            // Gradient: ∂B/∂p = -∂B/∂r
            // Use local variables to compute G_plus and G_minus for reuse
            Eigen::Matrix3d G_plus = MU_0_OVER_4PI * sigma * mesh_.dS[i] * kernel_gradient(r_plus);
            Eigen::Matrix3d G_minus = MU_0_OVER_4PI * sigma * mesh_.dS[i] * kernel_gradient(r_minus);

            result.dB_dp -= (G_plus - G_minus);

            // NEW: Analytical Rotation Gradient (G_rot)
            // G_rot = (-G_plus * [r_plus_prime]x) - (-G_minus * [r_minus_prime]x)
            // where r_prime = p_world - center
            Eigen::Vector3d r_plus_prime = p_plus_world - center;
            Eigen::Vector3d r_minus_prime = p_minus_world - center;

            result.dB_du -= G_plus * skew_symmetric(r_plus_prime);
            result.dB_du += G_minus * skew_symmetric(r_minus_prime);
        }

        // NEW: Analytical scale gradient: dB/ds = B / scale
        if (scale > 1e-12) {
            result.dB_dscale = result.B / scale;
        }
        else {
            // Should not happen if scale is initialized to 1.0, but for robustness
            result.dB_dscale.setZero();
        }

        return result;
    }

    // 批量计算 B 场和梯度（优化实现：将 R 和 sigma 提到循环外）
    std::vector<BiotResult> BiotSavartModel::compute_B_batch_with_gradients(
        const std::vector<Eigen::Vector3d>& sensors,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& u_hat,
        double scale
    ) const {
        std::vector<BiotResult> results(sensors.size());

        // --- 优化点：将坐标系和常数构建移到循环外 ---
        Eigen::Vector3d e1, e2, e3;
        build_frame(u_hat, e1, e2, e3);
        Eigen::Matrix3d R;
        R.col(0) = e1;
        R.col(1) = e2;
        R.col(2) = e3;
        const double sigma = scale * geom_.Br / MU_0;
        // ----------------------------------------


#ifdef USE_OPENMP
#pragma omp parallel for if(sensors.size() > 4)
#endif
        for (int i = 0; i < static_cast<int>(sensors.size()); ++i) {
            BiotResult result;
            const Eigen::Vector3d& sensor = sensors[i]; // 当前传感器

            result.B = Eigen::Vector3d::Zero();
            result.dB_dp = Eigen::Matrix3d::Zero();
            result.dB_du = Eigen::Matrix3d::Zero();

            for (size_t j = 0; j < mesh_.dS.size(); ++j) {
                Eigen::Vector3d p_plus_world = center + R * mesh_.p_local_plus[j];
                Eigen::Vector3d p_minus_world = center + R * mesh_.p_local_minus[j];

                Eigen::Vector3d r_plus = sensor - p_plus_world;
                Eigen::Vector3d r_minus = sensor - p_minus_world;

                // Field
                result.B += MU_0_OVER_4PI * sigma * mesh_.dS[j] *
                    (kernel(r_plus) - kernel(r_minus));

                // Gradient: ∂B/∂p = -∂B/∂r
                Eigen::Matrix3d G_plus = MU_0_OVER_4PI * sigma * mesh_.dS[j] * kernel_gradient(r_plus);
                Eigen::Matrix3d G_minus = MU_0_OVER_4PI * sigma * mesh_.dS[j] * kernel_gradient(r_minus);

                result.dB_dp -= (G_plus - G_minus);

                // Analytical Rotation Gradient (G_rot)
                Eigen::Vector3d r_plus_prime = p_plus_world - center;
                Eigen::Vector3d r_minus_prime = p_minus_world - center;

                result.dB_du -= G_plus * skew_symmetric(r_plus_prime);
                result.dB_du += G_minus * skew_symmetric(r_minus_prime);
            }

            // Analytical scale gradient: dB/ds = B / scale
            if (scale > 1e-12) {
                result.dB_dscale = result.B / scale;
            }
            else {
                result.dB_dscale.setZero();
            }

            results[i] = result;
        }

        return results;
    }

} // namespace biot