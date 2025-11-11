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

    std::vector<Eigen::Vector3d> BiotSavartModel::compute_B_batch(
        const std::vector<Eigen::Vector3d>& sensors,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& u_hat,
        double scale
    ) const {
        std::vector<Eigen::Vector3d> result(sensors.size());
        for (size_t i = 0; i < sensors.size(); ++i) {
            result[i] = compute_B(sensors[i], center, u_hat, scale);
        }
        return result;
    }

    // NEW: Compute B with gradient
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

        for (size_t i = 0; i < mesh_.dS.size(); ++i) {
            Eigen::Vector3d p_plus_world = center + R * mesh_.p_local_plus[i];
            Eigen::Vector3d p_minus_world = center + R * mesh_.p_local_minus[i];

            Eigen::Vector3d r_plus = sensor - p_plus_world;
            Eigen::Vector3d r_minus = sensor - p_minus_world;

            // Field
            result.B += MU_0_OVER_4PI * sigma * mesh_.dS[i] *
                (kernel(r_plus) - kernel(r_minus));

            // Gradient: ∂B/∂p = -∂B/∂r
            result.dB_dp -= MU_0_OVER_4PI * sigma * mesh_.dS[i] *
                (kernel_gradient(r_plus) - kernel_gradient(r_minus));
        }

        // Note: dB_du is computed numerically in optimizer
        result.dB_du = Eigen::Matrix3d::Zero();
        result.dB_dscale = Eigen::Vector3d::Zero();

        return result;
    }

    // NEW: Batch with gradients
    std::vector<BiotResult> BiotSavartModel::compute_B_batch_with_gradients(
        const std::vector<Eigen::Vector3d>& sensors,
        const Eigen::Vector3d& center,
        const Eigen::Vector3d& u_hat,
        double scale
    ) const {
        std::vector<BiotResult> results(sensors.size());

#ifdef USE_OPENMP
#pragma omp parallel for if(sensors.size() > 4)
#endif
        for (int i = 0; i < static_cast<int>(sensors.size()); ++i) {
            results[i] = compute_B_with_gradient(sensors[i], center, u_hat, scale);
        }

        return results;
    }

} // namespace biot