// biot_field.hpp
// Biot-Savart surface charge model with gradient computation
#pragma once

#include <Eigen/Dense>
#include <vector>
#include "geometry.hpp"

namespace biot {

    // End disk mesh data
    struct EndDisks {
        std::vector<Eigen::Vector3d> p_local_plus;
        std::vector<Eigen::Vector3d> p_local_minus;
        std::vector<double> dS;

        EndDisks(const CylinderGeom& geom, const DiscGrid& disc);

    private:
        void build(const CylinderGeom& geom, const DiscGrid& disc);
    };

    // Result with gradient information
    struct BiotResult {
        Eigen::Vector3d B;          // Magnetic field [T]
        Eigen::Matrix3d dB_dp;      // Gradient w.r.t. position (3x3)
        Eigen::Matrix3d dB_du;      // Gradient w.r.t. direction (3x3, approximate)
        Eigen::Vector3d dB_dscale;  // Gradient w.r.t. scale (3x1)
    };

    // Biot-Savart forward model
    class BiotSavartModel {
    public:
        BiotSavartModel(const CylinderGeom& geom, const DiscGrid& disc);

        // Compute B field at sensor position
        Eigen::Vector3d compute_B(
            const Eigen::Vector3d& sensor,
            const Eigen::Vector3d& center,
            const Eigen::Vector3d& u_hat,
            double scale = 1.0
        ) const;

        // Compute B field for multiple sensors
        std::vector<Eigen::Vector3d> compute_B_batch(
            const std::vector<Eigen::Vector3d>& sensors,
            const Eigen::Vector3d& center,
            const Eigen::Vector3d& u_hat,
            double scale = 1.0
        ) const;

        // NEW: Compute B field and gradient w.r.t. position
        BiotResult compute_B_with_gradient(
            const Eigen::Vector3d& sensor,
            const Eigen::Vector3d& center,
            const Eigen::Vector3d& u_hat,
            double scale = 1.0
        ) const;

        // NEW: Batch computation with gradients
        std::vector<BiotResult> compute_B_batch_with_gradients(
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

        // NEW: Kernel gradient ∂K/∂r
        Eigen::Matrix3d kernel_gradient(const Eigen::Vector3d& r) const;
    };

} // namespace biot