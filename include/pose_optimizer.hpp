#pragma once
// pose_optimizer.hpp
// Pose optimization using Gauss-Newton

#include <Eigen/Dense>
#include <vector>
#include "biot_field.hpp"
#include "geometry.hpp"

namespace biot {

    // Pose estimation result
    struct PoseResult {
        Eigen::Vector3d position;    // Estimated magnet center
        Eigen::Vector3d direction;   // Estimated unit direction
        double scale;                 // Estimated scale factor
        double rmse;                  // Root mean square error (cost)
        int iterations;               // Number of iterations (nfev)
        bool converged;               // Convergence flag
        int status;                   // Status: 1=converged, 0=not converged
    };

    // Pose optimizer
    class PoseOptimizer {
    public:
        PoseOptimizer(
            const CylinderGeom& geom,
            const DiscGrid& disc = DiscGrid(),
            int max_iterations = 20,
            double tolerance = 1e-8
        );

        // Estimate pose from sensor measurements
        // sensors: sensor positions [m]
        // B_meas: measured B fields [T]
        // p_init: initial position guess
        // u_init: initial direction guess (need not be unit)
        // scale_init: initial scale guess
        PoseResult estimate_pose(
            const std::vector<Eigen::Vector3d>& sensors,
            const std::vector<Eigen::Vector3d>& B_meas,
            const Eigen::Vector3d& p_init,
            const Eigen::Vector3d& u_init,
            double scale_init = 1.0
        );

        // Compute position and direction errors
        static double compute_position_error(
            const Eigen::Vector3d& p_est,
            const Eigen::Vector3d& p_ref
        );

        static double compute_direction_error_deg(
            const Eigen::Vector3d& u_est,
            const Eigen::Vector3d& u_ref
        );

    private:
        CylinderGeom geom_;
        BiotSavartModel model_;
        int max_iterations_;
        double tolerance_;

        // Compute residuals and Jacobian
        void compute_residuals_and_jacobian(
            const std::vector<Eigen::Vector3d>& sensors,
            const std::vector<Eigen::Vector3d>& B_meas,
            const Eigen::Vector3d& p,
            const Eigen::Vector3d& u,
            double scale,
            Eigen::VectorXd& residuals,
            Eigen::MatrixXd& jacobian
        );

        // Numerical Jacobian using finite differences
        void numerical_jacobian(
            const std::vector<Eigen::Vector3d>& sensors,
            const Eigen::Vector3d& p,
            const Eigen::Vector3d& u,
            double scale,
            const std::vector<Eigen::Vector3d>& B_pred,
            Eigen::MatrixXd& J
        );
    };


} // namespace biot