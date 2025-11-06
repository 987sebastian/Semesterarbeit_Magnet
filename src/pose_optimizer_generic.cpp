// pose_optimizer_generic.cpp
#include "pose_optimizer_generic.hpp"
#include "config.hpp"
#include <cmath>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace generic {

    GenericPoseOptimizer::GenericPoseOptimizer(const MagneticFieldModel* model, int max_iterations, double tolerance)
        : model_(model), max_iterations_(max_iterations), tolerance_(tolerance) {
    }

    void GenericPoseOptimizer::compute_residuals_and_jacobian(
        const std::vector<Eigen::Vector3d>& sensors, const std::vector<Eigen::Vector3d>& B_meas,
        const Eigen::Vector3d& p, const Eigen::Vector3d& u, double scale,
        Eigen::VectorXd& residuals, Eigen::MatrixXd& jacobian
    ) {
        std::vector<Eigen::Vector3d> B_fields;
        model_->compute_fields_with_gradients(sensors, p, u.normalized(), scale, B_fields, jacobian);
        residuals.resize(sensors.size() * 3);
        for (size_t i = 0; i < sensors.size(); ++i) {
            residuals.segment<3>(i * 3) = B_fields[i] - B_meas[i];
        }
    }

    PoseResult GenericPoseOptimizer::estimate_pose(
        const std::vector<Eigen::Vector3d>& sensors, const std::vector<Eigen::Vector3d>& B_meas,
        const Eigen::Vector3d& p_init, const Eigen::Vector3d& u_init, double scale_init
    ) {
        PoseResult result;
        result.converged = false;
        result.status = 0;

        Eigen::Vector3d p = p_init, u = u_init.normalized();
        double scale = scale_init;
        double lambda = biot::config::optimizer::LAMBDA_INIT;

        double best_rmse = std::numeric_limits<double>::max();
        Eigen::Vector3d best_p = p, best_u = u;
        double best_scale = scale;
        int no_improvement = 0;
        double initial_rmse = 0;

        for (int iter = 0; iter < max_iterations_; ++iter) {
            Eigen::VectorXd residuals;
            Eigen::MatrixXd J;
            compute_residuals_and_jacobian(sensors, B_meas, p, u, scale, residuals, J);

            double rmse = std::sqrt(residuals.squaredNorm() / sensors.size());
            if (iter == 0) initial_rmse = rmse;
            result.rmse = rmse;
            result.iterations = iter + 1;

            // Position drift check
            if ((p - p_init).norm() > 0.015 && iter > 3) {
                p = best_p; u = best_u; scale = best_scale;
                result.converged = true; result.status = 6; break;
            }

            // Convergence
            if (rmse < 4.2e-5) {
                result.converged = true; result.status = 1; break;
            }

            if (iter >= 5 && (initial_rmse - rmse) / initial_rmse < 0.05) {
                result.converged = true; result.status = 2; break;
            }

            // Track best
            if (rmse < best_rmse) {
                best_rmse = rmse; best_p = p; best_u = u; best_scale = scale; no_improvement = 0;
            }
            else {
                no_improvement++;
            }

            if (no_improvement >= 4 && iter >= 6) {
                result.converged = true; result.status = 4; break;
            }

            // LM step
            Eigen::MatrixXd JtJ = J.transpose() * J;
            Eigen::VectorXd Jtr = J.transpose() * residuals;
            Eigen::VectorXd delta = (JtJ + lambda * Eigen::MatrixXd::Identity(7, 7)).ldlt().solve(-Jtr);

            if (delta.norm() > biot::config::optimizer::TRUST_RADIUS) {
                delta *= biot::config::optimizer::TRUST_RADIUS / delta.norm();
            }

            Eigen::Vector3d p_new = p + delta.segment<3>(0);
            Eigen::Vector3d u_new = (u + delta.segment<3>(3)).normalized();
            double scale_new = std::max(0.01, std::min(10.0, scale + delta(6)));

            Eigen::VectorXd res_new;
            Eigen::MatrixXd J_new;
            compute_residuals_and_jacobian(sensors, B_meas, p_new, u_new, scale_new, res_new, J_new);

            double cost = residuals.squaredNorm(), cost_new = res_new.squaredNorm();
            double pred = -delta.dot(Jtr) + 0.5 * delta.dot(JtJ * delta);
            double rho = (pred > 1e-12) ? ((cost - cost_new) / pred) : -1.0;

            if (rho > biot::config::optimizer::MIN_STEP_QUALITY && cost_new < cost) {
                p = p_new; u = u_new; scale = scale_new;
                lambda = (rho > 0.75) ? (std::max)(lambda * 0.1, biot::config::optimizer::LAMBDA_MIN)
                    : std::max(lambda * biot::config::optimizer::LAMBDA_DOWN, biot::config::optimizer::LAMBDA_MIN);
            }
            else {
                lambda = std::min(lambda * biot::config::optimizer::LAMBDA_UP, biot::config::optimizer::LAMBDA_MAX);
            }
        }

        result.position = best_p;
        result.direction = best_u.normalized();
        result.scale = best_scale;
        result.rmse = best_rmse;

        if (!result.converged && best_rmse < 5e-5) {
            result.converged = true;
            result.status = 5;
        }

        return result;
    }

    double GenericPoseOptimizer::compute_position_error(const Eigen::Vector3d& p_est, const Eigen::Vector3d& p_ref) {
        return (p_est - p_ref).norm();
    }

    double GenericPoseOptimizer::compute_direction_error_deg(const Eigen::Vector3d& u_est, const Eigen::Vector3d& u_ref) {
        double cos_angle = std::max(-1.0, std::min(1.0, u_est.dot(u_ref) / (u_est.norm() * u_ref.norm())));
        return std::acos(cos_angle) * 180.0 / M_PI;
    }

} // namespace generic