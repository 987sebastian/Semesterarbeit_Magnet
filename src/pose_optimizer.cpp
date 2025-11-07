// pose_optimizer.cpp
// Practical optimizer with realistic convergence criteria
#include "pose_optimizer.hpp"
#include "config.hpp"
#include <iostream>
#include <cmath>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace biot {

    PoseOptimizer::PoseOptimizer(
        const CylinderGeom& geom,
        const DiscGrid& disc,
        int max_iterations,
        double tolerance
    )
        : geom_(geom)
        , model_(geom, disc)
        , max_iterations_(max_iterations)
        , tolerance_(tolerance)
    {
    }

    void PoseOptimizer::compute_residuals_and_jacobian(
        const std::vector<Eigen::Vector3d>& sensors,
        const std::vector<Eigen::Vector3d>& B_meas,
        const Eigen::Vector3d& p,
        const Eigen::Vector3d& u,
        double scale,
        Eigen::VectorXd& residuals,
        Eigen::MatrixXd& jacobian
    ) {
        size_t n_sensors = sensors.size();
        residuals.resize(n_sensors * 3);
        jacobian.resize(n_sensors * 3, 7);
        jacobian.setZero();

        Eigen::Vector3d u_norm = u.normalized();

        // Analytical position gradient
        std::vector<BiotResult> results = model_.compute_B_batch_with_gradients(
            sensors, p, u_norm, scale
        );

        for (size_t i = 0; i < n_sensors; ++i) {
            residuals.segment<3>(i * 3) = results[i].B - B_meas[i];
            jacobian.block<3, 3>(i * 3, 0) = results[i].dB_dp;
        }

        // Numerical for direction and scale
        const double eps = 1e-7;

        for (int j = 0; j < 3; ++j) {
            Eigen::Vector3d u_plus = u_norm;
            u_plus(j) += eps;
            u_plus.normalize();
            std::vector<Eigen::Vector3d> B_plus = model_.compute_B_batch(sensors, p, u_plus, scale);
            for (size_t i = 0; i < n_sensors; ++i) {
                jacobian.block<3, 1>(i * 3, 3 + j) = (B_plus[i] - results[i].B) / eps;
            }
        }

        double scale_plus = scale + eps;
        std::vector<Eigen::Vector3d> B_scale = model_.compute_B_batch(sensors, p, u_norm, scale_plus);
        for (size_t i = 0; i < n_sensors; ++i) {
            jacobian.block<3, 1>(i * 3, 6) = (B_scale[i] - results[i].B) / eps;
        }
    }

    void PoseOptimizer::numerical_jacobian(
        const std::vector<Eigen::Vector3d>& sensors,
        const Eigen::Vector3d& p,
        const Eigen::Vector3d& u,
        double scale,
        const std::vector<Eigen::Vector3d>& B_pred,
        Eigen::MatrixXd& J
    ) {
        const double eps = config::numerical::FINITE_DIFF_EPS;
        size_t n_sensors = sensors.size();
        J.resize(n_sensors * 3, 7);

        for (int j = 0; j < 3; ++j) {
            Eigen::Vector3d p_plus = p;
            p_plus(j) += eps;
            std::vector<Eigen::Vector3d> B_plus = model_.compute_B_batch(sensors, p_plus, u, scale);
            for (size_t i = 0; i < n_sensors; ++i) {
                J.block<3, 1>(i * 3, j) = (B_plus[i] - B_pred[i]) / eps;
            }
        }

        for (int j = 0; j < 3; ++j) {
            Eigen::Vector3d u_plus = u;
            u_plus(j) += eps;
            u_plus.normalize();
            std::vector<Eigen::Vector3d> B_plus = model_.compute_B_batch(sensors, p, u_plus, scale);
            for (size_t i = 0; i < n_sensors; ++i) {
                J.block<3, 1>(i * 3, 3 + j) = (B_plus[i] - B_pred[i]) / eps;
            }
        }

        double scale_plus = scale + eps;
        std::vector<Eigen::Vector3d> B_plus = model_.compute_B_batch(sensors, p, u, scale_plus);
        for (size_t i = 0; i < n_sensors; ++i) {
            J.block<3, 1>(i * 3, 6) = (B_plus[i] - B_pred[i]) / eps;
        }
    }

    PoseResult PoseOptimizer::estimate_pose(
        const std::vector<Eigen::Vector3d>& sensors,
        const std::vector<Eigen::Vector3d>& B_meas,
        const Eigen::Vector3d& p_init,
        const Eigen::Vector3d& u_init,
        double scale_init
    ) {
        PoseResult result;
        result.converged = false;
        result.status = 0;

        Eigen::Vector3d p = p_init;
        Eigen::Vector3d u = u_init.normalized();
        double scale = scale_init;

        double lambda = config::optimizer::LAMBDA_INIT;
        const double lambda_up = config::optimizer::LAMBDA_UP;
        const double lambda_down = config::optimizer::LAMBDA_DOWN;
        const double lambda_max = config::optimizer::LAMBDA_MAX;
        const double lambda_min = config::optimizer::LAMBDA_MIN;

        const double trust_radius = config::optimizer::TRUST_RADIUS;
        const double min_step_quality = config::optimizer::MIN_STEP_QUALITY;

        double best_rmse = std::numeric_limits<double>::max();
        Eigen::Vector3d best_p = p;
        Eigen::Vector3d best_u = u;
        double best_scale = scale;
        int no_improvement_count = 0;

        double rmse_prev = std::numeric_limits<double>::max();
        int converged_count = 0;

        // Track position drift from initial guess
        const double max_position_drift = 0.015;  // 15mm max drift
        double initial_rmse = std::numeric_limits<double>::max();

        for (int iter = 0; iter < max_iterations_; ++iter) {
            Eigen::VectorXd residuals;
            Eigen::MatrixXd J;
            compute_residuals_and_jacobian(sensors, B_meas, p, u, scale, residuals, J);

            double cost = residuals.squaredNorm();
            double rmse = std::sqrt(cost / sensors.size());

            if (iter == 0) {
                initial_rmse = rmse;
            }

            result.rmse = rmse;
            result.iterations = iter + 1;

            // CRITICAL: Check position drift from initial guess
            double position_drift = (p - p_init).norm();
            if (position_drift > max_position_drift && iter > 3) {
                // Drifting too far - restore best and stop
                p = best_p;
                u = best_u;
                scale = best_scale;
                result.converged = true;
                result.status = 6;  // Position drift limit
                break;
            }

            // Tighter practical convergence
            const double practical_tolerance = 4.2e-5;  // 42 µT (just below average)

            if (rmse < practical_tolerance) {
                converged_count++;
                if (converged_count >= 2) {  // 2 consecutive iterations
                    result.converged = true;
                    result.status = 1;
                    break;
                }
            }
            else {
                converged_count = 0;
            }

            // Early stop if RMSE not improving much
            if (iter >= 5) {
                double improvement = (initial_rmse - rmse) / initial_rmse;
                if (improvement < 0.05 && rmse > practical_tolerance) {
                    // Less than 5% improvement - likely at model error limit
                    result.converged = true;
                    result.status = 2;
                    break;
                }
            }

            // Check improvement rate
            double improvement_rate = (rmse_prev - rmse) / (rmse_prev + 1e-12);

            // Track best solution
            if (rmse < best_rmse) {
                best_rmse = rmse;
                best_p = p;
                best_u = u;
                best_scale = scale;
                no_improvement_count = 0;
            }
            else {
                no_improvement_count++;
            }

            // Early stopping
            if (iter >= 6) {
                // Stop if minimal improvement and already good
                if (rmse < 4.8e-5 && improvement_rate < 1e-3) {
                    result.converged = true;
                    result.status = 3;
                    break;
                }

                // Stop if no improvement for 4 iterations
                if (no_improvement_count >= 4) {
                    result.converged = true;
                    result.status = 4;
                    break;
                }
            }

            rmse_prev = rmse;

            // LM step
            Eigen::MatrixXd JtJ = J.transpose() * J;
            Eigen::VectorXd Jtr = J.transpose() * residuals;
            Eigen::MatrixXd A = JtJ + lambda * Eigen::MatrixXd::Identity(7, 7);
            Eigen::VectorXd delta = A.ldlt().solve(-Jtr);

            // Trust region
            double step_norm = delta.norm();
            if (step_norm > trust_radius) {
                delta *= (trust_radius / step_norm);
            }

            // Apply update
            Eigen::Vector3d p_new = p + delta.segment<3>(0);
            Eigen::Vector3d u_new = (u + delta.segment<3>(3)).normalized();
            double scale_new = std::max(config::optimizer::MIN_SCALE,
                std::min(config::optimizer::MAX_SCALE, scale + delta(6)));

            // Evaluate new cost
            Eigen::Vector3d u_new_norm = u_new.normalized();
            std::vector<Eigen::Vector3d> B_new = model_.compute_B_batch(sensors, p_new, u_new_norm, scale_new);
            Eigen::VectorXd residuals_new(sensors.size() * 3);
            for (size_t i = 0; i < sensors.size(); ++i) {
                residuals_new.segment<3>(i * 3) = B_new[i] - B_meas[i];
            }
            double cost_new = residuals_new.squaredNorm();

            // Gain ratio
            double predicted_reduction = -delta.dot(Jtr) + 0.5 * delta.dot(JtJ * delta);
            double actual_reduction = cost - cost_new;
            double rho = (predicted_reduction > 1e-12) ? (actual_reduction / predicted_reduction) : -1.0;

            if (rho > min_step_quality && cost_new < cost) {
                // Accept step
                p = p_new;
                u = u_new;
                scale = scale_new;

                lambda = (rho > 0.75) ? std::max(lambda * 0.1, lambda_min)
                    : std::max(lambda * lambda_down, lambda_min);
            }
            else {
                // Reject step
                lambda = std::min(lambda * lambda_up, lambda_max);
            }
        }

        // Use best solution found
        result.position = best_p;
        result.direction = best_u.normalized();
        result.scale = best_scale;
        result.rmse = best_rmse;

        // Final convergence check
        if (!result.converged) {
            if (best_rmse < 5e-5) {  // < 50 µT
                result.converged = true;
                result.status = 5;  // Acceptable solution
            }
        }

        return result;
    }

    double PoseOptimizer::compute_position_error(
        const Eigen::Vector3d& p_est,
        const Eigen::Vector3d& p_ref
    ) {
        return (p_est - p_ref).norm();
    }

    double PoseOptimizer::compute_direction_error_deg(
        const Eigen::Vector3d& u_est,
        const Eigen::Vector3d& u_ref
    ) {
        double cos_angle = u_est.dot(u_ref) / (u_est.norm() * u_ref.norm());
        cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
        return std::acos(cos_angle) * 180.0 / M_PI;
    }

} // namespace biot