// pose_optimizer.cpp
// Practical optimizer with analytical gradients
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

        // Compute B field with full analytical gradients / 计算磁场及完整解析梯度
        // All gradients (position, direction, scale) are now analytical
        // 所有梯度(位置、方向、尺度)现在都是解析的
        std::vector<BiotResult> results = model_.compute_B_batch_with_gradients(
            sensors, p, u_norm, scale
        );

        for (size_t i = 0; i < n_sensors; ++i) {
            residuals.segment<3>(i * 3) = results[i].B - B_meas[i];

            // Block 1: Analytical position gradient / 解析位置梯度
            jacobian.block<3, 3>(i * 3, 0) = results[i].dB_dp;

            // Block 2: Analytical direction gradient (G_rot) / 解析方向梯度
            jacobian.block<3, 3>(i * 3, 3) = results[i].dB_du;

            // Block 3: Analytical scale gradient / 解析尺度梯度
            jacobian.block<3, 1>(i * 3, 6) = results[i].dB_dscale;
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
        // NOTE: This function is kept for compatibility but is NOT used in the main optimizer
        // 注意: 此函数保留用于兼容性,但在主优化器中未使用
        const double eps = config::numerical::FINITE_DIFF_EPS;
        size_t n_sensors = sensors.size();
        J.resize(n_sensors * 3, 7);

        // Position gradients / 位置梯度
        for (int j = 0; j < 3; ++j) {
            Eigen::Vector3d p_plus = p;
            p_plus(j) += eps;
            std::vector<Eigen::Vector3d> B_plus = model_.compute_B_batch(sensors, p_plus, u, scale);
            for (size_t i = 0; i < n_sensors; ++i) {
                J.block<3, 1>(i * 3, j) = (B_plus[i] - B_pred[i]) / eps;
            }
        }

        // Direction gradients / 方向梯度
        for (int j = 0; j < 3; ++j) {
            Eigen::Vector3d u_plus = u;
            u_plus(j) += eps;
            u_plus.normalize();
            std::vector<Eigen::Vector3d> B_plus = model_.compute_B_batch(sensors, p, u_plus, scale);
            for (size_t i = 0; i < n_sensors; ++i) {
                J.block<3, 1>(i * 3, 3 + j) = (B_plus[i] - B_pred[i]) / eps;
            }
        }

        // Scale gradient / 尺度梯度
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

        // Levenberg-Marquardt parameters / LM算法参数
        double lambda = config::optimizer::LAMBDA_INIT;
        const double lambda_up = config::optimizer::LAMBDA_UP;
        const double lambda_down = config::optimizer::LAMBDA_DOWN;
        const double lambda_max = config::optimizer::LAMBDA_MAX;
        const double lambda_min = config::optimizer::LAMBDA_MIN;

        const double trust_radius = config::optimizer::TRUST_RADIUS;
        const double min_step_quality = config::optimizer::MIN_STEP_QUALITY;

        // Tracking variables / 跟踪变量
        double best_rmse = std::numeric_limits<double>::max();
        Eigen::Vector3d best_p = p;
        Eigen::Vector3d best_u = u;
        double best_scale = scale;
        int no_improvement_count = 0;

        double rmse_prev = std::numeric_limits<double>::max();
        int converged_count = 0;

        // Position drift monitoring / 位置漂移监控
        const double max_position_drift = 0.015;  // 15mm max drift
        double initial_rmse = std::numeric_limits<double>::max();

        // Main optimization loop / 主优化循环
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

            // CRITICAL: Check position drift from initial guess / 关键:检查位置漂移
            double position_drift = (p - p_init).norm();
            if (position_drift > max_position_drift && iter > 3) {
                p = best_p;
                u = best_u;
                scale = best_scale;
                result.converged = true;
                result.status = 6;  // Position drift limit
                break;
            }

            // Tighter practical convergence
            const double practical_tolerance = 2.5e-5;

            if (rmse < 5.0e-6) {
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

            // Early stop if RMSE not improving much / 改进不足提前停止
            if (iter >= 3) {
                double improvement = (initial_rmse - rmse) / initial_rmse;
                if (improvement < 0.1 && rmse > practical_tolerance) {
                    result.converged = true;
                    result.status = 2;
                    break;
                }
            }

            // Check improvement rate / 检查改进率
            double improvement_rate = (rmse_prev - rmse) / (rmse_prev + 1e-12);

            // Track best solution / 跟踪最佳解
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

            // Early stopping / 提前停止
            if (iter >= 6) {
                // Stop if minimal improvement and already good / 改进很小且已经足够好
                if (rmse < 1.0e-6 && improvement_rate < 1e-3) {
                    result.converged = true;
                    result.status = 3;
                    break;
                }

                // Stop if no improvement for 3 iterations
                if (no_improvement_count >= 3) {
                    result.converged = true;
                    result.status = 4;
                    break;
                }
            }

            rmse_prev = rmse;

            // Levenberg-Marquardt step / LM算法步骤
            Eigen::MatrixXd JtJ = J.transpose() * J;
            Eigen::VectorXd Jtr = J.transpose() * residuals;
            Eigen::MatrixXd A = JtJ + lambda * Eigen::MatrixXd::Identity(7, 7);
            Eigen::VectorXd delta = A.ldlt().solve(-Jtr);

            // Trust region constraint / 信赖域约束
            double step_norm = delta.norm();
            if (step_norm > trust_radius) {
                delta *= (trust_radius / step_norm);
            }

            // Apply update / 应用更新
            Eigen::Vector3d p_new = p + delta.segment<3>(0);
            Eigen::Vector3d u_new = (u + delta.segment<3>(3)).normalized();
            double scale_new = std::max(config::optimizer::MIN_SCALE,
                std::min(config::optimizer::MAX_SCALE, scale + delta(6)));

            // Evaluate new cost / 评估新代价
            Eigen::Vector3d u_new_norm = u_new.normalized();
            std::vector<Eigen::Vector3d> B_new = model_.compute_B_batch(sensors, p_new, u_new_norm, scale_new);
            Eigen::VectorXd residuals_new(sensors.size() * 3);
            for (size_t i = 0; i < sensors.size(); ++i) {
                residuals_new.segment<3>(i * 3) = B_new[i] - B_meas[i];
            }
            double cost_new = residuals_new.squaredNorm();

            // Gain ratio / 增益比
            double predicted_reduction = -delta.dot(Jtr) + 0.5 * delta.dot(JtJ * delta);
            double actual_reduction = cost - cost_new;
            double rho = (predicted_reduction > 1e-12)
                ? (actual_reduction / predicted_reduction) : -1.0;

            if (rho > min_step_quality && cost_new < cost) {
                // Accept step / 接受步骤
                p = p_new;
                u = u_new;
                scale = scale_new;

                lambda = (rho > 0.75)
                    ? std::max(lambda * 0.1, lambda_min)
                    : std::max(lambda * lambda_down, lambda_min);
            }
            else {
                // Reject step / 拒绝步骤
                lambda = std::min(lambda * lambda_up, lambda_max);
            }
        }

        // Use best solution found / 使用找到的最佳解
        result.position = best_p;
        result.direction = best_u.normalized();
        result.scale = best_scale;
        result.rmse = best_rmse;

        // Final convergence check / 最终收敛检查
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
